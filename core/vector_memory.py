# vector_memory.py - 完整增强版
# -*- coding: utf-8 -*-
"""
增强型向量记忆系统

特性：
- 语义嵌入检索（支持本地 Sentence-Transformers 或降级 TF-IDF 哈希）
- 时间衰减 + 重要性加权
- 重复检测（余弦相似度 > 0.95 跳过添加）
- MMR 多样性去重
- 分层存储（工作记忆 / 长期记忆）
- 智能压缩（基于聚类摘要）
- 工具调用链保留
- 自动重要性评分（基于关键词）
"""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


@dataclass
class MemoryEntry:
    """结构化记忆条目"""
    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
        }


class EmbeddingProvider:
    """嵌入提供者接口（支持多种后端）"""

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地轻量级嵌入（使用 sentence-transformers 或简化版 TF-IDF 哈希）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_cache_size: int = 512):
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.max_cache_size = max_cache_size
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ 使用 SentenceTransformer 模型: {model_name}")
        except Exception as e:
            # 降级到 TF-IDF 哈希
            self.model = None
            self.dimension = 384
            print(f"⚠️ SentenceTransformer 不可用，使用降级哈希嵌入: {e}")

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        emb = self._cache.get(text)
        if emb is None:
            return None
        self._cache.move_to_end(text)
        return emb.copy()

    def _store_cached_embedding(self, text: str, emb: np.ndarray) -> None:
        self._cache[text] = emb.copy()
        self._cache.move_to_end(text)
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_positions: Dict[str, List[int]] = {}

        for idx, text in enumerate(texts):
            normalized = text or ""
            cached = self._get_cached_embedding(normalized)
            if cached is not None:
                results[idx] = cached
            else:
                missing_positions.setdefault(normalized, []).append(idx)

        missing_texts = list(missing_positions.keys())

        if self.model:
            embeddings = self.model.encode(missing_texts, convert_to_numpy=True) if missing_texts else []
            for text, emb in zip(missing_texts, embeddings):
                self._store_cached_embedding(text, emb)
                for pos in missing_positions[text]:
                    results[pos] = emb.copy()
        else:
            for text in missing_texts:
                emb = self._simple_embed(text)
                self._store_cached_embedding(text, emb)
                for pos in missing_positions[text]:
                    results[pos] = emb.copy()

        return [emb if emb is not None else np.zeros(self.dimension) for emb in results]

    def _simple_embed(self, text: str) -> np.ndarray:
        """简化的哈希嵌入（无外部依赖时）"""
        vec = np.zeros(self.dimension)
        tokens = text.lower().split()
        for i, token in enumerate(tokens[:50]):  # 限制 token 数
            for j in range(3):  # 3个哈希位置
                hash_val = int(hashlib.md5(f"{token}_{j}".encode()).hexdigest(), 16)
                idx = hash_val % self.dimension
                vec[idx] += 1 / (i + 1)  # 位置加权
        # L2归一化
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class VectorMemory:
    """
    增强型向量记忆系统

    特性：
    - 语义嵌入检索
    - 时间衰减 + 重要性加权
    - 工具调用链保留
    - 分层存储（工作记忆/长期记忆）
    - 智能压缩（聚类摘要）
    - 自动重要性评分
    """

    # 重要性关键词权重
    _IMPORTANCE_KEYWORDS: Dict[str, float] = {
        # 高重要性：错误/决策/结论
        "错误": 0.3, "error": 0.3, "失败": 0.25, "成功": 0.2, "解决": 0.25,
        "结论": 0.3, "总结": 0.25, "原因": 0.2, "修复": 0.25, "修改": 0.2,
        "问题": 0.15, "结果": 0.15, "发现": 0.2, "注意": 0.2, "重要": 0.25,
        # 中等重要性：操作/技术
        "读取": 0.1, "写入": 0.1, "执行": 0.1, "运行": 0.1, "分析": 0.15,
    }

    def __init__(
            self,
            memory_dir: str = ".agent_memory",
            embedding_provider: Optional[EmbeddingProvider] = None,
            max_working_memory: int = 10,      # 工作记忆容量
            max_long_term_memory: int = 1000,   # 长期记忆容量
            enable_compression: bool = True
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # 嵌入提供者
        self.embedder = embedding_provider or LocalEmbeddingProvider()

        # 分层存储
        self.working_memory: List[MemoryEntry] = []   # 近期对话
        self.long_term_memory: List[MemoryEntry] = [] # 压缩后的历史

        self.max_working = max_working_memory
        self.max_long_term = max_long_term_memory
        self.enable_compression = enable_compression

        # 工具调用链索引
        self.tool_chains: Dict[str, List[str]] = {}  # task_id -> [memory_entry_ids]

        self._dirty = False  # 标记是否有未持久化的变更
        self._revision = 0
        self._search_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._search_cache_size = 128

        # 加载已有数据
        self._load_from_disk()

    def _invalidate_search_cache(self) -> None:
        self._revision += 1
        self._search_cache.clear()

    @staticmethod
    def _search_cache_key(
        query: str,
        top_k: int,
        recency_weight: float,
        importance_weight: float,
        semantic_weight: float,
        filter_metadata: Optional[Dict],
        min_score: float,
        mmr_lambda: float,
        keyword_boost: float,
        revision: int,
    ) -> str:
        payload = {
            "query": query,
            "top_k": top_k,
            "recency_weight": round(recency_weight, 4),
            "importance_weight": round(importance_weight, 4),
            "semantic_weight": round(semantic_weight, 4),
            "filter_metadata": filter_metadata or {},
            "min_score": round(min_score, 4),
            "mmr_lambda": round(mmr_lambda, 4),
            "keyword_boost": round(keyword_boost, 4),
            "revision": revision,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _touch_search_results(self, result_ids: List[str]) -> None:
        if not result_ids:
            return
        now = datetime.now()
        id_set = set(result_ids)
        for entry in self.working_memory + self.long_term_memory:
            if entry.id in id_set:
                entry.access_count += 1
                entry.last_accessed = now

    # ========== 辅助方法 ==========
    @staticmethod
    def _extract_keywords(content: str, top_n: int = 8) -> List[str]:
        """简单关键词提取：去停用词后按词频取 top_n。"""
        import re as _re
        stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
            "着", "没有", "看", "好", "自己", "这", "那", "user", "assistant",
            "tool", "action", "thought", "observation", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "it", "in", "to", "of",
        }
        words = _re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9_]{2,}', content.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]

    @staticmethod
    def _auto_importance(content: str, base: float = 0.5) -> float:
        """根据内容关键词自动调整重要性评分（在 base 基础上叠加）。"""
        score = base
        cl = content.lower()
        for kw, weight in VectorMemory._IMPORTANCE_KEYWORDS.items():
            if kw in cl:
                score = min(1.0, score + weight)
        # 内容越长，基础重要性略高
        length_bonus = min(0.15, len(content) / 2000)
        return min(1.0, score + length_bonus)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _is_duplicate(self, content: str, threshold: float = 0.95, precomputed_emb=None) -> tuple:
        """检测新内容是否与工作记忆中已有内容高度重复。

        Returns:
            (is_dup: bool, embedding: np.ndarray)
        """
        emb = precomputed_emb if precomputed_emb is not None else self.embedder.embed([content])[0]
        if not self.working_memory:
            return False, emb
        # 只检测最近 5 条
        for entry in self.working_memory[-5:]:
            sim = self._cosine_similarity(emb, entry.embedding)
            if sim >= threshold:
                return True, emb
        return False, emb

    # ========== 添加记忆 ==========
    def add(
            self,
            content: str,
            metadata: Dict[str, Any] = None,
            importance: float = 0.5,
            tool_chain_id: Optional[str] = None,
            auto_score: bool = True,
            skip_duplicate: bool = True,
            original_question: str = None,
    ) -> str:
        """
        添加记忆

        Args:
            content: 记忆内容
            metadata: 元数据（包含role, timestamp等）
            importance: 重要性基础分(0-1)，auto_score=True 时会自动上调
            tool_chain_id: 所属工具链ID
            auto_score: 是否根据内容关键词自动调整重要性
            skip_duplicate: 是否跳过与近期记忆高度相似的重复内容
            original_question: 原始用户提问，用于上下文检索时提供完整背景

        Returns:
            记忆条目ID，若跳过重复则返回 "dup_skip"
        """
        # 一次性计算 embedding
        embedding = self.embedder.embed([content])[0]

        # 去重检测
        if skip_duplicate:
            is_dup, _ = self._is_duplicate(content, precomputed_emb=embedding)
            if is_dup:
                return "dup_skip"

        # 重要性自动评分
        if auto_score:
            importance = self._auto_importance(content, base=importance)

        entry_id = hashlib.md5(
            f"{content}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # 提取关键词存入 metadata
        keywords = self._extract_keywords(content)
        final_meta = dict(metadata or {})
        if keywords:
            final_meta["keywords"] = keywords
        if original_question:
            final_meta["original_question"] = original_question

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            metadata=final_meta,
            importance_score=importance
        )

        # 添加到工作记忆
        self.working_memory.append(entry)

        # 关联工具链
        if tool_chain_id:
            if tool_chain_id not in self.tool_chains:
                self.tool_chains[tool_chain_id] = []
            self.tool_chains[tool_chain_id].append(entry_id)

        # 触发压缩
        if len(self.working_memory) > self.max_working:
            self._compress_working_memory()
            self._save_to_disk()
        else:
            self._dirty = True
        self._invalidate_search_cache()

        return entry_id

    # ========== 检索 ==========
    def search(
            self,
            query: str,
            top_k: int = 5,
            recency_weight: float = 0.3,
            importance_weight: float = 0.2,
            semantic_weight: float = 0.5,
            filter_metadata: Optional[Dict] = None,
            min_score: float = 0.0,
            mmr_lambda: float = 0.6,
            keyword_boost: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """混合检索：语义相似度 + 时间衰减 + 重要性 + 关键词匹配 + MMR 去重

        Args:
            query: 查询文本
            top_k: 返回结果数
            recency_weight: 时间权重
            importance_weight: 重要性权重
            semantic_weight: 语义权重
            filter_metadata: 元数据过滤条件（所有 k/v 均须匹配）
            min_score: 综合分数下限
            mmr_lambda: MMR 多样性参数（0=最大多样性，1=最大相关性）
            keyword_boost: 查询关键词与记忆 keywords 命中时的加分
        """
        if not self.working_memory and not self.long_term_memory:
            return []
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        cache_key = self._search_cache_key(
            query=normalized_query,
            top_k=top_k,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            semantic_weight=semantic_weight,
            filter_metadata=filter_metadata,
            min_score=min_score,
            mmr_lambda=mmr_lambda,
            keyword_boost=keyword_boost,
            revision=self._revision,
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            self._search_cache.move_to_end(cache_key)
            self._touch_search_results([item["id"] for item in cached])
            return [dict(item) for item in cached]

        query_emb = self.embedder.embed([normalized_query])[0]
        all_entries = self.working_memory + self.long_term_memory

        # 提取查询关键词
        query_keywords = set(self._extract_keywords(normalized_query, top_n=6))

        now = datetime.now()
        scored_entries = []

        for entry in all_entries:
            # 元数据过滤
            if filter_metadata:
                if not all(entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # 语义相似度
            semantic_score = self._cosine_similarity(query_emb, entry.embedding)

            # 时间衰减（指数衰减，48小时半衰期）
            time_diff = (now - entry.timestamp).total_seconds() / 3600
            recency_score = np.exp(-time_diff / 48)

            # 访问频率分数
            access_score = min(entry.access_count / 10, 1.0)

            # 关键词命中加分
            mem_keywords = set(entry.metadata.get("keywords", []))
            kw_hit = len(query_keywords & mem_keywords)
            kw_score = min(1.0, kw_hit * keyword_boost)

            # 综合分数
            total_score = (
                semantic_weight * semantic_score
                + recency_weight * recency_score
                + importance_weight * entry.importance_score
                + 0.05 * access_score
                + kw_score
            )

            if total_score >= min_score:
                scored_entries.append((entry, total_score, semantic_score))

        if not scored_entries:
            return []

        # MMR 去重
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = scored_entries[: min(top_k * 3, 50)]

        selected: List[tuple] = []
        selected_embs: List[np.ndarray] = []

        while len(selected) < top_k and candidate_pool:
            best_idx, best_score = -1, -1e9
            for idx, (entry, score, sem_score) in enumerate(candidate_pool):
                if selected_embs:
                    max_sim = max(
                        self._cosine_similarity(entry.embedding, s_emb)
                        for s_emb in selected_embs
                    )
                else:
                    max_sim = 0.0
                mmr_score = mmr_lambda * score - (1 - mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, idx

            if best_idx == -1:
                break
            chosen = candidate_pool.pop(best_idx)
            selected.append(chosen)
            selected_embs.append(chosen[0].embedding)

        results = []
        for entry, score, _ in selected:
            entry.access_count += 1
            entry.last_accessed = now
            results.append({
                "id": entry.id,
                "content": entry.content,
                "score": round(score, 4),
                "metadata": entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
                "keywords": entry.metadata.get("keywords", []),
            })

        self._search_cache[cache_key] = [dict(item) for item in results]
        self._search_cache.move_to_end(cache_key)
        while len(self._search_cache) > self._search_cache_size:
            self._search_cache.popitem(last=False)

        return results

    def search_by_types(
            self,
            query: str,
            types: List[str],
            top_k: int = 5,
            min_score: float = 0.0,
            **kwargs,
    ) -> List[Dict[str, Any]]:
        """按记忆类型依次检索并聚合结果，避免不同类型记忆互相污染。"""
        aggregated: List[Dict[str, Any]] = []
        seen_ids = set()
        for memory_type in types:
            results = self.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filter_metadata={"type": memory_type},
                **kwargs,
            )
            for item in results:
                item_id = item.get("id")
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                aggregated.append(item)
                if len(aggregated) >= top_k:
                    return aggregated
        return aggregated

    # ========== 工具链查询 ==========
    def get_tool_chain(self, chain_id: str) -> List[Dict]:
        """获取完整工具调用链"""
        entry_ids = self.tool_chains.get(chain_id, [])
        chain = []

        all_entries = {e.id: e for e in self.working_memory + self.long_term_memory}

        for eid in entry_ids:
            if eid in all_entries:
                entry = all_entries[eid]
                chain.append({
                    "id": entry.id,
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "metadata": entry.metadata
                })

        return chain

    # ========== 压缩 ==========
    def _compress_working_memory(self):
        """压缩工作记忆到长期记忆"""
        if not self.enable_compression:
            self.working_memory = self.working_memory[-self.max_working:]
            return

        to_compress = self.working_memory[:-self.max_working]
        self.working_memory = self.working_memory[-self.max_working:]

        compressed = self._cluster_compress(to_compress)
        self.long_term_memory.extend(compressed)
        self._invalidate_search_cache()

        # 限制长期记忆大小
        if len(self.long_term_memory) > self.max_long_term:
            self.long_term_memory.sort(
                key=lambda e: e.importance_score * (1 + e.access_count * 0.1),
                reverse=True
            )
            self.long_term_memory = self.long_term_memory[:self.max_long_term]

    def _cluster_compress(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """基于聚类的智能压缩"""
        if len(entries) <= 3:
            return entries

        clusters = []
        used = set()

        for i, entry in enumerate(entries):
            if i in used:
                continue

            cluster = [entry]
            used.add(i)

            for j, other in enumerate(entries[i + 1:], start=i + 1):
                if j in used:
                    continue
                sim = self._cosine_similarity(entry.embedding, other.embedding)
                if sim > 0.8:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) == 1:
                clusters.append(cluster[0])
            else:
                summary_content = self._generate_summary(cluster)
                summary_emb = self.embedder.embed([summary_content])[0]

                # 保留关键元数据
                representative = max(cluster, key=lambda e: e.importance_score)
                inherited_meta = {
                    k: v for k, v in representative.metadata.items()
                    if k in ("type", "role")
                }
                inherited_meta.update({
                    "compressed_cluster": True,
                    "original_count": len(cluster),
                    "original_ids": [e.id for e in cluster],
                })

                clusters.append(MemoryEntry(
                    id=f"cluster_{entry.id}",
                    content=summary_content,
                    embedding=summary_emb,
                    timestamp=cluster[-1].timestamp,
                    metadata=inherited_meta,
                    importance_score=max(e.importance_score for e in cluster)
                ))

        return clusters

    def _generate_summary(self, entries: List[MemoryEntry]) -> str:
        """生成聚类摘要（简化版）"""
        contents = [e.content[:100] for e in entries]
        return f"[压缩 {len(entries)} 条记忆] " + " | ".join(contents[:3])

    # ========== 持久化（优化版：embedding 存为 npz） ==========
    def save_to_disk(self):
        """公共持久化接口"""
        if self._dirty:
            self._save_to_disk()
            self._dirty = False

    def _save_to_disk(self):
        """持久化到磁盘，embedding 单独存为 .npz 文件"""
        # 保存元数据 JSON
        meta_data = {
            "working": [e.to_dict() for e in self.working_memory],
            "long_term": [e.to_dict() for e in self.long_term_memory],
            "tool_chains": self.tool_chains,
        }
        with open(self.memory_dir / "vector_memory_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        # 保存 embedding 为 numpy 压缩文件
        all_ids = []
        all_embeddings = []
        for e in self.working_memory + self.long_term_memory:
            all_ids.append(e.id)
            all_embeddings.append(e.embedding)
        if all_embeddings:
            np.savez_compressed(
                self.memory_dir / "vector_memory_emb.npz",
                ids=np.array(all_ids),
                embeddings=np.array(all_embeddings)
            )

    def _load_from_disk(self):
        """从磁盘加载，兼容旧版单一 JSON 格式"""
        meta_path = self.memory_dir / "vector_memory_meta.json"
        emb_path = self.memory_dir / "vector_memory_emb.npz"
        old_path = self.memory_dir / "vector_memory_v2.json"

        # 优先尝试新格式
        if meta_path.exists() and emb_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                loaded = np.load(emb_path, allow_pickle=True)
                ids = loaded["ids"]
                embs = loaded["embeddings"]
                embeddings_map = {str(i): e for i, e in zip(ids, embs)}

                for e_data in data.get("working", []):
                    entry = MemoryEntry(
                        id=e_data["id"],
                        content=e_data["content"],
                        embedding=embeddings_map.get(e_data["id"], np.zeros(self.embedder.dimension)),
                        timestamp=datetime.fromisoformat(e_data["timestamp"]),
                        metadata=e_data.get("metadata", {}),
                        importance_score=e_data.get("importance_score", 0.5),
                        access_count=e_data.get("access_count", 0)
                    )
                    self.working_memory.append(entry)

                for e_data in data.get("long_term", []):
                    entry = MemoryEntry(
                        id=e_data["id"],
                        content=e_data["content"],
                        embedding=embeddings_map.get(e_data["id"], np.zeros(self.embedder.dimension)),
                        timestamp=datetime.fromisoformat(e_data["timestamp"]),
                        metadata=e_data.get("metadata", {}),
                        importance_score=e_data.get("importance_score", 0.5),
                        access_count=e_data.get("access_count", 0)
                    )
                    self.long_term_memory.append(entry)

                self.tool_chains = data.get("tool_chains", {})
                self._invalidate_search_cache()
                return
            except Exception as e:
                print(f"加载新格式记忆失败，尝试旧格式: {e}")

        # 兼容旧版 JSON（内含 embedding 列表）
        if old_path.exists():
            try:
                with open(old_path, encoding="utf-8") as f:
                    data = json.load(f)
                embeddings_map = data.get("embeddings", {})

                for e_data in data.get("working", []):
                    entry = MemoryEntry(
                        id=e_data["id"],
                        content=e_data["content"],
                        embedding=np.array(embeddings_map.get(e_data["id"], [])),
                        timestamp=datetime.fromisoformat(e_data["timestamp"]),
                        metadata=e_data.get("metadata", {}),
                        importance_score=e_data.get("importance_score", 0.5),
                        access_count=e_data.get("access_count", 0)
                    )
                    self.working_memory.append(entry)

                for e_data in data.get("long_term", []):
                    entry = MemoryEntry(
                        id=e_data["id"],
                        content=e_data["content"],
                        embedding=np.array(embeddings_map.get(e_data["id"], [])),
                        timestamp=datetime.fromisoformat(e_data["timestamp"]),
                        metadata=e_data.get("metadata", {}),
                        importance_score=e_data.get("importance_score", 0.5),
                        access_count=e_data.get("access_count", 0)
                    )
                    self.long_term_memory.append(entry)

                self.tool_chains = data.get("tool_chains", {})
                self._invalidate_search_cache()
            except Exception as e:
                print(f"加载旧格式记忆失败: {e}")

    # ========== 兼容接口 ==========
    def add_context(self, key: str, value: str) -> None:
        """添加上下文键值对到记忆（兼容接口）"""
        if not value:
            return
        try:
            self.add(
                content=f"[context:{key}] {value}",
                metadata={"type": "context", "key": key},
                importance=0.6,
            )
        except Exception:
            pass
        if not hasattr(self, "_context_store"):
            self._context_store: Dict[str, str] = {}
        self._context_store[key] = value

    def update_tool_stats(self, tool_name: str, success: bool, exec_time: float) -> None:
        """更新工具调用统计（兼容接口）"""
        if not hasattr(self, "_tool_stats"):
            self._tool_stats: Dict[str, Dict] = {}
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {"success": 0, "failed": 0, "total_time": 0.0}
        stats = self._tool_stats[tool_name]
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        stats["total_time"] += exec_time

    def build_context_summary(self, messages: List[Dict]) -> str:
        """根据历史消息列表构建简短摘要（兼容接口）"""
        if not messages:
            return "（无历史上下文）"
        parts = []
        for msg in messages[-5:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:80]
            if role in ("user", "assistant") and content:
                prefix = "用户" if role == "user" else "助手"
                parts.append(f"{prefix}: {content}")
        return "；".join(parts) if parts else f"已压缩 {len(messages)} 条消息"

    def get_tool_recommendation(self, task_type: str) -> List[str]:
        """基于历史统计推荐工具（兼容接口）"""
        if not hasattr(self, "_tool_stats") or not self._tool_stats:
            return []
        ranked = sorted(
            self._tool_stats.items(),
            key=lambda x: x[1]["success"] / max(1, x[1]["success"] + x[1]["failed"]),
            reverse=True
        )
        return [tool for tool, _ in ranked[:3]]
