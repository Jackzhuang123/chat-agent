# -*- coding: utf-8 -*-
"""向量记忆模块 - 使用 embedding 存储和检索历史对话"""

import json
import pickle
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np


class VectorMemory:
    """
    向量记忆系统 - 基于 embedding 的语义检索

    特性：
    - 使用简化的 TF-IDF 作为 embedding（无需外部模型）
    - 支持语义相似度检索
    - 持久化到磁盘
    - 自动去重
    """

    def __init__(self, memory_dir: str = ".agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # 记忆存储
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

        # 词汇表（用于 TF-IDF）
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        # 持久化文件
        self.memory_file = self.memory_dir / "vector_memory.pkl"

        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载记忆"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memories = data.get("memories", [])
                    self.embeddings = data.get("embeddings", [])
                    self.vocabulary = data.get("vocabulary", {})
                    self.idf = data.get("idf", {})
            except Exception:
                pass

    def save_to_disk(self):
        """保存记忆到磁盘"""
        try:
            data = {
                "memories": self.memories,
                "embeddings": self.embeddings,
                "vocabulary": self.vocabulary,
                "idf": self.idf,
                "last_update": datetime.now().isoformat()
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        text = text.lower()
        # 保留中文、英文、数字
        tokens = re.findall(r'[\w]+', text)
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算词频（TF）"""
        tf = {}
        total = len(tokens)
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # 归一化
        for token in tf:
            tf[token] = tf[token] / total
        return tf

    def _update_vocabulary(self, tokens: List[str]):
        """更新词汇表"""
        for token in set(tokens):
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)

    def _compute_idf(self):
        """计算逆文档频率（IDF）"""
        if not self.memories:
            return

        doc_count = len(self.memories)
        token_doc_count = {}

        for memory in self.memories:
            tokens = set(self._tokenize(memory["content"]))
            for token in tokens:
                token_doc_count[token] = token_doc_count.get(token, 0) + 1

        # IDF = log(总文档数 / 包含该词的文档数)
        for token, count in token_doc_count.items():
            self.idf[token] = math.log(doc_count / count)

    def _compute_embedding(self, text: str) -> List[float]:
        """计算文本的 TF-IDF embedding"""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # 初始化向量
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            return [0.0] * 100  # 默认维度

        embedding = [0.0] * vocab_size

        for token, tf_value in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf_value = self.idf.get(token, 1.0)
                embedding[idx] = tf_value * idf_value

        # L2 归一化
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def add_memory(self, content: str, metadata: Optional[Dict] = None):
        """添加记忆"""
        # 去重：检查是否已存在相似记忆
        if self.memories:
            query_emb = self._compute_embedding(content)
            similarities = [
                self._cosine_similarity(query_emb, emb)
                for emb in self.embeddings
            ]
            if max(similarities) > 0.95:  # 相似度阈值
                return  # 跳过重复记忆

        # 更新词汇表
        tokens = self._tokenize(content)
        self._update_vocabulary(tokens)

        # 重新计算所有 IDF
        self._compute_idf()

        # 计算 embedding
        embedding = self._compute_embedding(content)

        # 添加记忆
        memory = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.memories.append(memory)
        self.embeddings.append(embedding)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        # 确保维度一致
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """语义检索"""
        if not self.memories:
            return []

        # 计算查询 embedding
        query_emb = self._compute_embedding(query)

        # 计算相似度
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, emb)
            similarities.append((i, sim))

        # 排序并返回 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in similarities[:top_k]:
            memory = self.memories[idx].copy()
            memory["similarity"] = sim
            results.append(memory)

        return results

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的 n 条记忆"""
        return self.memories[-n:] if self.memories else []

    def clear(self):
        """清空记忆"""
        self.memories = []
        self.embeddings = []
        self.vocabulary = {}
        self.idf = {}

    def __len__(self):
        return len(self.memories)
