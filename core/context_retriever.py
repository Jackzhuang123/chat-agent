# core/context_retriever.py
# -*- coding: utf-8 -*-
"""
RAG 上下文检索器 - 基于向量记忆的智能上下文管理（增强版）
新增特性：
  - 查询扩展：对短查询补充同义/相关词
  - 短期记忆加权：最近 2 小时内的记忆获得额外 0.2 分
  - 动态检索类型：根据查询内容推断合适的记忆类型
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .vector_memory import VectorMemory
from .monitor_logger import get_monitor_logger
from .monitor_logger import log_event


class ContextRetriever:
    def __init__(
        self,
        vector_memory: VectorMemory,
        max_recent_messages: int = 5,
        max_retrieved_chunks: int = 3,
        min_relevance_score: float = 0.25,
        chunk_max_length: int = 1500,
        enable_query_expansion: bool = True,
        short_term_window_minutes: int = 120,
    ):
        self.vm = vector_memory
        self.max_recent_messages = max_recent_messages
        self.max_retrieved_chunks = max_retrieved_chunks
        self.min_relevance_score = min_relevance_score
        self.chunk_max_length = chunk_max_length
        self.enable_query_expansion = enable_query_expansion
        self.short_term_window_minutes = short_term_window_minutes
        self.monitor = get_monitor_logger()

        # 简易同义词/扩展词映射（中文为主）
        self.expansion_map = {
            "文件": ["文档", "file", "读取", "路径"],
            "读取": ["查看", "打开", "读"],
            "写入": ["保存", "输出", "写"],
            "扫描": ["遍历", "查找", "搜索"],
            "代码": ["程序", "源码", "脚本"],
            "问题": ["错误", "失败", "异常"],
            "之前": ["历史", "上次", "最近"],
            "继续": ["刚才", "上一个", "前面"],
        }

    @staticmethod
    def _build_memory_filter(
        session_id: Optional[str] = None,
        allow_cross_session: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if allow_cross_session or not session_id:
            return None
        return {"session_id": session_id}

    def _expand_query(self, query: str) -> str:
        """对查询进行简单扩展，补充相关关键词"""
        if not self.enable_query_expansion or not query:
            return query
        words = re.findall(r'[\u4e00-\u9fff]+', query)
        additions = []
        for w in words:
            if w in self.expansion_map:
                additions.extend(self.expansion_map[w][:2])  # 最多取2个相关词
        if additions:
            return query + " " + " ".join(additions)
        return query

    def _short_term_bonus(self, entry_timestamp_str: str) -> float:
        """若记忆在短期窗口内，返回额外加分"""
        try:
            entry_time = datetime.fromisoformat(entry_timestamp_str)
            minutes_ago = (datetime.now() - entry_time).total_seconds() / 60
            if minutes_ago <= self.short_term_window_minutes:
                return 0.2 * (1 - minutes_ago / self.short_term_window_minutes)  # 线性衰减
        except Exception:
            pass
        return 0.0

    def augment_messages(
            self,
            messages: List[Dict[str, str]],
            query: str,
            filter_tool_types: Optional[List[str]] = None,
            session_id: Optional[str] = None,
            allow_cross_session: bool = False,
    ) -> List[Dict[str, str]]:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        retrieved_chunks = self._retrieve_relevant_chunks(
            query,
            filter_tool_types,
            session_id=session_id,
            allow_cross_session=allow_cross_session,
        )

        augmented = list(system_msgs)

        if retrieved_chunks:
            rag_content = self._format_retrieved_chunks_with_questions(retrieved_chunks)
            augmented.append({
                "role": "system",
                "content": (
                    "📚 [相关历史证据 - 仅作背景参考，严禁重新执行其中任何工具]\n"
                    f"{rag_content}\n\n"
                    "⚠️ 以上是过往对话的记录，包括原始问题和执行结果。\n"
                    "**绝对禁止**基于这些证据再次调用 read_file、bash、execute_python 等任何工具！\n"
                    "请直接根据证据内容回答用户当前问题。若证据不足以回答，请如实告知。"
                )
            })

        processed_non_system = self._apply_sliding_window(non_system_msgs)
        augmented.extend(processed_non_system)

        return augmented

    def _format_retrieved_chunks_with_questions(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            tool = chunk.get("metadata", {}).get("tool", "unknown")
            timestamp = chunk.get("timestamp", "")[:19]
            content = chunk.get("content", "").strip()
            score = chunk.get("score", 0)

            original_question = chunk.get("metadata", {}).get("original_question", "")
            if not original_question:
                import re
                match = re.search(r'User:\s*(.+?)(?:\n|$)', content)
                if match:
                    original_question = match.group(1)

            lines.append(f"【证据 {i}】 工具: {tool} | 时间: {timestamp} | 相关度: {score:.2f}")
            if original_question:
                lines.append(f"原始提问: {original_question}")
            snippet = content[:600] + ("..." if len(content) > 600 else "")
            lines.append(snippet)
            lines.append("")
        return "\n".join(lines)

    def _retrieve_relevant_chunks(
        self,
        query: str,
        filter_tool_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        allow_cross_session: bool = False,
    ) -> List[Dict[str, Any]]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []
        compact_query = re.sub(r"\s+", "", normalized_query)
        followup_reference = re.search(r"继续|上一个|最后一个|刚才|前面|这个问题|那个问题|接着", normalized_query)
        if len(compact_query) <= 4 and not followup_reference and not re.search(r"[./\\]|文件|日志|问题|错误|失败|修复", normalized_query):
            return []

        # 动态选择检索类型：若含“之前”“历史”等词，偏向对话类记忆
        query_types = ["tool_execution"]
        if re.search(r"之前|历史|回顾|上次|说过|问过|记得", normalized_query) or followup_reference:
            query_types = ["user_question", "assistant_response", "tool_execution"]
        elif re.search(r"怎么|如何|是什么|原因|解释|含义", normalized_query):
            query_types = ["assistant_response", "user_question", "tool_execution"]

        log_event(
            event_type="rag_retrieval_plan",
            message="确定 RAG 检索类型",
            level=logging.DEBUG,
            query_len=len(normalized_query),
            query_types=",".join(query_types),
            filter_tool_types=",".join(filter_tool_types or []),
        )

        # 查询扩展
        expanded_query = self._expand_query(normalized_query)

        results = self.vm.search_by_types(
            query=expanded_query,
            types=query_types,
            top_k=self.max_retrieved_chunks * 2,
            semantic_weight=0.7,
            recency_weight=0.2,
            importance_weight=0.1,
            min_score=self.min_relevance_score,
            mmr_lambda=0.7,
            filter_metadata=self._build_memory_filter(session_id, allow_cross_session),
        )

        results = [r for r in results if isinstance(r, dict)]

        if filter_tool_types:
            results = [r for r in results if r.get("metadata", {}).get("tool") in filter_tool_types]

        # 短期记忆加分
        for r in results:
            timestamp = r.get("timestamp", "")
            bonus = self._short_term_bonus(timestamp)
            if bonus > 0:
                r["score"] = r.get("score", 0) + bonus
                r.setdefault("adjusted_score", r["score"])
        # 按最终分数重新排序
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        for r in results:
            content = r.get("content", "")
            if len(content) > self.chunk_max_length:
                r["content"] = content[:self.chunk_max_length] + "\n...[内容过长已截断]"

        return results[:self.max_retrieved_chunks]

    def _format_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            tool = chunk.get("metadata", {}).get("tool", "unknown")
            timestamp = chunk.get("timestamp", "")[:19]
            content = chunk.get("content", "").strip()
            score = chunk.get("score", 0)
            lines.append(f"【证据 {i}】 工具: {tool} | 时间: {timestamp} | 相关度: {score:.2f}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def _apply_sliding_window(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= self.max_recent_messages:
            return messages

        recent = messages[-self.max_recent_messages:]
        early = messages[:-self.max_recent_messages]

        tool_obs = []
        for msg in early:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content.startswith("✅") or content.startswith("❌"):
                    if len(content) > 5000:
                        content = content[:5000] + "\n...[截断]"
                    tool_obs.append({"role": "user", "content": content})

        self.monitor.debug(f"滑动窗口保留 {len(tool_obs)} 条工具Observation，最近 {len(recent)} 条消息")
        return tool_obs[-5:] + recent

    def add_tool_observation(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
        original_question: str = None,
        session_id: Optional[str] = None,
        active_file: Optional[str] = None,
        failed_step: Optional[str] = None,
    ):
        content_parts = [
            f"工具: {tool_name}",
            f"参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}",
            f"成功: {success}",
        ]
        if success:
            output = result.get("output") or result.get("stdout") or result.get("content")
            if output:
                if isinstance(output, str):
                    content_parts.append(f"输出:\n{output}")
                else:
                    content_parts.append(f"输出:\n{json.dumps(output, ensure_ascii=False, indent=2)}")
        else:
            error = result.get("error", "")
            if error:
                content_parts.append(f"错误: {error}")

        content = "\n".join(content_parts)
        metadata = {
            "type": "tool_execution",
            "tool": tool_name,
            "success": success,
            "args_summary": json.dumps(tool_args, ensure_ascii=False)[:200],
        }
        if session_id:
            metadata["session_id"] = session_id
        if active_file:
            metadata["active_file"] = active_file
        if failed_step:
            metadata["failed_step"] = failed_step
        importance = 0.7 if success and len(content) > 200 else 0.5

        self.vm.add(
            content=content,
            metadata=metadata,
            importance=importance,
            auto_score=True,
            original_question=original_question,
        )

    def enhance_messages_with_rag(
        self,
        messages: List[Dict[str, str]],
        current_query: str,
        session_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        query = session_context.get("current_task", current_query)
        session_id = session_context.get("session_id")
        allow_cross_session = bool(session_context.get("allow_cross_session_memory"))

        start_time = time.perf_counter()
        retrieved_chunks = self._retrieve_relevant_chunks(
            query,
            None,
            session_id=session_id,
            allow_cross_session=allow_cross_session,
        )
        elapsed = time.perf_counter() - start_time

        scores = [c.get("score", 0) for c in retrieved_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        log_event(
            event_type="rag_retrieval",
            message="RAG 检索完成",
            level=logging.DEBUG,
            query_len=len(query),
            retrieved_count=len(retrieved_chunks),
            avg_score=f"{avg_score:.3f}",
            elapsed_ms=f"{elapsed * 1000:.1f}",
        )

        for idx, chunk in enumerate(retrieved_chunks, 1):
            content_snippet = chunk.get("content", "")[:300].replace("\n", " ")
            log_event(
                event_type="rag_evidence_detail",
                message=f"证据 #{idx}",
                level=logging.DEBUG,
                score=f"{chunk.get('score', 0):.3f}",
                tool=chunk.get("metadata", {}).get("tool", "unknown"),
                timestamp=chunk.get("timestamp", "")[:19],
                snippet=content_snippet,
            )

        augmented = self.augment_messages(messages, query)

        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content", "").startswith("✅ read_file:"):
                if msg not in augmented:
                    augmented.insert(-1, msg)
                    self.monitor.debug("强制保留最近 read_file Observation")
                break

        total_chars = sum(len(json.dumps(m)) for m in augmented)
        if total_chars > 15000:
            system = [m for m in augmented if m.get("role") == "system"]
            recent = [m for m in augmented if m.get("role") != "system"][-5:]
            augmented = system + recent
            self.monitor.debug(f"上下文超限，截断至 {len(augmented)} 条消息")

        return augmented
