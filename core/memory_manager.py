#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持久化记忆系统 - 借鉴 DeerFlow MemoryUpdater，适配轻量本地/API 模型

提供 MemoryManager：跨会话持久化记忆，支持：
  1. 记忆结构：用户上下文（工作/个人/当前关注）+ 历史摘要 + 事实条目
  2. 存储格式：JSON 文件，支持原子写入（临时文件 → rename）
  3. 缓存策略：文件修改时间缓存，避免频繁 I/O
  4. 内容过滤：剥离上传文件路径（会话级临时资源，不应持久化）
  5. 双路径更新：LLM 语义提取（有 API 时）/ 规则提取（本地小模型场景）

与 DeerFlow 的区别：
  DeerFlow → 异步队列更新，依赖 LangGraph Runtime
  本项目   → 同步轻量实现，适配 Qwen/GLM 本地 & API 模型
"""

import json
import re
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MemoryManager:
    """
    跨会话持久化记忆系统。

    设计哲学（借鉴 DeerFlow MemoryMiddleware）：
    ──────────────────────────────────────────────────────────────────
    1. 记忆结构：用户上下文（工作/个人/当前关注）+ 历史摘要 + 事实条目
    2. 存储格式：JSON 文件，支持原子写入（临时文件 → rename 保证一致性）
    3. 缓存策略：文件修改时间缓存，避免频繁 I/O
    4. 内容过滤：剥离上传文件路径（会话级临时资源，不应持久化）
    5. 注入方式：作为消息注入（不修改系统提示词，保留提示词缓存友好性）
    ──────────────────────────────────────────────────────────────────
    """

    VERSION = "1.0"

    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化记忆管理器。

        Args:
            storage_path: 记忆文件路径，默认 session_logs/memory.json
        """
        if storage_path:
            self._file_path = Path(storage_path)
        else:
            project_root = Path(__file__).parent.parent
            self._file_path = project_root / "session_logs" / "memory.json"

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_mtime: Optional[float] = None

    # ------------------------------------------------------------------ #
    #  记忆结构                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_memory() -> Dict[str, Any]:
        """返回空白记忆结构（与 DeerFlow 保持格式兼容）。"""
        now = datetime.utcnow().isoformat() + "Z"
        return {
            "version": MemoryManager.VERSION,
            "lastUpdated": now,
            "user": {
                "workContext":     {"summary": "", "updatedAt": ""},
                "personalContext": {"summary": "", "updatedAt": ""},
                "topOfMind":       {"summary": "", "updatedAt": ""},
            },
            "history": {
                "recentMonths":      {"summary": "", "updatedAt": ""},
                "earlierContext":    {"summary": "", "updatedAt": ""},
                "longTermBackground": {"summary": "", "updatedAt": ""},
            },
            "facts": [],
        }

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def load(self) -> Dict[str, Any]:
        """加载记忆（带文件修改时间缓存）。"""
        try:
            mtime = self._file_path.stat().st_mtime if self._file_path.exists() else None
        except OSError:
            mtime = None

        if self._cache is not None and self._cache_mtime == mtime:
            return self._cache

        if not self._file_path.exists():
            data = self._empty_memory()
        else:
            try:
                with open(self._file_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = self._empty_memory()

        self._cache = data
        self._cache_mtime = mtime
        return data

    def save(self, data: Dict[str, Any]) -> bool:
        """原子写入记忆文件（临时文件 → rename 保证一致性）。"""
        try:
            data["lastUpdated"] = datetime.utcnow().isoformat() + "Z"
            tmp = self._file_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._file_path)
            try:
                mtime = self._file_path.stat().st_mtime
            except OSError:
                mtime = None
            self._cache = data
            self._cache_mtime = mtime
            return True
        except OSError as e:
            print(f"[MemoryManager] 保存记忆失败: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  记忆注入（供中间件调用）                                              #
    # ------------------------------------------------------------------ #

    def format_for_injection(self, max_chars: int = 1200) -> str:
        """
        将记忆格式化为可注入上下文的文本。

        Args:
            max_chars: 最大字符数（防止超出上下文窗口）

        Returns:
            格式化后的记忆文本，若无内容则返回空字符串
        """
        data = self.load()
        sections: List[str] = []

        # 用户上下文
        user = data.get("user", {})
        user_lines: List[str] = []
        for key, label in [
            ("workContext", "工作背景"),
            ("personalContext", "个人背景"),
            ("topOfMind", "近期关注"),
        ]:
            summary = user.get(key, {}).get("summary", "").strip()
            if summary:
                user_lines.append(f"- {label}: {summary}")
        if user_lines:
            sections.append("用户上下文:\n" + "\n".join(user_lines))

        # 历史摘要
        history = data.get("history", {})
        hist_lines: List[str] = []
        recent = history.get("recentMonths", {}).get("summary", "").strip()
        if recent:
            hist_lines.append(f"- 近期: {recent}")
        earlier = history.get("earlierContext", {}).get("summary", "").strip()
        if earlier:
            hist_lines.append(f"- 历史: {earlier}")
        if hist_lines:
            sections.append("历史背景:\n" + "\n".join(hist_lines))

        # 事实条目（按置信度排序，取高置信度的前几条）
        facts = data.get("facts", [])
        if isinstance(facts, list) and facts:
            ranked = sorted(
                (f for f in facts if isinstance(f, dict) and f.get("content")),
                key=lambda x: float(x.get("confidence", 0)),
                reverse=True,
            )
            fact_lines: List[str] = []
            for fact in ranked[:8]:
                content = str(fact.get("content", "")).strip()
                category = str(fact.get("category", "context")).strip()
                conf = float(fact.get("confidence", 0))
                if content:
                    fact_lines.append(f"- [{category}|{conf:.1f}] {content}")
            if fact_lines:
                sections.append("已知事实:\n" + "\n".join(fact_lines))

        if not sections:
            return ""

        result = "\n\n".join(sections)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n..."
        return result

    # ------------------------------------------------------------------ #
    #  记忆更新                                                             #
    # ------------------------------------------------------------------ #

    _UPLOAD_RE = re.compile(
        r"[^.!?]*(?:upload(?:ed|ing)?|上传|/mnt/user-data/uploads/|<uploaded_files>)[^.!?]*[.!?]?\s*",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_upload_mentions(text: str) -> str:
        """剥离上传文件相关句子（会话级资源不应持久化）。"""
        return MemoryManager._UPLOAD_RE.sub("", text).strip()

    def update_from_conversation(
        self,
        conversation: List[Tuple[str, str]],
        session_id: Optional[str] = None,
        model_forward_fn=None,
    ) -> bool:
        """
        从对话历史提取信息更新记忆。

        实现策略：
          - 若提供 model_forward_fn（API 模型），调用 LLM 进行语义提取
          - 否则使用规则提取（关键词匹配），适配无 API 场景
          - 提取到的信息与现有记忆合并，避免重复

        Args:
            conversation: [(user_msg, assistant_msg), ...] 对话对列表
            session_id: 可选会话 ID，用于溯源
            model_forward_fn: 可选模型调用函数（支持 LLM 语义提取）

        Returns:
            是否成功更新
        """
        if not conversation:
            return False

        valid_pairs = [
            (u, a) for u, a in conversation
            if (u or "").strip() or (a or "").strip()
        ]
        if not valid_pairs:
            return False

        data = self.load()

        if model_forward_fn is not None:
            return self._llm_update(data, valid_pairs, session_id, model_forward_fn)
        else:
            return self._rule_update(data, valid_pairs, session_id)

    def _llm_update(
        self,
        data: Dict[str, Any],
        conversation: List[Tuple[str, str]],
        session_id: Optional[str],
        model_forward_fn,
    ) -> bool:
        """调用 LLM 进行语义记忆提取。"""
        SYSTEM_PROMPT = """\
你是一个记忆管理系统。分析对话，提取关于用户的重要信息，返回 JSON，不输出其他内容。

输出格式：
{
  "workContext": "工作背景摘要（若有新信息则填写，否则为空字符串）",
  "topOfMind": "用户当前关注点（若有新信息则填写，否则为空字符串）",
  "newFacts": [
    {"content": "具体事实", "category": "preference|knowledge|context|behavior|goal", "confidence": 0.9}
  ]
}

规则：
- 只提取明确或强烈暗示的信息，不猜测
- confidence: 明确陈述=0.9, 强烈暗示=0.7, 推断=0.5
- newFacts 最多 3 条，避免冗余
- 若无有价值信息，返回 {"workContext":"","topOfMind":"","newFacts":[]}"""

        recent = conversation[-3:]
        lines: List[str] = []
        for u, a in recent:
            u_clean = self._strip_upload_mentions(u or "")
            a_clean = (a or "")[:300]
            if u_clean:
                lines.append(f"用户: {u_clean[:200]}")
            if a_clean:
                lines.append(f"助手: {a_clean}")
        conv_text = "\n".join(lines)

        if not conv_text.strip():
            return False

        try:
            raw = model_forward_fn(
                [{"role": "user", "content": f"请分析以下对话并提取记忆：\n\n{conv_text}"}],
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
                top_p=0.9,
                max_tokens=400,
            )
            raw = raw.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return False

            update = json.loads(json_match.group())
        except Exception as e:
            print(f"[MemoryManager] LLM 提取失败: {e}")
            return self._rule_update(data, conversation, session_id)

        now = datetime.utcnow().isoformat() + "Z"
        changed = False

        for key in ("workContext", "topOfMind", "personalContext"):
            val = update.get(key, "").strip()
            if val:
                data["user"][key] = {"summary": val, "updatedAt": now}
                changed = True

        new_facts = update.get("newFacts", [])
        if isinstance(new_facts, list):
            changed |= self._merge_facts(data, new_facts, session_id or "llm", now)

        if changed:
            return self.save(data)
        return False

    def _rule_update(
        self,
        data: Dict[str, Any],
        conversation: List[Tuple[str, str]],
        session_id: Optional[str],
    ) -> bool:
        """规则提取记忆（无 LLM 开销，适配本地小模型场景）。"""
        PREF_PATTERNS = [
            (re.compile(r"我(?:喜欢|偏好|习惯|常用|一般用)\s*(.{2,20})", re.UNICODE), "preference"),
            (re.compile(r"我(?:不喜欢|讨厌|避免)\s*(.{2,20})", re.UNICODE), "preference"),
            (re.compile(r"我(?:是|作为|担任|负责)\s*(.{2,20})", re.UNICODE), "context"),
            (re.compile(r"我(?:在|工作于|就职于)\s*(.{2,20})", re.UNICODE), "context"),
            (re.compile(r"我(?:会|掌握|熟悉|了解)\s*(.{2,20})", re.UNICODE), "knowledge"),
        ]

        now = datetime.utcnow().isoformat() + "Z"
        new_facts: List[Dict[str, Any]] = []

        for user_msg, _ in conversation[-5:]:
            text = self._strip_upload_mentions(user_msg or "")
            for pattern, category in PREF_PATTERNS:
                for m in pattern.finditer(text):
                    content = m.group(1).strip().rstrip("。，,.")
                    if len(content) >= 2:
                        new_facts.append({
                            "content": f"用户{m.group(0)[1:]}",
                            "category": category,
                            "confidence": 0.7,
                        })

        if not new_facts:
            return False

        changed = self._merge_facts(data, new_facts, session_id or "rule", now)
        if changed:
            return self.save(data)
        return False

    def _merge_facts(
        self,
        data: Dict[str, Any],
        new_facts: List[Dict[str, Any]],
        source: str,
        now: str,
    ) -> bool:
        """将新事实合并入记忆，去重、限数量（最多 50 条，按置信度保留）。"""
        existing_contents = {
            f.get("content", "").strip()
            for f in data.get("facts", [])
            if isinstance(f, dict)
        }

        added = False
        for fact in new_facts:
            conf = float(fact.get("confidence", 0.5))
            if conf < 0.5:
                continue
            content = str(fact.get("content", "")).strip()
            if not content or content in existing_contents:
                continue
            data.setdefault("facts", []).append({
                "id": f"fact_{_uuid.uuid4().hex[:8]}",
                "content": content,
                "category": str(fact.get("category", "context")),
                "confidence": conf,
                "createdAt": now,
                "source": source,
            })
            existing_contents.add(content)
            added = True

        # 限制最大事实数量（按置信度保留高质量条目）
        MAX_FACTS = 50
        if len(data.get("facts", [])) > MAX_FACTS:
            data["facts"] = sorted(
                data["facts"],
                key=lambda f: float(f.get("confidence", 0)),
                reverse=True,
            )[:MAX_FACTS]

        return added

