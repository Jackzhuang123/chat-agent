# -*- coding: utf-8 -*-
"""基于向量记忆的 RAG 意图路由器（增强版）

新增特性：
- 前置规则层：文件操作/历史回顾指令强制路由，零延迟
- 查询关键词直接加权，避免过度依赖历史统计
- 修正置信度阈值边界条件（<=）
- LLM 二次确认时附带历史证据，提升准确性
"""

import json
import re
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from core.monitor_logger import get_monitor_logger
from core.vector_memory import VectorMemory


class IntentType(Enum):
    CHAT = "chat"
    TOOLS = "tools"
    SKILLS = "skills"
    HYBRID = "hybrid"
    PLAN = "plan"
    MEMORY_QUERY = "memory_query"   # 新增：历史记忆查询
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    reasoning: str
    evidence: List[Dict]          # 检索到的相关历史证据
    suggested_params: Dict[str, Any]


class RAGIntentRouter:
    """利用向量记忆检索相似历史对话，判断用户意图"""

    def __init__(
        self,
        vector_memory: VectorMemory,
        llm_forward_fn: Optional[Callable] = None,
        confidence_threshold: float = 0.7,
    ):
        self.vm = vector_memory
        self.llm_forward_fn = llm_forward_fn
        self.threshold = confidence_threshold
        self.monitor = get_monitor_logger()

        # 规则兜底库（关键词）
        self.rule_patterns = {
            IntentType.TOOLS: [
                r"读取|写入|修改|删除|列出|扫描|查找|调用工具|bash|grep|find|ls|cat",
                r"\.py|\.js|\.md|文件|目录|代码",
            ],
            IntentType.SKILLS: [r"技能|知识库|pdf|文档处理|代码审查"],
            IntentType.PLAN: [r"分析|重构|设计|实现|开发|生成|搭建|部署|复杂|多步骤"],
            IntentType.CHAT: [r"什么是|为什么|怎么样|解释|介绍|聊天|你好|谢谢"],
        }

    def route(self, user_input: str, context: Dict = None) -> IntentResult:
        """
        路由用户输入，返回意图结果。
        """
        context = context or {}
        self.monitor.debug(f"RAG路由开始，输入: {user_input[:100]}")
        normalized_input = user_input.strip()

        # ---------- L0 前置规则层（最高优先级） ----------
        # 0. 纯问候/致谢等低信息量对话，直接走 chat，避免无意义向量检索
        if self._is_low_information_chat(normalized_input):
            self.monitor.info("前置规则命中：低信息量寒暄 → chat")
            return IntentResult(
                intent=IntentType.CHAT,
                confidence=0.98,
                reasoning="用户输入是寒暄或简短致谢，无需工具或检索",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.CHAT)
            )

        # 1. 显式文件操作指令 → tools
        if self._has_explicit_file_operation(normalized_input):
            self.monitor.info("前置规则命中：显式文件操作 → tools")
            return IntentResult(
                intent=IntentType.TOOLS,
                confidence=0.99,
                reasoning="用户输入包含明确文件操作指令",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.TOOLS)
            )

        # 2. 历史回顾指令 → memory_query
        if self._is_memory_query(normalized_input):
            self.monitor.info("前置规则命中：历史回顾请求 → memory_query")
            return IntentResult(
                intent=IntentType.MEMORY_QUERY,
                confidence=0.99,
                reasoning="用户请求回顾历史问题",
                evidence=[],
                suggested_params={"temperature": 0.3, "max_tokens": 2048}
            )

        # 3. 上传文件且带有分析意图，优先走 tools
        if context.get("uploaded_files") and self._has_uploaded_file_analysis_intent(normalized_input):
            self.monitor.info("前置规则命中：上传文件分析 → tools")
            return IntentResult(
                intent=IntentType.TOOLS,
                confidence=0.95,
                reasoning="用户已上传文件并请求分析/提取内容",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.TOOLS)
            )

        # ---------- L1 向量检索相关历史 ----------
        evidence: List[Dict[str, Any]] = []
        if self._should_use_vector_retrieval(normalized_input):
            evidence = self.vm.search(
                query=normalized_input,
                top_k=3,
                semantic_weight=0.7,
                recency_weight=0.2,
                importance_weight=0.1,
                min_score=0.4,
            )
            evidence = [e for e in evidence if isinstance(e, dict)]
            self.monitor.debug(f"检索到 {len(evidence)} 条历史证据")
        else:
            self.monitor.debug("当前查询信息量较低，跳过向量检索")

        # ---------- L2 基于证据推断意图 ----------
        intent, confidence, reasoning = self._infer_from_evidence(normalized_input, evidence)

        # ---------- L3 LLM 二次确认（置信度不足时）----------
        # 修正：使用 <= 使得等于阈值时也触发 LLM 纠正
        if confidence <= self.threshold and self.llm_forward_fn and self._should_use_llm_recheck(normalized_input, evidence):
            self.monitor.info(f"置信度 {confidence:.2f} 低于/等于阈值，调用 LLM 二次确认")
            llm_intent, llm_conf, llm_reason = self._llm_route(normalized_input, evidence)
            if llm_conf > confidence:
                intent, confidence, reasoning = llm_intent, llm_conf, llm_reason

        # ---------- L4 兜底规则 ----------
        if confidence < 0.5:
            intent = self._rule_fallback(normalized_input)
            confidence = 0.5
            reasoning = "规则兜底（低置信度）"
            self.monitor.warning(f"低置信度，规则兜底为 {intent.value}")

        suggested = self._suggest_params(intent)
        self.monitor.info(f"RAG路由结果: {intent.value} (置信度 {confidence:.2f}) - {reasoning}")

        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            suggested_params=suggested,
        )

    def _has_explicit_file_operation(self, text: str) -> bool:
        if not text:
            return False
        file_like = bool(re.search(r'[/\\]|[\w\-]+\.(json|log|txt|md|py|yaml|yml|csv|pdf|docx?)\b', text, re.I))
        patterns = [
            r'查找.*文件', r'解析.*文件', r'读取.*文件',
            r'查看.*文件', r'打开.*文件', r'列出.*目录',
            r'扫描.*(目录|文件|代码)', r'写入.*文件', r'编辑.*文件',
            r'运行.*\.py', r'执行.*命令', r'分析.*\.json', r'分析.*日志'
        ]
        if any(re.search(p, text, re.I) for p in patterns):
            return True
        return file_like and any(kw in text for kw in ("查找", "读取", "查看", "解析", "分析", "打开", "扫描", "编辑", "修改"))

    def _is_memory_query(self, text: str) -> bool:
        patterns = [
            r'我之前.*问', r'回顾.*问题', r'历史.*记录',
            r'之前.*聊', r'上次.*说', r'过去.*问过',
            r'之前.*说了什么', r'最近.*问题', r'历史.*对话'
        ]
        return any(re.search(p, text, re.I) for p in patterns)

    def _is_low_information_chat(self, text: str) -> bool:
        if not text:
            return True
        normalized = re.sub(r'\s+', '', text.lower())
        low_info_patterns = [
            r'^(你好|您好|hi|hello|hey)$',
            r'^(谢谢|感谢|thanks|thankyou|thankyou!|thx)[!！]*$',
            r'^(在吗|忙吗|收到|ok|okay|好的|嗯嗯|嗯|哈喽)[!！]*$',
        ]
        return any(re.search(p, normalized, re.I) for p in low_info_patterns)

    def _has_uploaded_file_analysis_intent(self, text: str) -> bool:
        if not text:
            return False
        return any(kw in text for kw in ("总结", "概括", "分析", "提取", "读取", "查看", "内容", "翻译", "解释"))

    def _should_use_vector_retrieval(self, text: str) -> bool:
        if not text:
            return False
        if self._is_low_information_chat(text):
            return False
        compact = re.sub(r'\s+', '', text)
        if len(compact) <= 4 and not self._is_memory_query(text):
            return False
        return True

    def _should_use_llm_recheck(self, text: str, evidence: List[Dict[str, Any]]) -> bool:
        if self._is_low_information_chat(text):
            return False
        if self._has_explicit_file_operation(text) or self._is_memory_query(text):
            return False
        if not evidence and len(re.sub(r'\s+', '', text)) < 8:
            return False
        return True

    def _infer_from_evidence(self, query: str, evidence: List[Dict]) -> tuple:
        """
        基于检索到的证据启发式推断意图，同时考虑当前查询的关键词。
        """
        if not evidence:
            # 无证据时检查查询本身是否包含强工具信号
            if self._has_explicit_file_operation(query):
                return IntentType.TOOLS, 0.85, "查询包含文件操作指令，但无历史证据"
            return IntentType.CHAT, 0.3, "无历史证据，默认为对话模式"

        tool_hints = 0
        skill_hints = 0
        total_score = 0.0

        # 查询本身的关键词加权
        query_lower = query.lower()
        query_tool_bonus = sum(
            1 for kw in ['read_file', 'bash', 'write_file', '文件', '读取', '解析', '扫描', '执行', '日志', '.json', '.py']
            if kw in query_lower
        )
        tool_hints += query_tool_bonus

        for e in evidence:
            meta = e.get("metadata", {})
            score = e.get("score", 0.0)
            total_score += score
            if meta.get("type") == "tool_execution":
                tool_hints += 1
            if meta.get("skill_used"):
                skill_hints += 1

        avg_score = total_score / len(evidence) if evidence else 0.0

        # 启发式判断
        if tool_hints >= 2:
            conf = min(0.95, 0.7 + avg_score * 0.2)
            return IntentType.TOOLS, conf, f"历史及当前查询均频繁使用工具 (工具信号 {tool_hints}次)"
        elif skill_hints >= 1:
            conf = min(0.85, 0.6 + avg_score * 0.2)
            return IntentType.SKILLS, conf, "历史中使用过相关技能"
        elif avg_score > 0.7:
            # 高相似度但无工具/技能信号，可能是常规对话
            return IntentType.CHAT, 0.7, "高度相似的历史对话"
        else:
            return IntentType.CHAT, 0.5, "历史证据不足，偏向对话模式"

    def _llm_route(self, query: str, evidence: List[Dict]) -> tuple:
        """
        调用 LLM 进行意图识别。
        """
        evidence_text = "\n".join([f"- {e['content'][:200]}" for e in evidence[:2]])
        prompt = f"""分析用户意图，选择最匹配的类别（chat/tools/skills/plan/hybrid/memory_query）。

用户输入: {query}

相关历史记录:
{evidence_text if evidence_text else "（无历史记录）"}

请输出 JSON 格式，包含 intent, confidence (0-1), reason。
例如: {{"intent": "tools", "confidence": 0.85, "reason": "用户要求读取文件"}}
"""
        try:
            resp = self.llm_forward_fn([{"role": "user", "content": prompt}], system_prompt="")
            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
            if not json_match:
                raise ValueError("未找到 JSON")
            data = json.loads(json_match.group())
            intent_str = data.get("intent", "chat").lower()
            intent_map = {
                "chat": IntentType.CHAT,
                "tools": IntentType.TOOLS,
                "skills": IntentType.SKILLS,
                "plan": IntentType.PLAN,
                "hybrid": IntentType.HYBRID,
                "memory_query": IntentType.MEMORY_QUERY,
            }
            intent = intent_map.get(intent_str, IntentType.CHAT)
            confidence = float(data.get("confidence", 0.6))
            reason = data.get("reason", "LLM判断")
            return intent, min(0.95, confidence), reason
        except Exception as e:
            self.monitor.warning(f"LLM 意图识别失败: {e}")
            return IntentType.CHAT, 0.5, f"LLM解析失败 ({e})"

    def _rule_fallback(self, text: str) -> IntentType:
        """基于关键词的规则兜底"""
        text_lower = text.lower()
        scores = {intent: 0 for intent in IntentType}
        for intent, patterns in self.rule_patterns.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    scores[intent] += 1

        if scores[IntentType.TOOLS] > 0:
            return IntentType.TOOLS
        elif scores[IntentType.PLAN] > 0:
            return IntentType.PLAN
        elif scores[IntentType.SKILLS] > 0:
            return IntentType.SKILLS
        else:
            return IntentType.CHAT

    def _suggest_params(self, intent: IntentType) -> Dict[str, Any]:
        """根据意图返回建议的模型参数"""
        defaults = {
            IntentType.CHAT: {"temperature": 0.8, "max_tokens": 1024, "tools_enabled": False},
            IntentType.TOOLS: {"temperature": 0.3, "max_tokens": 2048, "tools_enabled": True},
            IntentType.SKILLS: {"temperature": 0.5, "max_tokens": 2048, "tools_enabled": True},
            IntentType.PLAN: {"temperature": 0.4, "max_tokens": 4096, "plan_mode": True},
            IntentType.HYBRID: {"temperature": 0.6, "max_tokens": 2048, "tools_enabled": True},
            IntentType.MEMORY_QUERY: {"temperature": 0.3, "max_tokens": 2048, "tools_enabled": False},
        }
        return defaults.get(intent, defaults[IntentType.CHAT])
