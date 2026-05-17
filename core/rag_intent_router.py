# -*- coding: utf-8 -*-
"""基于向量记忆的 RAG 意图路由器（增强版 + 修复）

修复点：
- 移除错误的 sympy 导入
- 修正 LLM 异常分支中未定义的 query 变量
- 强化 JSON 提取容错（支持多行及尾部多余字符）
- 优化 memory_query 的 LLM 防误判机制
"""

import json
import re
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from core.monitor_logger import get_monitor_logger
from core.multi_agent import _parse_json_like_payload
from core.vector_memory import VectorMemory


class IntentType(Enum):
    CHAT = "chat"
    TOOLS = "tools"
    SKILLS = "skills"
    HYBRID = "hybrid"
    PLAN = "plan"
    MEMORY_QUERY = "memory_query"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    reasoning: str
    evidence: List[Dict]
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
        context = context or {}
        self.monitor.debug(f"RAG路由开始，输入: {user_input[:100]}")
        normalized_input = user_input.strip()
        session_id = context.get("session_id")
        allow_cross_session = bool(context.get("allow_cross_session_memory"))

        # ---------- L0 前置规则层 ----------
        # 0. 历史回顾指令 → memory_query
        if self._is_memory_query(normalized_input):
            self.monitor.info("前置规则命中：历史回顾请求 → memory_query")
            return IntentResult(
                intent=IntentType.MEMORY_QUERY,
                confidence=0.99,
                reasoning="用户明确请求回顾历史问题",
                evidence=[],
                suggested_params={"temperature": 0.3, "max_tokens": 2048}
            )

        # 1. 纯问候/情绪表达等低信息量对话
        if self._is_file_followup_request(normalized_input, context):
            self.monitor.info("前置规则命中：文件任务追问 → tools")
            return IntentResult(
                intent=IntentType.TOOLS,
                confidence=0.97,
                reasoning="当前会话存在活跃文件，用户请求继续任务",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.TOOLS)
            )

        if self._is_low_information_chat(normalized_input):
            self.monitor.info("前置规则命中：低信息量寒暄/情绪表达 → chat")
            return IntentResult(
                intent=IntentType.CHAT,
                confidence=0.98,
                reasoning="用户输入是寒暄、情绪表达或极短无明确意图文本，无需工具或检索",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.CHAT)
            )

        # 2. 模糊请求（缺少目标）直接触发澄清
        if self._is_ambiguous_request(normalized_input):
            self.monitor.info("前置规则命中：模糊请求（缺少目标） → 触发澄清")
            return IntentResult(
                intent=IntentType.CHAT,
                confidence=0.4,
                reasoning="用户请求不完整，缺少具体目标（如文件名）",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.CHAT)
            )

        # 3. 显式文件操作指令 → tools
        if self._has_explicit_file_operation(normalized_input):
            self.monitor.info("前置规则命中：显式文件操作 → tools")
            return IntentResult(
                intent=IntentType.TOOLS,
                confidence=0.99,
                reasoning="用户输入包含明确文件操作指令",
                evidence=[],
                suggested_params=self._suggest_params(IntentType.TOOLS)
            )

        # 4. 上传文件且带有分析意图
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
        evidence = []
        if self._should_use_vector_retrieval(normalized_input):
            retrieval_filter = self._build_retrieval_filter(
                normalized_input,
                session_id=session_id,
                allow_cross_session=allow_cross_session,
            )
            evidence = self.vm.search(
                query=normalized_input,
                top_k=3,
                semantic_weight=0.7,
                recency_weight=0.2,
                importance_weight=0.1,
                min_score=0.4,
                filter_metadata=retrieval_filter,
            )
            evidence = [e for e in evidence if isinstance(e, dict)]
            self.monitor.debug(f"检索到 {len(evidence)} 条历史证据")
        else:
            self.monitor.debug("当前查询信息量较低，跳过向量检索")

        # ---------- L2 基于证据推断意图 ----------
        intent, confidence, reasoning = self._infer_from_evidence(normalized_input, evidence)

        # ---------- L3 LLM 二次确认 ----------
        if confidence <= self.threshold and self.llm_forward_fn and self._should_use_llm_recheck(normalized_input,
                                                                                                 evidence):
            self.monitor.info(f"置信度 {confidence:.2f} 低于/等于阈值，调用 LLM 二次确认")
            try:
                llm_intent, llm_conf, llm_reason = self._llm_route(normalized_input, evidence)
                if llm_intent == IntentType.MEMORY_QUERY and not self._has_memory_query_signal(normalized_input):
                    self.monitor.info("LLM 误判为 memory_query，但用户输入无明确记忆信号，强制降级为 chat")
                    llm_intent = IntentType.CHAT
                    llm_conf = 0.5
                    llm_reason = "无明确记忆查询关键词，降级为对话"
                if llm_conf > confidence:
                    intent, confidence, reasoning = llm_intent, llm_conf, llm_reason
            except Exception as e:
                self.monitor.warning(f"LLM 意图识别异常: {e}，使用规则强制判断")
                # 若用户输入明显包含文件操作或编号任务，强制设为 tools 模式
                if self._has_explicit_file_operation(normalized_input) or re.search(r'\d+[\.、]', normalized_input):
                    intent = IntentType.TOOLS
                    confidence = 0.85
                    reasoning = "LLM 识别失败，根据文件操作/编号任务强制设为 tools 模式"

        # ---------- L4 兜底规则 ----------
        if confidence < 0.5:
            intent = self._rule_fallback(normalized_input)
            confidence = 0.5
            reasoning = "规则兜底（低置信度）"

        suggested = self._suggest_params(intent)
        self.monitor.info(f"RAG路由结果: {intent.value} (置信度 {confidence:.2f}) - {reasoning}")

        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            suggested_params=suggested,
        )

    @staticmethod
    def _is_ambiguous_request(text: str) -> bool:
        if not text:
            return False
        operation_verbs = ['读取', '分析', '理解', '处理', '查看', '解析', '运行', '执行']
        has_verb = any(verb in text for verb in operation_verbs)
        if not has_verb:
            return False
        has_file_indicator = bool(re.search(r'[\w\-]+\.\w{2,}', text))
        has_path = bool(re.search(r'[/\\]', text))
        if has_file_indicator or has_path:
            return False
        if re.search(r'(该|这个|那个|它)\s*$', text):
            return True
        if len(text) <= 6 and has_verb:
            return True
        return False

    @staticmethod
    def _has_explicit_file_operation(text: str) -> bool:
        if not text:
            return False
        file_like = bool(re.search(r'[/\\]|[\w\-]+\.(json|log|txt|md|py|yaml|yml|csv|pdf|docx?)\b', text, re.I))
        if not file_like:
            return False
        # 仅提供文件名、相对路径或绝对路径，也应视为明确的本地文件请求，
        # 否则这类输入会被历史检索或 LLM 误带到 chat / memory_query。
        compact = re.sub(r'\s+', '', text)
        if compact and re.fullmatch(r'.*[\w\-./\\]+\.(json|log|txt|md|py|yaml|yml|csv|pdf|docx?)([，,].*)?$', compact, re.I):
            return True
        operation_keywords = ['读取', '解析', '分析', '查看', '打开', '写入', '修改', '删除', '运行', '执行',
                              '查找', '扫描', '提取', '编辑', 'read', 'write', 'analyze', 'parse', 'open']
        return any(kw in text.lower() for kw in operation_keywords)

    def _is_memory_query(self, text: str) -> bool:
        patterns = [
            r'我之前.*问', r'回顾.*问题', r'历史.*记录',
            r'之前.*聊', r'上次.*说', r'过去.*问过',
            r'之前.*说了什么', r'最近.*问题', r'历史.*对话',
            r'之前.*问', r'以前.*问', r'上次.*问',
            r'之前.*说过', r'我问过.*什么', r'我问了.*什么',
            r'我.*之前.*问题', r'聊天记录', r'历史问题', r'曾经问过'
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
            r'^(今天|最近|我|身体|感觉|心情|头|胃|累|困|难过|开心|无聊|烦|不舒服|生病|疼|晕)',
            r'^(好|有点|非常|特别)(累|困|烦|难过|疼|晕|无聊)',
            r'^(哎|唉|唔|嗯|哦|啊)[~～!！]*$',
            r'^(?!.*(文件|代码|读取|写入|运行|扫描|执行|查看|bash|python|api|日志|json|\.py|\.md|：|:)).{1,6}$',
        ]
        if len(text) > 15:
            return False
        return any(re.search(p, normalized, re.I) for p in low_info_patterns)

    @staticmethod
    def _is_file_followup_request(text: str, context: Dict[str, Any]) -> bool:
        normalized = re.sub(r"\s+", "", text or "")
        if not normalized:
            return False
        if not context.get("active_file"):
            return False
        return normalized in {"继续任务", "继续", "接着", "然后呢", "继续处理", "继续这个任务"}

    def _has_uploaded_file_analysis_intent(self, text: str) -> bool:
        if not text:
            return False
        return any(kw in text for kw in ("总结", "概括", "分析", "提取", "读取", "查看", "内容", "翻译", "解释"))

    def _build_retrieval_filter(self, text: str, session_id: Optional[str], allow_cross_session: bool) -> Optional[Dict[str, Any]]:
        retrieval_filter = None if allow_cross_session or not session_id else {"session_id": session_id}
        if self._has_explicit_file_operation(text):
            retrieval_filter = dict(retrieval_filter or {})
            retrieval_filter["type"] = "tool_execution"
        return retrieval_filter

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

    def _has_memory_query_signal(self, text: str) -> bool:
        text_lower = text.lower()
        signals = [
            r'之前', r'上次', r'历史', r'回顾', r'问过',
            r'说过', r'聊过', r'记录', r'以前', r'过去',
            r'记得', r'忘记', r'提醒'
        ]
        return any(re.search(sig, text_lower) for sig in signals)

    def _infer_from_evidence(self, query: str, evidence: List[Dict]) -> tuple:
        if self._has_explicit_file_operation(query):
            return IntentType.TOOLS, 0.9, "查询包含明确文件名或路径，优先按 tools 处理"
        if not evidence:
            if self._has_explicit_file_operation(query):
                return IntentType.TOOLS, 0.85, "查询包含文件操作指令，但无历史证据"
            return IntentType.CHAT, 0.3, "无历史证据，默认为对话模式"

        tool_hints = 0
        skill_hints = 0
        total_score = 0.0

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

        if tool_hints >= 2:
            conf = min(0.95, 0.7 + avg_score * 0.2)
            return IntentType.TOOLS, conf, f"历史及当前查询均频繁使用工具 (工具信号 {tool_hints}次)"
        elif skill_hints >= 1:
            conf = min(0.85, 0.6 + avg_score * 0.2)
            return IntentType.SKILLS, conf, "历史中使用过相关技能"
        elif avg_score > 0.7:
            return IntentType.CHAT, 0.7, "高度相似的历史对话"
        else:
            return IntentType.CHAT, 0.5, "历史证据不足，偏向对话模式"

    def _llm_route(self, query: str, evidence: List[Dict]) -> tuple:
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
            data, parsed_payload = _parse_json_like_payload(resp)
            if isinstance(data, list):
                data = self._select_llm_intent_candidate(query, data)
            if not isinstance(data, dict):
                raise ValueError("未找到可用的意图 JSON")
            intent_str = str(data.get("intent", "chat")).lower()
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
            snippet = (parsed_payload if 'parsed_payload' in locals() else resp if 'resp' in locals() else "")[:300]
            self.monitor.warning(f"LLM 意图识别失败: {e} | 响应片段: {snippet}")
            # 注意：原代码中错误分支引用了未定义的 query 变量，现已修正为 query 参数
            return IntentType.CHAT, 0.5, f"LLM解析失败 ({e})"

    def _select_llm_intent_candidate(self, query: str, payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [item for item in payloads if isinstance(item, dict)]
        if not candidates:
            return None

        has_tool_signal = self._has_explicit_file_operation(query) or any(
            kw in query.lower() for kw in ("read_file", "write_file", "bash", ".py", ".md", ".json", "日志")
        )
        has_memory_signal = self._has_memory_query_signal(query) or self._is_memory_query(query)
        has_numbered_tasks = bool(re.search(r'\d+\s*[\.、]', query))
        has_high_risk_knowledge = any(
            kw in query for kw in ("歌词", "片段", "原文", "引用", "十首", "排名", "最出名", "最有名", "名单")
        )

        def score(item: Dict[str, Any]) -> float:
            intent = str(item.get("intent", "chat")).lower()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            bonus = 0.0
            if intent == "tools" and has_tool_signal:
                bonus += 0.18
            if intent == "hybrid" and has_tool_signal and (has_numbered_tasks or has_high_risk_knowledge):
                bonus += 0.20
            if intent == "chat" and not has_tool_signal:
                bonus += 0.05
            if intent == "memory_query" and not has_memory_signal:
                bonus -= 0.45
            if intent == "memory_query" and has_memory_signal:
                bonus += 0.20
            return confidence + bonus

        return max(candidates, key=score)

    def _rule_fallback(self, text: str) -> IntentType:
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
        defaults = {
            IntentType.CHAT: {"temperature": 0.8, "max_tokens": 1024, "tools_enabled": False},
            IntentType.TOOLS: {"temperature": 0.3, "max_tokens": 2048, "tools_enabled": True},
            IntentType.SKILLS: {"temperature": 0.5, "max_tokens": 2048, "tools_enabled": True},
            IntentType.PLAN: {"temperature": 0.4, "max_tokens": 4096, "plan_mode": True},
            IntentType.HYBRID: {"temperature": 0.6, "max_tokens": 2048, "tools_enabled": True},
            IntentType.MEMORY_QUERY: {"temperature": 0.3, "max_tokens": 2048, "tools_enabled": False},
        }
        return defaults.get(intent, defaults[IntentType.CHAT])
