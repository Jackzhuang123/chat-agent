#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from core.multi_agent import safe_fstr, WorkflowPlanState   # 添加 WorkflowPlanState

import gradio as gr

from core import (
    ConversationSummaryMiddleware,
    ContextWindowMiddleware,
    PlanModeMiddleware,
    RuntimeModeMiddleware,
    SkillInjector,
    SkillManager,
    SkillsContextMiddleware,
    ToolResultGuardMiddleware,
    UploadedFilesMiddleware,
    create_example_skills,
    create_qwen_model_forward,
    ReActMultiAgentOrchestrator,
    ToolEnforcementMiddleware,
)
from core.components import clean_react_tags
from core.components.output_cleaner import summarize_long_response
from core.langgraph_agent import LangGraphAgent as QwenAgentFramework
from core.prompts import get_system_prompt, inject_few_shot_examples
from core.state_manager import SessionContext
from ui.qwen_agent import QwenAgent
from ui.session_logger import get_logger

from core.rag_intent_router import RAGIntentRouter, IntentType
from core.vector_memory import VectorMemory
from core.context_retriever import ContextRetriever
from core.monitor_logger import get_monitor_logger, log_async_execution_time, log_event, make_trace_id

try:
    from ui.glm_agent import GLMAgent
    HAS_GLM = True
except ImportError:
    HAS_GLM = False

try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from core.clarification import ClarificationManager


class ChatController:
    """负责意图路由、会话上下文和执行编排。

    这里最容易失稳的点是“路由结果”和“控制器是否强制拆解”不一致，
    因此相关判定统一收敛到独立策略函数中，避免条件分散在主流程里。
    """
    _MEMORY_META_PATTERNS = (
        r'我之前.*问', r'回顾.*问题', r'历史.*记录', r'之前.*聊', r'上次.*说',
        r'过去.*问过', r'之前.*说了什么', r'最近.*问题', r'历史.*对话',
        r'以前.*问', r'我问过.*什么', r'我问了.*什么', r'聊天记录', r'历史问题',
    )
    _FOLLOWUP_REFERENCE_PATTERNS = (
        r'继续回答', r'继续说', r'接着说', r'继续分析', r'继续讲',
        r'上一个问题', r'最后一个问题', r'刚才那个问题', r'前面那个问题',
        r'这个问题', r'那个问题', r'回答最后一个', r'回答上一个',
    )
    _TOPIC_CARRYOVER_PATTERNS = (
        r'^继续$', r'^继续说$', r'^接着说$', r'^展开讲讲$', r'^展开一下$', r'^详细讲讲$',
        r'^那这个怎么修$', r'^这个怎么修$', r'^然后呢$', r'^细说一下$', r'^继续分析$', r'^继续任务$',
    )

    def __init__(self):
        self.logger = get_logger()
        self.monitor = get_monitor_logger()
        self._active_agent = {"instance": None, "type": "local"}
        self._GLM_API_KEY = os.environ.get("GLM_API_KEY", "").strip()
        self._GLM_AUTO_ENABLED = HAS_GLM and bool(self._GLM_API_KEY)

        self.vector_memory = VectorMemory()

        self.mode_router = RAGIntentRouter(
            vector_memory=self.vector_memory,
            llm_forward_fn=self.dynamic_routing_forward,
            confidence_threshold=0.7
        )

        self.user_sessions = {}
        self.session_states = {}
        self._init_skills()
        self.clarification_mgr = ClarificationManager()

        if self._GLM_AUTO_ENABLED:
            try:
                self._active_agent["instance"] = GLMAgent(
                    api_key=self._GLM_API_KEY,
                    model="glm-4-flash",
                    logger=self.logger,
                )
                self._active_agent["type"] = "glm"
                self.monitor.info("GLM-4-Flash 已自动启用")
            except Exception as e:
                self.monitor.warning(f"GLM 初始化失败，将使用本地模型: {e}")
                self._GLM_AUTO_ENABLED = False
        else:
            self.monitor.info("未检测到 GLM_API_KEY，使用本地 Qwen 模型")

    def _init_skills(self):
        self.monitor.info("初始化 Skills 系统...")
        create_example_skills()
        self.skill_manager = SkillManager()
        self.skill_injector = SkillInjector(self.skill_manager)
        self._available_skill_ids = list(self.skill_manager.skills_metadata.keys())
        self.monitor.info(f"发现 {len(self.skill_manager.skills_metadata)} 个技能")

    @staticmethod
    def _has_local_execution_signal(user_input: str, uploaded_files_meta: Optional[List[Dict]] = None) -> bool:
        if uploaded_files_meta:
            return True
        return bool(re.search(r'[/\\]|[\w\-]+\.\w{2,}', user_input or ""))

    @staticmethod
    def _build_request_profile(user_input: str, uploaded_files_meta: Optional[List[Dict]] = None) -> Dict[str, bool]:
        text = user_input or ""
        has_local_signal = ChatController._has_local_execution_signal(text, uploaded_files_meta)
        high_risk_knowledge = ChatController._is_high_risk_knowledge_request(text)
        numbered_tasks = re.findall(r'(?:^|[\n\r])\s*(\d+)\s*[.、]', text)
        return {
            "has_local_signal": has_local_signal,
            "high_risk_knowledge": high_risk_knowledge,
            "numbered_tasks": len(set(numbered_tasks)) >= 2,
            "avoid_breakdown": high_risk_knowledge and not has_local_signal,
            "external_only_high_risk": high_risk_knowledge and not has_local_signal,
        }

    @staticmethod
    def _should_auto_plan_request(run_mode: str, intent_type, request_profile: Dict[str, bool]) -> bool:
        """Only auto-plan requests that clearly require local execution or explicit planning."""
        if request_profile.get("avoid_breakdown"):
            return False
        if run_mode in ("tools", "plan"):
            return True
        if intent_type == IntentType.PLAN:
            return True
        return bool(request_profile.get("numbered_tasks") and request_profile.get("has_local_signal"))

    @staticmethod
    def _should_break_down_request(run_mode: str, intent_type, request_profile: Dict[str, bool],
                                   explicit_plan_mode: bool = False) -> bool:
        """Keep breakdown aligned with the routed intent and actual execution signals."""
        if request_profile.get("avoid_breakdown"):
            return False
        if explicit_plan_mode:
            return True
        if run_mode in ("tools", "plan"):
            return True
        if intent_type == IntentType.PLAN:
            return True
        return bool(run_mode == "hybrid" and request_profile.get("has_local_signal"))

    @staticmethod
    def _describe_execution_strategy(run_mode: str, intent_type, request_profile: Dict[str, bool],
                                     auto_plan: bool, needs_breakdown: bool) -> str:
        flags = []
        if request_profile.get("has_local_signal"):
            flags.append("local_signal")
        if request_profile.get("high_risk_knowledge"):
            flags.append("high_risk")
        if request_profile.get("numbered_tasks"):
            flags.append("numbered")
        if request_profile.get("external_only_high_risk"):
            flags.append("external_only")
        mode = "breakdown" if needs_breakdown else "direct"
        plan = "auto_plan" if auto_plan else "no_auto_plan"
        flag_text = ",".join(flags) if flags else "plain"
        return f"{mode} | {plan} | run_mode={run_mode} | intent={intent_type.value} | flags={flag_text}"

    @staticmethod
    def _format_evidence_labeled_response(response_text: str, run_mode: str,
                                          runtime_context: Optional[Dict] = None,
                                          tool_calls: Optional[List[Dict]] = None) -> str:
        text = (response_text or "").strip()
        if not text:
            return text
        runtime_context = runtime_context or {}
        tool_calls = tool_calls or []
        if runtime_context.get("external_only_high_risk"):
            return (
                "## 谨慎结论 / 待核实\n"
                "以下内容缺少本地或可信证据支持，因此仅保留概括，不应视为已核实事实。\n\n"
                f"{text}"
            )
        if run_mode in ("tools", "hybrid", "plan") and tool_calls:
            return (
                "## 基于工具执行\n"
                "以下结论来自本次工具调用与执行结果。\n\n"
                f"{text}"
            )
        return text

    @classmethod
    def _is_memory_meta_request(cls, text: str) -> bool:
        normalized = (text or "").strip()
        return any(re.search(pattern, normalized, re.I) for pattern in cls._MEMORY_META_PATTERNS)

    @classmethod
    def _is_followup_reference_request(cls, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        return any(re.search(pattern, normalized, re.I) for pattern in cls._FOLLOWUP_REFERENCE_PATTERNS)

    @classmethod
    def _should_replay_followup_answer(cls, text: str) -> bool:
        normalized = re.sub(r'\s+', '', (text or ""))
        if not normalized:
            return False
        pure_patterns = (
            r'^回答最后一个问题$',
            r'^回答上一个问题$',
            r'^继续回答最后一个问题$',
            r'^继续回答上一个问题$',
            r'^继续说$',
            r'^接着说$',
            r'^继续讲$',
            r'^继续回答$',
        )
        return any(re.match(pattern, normalized, re.I) for pattern in pure_patterns)

    @classmethod
    def _resolve_followup_turn(cls, history_pairs: List[List[str]],
                               user_message: str,
                               fallback_turns: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, str]]:
        if not cls._is_followup_reference_request(user_message):
            return None

        def _candidate_from_pair(user_text: str, bot_text: str) -> Optional[Dict[str, str]]:
            normalized = (user_text or "").strip()
            if not normalized:
                return None
            if cls._is_memory_meta_request(normalized):
                return None
            if cls._is_followup_reference_request(normalized):
                return None
            if normalized in ("继续", "下一个", "继续其他问题"):
                return None
            return {
                "user_message": normalized,
                "bot_response": (bot_text or "").strip(),
            }

        for pair in reversed(history_pairs or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            candidate = _candidate_from_pair(pair[0], pair[1])
            if candidate:
                return candidate

        for turn in reversed(fallback_turns or []):
            candidate = _candidate_from_pair(turn.get("user_message", ""), turn.get("bot_response", ""))
            if candidate:
                return candidate
        return None

    @staticmethod
    def _build_followup_context_message(followup_turn: Dict[str, str], current_user_message: str) -> str:
        prior_question = (followup_turn or {}).get("user_message", "").strip()
        prior_answer = (followup_turn or {}).get("bot_response", "").strip()
        answer_excerpt = prior_answer[:800] + ("..." if len(prior_answer) > 800 else "")
        return (
            "上下文：用户当前消息是在继续追问之前的相关问题。\n"
            f"上一条相关用户问题：{prior_question}\n"
            f"上一条助手回答：{answer_excerpt or '（上一条回答为空）'}\n"
            f"当前追问：{current_user_message}\n"
            "要求：请围绕“上一条相关用户问题”继续回答当前追问，不要把“最后一个问题/上一个问题”当作字面待解释对象。"
        )

    @staticmethod
    def _build_followup_replay_answer(followup_turn: Dict[str, str]) -> str:
        prior_question = (followup_turn or {}).get("user_message", "").strip()
        prior_answer = (followup_turn or {}).get("bot_response", "").strip()
        if prior_answer:
            return (
                "## 上一条相关问题的回答\n"
                f"对应问题：{prior_question}\n\n"
                f"{prior_answer}"
            )
        return f"上一条相关问题是：{prior_question}"

    @classmethod
    def _is_topic_carryover_request(cls, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        if cls._is_followup_reference_request(normalized):
            return True
        return any(re.match(pattern, normalized, re.I) for pattern in cls._TOPIC_CARRYOVER_PATTERNS)

    @classmethod
    def _extract_topic_from_message(cls, text: str) -> Optional[str]:
        normalized = re.sub(r'\s+', ' ', (text or '').strip())
        if not normalized:
            return None
        if cls._is_memory_meta_request(normalized):
            return None
        if cls._is_topic_carryover_request(normalized):
            return None
        return normalized[:160]

    @staticmethod
    def _extract_active_file_from_message(text: str) -> Optional[str]:
        normalized = (text or "").strip()
        if not normalized:
            return None
        match = re.search(r'([A-Za-z0-9_./\\-]+\.(?:py|json|md|txt|log|yaml|yml|csv|pdf|doc|docx))', normalized, re.I)
        return match.group(1) if match else None

    @classmethod
    def _update_active_file(cls, session_context: SessionContext, user_message: str,
                            followup_turn: Optional[Dict[str, str]] = None) -> Optional[str]:
        task_context = session_context.task_context
        candidate = cls._extract_active_file_from_message(user_message)
        if not candidate and followup_turn:
            candidate = cls._extract_active_file_from_message(followup_turn.get("user_message", ""))
        if not candidate:
            return task_context.get("active_file")
        previous = task_context.get("active_file")
        task_context["active_file"] = candidate
        task_context["active_file_updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        file_history = task_context.setdefault("file_history", [])
        if not file_history or file_history[-1].get("path") != candidate:
            file_history.append({
                "path": candidate,
                "source": "followup" if followup_turn else "user_message",
                "timestamp": task_context["active_file_updated_at"],
            })
            if len(file_history) > 8:
                del file_history[:-8]
        if previous and previous != candidate:
            task_context["previous_file"] = previous
        return candidate

    @classmethod
    def _build_file_followup_context(cls, session_context: SessionContext, user_message: str) -> Optional[str]:
        task_context = getattr(session_context, "task_context", {}) or {}
        active_file = task_context.get("active_file")
        if not active_file or not cls._is_topic_carryover_request(user_message):
            return None
        if cls._extract_active_file_from_message(user_message):
            return None
        return (
            f"上下文：当前正在处理的文件是 {active_file}。\n"
            f"当前用户追问：{user_message}\n"
            "要求：默认继续围绕这个文件执行或解释，除非用户明确指定了新文件。"
        )

    @classmethod
    def _resolve_topic_for_turn(cls, session_context: SessionContext, user_message: str,
                                followup_turn: Optional[Dict[str, str]] = None) -> Optional[str]:
        task_context = getattr(session_context, "task_context", {}) or {}
        current_topic = task_context.get("current_topic")
        if followup_turn:
            return followup_turn.get("user_message") or current_topic
        if cls._is_topic_carryover_request(user_message):
            return current_topic
        return cls._extract_topic_from_message(user_message) or current_topic

    @classmethod
    def _update_session_topic(cls, session_context: SessionContext, user_message: str,
                              followup_turn: Optional[Dict[str, str]] = None) -> Optional[str]:
        task_context = session_context.task_context
        resolved_topic = cls._resolve_topic_for_turn(session_context, user_message, followup_turn)
        if not resolved_topic:
            return task_context.get("current_topic")
        previous_topic = task_context.get("current_topic")
        task_context["current_topic"] = resolved_topic
        task_context["topic_updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        topic_history = task_context.setdefault("topic_history", [])
        if not topic_history or topic_history[-1].get("topic") != resolved_topic:
            topic_history.append({
                "topic": resolved_topic,
                "source": "followup" if followup_turn or cls._is_topic_carryover_request(user_message) else "user_message",
                "timestamp": task_context["topic_updated_at"],
            })
            if len(topic_history) > 8:
                del topic_history[:-8]
        if previous_topic and previous_topic != resolved_topic:
            task_context["previous_topic"] = previous_topic
        return resolved_topic

    def get_session_state(self, session_hash):
        if session_hash not in self.session_states:
            middlewares = [
                RuntimeModeMiddleware(),
                PlanModeMiddleware(),
                SkillsContextMiddleware(),
                UploadedFilesMiddleware(),
                ConversationSummaryMiddleware(
                    max_history_pairs=10,
                    keep_recent_pairs=4,
                    model_forward_fn=self.dynamic_routing_forward,
                ),
                ContextWindowMiddleware(max_chars=12000, max_messages=16),
                ToolResultGuardMiddleware(),
            ]

            agent_framework = QwenAgentFramework(
                model_forward_fn=self.dynamic_model_forward,
                work_dir=str(Path(__file__).parent.parent),
                enable_bash=True,
                max_iterations=20,
                middlewares=middlewares + [ToolEnforcementMiddleware(max_retries=2)],
                default_runtime_context={"run_mode": "chat", "plan_mode": False},
                enable_memory=True,
                enable_reflection=True,
                enable_tool_learning=True,
                checkpoint_db_path="checkpoints.db",
            )

            agent_framework.vector_memory = self.vector_memory
            agent_framework.memory = self.vector_memory
            agent_framework.context_retriever = ContextRetriever(
                vector_memory=self.vector_memory,
                max_recent_messages=5,
                max_retrieved_chunks=3,
            )

            self.session_states[session_hash] = {
                "agent_framework": agent_framework,
                "session_context": SessionContext()
            }
        return self.session_states[session_hash]

    def _ensure_log_session(self, session_hash: str) -> str:
        existing = self.user_sessions.get(session_hash)
        if existing:
            return self.logger.bind_session(existing)
        session_id = self.logger.create_session()
        self.user_sessions[session_hash] = session_id
        self.monitor.info(f"为 Gradio 会话 {session_hash} 绑定日志会话 {session_id}")
        return session_id

    def get_or_init_local_agent(self):
        if self._active_agent["instance"] is None or self._active_agent["type"] != "local":
            self.monitor.info("正在加载本地 Qwen2.5-0.5B 模型...")
            self._active_agent["instance"] = QwenAgent(logger=self.logger)
            self._active_agent["type"] = "local"
            self.monitor.info("本地模型加载完成")
        return self._active_agent["instance"]

    def dynamic_model_forward(self, messages, system_prompt="", **kwargs):
        current_agent = self._active_agent.get("instance")
        if current_agent is None:
            current_agent = self.get_or_init_local_agent()
        forward_fn = create_qwen_model_forward(current_agent)
        return forward_fn(messages, system_prompt=system_prompt, **kwargs)

    def dynamic_routing_forward(self, messages, system_prompt="", **kwargs):
        current_agent = self._active_agent.get("instance")
        if current_agent is None:
            current_agent = self.get_or_init_local_agent()
        _orig_logger = getattr(current_agent, "logger", None)
        try:
            if _orig_logger is not None:
                current_agent.logger = None
            forward_fn = create_qwen_model_forward(current_agent)
            return forward_fn(messages, system_prompt=system_prompt, **kwargs)
        finally:
            if _orig_logger is not None:
                current_agent.logger = _orig_logger

    def dynamic_stream_forward(self, messages, system_prompt="", **kwargs):
        current_agent = self._active_agent.get("instance")
        if current_agent is None:
            current_agent = self.get_or_init_local_agent()

        full_messages = list(messages)
        if system_prompt and system_prompt.strip():
            _already = (
                full_messages
                and full_messages[0].get("role") == "system"
                and full_messages[0].get("content", "").strip() == system_prompt.strip()
            )
            if not _already:
                full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        if hasattr(current_agent, "generate_stream_with_messages"):
            yield from current_agent.generate_stream_with_messages(
                full_messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
            )
        else:
            yield self.dynamic_model_forward(messages, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def extract_pdf_text(pdf_files):
        if not pdf_files or not HAS_PYPDF:
            return ""
        pdf_content = ""
        files_to_process = pdf_files if isinstance(pdf_files, list) else [pdf_files]
        for pdf_item in files_to_process:
            try:
                pdf_path = pdf_item.get('name') if isinstance(pdf_item, dict) else pdf_item
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    filename = Path(pdf_path).name if pdf_path else "unknown.pdf"
                    pdf_content += f"\n【文件: {filename}】\n"
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        if text.strip():
                            pdf_content += f"【第 {page_num} 页】\n{text}\n"
            except Exception as e:
                pdf_content += f"\n❌ 提取失败: {str(e)}\n"
        return pdf_content.strip()

    @staticmethod
    def extract_uploaded_file_meta(pdf_files):
        if not pdf_files:
            return []
        files_to_process = pdf_files if isinstance(pdf_files, list) else [pdf_files]
        uploaded_files = []
        for pdf_item in files_to_process:
            pdf_path = pdf_item.get('name') if isinstance(pdf_item, dict) else pdf_item
            if not pdf_path:
                continue
            path_obj = Path(pdf_path)
            try:
                file_size = path_obj.stat().st_size if path_obj.exists() else 0
            except Exception:
                file_size = 0
            uploaded_files.append({
                "filename": path_obj.name,
                "path": str(path_obj),
                "size": file_size,
            })
        return uploaded_files

    @log_async_execution_time
    async def bot_response(
            self,
            history,
            sys_prompt,
            temp,
            top_p_val,
            max_tok,
            plan_mode_enabled,
            pdf_files,
            gr_request: gr.Request,
    ):
        global _asyncio, partial_results
        if not history or history[-1][1] is not None:
            yield history, "💬 等待输入..."
            return

        user_message = history[-1][0]
        history_without_last = history[:-1]

        session_hash = gr_request.session_hash
        self._ensure_log_session(session_hash)
        trace_id = make_trace_id()
        session_state = self.get_session_state(session_hash)
        agent_framework = session_state["agent_framework"]
        session_context = session_state["session_context"]
        session_context.task_context["session_id"] = session_hash
        session_context.runtime_context["session_id"] = session_hash

        # --- 向量记忆存储 ---
        if agent_framework.vector_memory and user_message:
            if user_message != "[未设置]" and not user_message.startswith("["):
                agent_framework.vector_memory.add(
                    content=f"User: {user_message}",
                    metadata={
                        "type": "user_question",
                        "original_question": user_message,
                        "session_id": session_hash,
                        "active_file": session_context.task_context.get("active_file"),
                    },
                    importance=0.95,
                    auto_score=False,
                    skip_duplicate=True,
                )

        # --- PDF 处理 ---
        pdf_info = ""
        uploaded_files_meta = self.extract_uploaded_file_meta(pdf_files)
        if uploaded_files_meta and HAS_PYPDF:
            pdf_text = self.extract_pdf_text(pdf_files)
            if pdf_text:
                pdf_info = f"\n\n【PDF内容】:\n{pdf_text[:1000]}..."

        # --- 清理历史 ---
        _filter_prefixes = (
            "已达到最大迭代次数",
            "[⚠️ 工具模式错误]",
            "[GLM API 错误]",
            "[在进行中...]",
            "[未设置]",
        )

        def _is_stale_tool_plan(a_text: str) -> bool:
            if not a_text:
                return False
            if "<tool>" in a_text and "</tool>" in a_text:
                return True
            if "<input>" in a_text and "</input>" in a_text:
                return True
            return False

        _clean_pairs = []
        for u, a in history_without_last:
            if not u:
                continue
            _a_invalid = (
                    not a
                    or any(a.startswith(p) for p in _filter_prefixes)
                    or _is_stale_tool_plan(a)
            )
            if _a_invalid:
                _clean_pairs.append([u, None])
            else:
                _clean_pairs.append([u, a])
        chat_history = [[u, a] for u, a in _clean_pairs if u and a]
        current_log_session = self.user_sessions.get(session_hash)
        recent_logged_turns = self.logger.get_recent_turns(current_log_session, limit=12) if current_log_session else []
        followup_turn = self._resolve_followup_turn(_clean_pairs, user_message, recent_logged_turns)
        current_topic = self._update_session_topic(session_context, user_message, followup_turn)
        active_file = self._update_active_file(session_context, user_message, followup_turn)

        # --- 路由器上下文 ---
        skill_list_for_router = []
        for skill_id in self._available_skill_ids:
            meta = self.skill_manager.skills_metadata.get(skill_id, {})
            skill_list_for_router.append({
                "id": skill_id,
                "name": meta.get("name", skill_id),
                "tags": meta.get("tags", []),
                "description": meta.get("description", ""),
            })
        router_context = {
            "session_id": session_hash,
            "available_skills": skill_list_for_router,
            "uploaded_files": uploaded_files_meta,
            "history": [[u, a] for u, a in _clean_pairs[-3:] if u],
            "current_topic": current_topic,
            "active_file": active_file,
        }

        # --- 意图路由 ---
        intent_result = self.mode_router.route(user_message + pdf_info, router_context)
        run_mode = intent_result.intent.value
        confidence = intent_result.confidence

        if run_mode == "memory_query":
            current_session_questions = []
            for turn in recent_logged_turns:
                q = turn.get("user_message", "")
                if q and not self._is_memory_meta_request(q):
                    current_session_questions.append({
                        "timestamp": turn.get("timestamp", ""),
                        "user_message": q,
                    })
            questions = current_session_questions or self.logger.get_all_user_questions(skip_placeholders=True)
            memory_answer = "\n".join(
                f"- {q.get('timestamp', '')[:19]}：{q.get('user_message', '')}"
                for q in (questions[-10:] if questions else [])
            ) if questions else "暂无历史问题记录。"
            history[-1][1] = memory_answer
            yield history, "📚 历史记忆查询"
            return

        if followup_turn and self._should_replay_followup_answer(user_message):
            replay_answer = self._build_followup_replay_answer(followup_turn)
            history[-1][1] = replay_answer
            self.logger.log_message(
                user_message=user_message,
                bot_response=replay_answer,
                execution_time=0.0,
                tokens_used=0,
                model="followup_replay",
                runtime_context={
                    "trace_id": trace_id,
                    "session_id": session_hash,
                    "run_mode": "chat",
                    "followup_turn": followup_turn,
                    "followup_replay": True,
                },
                execution_log=[{
                    "iteration": -1,
                    "type": "followup_replay",
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }],
            )
            yield history, "💬 对话模式（历史追问回放）"
            return

        # ===== 文件操作缺少路径时使用统一的澄清卡片 =====
        if run_mode == "tools":
            has_path = bool(re.search(r'[/\\]|[\w\-]+\.\w{2,}', user_message))
            if not has_path and not uploaded_files_meta:
                missing_items = ["请提供要处理的文件路径或文件名"]
                clarification_card = self.clarification_mgr.trigger_clarification(
                    runtime_context={},
                    missing_info=missing_items,
                    source="pre_check"
                )
                history[-1][1] = clarification_card
                yield history, "❓ 需要澄清（缺少文件路径）"
                return
        # ==========================================================

        suggested = intent_result.suggested_params
        if run_mode in ("tools", "skills"):
            actual_temp = 0.2
            actual_max_tokens = suggested.get("max_tokens", 4096)
        elif run_mode == "plan":
            actual_temp = 0.4
            actual_max_tokens = suggested.get("max_tokens", 4096)
        else:
            actual_temp = suggested.get("temperature", temp)
            actual_max_tokens = suggested.get("max_tokens", max_tok)

        skill_ids = []
        if run_mode in ("skills", "hybrid"):
            msg_lower = user_message.lower()
            for sid in self._available_skill_ids:
                if sid.lower() in msg_lower or any(
                        kw in msg_lower for kw in sid.lower().replace("-", " ").split()
                ):
                    skill_ids.append(sid)
            if not skill_ids and run_mode == "skills":
                skill_ids = self._available_skill_ids[:1]

        request_profile = self._build_request_profile(user_message, uploaded_files_meta)
        if request_profile["external_only_high_risk"] and run_mode == "hybrid":
            run_mode = "chat"
        _numbered_tasks = re.findall(r'(?:^|[\n\r])\s*(\d+)\s*[.、]', user_message)
        _auto_plan = self._should_auto_plan_request(run_mode, intent_result.intent, request_profile)
        _effective_plan_mode = bool(plan_mode_enabled) or _auto_plan
        needs_breakdown = self._should_break_down_request(
            run_mode,
            intent_result.intent,
            request_profile,
            explicit_plan_mode=bool(plan_mode_enabled),
        )
        execution_strategy = self._describe_execution_strategy(
            run_mode,
            intent_result.intent,
            request_profile,
            _auto_plan,
            needs_breakdown,
        )
        self.monitor.info("执行策略决策: %s", execution_strategy)

        runtime_context = {
            "trace_id": trace_id,
            "session_id": session_hash,
            "run_mode": run_mode,
            "plan_mode": _effective_plan_mode,
            "request_profile": request_profile,
            "external_only_high_risk": request_profile["external_only_high_risk"],
            "followup_turn": followup_turn,
            "current_topic": current_topic,
            "active_file": active_file,
            "uploaded_files": uploaded_files_meta,
            "selected_skills": skill_ids,
            "intent_reasons": [intent_result.reasoning],
            "router_evidence": intent_result.evidence,
            "execution_strategy": execution_strategy,
        }

        if self.clarification_mgr.is_clarification_active(runtime_context):
            if self.clarification_mgr.process_user_clarification(
                    user_message,
                    runtime_context,
                    agent_framework.vector_memory if hasattr(agent_framework, 'vector_memory') else None
            ):
                self.monitor.info("用户澄清信息已处理，即将继续任务")
            else:
                self.monitor.warning("处理用户澄清回复失败")

        if not sys_prompt or sys_prompt.strip() == "":
            sys_prompt = get_system_prompt(
                mode=run_mode,
                work_dir=str(Path.cwd()),
                skills_context="\n".join(skill_ids) if skill_ids else "",
            )

        base_messages = [{"role": "system", "content": sys_prompt}]
        if run_mode == "tools":
            base_messages = inject_few_shot_examples(base_messages, max_examples=2)
        for u, a in chat_history:
            base_messages.append({"role": "user", "content": u})
            base_messages.append({"role": "assistant", "content": a})
        _user_msg_with_ctx = user_message + pdf_info
        base_messages.append({"role": "user", "content": _user_msg_with_ctx})

        if self.clarification_mgr.get_clarified_facts(runtime_context):
            facts_msg = self.clarification_mgr.build_clarification_context_message(runtime_context)
            base_messages.append(facts_msg)

        if skill_ids:
            skill_content_parts = []
            skill_infos = []
            for sid in skill_ids:
                content = self.skill_manager.get_skill_detail(sid)
                if not content:
                    continue
                meta = self.skill_manager.skills_metadata.get(sid, {})
                skill_name = meta.get("name", sid)
                skill_content_parts.append(
                    f'<skill-loaded name="{sid}">\n# 技能: {skill_name}\n\n{content}\n</skill-loaded>'
                )
                skill_infos.append({
                    "id": sid,
                    "name": skill_name,
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                })
            if skill_content_parts:
                runtime_context["skill_contexts"] = skill_infos
                skill_inject_msg = "\n\n".join(skill_content_parts) + "\n\n请参考上方技能知识完成任务。"
                base_messages.append({"role": "user", "content": skill_inject_msg})

        start_time = time.time()
        execution_log = [{
            "iteration": -1,
            "type": "intent_analysis",
            "run_mode": run_mode,
            "skill_ids": skill_ids,
            "reasons": [intent_result.reasoning],
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }]

        review_info = {}
        _react_orchestrator = None              # 修正：预置 orchestrator 变量，避免 NameError

        context_prefixes = []
        clarified_facts = self.clarification_mgr.get_clarified_facts(runtime_context)
        if clarified_facts:
            facts_text = "；".join(clarified_facts)
            context_prefixes.append(f"背景：上一轮任务需要澄清，用户补充了以下信息：{facts_text}。")
        if followup_turn:
            context_prefixes.append(self._build_followup_context_message(followup_turn, user_message))
        file_followup_context = self._build_file_followup_context(session_context, user_message)
        if file_followup_context:
            context_prefixes.append(file_followup_context)
        elif self._is_topic_carryover_request(user_message) and current_topic:
            context_prefixes.append(
                f"上下文：当前会话的主话题是：{current_topic}。\n"
                f"当前用户追问：{user_message}\n"
                "要求：请围绕这个主话题继续回答，不要把代词或省略句单独理解成新任务。"
            )
        if context_prefixes:
            user_message_with_context = "\n\n".join(context_prefixes + [f"现在用户说：{user_message}"]) + pdf_info
        else:
            user_message_with_context = user_message + pdf_info

        if needs_breakdown:
            self.monitor.info("进入 ReAct 多 Agent 模式")
            try:
                _react_orchestrator = ReActMultiAgentOrchestrator(
                    react_framework=agent_framework,
                    max_plan_steps=100,
                    max_retries=1,
                    clarification_mgr=self.clarification_mgr
                )
                session_state["orchestrator"] = _react_orchestrator

                partial_results = []
                final_response = ""

                # 直接迭代，不添加总超时（内部已有步骤级超时）
                async for event in _react_orchestrator.run_stream(
                        user_input=user_message_with_context,
                        session=session_context,
                        context=runtime_context,
                        temperature=actual_temp,
                        top_p=top_p_val,
                        max_tokens=actual_max_tokens,
                ):
                    if event["type"] == "step_complete":
                        partial_results.append(
                            f"✅ 步骤 {event['step_id']}: {safe_fstr(event['action'])}\n{safe_fstr(event['result'])}")
                    elif event["type"] == "step_failed":
                        partial_results.append(f"❌ 步骤 {event['step_id']} 失败: {safe_fstr(event['error'])}")
                    elif event["type"] == "step_blocked":
                        # ✅ 新增 safe_fstr
                        partial_results.append(f"🔒 步骤 {event['step_id']} 需要澄清: {safe_fstr(event['message'])}")
                    elif event["type"] == "error":
                        error_msg = event.get("message", "未知错误")
                        # ✅ 新增 safe_fstr
                        final_response = f"❌ 任务执行失败：{safe_fstr(error_msg)}"
                        self.monitor.error(f"多 Agent 执行错误: {safe_fstr(error_msg)}")
                        break
                    elif event["type"] == "final":
                        final_response = event["response"]  # 直接赋值，无需转义

                raw_agent_response = final_response if final_response.strip() else (
                    "\n\n".join(partial_results) if partial_results else "（无输出）")

                # 检查是否有步骤需要澄清
                workflow_state = _react_orchestrator.workflow_states.get(trace_id)
                if isinstance(workflow_state, WorkflowPlanState) and workflow_state.pending_clarifications:
                    clarification_card = self.clarification_mgr.trigger_clarification(
                        runtime_context, missing_items=["请提供更多信息"], source="multi_agent"
                    )
                    history[-1][1] = clarification_card
                    yield history, f"❓ 需要澄清"
                    return

            except Exception as e:
                import traceback
                traceback.print_exc()  # <-- 加这一行
                raw_agent_response = f"❌ 任务执行失败：{str(e)}"
        else:
            formatted_history = []
            for user_msg, bot_msg in chat_history:
                if user_msg:
                    formatted_history.append({"role": "user", "content": user_msg})
                if bot_msg:
                    formatted_history.append({"role": "assistant", "content": bot_msg})
            request_thread_id = f"{session_hash}_{trace_id}"
            res = await agent_framework.run(
                user_input=user_message_with_context,
                session=session_context,
                history=formatted_history,
                runtime_context=runtime_context,
                temperature=actual_temp,
                top_p=top_p_val,
                max_tokens=actual_max_tokens,
                thread_id=request_thread_id,
            )
            raw_agent_response = self._format_evidence_labeled_response(
                res["response"],
                run_mode=run_mode,
                runtime_context=runtime_context,
                tool_calls=res.get("tool_calls", []),
            )

        # ========== 澄清检测（高风险知识请求不弹卡片） ==========
        should_clarify = False
        if isinstance(raw_agent_response, str) and any(
                status in raw_agent_response for status in ("STATUS: NEEDS_CONTEXT", "STATUS: BLOCKED")):
            if not self._is_high_risk_knowledge_request(user_message):
                should_clarify = True
            else:
                self.monitor.info("高风险知识请求，忽略 STATUS，保留当前回答")
        elif review_info.get("completed") is False and review_info.get("quality") == "poor":
            should_clarify = True
        elif run_mode in ("tools", "hybrid") and len(raw_agent_response.strip()) < 50 and any(
                kw in raw_agent_response for kw in ["请提供", "缺少", "不知道", "需要更多"]):
            should_clarify = True

        execution_time = time.time() - start_time

        if should_clarify and self.clarification_mgr.get_clarify_round(
                runtime_context) < self.clarification_mgr.MAX_CLARIFY_ROUNDS:
            missing_items = ["请提供缺失的文件路径或参数", "或描述您希望完成的具体操作"]
            if "file_not_found" in raw_agent_response.lower():
                missing_items = ["文件未找到，请提供正确的文件名或路径"]
            elif "NEEDS_CONTEXT" in raw_agent_response:
                missing_items = ["根据模型要求，请补充上下文信息"]

            clarification_card = self.clarification_mgr.trigger_clarification(
                runtime_context, missing_items, source="agent_response"
            )
            history[-1][1] = clarification_card

            try:
                self.logger.log_message(
                    user_message=user_message,
                    bot_response=raw_agent_response.strip(),
                    execution_time=execution_time,
                    tokens_used=0,
                    model="glm-4-flash" if self._active_agent["type"] == "glm" else "Qwen2.5-0.5B",
                    runtime_context=runtime_context,
                    execution_log=execution_log,
                )
            except Exception as log_err:
                self.monitor.warning(f"澄清轮次日志记录失败: {log_err}")

            yield history, f"❓ 需要澄清（第 {runtime_context.get('clarify_round', 1)} 次）"
            return

        # --- 最终输出与日志 ---
        final_response = raw_agent_response
        final_response = clean_react_tags(final_response)
        #final_response = summarize_long_response(final_response, max_chars=0)
        # 增加兜底：若最终响应仍为空，给一个默认提示
        if not final_response.strip():
            final_response = "✅ 任务已处理，但未生成文本结果。请查看控制台日志。"
        history[-1][1] = final_response

        _mode_icons = {"chat": "💬", "tools": "🔧", "skills": "🎓", "hybrid": "⚡", "plan": "📋"}
        _mode_labels = {"chat": "对话模式", "tools": "工具模式", "skills": "技能模式", "hybrid": "混合模式",
                        "plan": "计划模式"}
        mode_display = f"{_mode_icons.get(run_mode, '🤖')} {_mode_labels.get(run_mode, run_mode)} (置信度: {confidence:.2f})"
        if _auto_plan and not plan_mode_enabled:
            mode_display += f" | 📋自动计划（{len(set(_numbered_tasks))}个子任务）"
        if skill_ids:
            mode_display += f" | 技能: {', '.join(skill_ids)}"

        self.logger.log_message(
            user_message=user_message,
            bot_response=final_response,
            execution_time=execution_time,
            tokens_used=0,
            model="glm-4-flash" if self._active_agent["type"] == "glm" else "Qwen2.5-0.5B",
            runtime_context=runtime_context,
            execution_log=execution_log,
        )
        self._check_performance_alert(execution_time, raw_agent_response, trace_id)  # 修正：使用原始响应

        # ---------- 恢复未完成工作流 ----------
        if _react_orchestrator is not None and user_message.strip() in ("继续", "继续其他问题", "下一个"):
            resume_trace_id = self._get_or_resume_workflow(session_hash)
            if resume_trace_id:
                result = await _react_orchestrator.run(
                    user_input=user_message,
                    session=session_context,
                    context=runtime_context,
                    resume=True,
                    temperature=actual_temp,
                    top_p=top_p_val,
                    max_tokens=actual_max_tokens
                )
                final_response = result.get("final_response", "")
                history[-1][1] = final_response
                # 补充日志记录
                self.logger.log_message(
                    user_message=user_message,
                    bot_response=final_response,
                    execution_time=time.time() - start_time,
                    tokens_used=0,
                    model="glm-4-flash" if self._active_agent["type"] == "glm" else "Qwen2.5-0.5B",
                    runtime_context=runtime_context,
                    execution_log=[],
                )
                yield history, "🔄 继续执行剩余任务"
                return
        # -------------------------------------------------------------

        yield history, mode_display

    @staticmethod
    def _extract_missing_from_status(text: str) -> List[str]:
        """从 STATUS 响应中提取 REASON 和 RECOMMENDATION"""
        reasons = []
        for line in text.split('\n'):
            if line.startswith("REASON:") or line.startswith("RECOMMENDATION:"):
                reasons.append(line.split(":", 1)[1].strip())
        return reasons if reasons else ["需要更多信息"]

    @staticmethod
    def _is_high_risk_knowledge_request(user_input: str) -> bool:
        markers = ["歌词", "片段", "最出名", "十首", "top", "排名", "排行榜", "具体数据",
                   "列出", "给出", "截取"]
        return any(m in user_input for m in markers)

    def _get_or_resume_workflow(self, session_hash: str) -> Optional[str]:
        session_state = self.session_states.get(session_hash)
        if not session_state:
            return None
        orchestrator = session_state.get("orchestrator")
        if not orchestrator:
            return None
        for trace_id, state in orchestrator.workflow_states.items():
            if not state.is_all_done() and not state.has_blocked():
                return trace_id
        return None

    def on_engine_change(self, engine_choice):
        is_glm = "GLM" in engine_choice
        if is_glm:
            if not self._GLM_AUTO_ENABLED:
                return "⚪ 未配置（请设置环境变量 GLM_API_KEY）"
            current_model = self._active_agent["instance"].model if (
                self._active_agent.get("instance") and self._active_agent["type"] == "glm"
            ) else "glm-4-flash"
            if self._active_agent["type"] != "glm":
                try:
                    self._active_agent["instance"] = GLMAgent(
                        api_key=self._GLM_API_KEY,
                        model=current_model,
                        logger=self.logger,
                    )
                    self._active_agent["type"] = "glm"
                    self.monitor.info(f"切换至 GLM 引擎，模型: {current_model}")
                except Exception as e:
                    self.monitor.error(f"切换 GLM 引擎失败: {e}")
                    return f"❌ 切换失败: {e}"
            return f"✅ 已启用（模型: {current_model}）"
        else:
            self._active_agent["instance"] = None
            self._active_agent["type"] = "local"
            self.monitor.info("切换至本地 Qwen 引擎")
            return "⚪ 本地 Qwen 模式（将在首次对话时加载模型）"

    def on_model_change(self, model_name):
        if not self._GLM_AUTO_ENABLED:
            return "⚪ 未配置 GLM_API_KEY，无法切换"
        try:
            self._active_agent["instance"] = GLMAgent(
                api_key=self._GLM_API_KEY,
                model=model_name,
                logger=self.logger,
            )
            self._active_agent["type"] = "glm"
            self.monitor.info(f"GLM 模型切换为: {model_name}")
            return f"✅ 已切换模型: {model_name}"
        except Exception as e:
            self.monitor.error(f"GLM 模型切换失败: {e}")
            return f"❌ 切换失败: {e}"

    def _check_performance_alert(self, execution_time: float, raw_agent_response: str, trace_id: str):
        if execution_time > 120:
            self.monitor.warning(f"请求 {trace_id} 执行超时 ({execution_time:.2f}s)")

        session_hash = trace_id.split('_')[0] if '_' in trace_id else "global"
        if not hasattr(self, '_failure_counters'):
            self._failure_counters: Dict[str, int] = {}
        counter = self._failure_counters.get(session_hash, 0)
        if "❌" in raw_agent_response or "失败" in raw_agent_response:
            counter += 1
            self._failure_counters[session_hash] = counter
            if counter >= 3:
                self.monitor.error(f"会话 {session_hash} 连续失败 {counter} 次，触发降级")
        else:
            self._failure_counters[session_hash] = 0

    async def handle_message_with_workflow_resume(
        self,
        history,
        sys_prompt,
        temp,
        top_p_val,
        max_tok,
        plan_mode_enabled,
        pdf_files,
        gr_request: gr.Request
    ):
        async for response in self.bot_response(
            history, sys_prompt, temp, top_p_val, max_tok,
            plan_mode_enabled, pdf_files, gr_request
        ):
            yield response
