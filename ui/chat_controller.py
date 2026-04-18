#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
from core.langgraph_agent import LangGraphAgent as QwenAgentFramework
from core.prompts import get_system_prompt, inject_few_shot_examples
from core.state_manager import SessionContext
from ui.qwen_agent import QwenAgent
from ui.session_logger import get_logger

# 新增：RAG 意图路由器
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


class ChatController:
    def __init__(self):
        self.logger = get_logger()                     # 业务会话日志
        self.monitor = get_monitor_logger()            # 监控日志
        self._active_agent = {"instance": None, "type": "local"}
        self._GLM_API_KEY = os.environ.get("GLM_API_KEY", "").strip()
        self._GLM_AUTO_ENABLED = HAS_GLM and bool(self._GLM_API_KEY)

        # 初始化向量记忆（供 RAG 路由器使用）
        self.vector_memory = VectorMemory()

        # 使用 RAG 意图路由器
        self.mode_router = RAGIntentRouter(
            vector_memory=self.vector_memory,
            llm_forward_fn=self.dynamic_routing_forward,
            confidence_threshold=0.7
        )

        self.user_sessions = {}
        self.session_states = {}
        self._init_skills()

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

            # 强制共享同一份向量记忆，避免路由器与执行框架各自检索不同记忆导致结果不一致。
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
    async def bot_response(self, history, sys_prompt, temp, top_p_val, max_tok, plan_mode_enabled, pdf_files, gr_request: gr.Request):
        if not history or history[-1][1] is not None:
            yield history, "💬 等待输入..."
            return

        user_message = history[-1][0]
        history_without_last = history[:-1]
        self.logger.create_session()

        session_hash = gr_request.session_hash
        trace_id = make_trace_id()
        session_state = self.get_session_state(session_hash)
        agent_framework = session_state["agent_framework"]
        session_context = session_state["session_context"]

        # 存储用户消息到向量记忆
        if agent_framework.vector_memory and user_message:
            if user_message != "[未设置]" and not user_message.startswith("["):
                agent_framework.vector_memory.add(
                    content=f"User: {user_message}",
                    metadata={
                        "type": "user_question",
                        "original_question": user_message,
                        "session_id": session_hash,
                    },
                    importance=0.95,
                    auto_score=False,
                    skip_duplicate=True
                )

        pdf_info = ""
        uploaded_files_meta = self.extract_uploaded_file_meta(pdf_files)
        self.monitor.info(
            f"收到用户消息，trace_id={trace_id}, session={session_hash[:8]}, 消息预览: {user_message[:100]}"
        )
        log_event(
            "request_received",
            "收到用户请求",
            trace_id=trace_id,
            session=session_hash[:8],
            user_len=len(user_message),
            has_uploads=bool(uploaded_files_meta),
        )
        if uploaded_files_meta and HAS_PYPDF:
            pdf_text = self.extract_pdf_text(pdf_files)
            if pdf_text:
                pdf_info = f"\n\n【PDF内容】:\n{pdf_text[:1000]}..."

        # 清理历史中无效的占位消息
        _filter_prefixes = ("已达到最大迭代次数", "[⚠️ 工具模式错误]", "[GLM API 错误]", "[在进行中...]", "[未设置]")
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

        # 准备路由器上下文
        skill_list_for_router = []
        for skill_id in self._available_skill_ids:
            meta = self.skill_manager.skills_metadata.get(skill_id, {})
            skill_list_for_router.append({
                "id": skill_id,
                "name": meta.get("name", skill_id),
                "tags": meta.get("tags", []),
                "description": meta.get("description", "")
            })

        router_context = {
            "available_skills": skill_list_for_router,
            "uploaded_files": uploaded_files_meta,
            "history": [[u, a] for u, a in _clean_pairs[-3:] if u]
        }

        # ---------- RAG 意图路由 ----------
        intent_result = self.mode_router.route(user_message + pdf_info, router_context)
        run_mode = intent_result.intent.value
        confidence = intent_result.confidence

        # ===== 新增：处理 MEMORY_QUERY 意图 =====
        if run_mode == "memory_query":
            questions = self.logger.get_all_user_questions(skip_placeholders=True)
            if questions:
                lines = ["📚 根据历史记录，您曾问过以下问题："]
                for q in questions[-10:]:  # 最近10条
                    timestamp = q.get("timestamp", "")[:19]
                    user_msg = q.get("user_message", "")
                    lines.append(f"- {timestamp}：{user_msg}")
                memory_answer = "\n".join(lines)
            else:
                memory_answer = "暂无历史问题记录。"
            history[-1][1] = memory_answer
            yield history, "📚 历史记忆查询"
            return

        suggested = intent_result.suggested_params

        actual_temp = suggested.get("temperature", temp)
        actual_max_tokens = suggested.get("max_tokens", max_tok)

        # 技能匹配
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

        # 自动计划检测
        import re as _re
        _numbered_tasks = _re.findall(r'(?:^|[\n\r])\s*(\d+)\s*[.、]', user_message)
        _auto_plan = len(set(_numbered_tasks)) >= 2
        _effective_plan_mode = bool(plan_mode_enabled) or _auto_plan

        needs_breakdown = (run_mode == "plan" or run_mode == "multi_agent"
                           or intent_result.intent == IntentType.PLAN or _auto_plan)

        _mode_icons = {"chat": "💬", "tools": "🔧", "skills": "🎓", "hybrid": "⚡", "plan": "📋"}
        _mode_labels = {"chat": "对话模式", "tools": "工具模式", "skills": "技能模式", "hybrid": "混合模式", "plan": "计划模式"}
        mode_display_text = f"{_mode_icons.get(run_mode, '🤖')} {_mode_labels.get(run_mode, run_mode)} (置信度: {confidence:.2f})"
        if _auto_plan and not plan_mode_enabled:
            mode_display_text += f" | 📋自动计划模式（检测到{len(set(_numbered_tasks))}个子任务）"
        if skill_ids:
            mode_display_text += f" | 技能: {', '.join(skill_ids)}"
        if intent_result.reasoning:
            mode_display_text += f"\n原因: {intent_result.reasoning[:100]}"
        if needs_breakdown:
            mode_display_text += " | 📋任务拆解中..."

        self.monitor.info(f"意图路由结果: {run_mode}, 置信度: {confidence:.2f}, 计划模式: {_effective_plan_mode}")
        log_event(
            "intent_route",
            "完成意图路由",
            trace_id=trace_id,
            session=session_hash[:8],
            run_mode=run_mode,
            confidence=f"{confidence:.2f}",
            plan_mode=_effective_plan_mode,
            auto_plan=_auto_plan,
            skills=",".join(skill_ids) if skill_ids else "-",
        )

        # 关键修改：将路由阶段检索到的证据传递给 Agent
        runtime_context = {
            "trace_id": trace_id,
            "run_mode": run_mode,
            "plan_mode": _effective_plan_mode,
            "uploaded_files": uploaded_files_meta,
            "selected_skills": skill_ids,
            "intent_reasons": [intent_result.reasoning],
            "router_evidence": intent_result.evidence,  # 新增：预检索证据
        }

        # 构建系统提示词
        if not sys_prompt or sys_prompt.strip() == "":
            skills_text = ""
            if skill_ids:
                skills_text = "\n".join([f"- {sid}" for sid in skill_ids])
            sys_prompt = get_system_prompt(
                mode=run_mode,
                work_dir=str(Path.cwd()),
                skills_context=skills_text
            )

        # 构建消息
        base_messages = [{"role": "system", "content": sys_prompt}]
        if run_mode == "tools":
            base_messages = inject_few_shot_examples(base_messages, max_examples=2)

        for u, a in chat_history:
            base_messages.append({"role": "user", "content": u})
            base_messages.append({"role": "assistant", "content": a})

        _user_msg_with_ctx = user_message + pdf_info
        base_messages.append({"role": "user", "content": _user_msg_with_ctx})

        # 注入技能内容
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
                skill_inject_msg = (
                    "\n\n".join(skill_content_parts)
                    + "\n\n请参考上方技能知识完成任务。"
                )
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

        # ---------- 核心执行：根据是否多步骤决定调用方式 ----------
        if needs_breakdown:
            # 使用 ReActMultiAgentOrchestrator 进行多步骤执行
            self.monitor.info("进入 ReAct 多 Agent 模式")
            try:
                import asyncio as _asyncio
                _react_orchestrator = ReActMultiAgentOrchestrator(
                    react_framework=agent_framework,
                    max_plan_steps=6,
                    max_retries=1,
                )
                _ma_result = await _asyncio.wait_for(
                    _react_orchestrator.run(
                        user_input=user_message + pdf_info,
                        session=session_context,
                        context=runtime_context,
                        temperature=actual_temp,
                        top_p=top_p_val,
                        max_tokens=actual_max_tokens,
                    ),
                    timeout=240,
                )
                final_response = _ma_result.get("final_response", "")
                if not final_response:
                    final_response = "多 Agent 执行完成，但未能生成最终回答。"
                _plan = _ma_result.get("plan", {})
                _steps = _ma_result.get("step_results", [])
                _n_success = sum(1 for s in _steps if s.get("success"))
                execution_log.append({
                    "iteration": 0,
                    "type": "react_multi_agent_run",
                    "plan_complexity": _plan.get("complexity", "unknown"),
                    "steps_total": len(_steps),
                    "steps_success": _n_success,
                    "duration": _ma_result.get("duration", 0),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                if _ma_result.get("final_artifact"):
                    execution_log.append({
                        "type": "final_artifact",
                        "final_facts_count": len(_ma_result["final_artifact"].get("final_facts", [])),
                        "unresolved_count": len(_ma_result["final_artifact"].get("unresolved_issues", [])),
                        "evidence_count": len(_ma_result["final_artifact"].get("evidence_used", [])),
                    })
                history[-1][1] = final_response
                self.monitor.info(f"多 Agent 执行完成，成功步骤: {_n_success}/{len(_steps)}")
                log_event(
                    "request_completed",
                    "多 Agent 请求完成",
                    trace_id=trace_id,
                    session=session_hash[:8],
                    mode=run_mode,
                    steps_total=len(_steps),
                    steps_success=_n_success,
                    duration_s=f"{_ma_result.get('duration', 0):.2f}",
                )
                yield history, mode_display_text
            except _asyncio.TimeoutError:
                _timeout_msg = "⏰ 多步骤任务执行超时（超过4分钟），已强制中断。"
                self.monitor.error("ReAct多Agent模式超时")
                history[-1][1] = _timeout_msg
                yield history, "⏰ 执行超时"
                execution_log.append({
                    "type": "multi_agent_timeout",
                    "error": "asyncio.TimeoutError after 240s",
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                log_event(
                    "request_failed",
                    "多 Agent 请求超时",
                    trace_id=trace_id,
                    session=session_hash[:8],
                    mode=run_mode,
                    error="timeout",
                )
            except Exception as e:
                self.monitor.warning(f"ReAct多Agent模式异常: {e}")
                error_msg = f"❌ 任务执行失败：{str(e)}。"
                history[-1][1] = error_msg
                yield history, mode_display_text
                execution_log.append({
                    "type": "multi_agent_failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                log_event(
                    "request_failed",
                    "多 Agent 请求失败",
                    trace_id=trace_id,
                    session=session_hash[:8],
                    mode=run_mode,
                    error=str(e)[:200],
                )
        else:
            # 直接调用 Agent
            try:
                # 将 Gradio 格式的二维历史 [[user, assistant], ...] 转换为消息字典列表
                formatted_history = []
                for user_msg, bot_msg in chat_history:
                    if user_msg:
                        formatted_history.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        formatted_history.append({"role": "assistant", "content": bot_msg})

                res = await agent_framework.run(
                    user_input=user_message + pdf_info,
                    session=session_context,
                    history=formatted_history,
                    runtime_context=runtime_context,
                    temperature=actual_temp,
                    top_p=top_p_val,
                    max_tokens=actual_max_tokens,
                    thread_id=session_hash,
                )
                final_response = res["response"]
                if isinstance(res.get("tool_calls"), list):
                    for i, tc in enumerate(res["tool_calls"]):
                        if isinstance(tc, dict):
                            execution_log.append({
                                "iteration": i,
                                "type": "tool_call",
                                "tool": tc.get("tool"),
                                "success": tc.get("success")
                            })
                        else:
                            # 记录异常格式，避免崩溃
                            self.monitor.warning(f"工具调用记录格式异常，已跳过：{type(tc)} - {tc}")

                history[-1][1] = final_response
                yield history, mode_display_text
                self.monitor.info(f"Agent 执行完成，工具调用数: {len(res['tool_calls'])}")
                execution_log.append({
                    "type": "agent_summary",
                    "iterations": res.get("iterations", 0),
                    "duration": res.get("duration", 0),
                    "thread_id": res.get("thread_id"),
                    "reflection_summary_present": bool(res.get("reflection_summary")),
                })
                log_event(
                    "request_completed",
                    "Agent 请求完成",
                    trace_id=trace_id,
                    session=session_hash[:8],
                    mode=run_mode,
                    tool_calls=len(res["tool_calls"]),
                    iterations=res.get("iterations", 0),
                    duration_s=f"{res.get('duration', 0):.2f}",
                )
            except Exception as e:
                self.monitor.error(f"Agent 执行异常: {e}")
                error_msg = f"❌ 执行失败: {str(e)}"
                history[-1][1] = error_msg
                yield history, mode_display_text
                execution_log.append({
                    "type": "runtime_error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                log_event(
                    "request_failed",
                    "Agent 请求失败",
                    trace_id=trace_id,
                    session=session_hash[:8],
                    mode=run_mode,
                    error=str(e)[:200],
                )

        # 记录日志
        execution_time = time.time() - start_time
        _cur_agent = self._active_agent.get("instance")
        _agent_type = self._active_agent.get("type", "local")
        _cur_model_name = "glm-4-flash" if _agent_type == "glm" else "Qwen2.5-0.5B"
        self.logger.log_message(
            user_message=user_message,
            bot_response=history[-1][1],
            execution_time=execution_time,
            tokens_used=0,
            model=_cur_model_name,
            runtime_context=runtime_context,
            execution_log=execution_log,
        )
        self.monitor.info(f"请求处理完成，trace_id={trace_id}, 总耗时 {execution_time:.2f}s")

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
        """
        兼容 web_agent_with_skills.py 中绑定的事件名称。
        直接调用 bot_response 并转发其流式输出。
        """
        async for response in self.bot_response(
            history, sys_prompt, temp, top_p_val, max_tok,
            plan_mode_enabled, pdf_files, gr_request
        ):
            yield response
