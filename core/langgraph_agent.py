# -*- coding: utf-8 -*-
"""基于 LangGraph 的增强 Agent 核心，支持 Checkpoint、断点续跑、状态查询"""

import asyncio
import inspect
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, TypedDict, Literal

import aiosqlite
import numpy as np
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END


def _is_alive_patch(self) -> bool:
    """检查 aiosqlite 连接是否存活"""
    try:
        _ = self._conn
        return True
    except (ValueError, AttributeError, aiosqlite.Error):
        return False


if not hasattr(aiosqlite.core.Connection, 'is_alive'):
    aiosqlite.core.Connection.is_alive = _is_alive_patch

from .agent_tools import ToolExecutor, ToolParser
from .state_manager import SessionContext
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory
from .agent_middlewares import (
    AgentMiddleware,
    SearchBeforeBuildingMiddleware,
    CompletionStatusMiddleware,
)
from .context_retriever import ContextRetriever
from .reflection import EnhancedReflectionEngine
from .monitor_logger import get_monitor_logger, log_event
from .prompts import get_system_prompt

from .components import (
    clean_react_tags,
    strip_trailing_tool_call,
    inject_task_context,
    detect_loop,
    should_retry_tool_format,
)


# ============================================================================
# 状态定义（LangGraph 核心）
# ============================================================================
class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    user_input: str
    session_id: str
    thread_id: str
    iteration: int
    max_iterations: int
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_response: Optional[str]
    runtime_context: Dict[str, Any]
    session_context: Dict[str, Any]
    tool_call_history: List[Dict[str, Any]]
    reflection_history: List[Dict[str, Any]]
    raw_response_cache: List[str]
    _needs_retry: bool
    _tool_enforcement_retry: int
    _checkpoint_id: Optional[str]


def _safe_model_call(model_forward_fn: Callable, messages: list,
                     system_prompt: str = "", **kwargs) -> str:
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        raise

    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        full_text = ""
        try:
            for chunk in result:
                if isinstance(chunk, str):
                    if not full_text:
                        full_text = chunk
                    elif chunk.startswith(full_text):
                        full_text = chunk
                    elif len(chunk) >= max(16, len(full_text) // 2) and chunk[:1] in "{[`\"":
                        full_text = chunk
                    else:
                        full_text += chunk
        except Exception:
            pass
        return full_text

    if not isinstance(result, str):
        return str(result) if result is not None else ""
    return result


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


def _sanitize_state_update(update: Dict[str, Any]) -> Dict[str, Any]:
    return _make_json_serializable(update)


def _should_skip_rag_for_file_request(task: str) -> bool:
    if not task:
        return False
    lowered = task.lower()
    has_file_action = any(keyword in task for keyword in ("查看", "读取", "读", "解析", "分析", "打开", "查找"))
    has_file_hint = bool(re.search(r"\b[\w\-]+\.(json|log|txt|md|py|yaml|yml|csv)\b", lowered))
    has_path_hint = "/" in task or "\\" in task
    return has_file_action and (has_file_hint or has_path_hint)


def _requires_tool_evidence(task: str, run_mode: str) -> bool:
    if run_mode not in {"tools", "hybrid", "plan"}:
        return False
    text = task or ""
    lowered = text.lower()
    file_or_path = bool(re.search(r'[/\\]|[\w\-]+\.(json|log|txt|md|py|yaml|yml|csv|pdf|docx?)\b', lowered))
    execution_keywords = (
        "读取", "查看", "分析", "解析", "打开", "查找", "扫描", "列出", "写入", "修改",
        "删除", "运行", "执行", "提取", "生成文件", "保存", "grep", "find", "ls", "cat",
    )
    return file_or_path or any(keyword in text for keyword in execution_keywords)


def _normalize_run_mode(run_mode: str, plan_mode: bool = False) -> str:
    mode = (run_mode or "chat").lower()
    if mode == "skills":
        return "hybrid"
    if plan_mode and mode == "chat":
        return "plan"
    if mode not in {"chat", "tools", "plan", "hybrid"}:
        return "chat"
    return mode


def _append_unique_fact(facts: List[Dict[str, Any]], fact: Dict[str, Any], max_items: int = 12) -> None:
    signature = json.dumps(fact, ensure_ascii=False, sort_keys=True)
    if any(json.dumps(item, ensure_ascii=False, sort_keys=True) == signature for item in facts):
        return
    facts.append(fact)
    if len(facts) > max_items:
        del facts[:-max_items]


def _update_facts_ledger(session: SessionContext, tool_name: str, tool_args: Dict[str, Any], result_obj: Dict[str, Any], success: bool) -> None:
    ledger = session.task_context.setdefault("facts_ledger", {
        "confirmed_facts": [],
        "file_facts": [],
        "failed_actions": [],
        "open_questions": [],
    })
    for key in ("confirmed_facts", "file_facts", "failed_actions", "open_questions"):
        ledger.setdefault(key, [])

    if success:
        if tool_name == "read_file":
            path = result_obj.get("path", tool_args.get("path", ""))
            file_fact = result_obj.get("file_facts") or {"path": path, "summary": ""}
            _append_unique_fact(ledger["file_facts"], file_fact, max_items=8)
            _append_unique_fact(ledger["confirmed_facts"], {
                "kind": "file_read",
                "tool": tool_name,
                "path": path,
            })
        elif tool_name == "bash":
            stdout = (result_obj.get("stdout") or result_obj.get("output") or "").strip()
            _append_unique_fact(ledger["confirmed_facts"], {
                "kind": "command_executed",
                "tool": tool_name,
                "command": tool_args.get("command", "")[:160],
                "output": stdout[:160],
            })
        else:
            _append_unique_fact(ledger["confirmed_facts"], {
                "kind": "tool_success",
                "tool": tool_name,
                "args": {k: str(v)[:80] for k, v in tool_args.items()},
            })
    else:
        _append_unique_fact(ledger["failed_actions"], {
            "tool": tool_name,
            "error": str(result_obj.get("error", ""))[:200],
            "args": {k: str(v)[:80] for k, v in tool_args.items()},
        }, max_items=10)


def _summarize_facts_ledger(session: SessionContext) -> Dict[str, int]:
    ledger = session.task_context.get("facts_ledger", {})
    return {
        "confirmed_facts": len(ledger.get("confirmed_facts", [])),
        "file_facts": len(ledger.get("file_facts", [])),
        "failed_actions": len(ledger.get("failed_actions", [])),
        "open_questions": len(ledger.get("open_questions", [])),
    }


def session_to_state(session: SessionContext) -> Dict[str, Any]:
    return {
        "tool_history": session.tool_history,
        "reflection_history": session.reflection_history,
        "read_files_cache": dict(session.read_files_cache),
        "task_context": _make_json_serializable(session.task_context),
        "current_tool_chain_id": session.current_tool_chain_id,
        "runtime_context": _make_json_serializable(session.runtime_context),
    }


def state_to_session(state_data: Dict[str, Any], session: SessionContext) -> None:
    session.tool_history = state_data.get("tool_history", [])
    session.reflection_history = state_data.get("reflection_history", [])
    session.read_files_cache = state_data.get("read_files_cache", {})
    task_ctx = state_data.get("task_context", {})
    task_ctx.setdefault("completed_steps", [])
    task_ctx.setdefault("failed_attempts", [])
    task_ctx.setdefault("current_topic", None)
    task_ctx.setdefault("topic_history", [])
    task_ctx.setdefault("topic_updated_at", None)
    task_ctx.setdefault("subtask_status", {})
    task_ctx.setdefault("facts_ledger", {
        "confirmed_facts": [],
        "file_facts": [],
        "failed_actions": [],
        "open_questions": [],
    })
    session.task_context = task_ctx
    session.current_tool_chain_id = state_data.get("current_tool_chain_id")
    session.runtime_context = state_data.get("runtime_context", {})


# ============================================================================
# 节点类（异步节点）
# ============================================================================
class AgentNode:
    def __init__(self, fw: 'LangGraphAgent'):
        self.fw = fw

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        fw = self.fw
        runtime_context = state["runtime_context"]
        trace_id = runtime_context.get("trace_id", "-")
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        messages = state["messages"].copy()
        iteration = state.get("iteration", 0)
        # 提前获取 run_mode，确保全局可见
        run_mode = runtime_context.get("run_mode", "chat")

        # 反思检查
        if fw.reflection:
            should_cont, reason = fw.reflection.should_continue(session.reflection_history)
            if not should_cont:
                fw.monitor.warning(f"反思引擎建议停止: {reason}")
                return _sanitize_state_update({"final_response": f"⚠️ 任务中断: {reason}"})

        # RAG 上下文增强 - 仅首次迭代执行。
        if fw.context_retriever and iteration == 0:
            current_task = session.task_context.get("current_task", "")
            session_id = session.task_context.get("session_id") or runtime_context.get("session_id")
            if _should_skip_rag_for_file_request(current_task):
                fw.monitor.info("检测到明确文件读取请求，仅注入工具操作历史作为参考")
                messages = fw.context_retriever.augment_messages(
                    messages=messages,
                    query=current_task,
                    filter_tool_types=["bash", "read_file", "write_file"],
                    session_id=session_id,
                )
            else:
                session.task_context["session_id"] = session_id
                messages = fw.context_retriever.enhance_messages_with_rag(
                    messages=messages,
                    current_query=current_task,
                    session_context=session.task_context,
                )
            fw.monitor.debug(f"RAG 增强后消息数: {len(messages)}")

        # 注入任务上下文
        messages = inject_task_context(messages, session, fw.vector_memory)

        # 注入反思摘要
        if fw.reflection and session.reflection_history:
            refl_summary = fw.reflection.get_reflection_summary(session.reflection_history)
            messages.append({"role": "system", "content": refl_summary})

        strategy = runtime_context.get("_strategy_switch")
        if strategy:
            action = strategy.get("action")
            if action == "switch_tool":
                new_tool = strategy["new_tool"]
                reason = strategy.get("reason", "")
                switch_msg = {
                    "role": "user",
                    "content": (
                        f"⚠️ 系统指令：由于 {reason}，"
                        f"请在本次回复中**仅使用**工具 '{new_tool}' 完成任务。"
                        f"不要尝试之前失败的工具。"
                    )
                }
                messages.append(switch_msg)
            elif action == "suggest_search":
                template = strategy.get("query_template", "find . -name '*'")
                search_msg = {
                    "role": "user",
                    "content": (
                        f"⚠️ 你刚才多次未找到目标文件。"
                        f"请先使用 bash 执行以下搜索命令：\n"
                        f"bash\n{{\"command\": \"{template}\"}}"
                    )
                }
                messages.append(search_msg)
            elif action == "abort":
                fw.monitor.warning(f"策略要求中止，返回 NEEDS_CONTEXT。原因: {strategy.get('reason')}")
                return _sanitize_state_update({
                    "final_response": (
                        "STATUS: NEEDS_CONTEXT\n"
                        f"REASON: {strategy.get('reason')}\n"
                        "RECOMMENDATION: 请提供更具体的指示或修正任务。"
                    )
                })

            # 使用后立即清除，避免下一轮重复应用
            runtime_context.pop("_strategy_switch", None)
        # =========================================

        # 中间件依赖这些字段判断“是否必须先调工具”以及“是否属于文件任务追问”。
        runtime_context.setdefault("user_input", state.get("user_input", "") or session.task_context.get("current_task", ""))
        runtime_context.setdefault("current_task", session.task_context.get("current_task", ""))
        runtime_context.setdefault("active_file", session.task_context.get("active_file"))

        # ---------- 强化中间件：确保搜索优先和完成报告默认注入 ----------
        if run_mode in ("tools", "hybrid"):
            # 搜索优先中间件（防止模型自创代码）
            if not any(isinstance(mw, SearchBeforeBuildingMiddleware) for mw in fw.middlewares):
                fw.middlewares.insert(0, SearchBeforeBuildingMiddleware())
            # 完成状态报告中间件（要求模型明确输出 DONE/BLOCKED）
            if not any(isinstance(mw, CompletionStatusMiddleware) for mw in fw.middlewares):
                fw.middlewares.insert(0, CompletionStatusMiddleware())
        # ----------------------------------------------------------------

        # ---------- 新增：工具推荐注入 ----------
        if fw.tool_learner and len(session.tool_history) > 0:
            executed_tools = [h["tool"] for h in session.tool_history]
            recommendations = fw.tool_learner.recommend_next_tools(
                current_task=session.task_context.get("current_task", ""),
                executed_tools=executed_tools,
                top_k=2
            )
            if recommendations:
                rec_text = "💡 基于历史，建议优先考虑以下工具：\n"
                for rec in recommendations:
                    rec_text += f"- {rec['tool']}（置信度 {rec['confidence']:.2f}）\n"
                rec_text += "你可以根据实际情况选择其他工具。"
                messages.append({"role": "system", "content": rec_text})
        # ----------------------------------------

        # ---------- 新增：全局工具成功率提示 ----------
        if fw.tool_learner and run_mode in ("tools", "hybrid"):
            top_tools = fw.tool_learner.recommend_next_tools(
                current_task=session.task_context.get("current_task", ""),
                executed_tools=[],  # 未执行任何工具时，获得基于任务类型的初始推荐
                top_k=3
            )
            if top_tools:
                hint_lines = ["💡 基于当前任务类型和历史成功率，工具选择建议："]
                for rec in top_tools:
                    hint_lines.append(f"- {rec['tool']}（推荐度 {rec['confidence']:.2f}）")
                hint_lines.append("请优先考虑上述工具，但可根据实际情况调整。")
                messages.append({"role": "system", "content": "\n".join(hint_lines)})
        # ------------------------------------------------

        # 前置中间件
        for mw in fw.middlewares:
            if hasattr(mw, "process_before_llm"):
                messages = await mw.process_before_llm(messages, runtime_context)

        temp = runtime_context.get("temperature", 0.7)
        top_p = runtime_context.get("top_p", 0.9)
        max_tok = runtime_context.get("max_tokens", 8192)

        # 调用 LLM
        try:
            effective_system_prompt = fw.build_system_prompt(runtime_context)
            log_event(
                "agent_llm_call",
                "开始调用 LLM",
                trace_id=trace_id,
                iteration=iteration,
                run_mode=run_mode,
                message_count=len(messages),
                system_prompt_mode=_normalize_run_mode(run_mode, plan_mode=bool(runtime_context.get("plan_mode"))),
            )
            response = _safe_model_call(
                fw.model_forward_fn, messages, effective_system_prompt,
                temperature=temp, top_p=top_p, max_tokens=max_tok
            )
            fw.monitor.debug(f"LLM 响应长度: {len(response)} 字符")
        except Exception as e:
            fw.monitor.error(f"模型调用失败: {e}")
            return _sanitize_state_update({"final_response": f"❌ 模型调用失败: {e}"})

        # 后置中间件
        for mw in fw.middlewares:
            if hasattr(mw, "process_after_llm"):
                response = await mw.process_after_llm(response, runtime_context)

        if runtime_context.get("_needs_retry"):
            runtime_context["_needs_retry"] = False
            return _sanitize_state_update({"iteration": iteration, "runtime_context": runtime_context})

        # 更新 raw_response_cache
        raw_cache = state.get("raw_response_cache", [])
        raw_cache.append(response)
        if len(raw_cache) > 20:
            raw_cache = raw_cache[-20:]

        # 纯对话模式
        if run_mode == "chat":
            final_response = clean_react_tags(strip_trailing_tool_call(response))
            if fw.vector_memory:
                fw.vector_memory.add(
                    content=f"Assistant: {final_response}",
                    metadata={
                        "role": "assistant",
                        "type": "assistant_response",
                        "session_id": session.task_context.get("session_id") or runtime_context.get("session_id"),
                        "active_file": session.task_context.get("active_file") or runtime_context.get("active_file"),
                    },
                )
            fw.monitor.info("对话模式，直接返回最终回答")
            return _sanitize_state_update({"final_response": final_response, "raw_response_cache": raw_cache})

        # 解析工具调用
        tool_calls_raw = fw.tool_parser.parse_tool_calls(response)
        tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

        if not tool_calls:
            has_successful_tool_result = any(
                bool(item.get("success"))
                for item in (state.get("tool_results", []) or [])
                if isinstance(item, dict)
            )
            # ----- 新增：检测到 STATUS: BLOCKED/NEEDS_CONTEXT 直接返回最终回答 -----
            if re.search(r'STATUS:\s*(BLOCKED|NEEDS_CONTEXT)', response, re.IGNORECASE):
                fw.monitor.info("检测到 STATUS: BLOCKED/NEEDS_CONTEXT，直接作为最终回答")
                return _sanitize_state_update({
                    "final_response": response,
                    "raw_response_cache": raw_cache,
                    "messages": [{"role": "assistant", "content": response}],
                    "session_context": session_to_state(session),
                })
            # -----------------------------------------------------------------
            tool_intent = should_retry_tool_format(
                response,
                has_successful_tool_result=has_successful_tool_result,
            )

            if tool_intent:
                fw.monitor.warning("检测到工具调用意图但解析失败，尝试格式纠正")

                # ====== 从最后一条用户消息中提取文件名 ======
                target_filename = None
                for msg_dict in reversed(messages):
                    if msg_dict.get("role") == "user":
                        content = msg_dict.get("content", "")
                        # 跳过系统注入的上下文消息（以特定标记开头）
                        if not content.startswith("✅") and not content.startswith("❌") \
                                and not content.startswith("<") and not content.startswith("📚"):
                            match = re.search(r'[\w\-.]+\.(json|log|txt|md|py|yaml|yml|csv)', content)
                            if match:
                                target_filename = match.group(0)
                            break

                if target_filename:
                    correction = (
                        "⚠️ 系统检测到你尝试调用工具，但格式不正确。\n"
                        f"请严格按以下格式重新输出，必须读取文件：{target_filename}\n\n"
                        f"read_file\n"
                        f'{{"path": "{target_filename}"}}\n\n'
                        "不要使用 read_file(\"...\") 或任何函数调用格式。"
                    )
                else:
                    correction = (
                        "⚠️ 系统检测到你尝试调用工具，但格式不正确。\n"
                        "请严格按以下格式重新输出（工具名独占一行，紧接着 JSON 参数）：\n\n"
                        "execute_python\n"
                        "{\"code\": \"你的代码\"}\n\n"
                        "read_file\n"
                        "{\"path\": \"文件路径\"}\n\n"
                        "不要使用 execute_python -c \"...\" 或 read_file(\"...\") 等格式。"
                    )
                # ====== 提取文件名结束 ======

                retry_messages = messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": correction},
                ]
                try:
                    effective_system_prompt = fw.build_system_prompt(runtime_context)
                    retry_response = _safe_model_call(
                        fw.model_forward_fn, retry_messages, effective_system_prompt,
                        temperature=temp, top_p=top_p, max_tokens=max_tok
                    )
                    retry_tool_calls_raw = fw.tool_parser.parse_tool_calls(retry_response)
                    retry_tool_calls = [{"name": name, "args": args} for name, args in retry_tool_calls_raw]
                    if retry_tool_calls:
                        fw.monitor.info("格式纠正成功，解析到工具调用")
                        raw_cache.append(retry_response)
                        return _sanitize_state_update({
                            "tool_calls": retry_tool_calls,
                            "iteration": iteration + 1,
                            "session_context": session_to_state(session),
                            "runtime_context": runtime_context,
                            "messages": [{"role": "assistant", "content": retry_response}],
                            "raw_response_cache": raw_cache,
                        })
                    fw.monitor.warning("格式纠正后仍未解析到工具调用")
                    return _sanitize_state_update({"final_response": f"⚠️ 工具调用格式纠正后仍未解析成功。\n{clean_react_tags(strip_trailing_tool_call(retry_response))}",
                            "raw_response_cache": raw_cache})
                except Exception as e:
                    fw.monitor.error(f"工具格式纠正重试失败: {e}")
                    return _sanitize_state_update({"final_response": f"❌ 工具格式纠正重试失败: {e}", "raw_response_cache": raw_cache})
            elif has_successful_tool_result:
                fw.monitor.info("当前链路已有成功工具结果，含工具名的自然语言将按结果总结处理，不再触发格式纠正")

            current_task = session.task_context.get("current_task", "")
            if _requires_tool_evidence(current_task, run_mode) and not has_successful_tool_result:
                if "此步骤为知识问答，不需要调用工具，直接输出完整回答。" in current_task:
                    fw.monitor.info("知识回答步骤未使用工具，保留模型原始自然语言回答")
                else:
                    fw.monitor.warning("当前任务需要工具证据，但模型未产生有效工具调用")
                    hint = (
                        "⚠️ 当前任务需要先通过工具获取事实，不能直接给出自然语言结论。\n"
                        "请改为显式调用合适的工具，例如 `read_file`、`bash`、`list_dir` 或 `write_file`。"
                    )
                    return _sanitize_state_update({
                        "final_response": hint,
                        "raw_response_cache": raw_cache,
                        "messages": messages + [{"role": "assistant", "content": response}],
                        "session_context": session_to_state(session),
                    })

            # 无工具调用且无工具意图，直接采纳回答
            final = clean_react_tags(strip_trailing_tool_call(response))
            if fw.vector_memory:
                fw.vector_memory.add(
                    content=f"Assistant: {final}",
                    metadata={
                        "role": "assistant",
                        "type": "assistant_response",
                        "session_id": session.task_context.get("session_id") or runtime_context.get("session_id"),
                        "active_file": session.task_context.get("active_file") or runtime_context.get("active_file"),
                    },
                )
            fw.monitor.info("模型返回自然语言回答，直接作为最终输出")
            updated_messages = messages + [{"role": "assistant", "content": response}]
            return _sanitize_state_update({
                "final_response": final,
                "raw_response_cache": raw_cache,
                "messages": updated_messages,
                "session_context": session_to_state(session),
            })

        fw.monitor.info(f"解析到 {len(tool_calls)} 个工具调用")
        log_event(
            "agent_tool_calls_parsed",
            "模型输出已解析为工具调用",
            trace_id=trace_id,
            iteration=iteration,
            tool_count=len(tool_calls),
        )
        return _sanitize_state_update({
            "tool_calls": tool_calls,
            "iteration": iteration + 1,
            "session_context": session_to_state(session),
            "runtime_context": runtime_context,
            "messages": [{"role": "assistant", "content": response}],
            "raw_response_cache": raw_cache,
        })


# ============================================================================
# ToolNode 增强版（预检 + 反思建议）
# ============================================================================
class ToolNode:
    def __init__(self, fw: 'LangGraphAgent'):
        self.fw = fw

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        fw = self.fw
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return _sanitize_state_update({})
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        runtime_context = state["runtime_context"]
        runtime_context.setdefault("user_input", state.get("user_input", "") or session.task_context.get("current_task", ""))
        runtime_context.setdefault("current_task", session.task_context.get("current_task", ""))
        trace_id = runtime_context.get("trace_id", "-")

        async def _run_one(tc: Dict) -> Dict:
            tool_name, tool_args = tc["name"], tc["args"]
            log_event(
                "tool_started",
                "开始执行工具",
                trace_id=trace_id,
                tool=tool_name,
                args=json.dumps(tool_args, ensure_ascii=False)[:200],
            )

            for mw in fw.middlewares:
                if hasattr(mw, "process_before_tool"):
                    tool_name, tool_args = await mw.process_before_tool(tool_name, tool_args, runtime_context)

            # ---------- read_file 预检 ----------
            if tool_name == "read_file" and "path" in tool_args:
                path = tool_args["path"]
                found = await asyncio.to_thread(fw.tool_executor._fuzzy_find_file, str(path), True, 3)
                if not found and not await asyncio.to_thread(fw.tool_executor._is_direct_path, str(path)):
                    result_obj = {
                        "error": f"文件未找到: {path}",
                        "hint": f"文件 {path} 不存在。将尝试全局搜索，若仍找不到请用户提供正确路径。"
                    }
                    return {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result_obj,
                        "success": False,
                        "exec_time": 0,
                    }

            start_time = time.perf_counter()
            result_str = await asyncio.to_thread(
                fw.tool_executor.execute_tool, tool_name, tool_args
            )
            exec_time = time.perf_counter() - start_time

            for mw in fw.middlewares:
                if hasattr(mw, "process_after_tool"):
                    result_str = await mw.process_after_tool(tool_name, tool_args, result_str, runtime_context)

            try:
                result_obj = json.loads(result_str)
            except Exception:
                result_obj = {"output": result_str}

            success = not result_obj.get("error")
            fw.monitor.debug(f"工具 {tool_name} 执行{'成功' if success else '失败'}，耗时 {exec_time:.3f}s")

            if fw.reflection:
                fw.reflection.record_result(
                    tool_name, result_obj,
                    context={"recent_tools": [h.get("tool") for h in session.tool_history[-5:]]},
                    history=session.reflection_history
                )

            session.tool_history.append({
                "tool": tool_name,
                "args": json.dumps(tool_args),
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "chain_id": session.current_tool_chain_id,
            })
            if success:
                session.task_context["completed_steps"].append(tool_name)
                if tool_name == "read_file" and tool_args.get("path"):
                    resolved_path = Path(tool_args["path"]).resolve().as_posix()
                    session.read_files_cache[resolved_path] = datetime.utcnow().isoformat()
                    session.task_context["active_file"] = result_obj.get("path") or resolved_path
                    session.task_context["active_file_updated_at"] = datetime.utcnow().isoformat()
            else:
                session.task_context["failed_attempts"].append(tool_name)
                session.task_context["last_failed_tool"] = tool_name

            _update_facts_ledger(session, tool_name, tool_args, result_obj, success)
            log_event(
                "tool_completed",
                "工具执行完成",
                trace_id=trace_id,
                tool=tool_name,
                success=success,
                exec_ms=f"{exec_time * 1000:.1f}",
                ledger=json.dumps(_summarize_facts_ledger(session), ensure_ascii=False),
            )

            return {"tool": tool_name, "args": tool_args, "result": result_obj, "success": success, "exec_time": exec_time}

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])

        # ========== 插入：策略切换检测 ==========
        recent_failures = []
        for r in results:
            if not r.get("success"):
                error_msg = r.get("result", {}).get("error", "")
                if fw.reflection:
                    analysis = fw.reflection._analyze_error(error_msg)
                    recent_failures.append({
                        "tool": r.get("tool"),
                        "error_type": analysis.get("type"),
                        "error": error_msg
                    })

        if recent_failures and fw.reflection:
            # 合并之前的失败记录（只保留最近3条）
            previous_failures = runtime_context.get("_recent_failures", [])
            all_failures = (previous_failures + recent_failures)[-3:]
            runtime_context["_recent_failures"] = all_failures

            action_plan = fw.reflection.get_action_plan(all_failures, max_errors=3)
            if action_plan and action_plan.get("action") != "none":
                runtime_context["_strategy_switch"] = action_plan
                fw.monitor.info(f"策略切换触发: {action_plan['action']} ({action_plan.get('reason')})")
            else:
                runtime_context.pop("_strategy_switch", None)  # 清除旧指令
        # =========================================

        # 将成功的 read_file 内容存入向量记忆
        if fw.vector_memory:
            for r in results:
                if r.get("tool") == "read_file" and r.get("success"):
                    tool_name = r["tool"]
                    result_obj = r["result"]
                    path = result_obj.get("path", "")
                    content = result_obj.get("content", "")
                    if content:
                        snippet = content[:800] + ("..." if len(content) > 800 else "")
                        current_task = session.task_context.get("current_task", "")
                        fw.vector_memory.add(
                            content=f"文件 {path} 内容摘要：{snippet}",
                            metadata={
                                "type": "file_content",
                                "tool": "read_file",
                                "path": path,
                                "success": True,
                                "original_question": current_task,
                                "session_id": session.task_context.get("session_id") or runtime_context.get("session_id"),
                                "active_file": session.task_context.get("active_file") or path,
                            },
                            importance=0.8,
                            auto_score=False,
                            skip_duplicate=True,
                        )
                        fw.monitor.debug(f"已将文件 {path} 摘要存入向量记忆")

        if fw.context_retriever:
            for r in results:
                fw.context_retriever.add_tool_observation(
                    tool_name=r["tool"],
                    tool_args=r.get("args", {}),
                    result=r["result"],
                    success=r["success"],
                    original_question=session.task_context.get("current_task"),
                    session_id=session.task_context.get("session_id") or runtime_context.get("session_id"),
                    active_file=session.task_context.get("active_file"),
                    failed_step=session.task_context.get("last_failed_tool"),
                )
                fw.monitor.debug(f"工具结果已写入向量库: {r['tool']}")

        result_messages = []
        for r in results:
            tool = r["tool"]
            success = r["success"]
            result_obj = r["result"]

            if tool == "read_file" and success:
                content = result_obj.get("content", "")
                path = result_obj.get("path", "")
                if len(content) > 2000:
                    content = content[:2000] + "\n...[内容过长已截断]"
                result_messages.append({
                    "role": "user",
                    "content": (
                        f"✅ 工具执行成功：已读取文件 {path}\n"
                        f"【当前任务唯一可信依据】以下是刚刚读取到的真实文件内容。\n"
                        f"你接下来必须只依据这份内容回答用户问题，禁止引用其他历史记录、记忆或猜测。\n"
                        f"如果文件内容里没有答案，就明确说明内容不足。\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"{content}"
                    )
                })
            else:
                summary = json.dumps(result_obj, ensure_ascii=False)[:500]
                msg_content = f"{'✅' if success else '❌'} {tool}: {summary}"

                if not success and fw.reflection:
                    suggestion = fw.reflection.get_last_suggestion(tool)
                    if suggestion:
                        msg_content += f"\n\n💡 系统建议：{suggestion}"

                result_messages.append({"role": "user", "content": msg_content})

        return _sanitize_state_update({
            "tool_results": list(results),
            "tool_calls": [],
            "session_context": session_to_state(session),
            "messages": result_messages,
        })


class FinalizeNode:
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        if state.get("final_response"):
            return _sanitize_state_update({})

        raw_cache = state.get("raw_response_cache", [])
        tool_keywords = {"execute_python", "read_file", "write_file", "edit_file", "list_dir", "bash"}

        for resp in reversed(raw_cache):
            has_tool_keyword = any(kw in resp for kw in tool_keywords)
            if not has_tool_keyword:
                cleaned = clean_react_tags(strip_trailing_tool_call(resp))
                if len(cleaned.strip()) > 5:
                    return _sanitize_state_update({"final_response": cleaned})

        if raw_cache:
            last_resp = raw_cache[-1]
            cleaned = clean_react_tags(strip_trailing_tool_call(last_resp))
            if len(cleaned.strip()) > 5:
                return _sanitize_state_update({"final_response": cleaned})

        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                cleaned = clean_react_tags(strip_trailing_tool_call(msg["content"]))
                if cleaned.strip():
                    return _sanitize_state_update({"final_response": cleaned})

        return _sanitize_state_update({"final_response": "任务完成"})


class LoopDetectedNode:
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        completed = session.task_context.get("completed_steps", [])
        read_files = list(session.read_files_cache.keys())
        hint = (
            f"⚠️ 检测到工具调用循环。已完成步骤：{completed}。"
            f"已读文件：{read_files}。请基于已有信息作答。"
        )
        return _sanitize_state_update({"final_response": hint})


# ============================================================================
# 主类：LangGraphAgent
# ============================================================================
class LangGraphAgent:
    def __init__(
        self,
        model_forward_fn: Callable,
        work_dir: Optional[str] = None,
        enable_bash: bool = False,
        max_iterations: int = 20,
        middlewares: Optional[List[AgentMiddleware]] = None,
        enable_memory: bool = True,
        enable_reflection: bool = True,
        enable_parallel: bool = True,
        enable_tool_learning: bool = True,
        default_runtime_context: Optional[Dict] = None,
        checkpoint_db_path: str = "checkpoints.db",
    ):
        self.model_forward_fn = model_forward_fn
        self.max_iterations = max_iterations
        self.middlewares = middlewares or []
        self.default_runtime_context = default_runtime_context or {}
        self.tool_executor = ToolExecutor(work_dir=work_dir, enable_bash=enable_bash)
        self.tool_parser = ToolParser()
        self.vector_memory = VectorMemory() if enable_memory else None
        self.memory = self.vector_memory
        self.tool_learner = AdaptiveToolLearner() if enable_tool_learning else None
        self.reflection = EnhancedReflectionEngine() if enable_reflection else None
        if self.reflection and self.tool_learner:
            self.reflection.attach_tool_learner(self.tool_learner)

        self.system_prompt = get_system_prompt(
            mode="tools",
            work_dir=str(self.tool_executor.work_dir.resolve())
        )

        self.checkpoint_db_path = Path(checkpoint_db_path)
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = None
        self._checkpoint_cm = None
        self.graph = None
        self._conn = None
        self.monitor = get_monitor_logger()

        if self.vector_memory:
            self.context_retriever = ContextRetriever(
                vector_memory=self.vector_memory,
                max_recent_messages=5,
                max_retrieved_chunks=3,
            )
        else:
            self.context_retriever = None

        self.monitor.info(f"LangGraphAgent 初始化完成，Checkpoint DB: {self.checkpoint_db_path}")

    def build_system_prompt(self, runtime_context: Optional[Dict[str, Any]] = None) -> str:
        runtime_context = runtime_context or {}
        plan_mode = bool(runtime_context.get("plan_mode"))
        mode = _normalize_run_mode(runtime_context.get("run_mode", "chat"), plan_mode=plan_mode)
        skills_context = runtime_context.get("skills_context_text", "暂无激活的技能")
        return get_system_prompt(
            mode=mode,
            work_dir=str(self.tool_executor.work_dir.resolve()),
            skills_context=skills_context,
        )

    async def _ensure_graph(self):
        if self.graph is not None:
            return
        self.monitor.debug("正在初始化异步 SQLite 检查点...")
        self._checkpoint_cm = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_db_path))
        self.checkpointer = await self._checkpoint_cm.__aenter__()
        self.graph = self._build_graph()
        self.monitor.debug("异步检查点初始化完成")

    async def close(self):
        if self._checkpoint_cm:
            await self._checkpoint_cm.__aexit__(None, None, None)
            self._checkpoint_cm = None
            self.checkpointer = None
            self.graph = None

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", AgentNode(self))
        builder.add_node("tools", ToolNode(self))
        builder.add_node("finalize", FinalizeNode())
        builder.add_node("loop_detected", LoopDetectedNode())
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", self._should_continue)
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        builder.add_edge("loop_detected", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _should_continue(self, state: AgentState) -> Literal["tools", "finalize", "loop_detected"]:
        if state.get("final_response"):
            self.monitor.debug("已有最终回答，进入 finalize")
            return "finalize"
        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            self.monitor.info(f"达到最大迭代次数 {self.max_iterations}，进入 finalize")
            return "finalize"
        if not state.get("tool_calls"):
            self.monitor.debug("无工具调用，进入 finalize")
            return "finalize"
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        session.raw_response_cache = state.get("raw_response_cache", [])
        if detect_loop(session):
            self.monitor.warning("检测到工具调用循环，进入 loop_detected")
            return "loop_detected"
        return "tools"

    def _build_messages(self, user_input: str, history: Optional[List[Dict]]) -> List[Dict]:
        msgs = history[:] if history else []
        msgs.append({"role": "user", "content": user_input})
        return msgs

    async def run(
            self,
            user_input: str,
            session: SessionContext,
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
            thread_id: Optional[str] = None,
            resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(user_input, str):
            user_input = str(user_input) if user_input is not None else ""
        start = time.time()
        trace_id = (runtime_context or {}).get("trace_id", "-")
        self.monitor.info(f"LangGraphAgent.run 开始，trace_id={trace_id}, 输入长度: {len(user_input)}")

        await self._ensure_graph()

        if thread_id is None:
            thread_id = f"thread_{int(start)}_{uuid.uuid4().hex[:6]}"
        session.current_tool_chain_id = thread_id

        messages = self._build_messages(user_input, history)
        runtime_ctx = {
            **self.default_runtime_context,
            **(runtime_context or {}),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        session.task_context["current_task"] = user_input
        session.runtime_context = runtime_ctx

        # ---------- 从历史检查点恢复时，注入 facts_ledger ----------
        if resume_from_checkpoint and session.task_context.get("facts_ledger"):
            ledger = session.task_context["facts_ledger"]
            confirmed_facts = ledger.get("confirmed_facts", [])
            if confirmed_facts:
                facts_text = "📋 已确认的历史事实：\n"
                for fact in confirmed_facts[-6:]:
                    facts_text += f"- {str(fact)[:120]}\n"
                messages.append({"role": "system", "content": facts_text})
        # ----------------------------------------------------------

        log_event(
            "agent_run_started",
            "LangGraphAgent 开始执行",
            trace_id=trace_id,
            thread_id=thread_id or "-",
            run_mode=runtime_ctx.get("run_mode", "chat"),
            plan_mode=bool(runtime_ctx.get("plan_mode")),
        )

        initial: AgentState = {
            "messages": messages,
            "user_input": user_input,
            "session_id": thread_id,
            "thread_id": thread_id,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "tool_calls": [],
            "tool_results": [],
            "final_response": None,
            "runtime_context": runtime_ctx,
            "session_context": session_to_state(session),
            "tool_call_history": [],
            "reflection_history": session.reflection_history,
            "raw_response_cache": [],
        }

        initial = _sanitize_state_update(initial)

        config = {"configurable": {"thread_id": thread_id}}
        final = await self.graph.ainvoke(initial, config)

        state_to_session(final.get("session_context", {}), session)

        if self.vector_memory:
            self.vector_memory.save_to_disk()

        duration = time.time() - start
        self.monitor.info(
            f"LangGraphAgent.run 完成，trace_id={trace_id}, 迭代: {final.get('iteration', 0)}，"
            f"工具调用数: {len(final.get('tool_results', []))}，耗时: {duration:.2f}s，thread_id: {thread_id}"
        )
        log_event(
            "agent_run_completed",
            "LangGraphAgent 执行完成",
            trace_id=trace_id,
            thread_id=thread_id,
            iterations=final.get("iteration", 0),
            tool_results=len(final.get("tool_results", [])),
            duration_s=f"{duration:.2f}",
            final_response_len=len(final.get("final_response", "") or ""),
            ledger=json.dumps(_summarize_facts_ledger(session), ensure_ascii=False),
        )

        raw_tool_calls = final.get("tool_results", [])
        clean_tool_calls = []
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                clean_tool_calls.append(tc)
            else:
                self.monitor.warning(f"跳过非字典的工具结果: {type(tc)}")

        return {
            "response": final.get("final_response", ""),
            "tool_calls": clean_tool_calls,
            "iterations": final.get("iteration", 0),
            "duration": duration,
            "thread_id": thread_id,
            "checkpoint_id": final.get("_checkpoint_id"),
            "reflection_summary": self.reflection.get_reflection_summary(session.reflection_history) if self.reflection else "",
        }

    async def process_message(
        self,
        user_input: str,
        session: SessionContext,
        chat_history=None,
        runtime_context: Optional[Dict] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        hist = []
        if chat_history:
            for u, a in chat_history:
                if u:
                    hist.append({"role": "user", "content": u})
                if a:
                    hist.append({"role": "assistant", "content": a})
        res = await self.run(user_input, session, hist, runtime_context, temperature, top_p, max_tokens)
        exec_log = [
            {"iteration": i, "type": "tool_call", "tool": r.get("tool"), "success": r.get("success")}
            for i, r in enumerate(res["tool_calls"])
        ]
        return res["response"], exec_log, runtime_context

    async def process_message_direct_stream(
        self,
        messages: List[Dict],
        session: SessionContext,
        runtime_context: Optional[Dict] = None,
        stream_forward_fn=None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        user_input = messages[-1]["content"] if messages else ""
        hist = messages[:-1]
        res = await self.run(
            user_input=user_input, session=session, history=hist,
            runtime_context=runtime_context, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        final_response = res.get("response", "")
        tool_results = res.get("tool_calls", [])

        for tc in tool_results:
            yield "", {"type": "tool_result", "tool": tc.get("tool"), "success": tc.get("success")}

        yield final_response, {}

    def get_checkpoint_state(self, thread_id: str) -> Optional[Dict]:
        if self.graph is None:
            return None
        config = {"configurable": {"thread_id": thread_id}}
        try:
            checkpoint = self.graph.get_state(config)
            if checkpoint and checkpoint.values:
                return checkpoint.values
        except Exception as e:
            self.monitor.error(f"获取检查点失败: {e}")
        return None

    async def list_threads(self) -> List[str]:
        if self.checkpointer is None:
            await self._ensure_graph()
        import aiosqlite
        async with aiosqlite.connect(str(self.checkpoint_db_path)) as conn:
            cursor = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cursor.fetchall()
            await cursor.close()
        return [r[0] for r in rows]
