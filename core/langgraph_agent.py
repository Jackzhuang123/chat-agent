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
        # 尝试执行一个简单的查询来检查连接状态
        # 如果连接已关闭，_conn 属性会抛出 ValueError
        _ = self._conn
        return True
    except (ValueError, AttributeError, aiosqlite.Error):
        return False


# 应用 monkey patch
if not hasattr(aiosqlite.core.Connection, 'is_alive'):
    aiosqlite.core.Connection.is_alive = _is_alive_patch

from .agent_tools import ToolExecutor, ToolParser
from .state_manager import SessionContext
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory
from .agent_middlewares import AgentMiddleware
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
    _checkpoint_id: Optional[str]   # 避免与内部字段冲突


def _safe_model_call(model_forward_fn: Callable, messages: list,
                     system_prompt: str = "", **kwargs) -> str:
    """安全调用模型前向函数，兼容返回字符串或生成器的情况。"""
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        raise

    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        chunks = []
        try:
            for chunk in result:
                if isinstance(chunk, str):
                    chunks.append(chunk)
        except Exception:
            pass
        return chunks[-1] if chunks else ""

    if not isinstance(result, str):
        return str(result) if result is not None else ""
    return result


def _make_json_serializable(obj: Any) -> Any:
    """递归地将不可 JSON 序列化的对象（包括 NumPy 类型）转换为可序列化形式。"""
    # 处理 NumPy 浮点数
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    # 处理 NumPy 整数
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    # 处理 NumPy 布尔值
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # 处理 NumPy 数组
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
    """清洗状态更新字典，确保所有值均可被 msgpack 序列化（用于检查点保存）。"""
    return _make_json_serializable(update)


def _should_skip_rag_for_file_request(task: str) -> bool:
    """检测明确文件读取请求，避免历史工具证据污染当前文件解析。"""
    if not task:
        return False

    lowered = task.lower()
    has_file_action = any(keyword in task for keyword in ("查看", "读取", "读", "解析", "分析", "打开", "查找"))
    has_file_hint = bool(re.search(r"\b[\w\-]+\.(json|log|txt|md|py|yaml|yml|csv)\b", lowered))
    has_path_hint = "/" in task or "\\" in task
    return has_file_action and (has_file_hint or has_path_hint)


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
            file_fact = result_obj.get("file_facts") or {
                "path": path,
                "summary": "",
            }
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

        # 反思检查
        if fw.reflection:
            should_cont, reason = fw.reflection.should_continue(session.reflection_history)
            if not should_cont:
                fw.monitor.warning(f"反思引擎建议停止: {reason}")
                return _sanitize_state_update({"final_response": f"⚠️ 任务中断: {reason}"})

        # RAG 上下文增强 - 仅首次迭代执行。
        # 对明确的文件读取/解析请求禁用历史工具证据，避免旧 read_file 结果污染当前任务。
        if fw.context_retriever and iteration == 0:
            current_task = session.task_context.get("current_task", "")
            if _should_skip_rag_for_file_request(current_task):
                fw.monitor.info("检测到明确文件读取请求，跳过历史 RAG 注入")
            else:
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
                run_mode=runtime_context.get("run_mode", "chat"),
                message_count=len(messages),
                system_prompt_mode=_normalize_run_mode(
                    runtime_context.get("run_mode", "chat"),
                    plan_mode=bool(runtime_context.get("plan_mode")),
                ),
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
        run_mode = runtime_context.get("run_mode", "chat")
        if run_mode == "chat":
            final_response = clean_react_tags(strip_trailing_tool_call(response))
            if fw.vector_memory:
                fw.vector_memory.add(
                    content=f"Assistant: {final_response}",
                    metadata={"role": "assistant", "type": "assistant_response"},
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
            tool_intent = should_retry_tool_format(
                response,
                has_successful_tool_result=has_successful_tool_result,
            )
            if tool_intent:
                fw.monitor.warning("检测到工具调用意图但解析失败，尝试格式纠正")
                correction = (
                    "⚠️ 系统检测到你尝试调用工具，但格式不正确。\n"
                    "请严格按以下格式重新输出（工具名独占一行，紧接着 JSON 参数）：\n\n"
                    "execute_python\n"
                    "{\"code\": \"你的代码\"}\n\n"
                    "read_file\n"
                    "{\"path\": \"文件路径\"}\n\n"
                    "不要使用 execute_python -c \"...\" 或 read_file(\"...\") 等格式。"
                )
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

            # ---------- 修复：无工具调用且无工具意图时，直接采纳回答 ----------
            final = clean_react_tags(strip_trailing_tool_call(response))
            if fw.vector_memory:
                fw.vector_memory.add(
                    content=f"Assistant: {final}",
                    metadata={"role": "assistant", "type": "assistant_response"},
                )
            fw.monitor.info("模型返回自然语言回答，直接作为最终输出")
            # ⭐ 关键修复：将当前 assistant 消息添加到 messages 状态中
            updated_messages = messages + [{"role": "assistant", "content": response}]
            return _sanitize_state_update({
                "final_response": final,
                "raw_response_cache": raw_cache,
                "messages": updated_messages,
                "session_context": session_to_state(session),  # 保证状态流转正常
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


# core/langgraph_agent.py （仅 ToolNode 部分，其余文件内容不变）

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
                    session.read_files_cache[Path(tool_args["path"]).resolve().as_posix()] = datetime.utcnow().isoformat()
            else:
                session.task_context["failed_attempts"].append(tool_name)
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
                # 使用普通消息回灌工具结果，兼容只接受单个 system prompt 的模型接口。
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
                result_messages.append({
                    "role": "user",
                    "content": f"{'✅' if success else '❌'} {tool}: {summary}",
                })

        return _sanitize_state_update({
            "tool_results": list(results),
            "tool_calls": [],
            "session_context": session_to_state(session),
            "messages": result_messages,
        })


class FinalizeNode:
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. 已有 final_response，直接结束
        if state.get("final_response"):
            return _sanitize_state_update({})

        # 2. 从 raw_response_cache 中提取最后一条非工具调用内容
        raw_cache = state.get("raw_response_cache", [])
        tool_keywords = {"execute_python", "read_file", "write_file", "edit_file", "list_dir", "bash"}

        # 添加调试日志
        self._log_debug = lambda msg: None  # 兼容无 logger 的情况

        for resp in reversed(raw_cache):
            # 检查是否包含工具关键词
            has_tool_keyword = any(kw in resp for kw in tool_keywords)
            if not has_tool_keyword:
                cleaned = clean_react_tags(strip_trailing_tool_call(resp))
                if len(cleaned.strip()) > 5:  # 有效内容
                    return _sanitize_state_update({"final_response": cleaned})

        # 3. 如果 raw_response_cache 中有内容但都包含工具关键词，
        #    尝试提取最后一条作为最终回答（可能是模型对工具结果的总结）
        if raw_cache:
            last_resp = raw_cache[-1]
            cleaned = clean_react_tags(strip_trailing_tool_call(last_resp))
            if len(cleaned.strip()) > 5:
                return _sanitize_state_update({"final_response": cleaned})

        # 4. 回退：从 messages 中提取最后一条 assistant 消息
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                cleaned = clean_react_tags(strip_trailing_tool_call(msg["content"]))
                if cleaned.strip():
                    return _sanitize_state_update({"final_response": cleaned})

        # 5. 最终兜底
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

        # 保存数据库路径，但不立即创建连接（异步懒加载）
        self.checkpoint_db_path = Path(checkpoint_db_path)
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = None
        self._checkpoint_cm = None  # 保存上下文管理器以便退出
        self.graph = None
        self._conn = None  # 保存 aiosqlite 连接用于 list_threads
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
        # 使用 async with 正确初始化 AsyncSqliteSaver
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
        # 如果已经有最终回答，直接结束
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

        # 确保异步检查点和图已就绪
        await self._ensure_graph()

        if thread_id is None:
            # 每次 run 默认生成新的执行链，避免跨请求复用旧工具历史，误触发循环检测。
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

        # 初始状态也需要清洗（以防传入的 runtime_ctx 有 numpy）
        initial = _sanitize_state_update(initial)

        config = {"configurable": {"thread_id": thread_id}}
        # 使用异步调用
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

        # 清洗工具调用结果，确保均为字典（防止列表内嵌套非 dict 导致调用 .get() 报错）
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
        """同步获取检查点状态（仅供调试）"""
        # 注意：graph.get_state 是同步方法，但由于我们在异步上下文中，建议外部使用异步方式
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
        """异步列出所有线程ID"""
        if self.checkpointer is None:
            await self._ensure_graph()
        # 使用 aiosqlite 直接查询数据库
        import aiosqlite
        async with aiosqlite.connect(str(self.checkpoint_db_path)) as conn:
            cursor = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cursor.fetchall()
            await cursor.close()
        return [r[0] for r in rows]
