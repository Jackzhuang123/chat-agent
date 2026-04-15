# -*- coding: utf-8 -*-
"""基于 LangGraph 的 Agent 核心实现（完全兼容 QwenAgentFramework）"""

import asyncio
import inspect
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Callable, TypedDict
from core.monitor_logger import get_monitor_logger

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    raise ImportError("请安装 langgraph: pip install langgraph>=0.2.0") from e

from .agent_tools import ToolExecutor, ToolParser
from .state_manager import SessionContext
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory
from .agent_middlewares import AgentMiddleware

from .components import (
    DeepReflectionEngine,
    compress_context_smart,
    detect_loop,
    looks_finished,
    clean_react_tags,
    strip_trailing_tool_call,
    inject_task_context,
    inject_format_correction,
)


# ============================================================================
# 安全模型调用：统一处理生成器返回值
# ============================================================================
def _safe_model_call(model_forward_fn: Callable, messages: list,
                     system_prompt: str = "", **kwargs) -> str:
    """安全调用 model_forward_fn，兼容返回生成器或字符串的模型。

    - 若返回字符串：直接返回。
    - 若返回生成器（流式模型）：消费到结束，取最后一个 chunk 作为完整响应。
    - 异常时返回空字符串并不抛出，由上层感知。
    """
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        raise  # 让上层 AgentNode 捕获并写入 final_response

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


# ============================================================================
# 状态定义
# ============================================================================
class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    user_input: str
    session_id: str
    iteration: int
    max_iterations: int
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_response: Optional[str]
    runtime_context: Dict[str, Any]
    session_context: Dict[str, Any]
    tool_call_history: List[Dict[str, Any]]
    reflection_history: List[Dict[str, Any]]
    _needs_retry: bool
    _tool_enforcement_retry: int


# ============================================================================
# 辅助函数
# ============================================================================
def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


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
    # 保证 task_context 中的关键列表字段始终存在
    task_ctx = state_data.get("task_context", {})
    task_ctx.setdefault("completed_steps", [])
    task_ctx.setdefault("failed_attempts", [])
    task_ctx.setdefault("subtask_status", {})
    session.task_context = task_ctx
    session.current_tool_chain_id = state_data.get("current_tool_chain_id")
    session.runtime_context = state_data.get("runtime_context", {})


# ============================================================================
# 节点类
# ============================================================================
class AgentNode:
    def __init__(self, fw: 'LangGraphAgent'):
        self.fw = fw

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        fw = self.fw
        runtime_context = state["runtime_context"]
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        messages = state["messages"].copy()
        iteration = state.get("iteration", 0)

        # 反思检查
        if fw.reflection:
            should_cont, reason = fw.reflection.should_continue(session.reflection_history)
            if not should_cont:
                return {"final_response": f"⚠️ 任务中断: {reason}"}

        # 上下文压缩与注入
        limit = 16000 if session.task_context.get("_plan_step_id") else 12000
        messages = compress_context_smart(messages, session, fw.model_forward_fn, fw.vector_memory, limit)
        messages = inject_task_context(messages, session, fw.vector_memory)
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

        # LLM 调用（_safe_model_call 统一处理生成器/字符串返回）
        try:
            response = _safe_model_call(
                fw.model_forward_fn, messages, fw.system_prompt,
                temperature=temp, top_p=top_p, max_tokens=max_tok
            )
        except Exception as e:
            return {"final_response": f"❌ 模型调用失败: {e}"}

        # 后置中间件
        for mw in fw.middlewares:
            if hasattr(mw, "process_after_llm"):
                response = await mw.process_after_llm(response, runtime_context)

        if runtime_context.get("_needs_retry"):
            runtime_context["_needs_retry"] = False
            return {"iteration": iteration, "runtime_context": runtime_context}

        # 解析工具调用
        tool_calls_raw = fw.tool_parser.parse_tool_calls(response)
        tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

        if not tool_calls:
            # 检测是否存在工具意图但解析失败
            tool_intent = any(kw in response for kw in ["execute_python", "read_file", "write_file", "bash"])
            if tool_intent:
                # 注入格式纠正消息，要求重新输出
                correction = (
                    "⚠️ 系统检测到你尝试调用工具，但格式不正确。\n"
                    "请严格按以下格式重新输出（工具名独占一行，紧接着 JSON 参数）：\n\n"
                    "execute_python\n"
                    "{\"code\": \"你的代码\"}\n\n"
                    "read_file\n"
                    "{\"path\": \"文件路径\"}\n\n"
                    "不要使用 execute_python -c \"...\" 或 read_file(\"...\") 等格式。"
                )
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": correction})
                return {"messages": messages, "iteration": iteration}
            if looks_finished(response, session, runtime_context):
                final = clean_react_tags(strip_trailing_tool_call(response))
                if fw.vector_memory:
                    fw.vector_memory.add(content=f"Assistant: {final}", metadata={"role": "assistant"})
                return {"final_response": final}
            else:
                messages = inject_format_correction(messages, response, str(fw.tool_executor.work_dir))
                return {"messages": messages, "iteration": iteration}

        # 记录原始响应（用于循环检测）
        session.raw_response_cache = getattr(session, 'raw_response_cache', []) + [response]

        return {
            "tool_calls": tool_calls,
            "iteration": iteration + 1,
            "session_context": session_to_state(session),
            "runtime_context": runtime_context,
            "messages": [{"role": "assistant", "content": response}],
        }


class ToolNode:
    def __init__(self, fw: 'LangGraphAgent'):
        self.fw = fw

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        fw = self.fw
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {}
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        runtime_context = state["runtime_context"]

        async def _run_one(tc: Dict) -> Dict:
            tool_name, tool_args = tc["name"], tc["args"]

            # 前置中间件
            for mw in fw.middlewares:
                if hasattr(mw, "process_before_tool"):
                    tool_name, tool_args = await mw.process_before_tool(
                        tool_name, tool_args, runtime_context)

            result_str = fw.tool_executor.execute_tool(tool_name, tool_args)

            # 后置中间件
            for mw in fw.middlewares:
                if hasattr(mw, "process_after_tool"):
                    result_str = await mw.process_after_tool(
                        tool_name, tool_args, result_str, runtime_context)

            try:
                result_obj = json.loads(result_str)
            except Exception:
                result_obj = {"output": result_str}

            success = not result_obj.get("error")

            # 反思引擎
            if fw.reflection:
                fw.reflection.reflect_on_result(
                    tool_name, result_obj,
                    context={"recent_tools": [h.get("tool") for h in session.tool_history[-5:]]},
                    history=session.reflection_history
                )

            # 更新会话状态
            session.tool_history.append({
                "tool": tool_name,
                "args": json.dumps(tool_args),
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
            })
            if success:
                session.task_context["completed_steps"].append(tool_name)
                # 缓存已读文件路径
                if tool_name == "read_file" and tool_args.get("path"):
                    session.read_files_cache[tool_args["path"]] = datetime.utcnow().isoformat()
            else:
                session.task_context["failed_attempts"].append(tool_name)

            return {"tool": tool_name, "result": result_obj, "success": success}

        # 并发执行所有工具调用
        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls],
                                       return_exceptions=False)
        # 新增日志摘要
        for r in results:
            status = "✅" if r["success"] else "❌"
            self.fw.monitor.debug(f"工具执行: {status} {r['tool']}")

        result_msg = "\n".join(
            f"{'✅' if r['success'] else '❌'} {r['tool']}: {json.dumps(r['result'], ensure_ascii=False)[:300]}"
            for r in results
        )

        return {
            "tool_results": list(results),
            "tool_calls": [],
            "session_context": session_to_state(session),
            "messages": [{"role": "user", "content": result_msg}],
        }


class FinalizeNode:
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        if state.get("final_response"):
            return {}
        # 尝试从最后一条 assistant 消息提取内容作为最终回复
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return {"final_response": clean_react_tags(strip_trailing_tool_call(msg["content"]))}
        return {"final_response": "任务完成"}


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
        return {"final_response": hint}


# ============================================================================
# 条件边工厂
# ============================================================================
def make_should_continue(fw: 'LangGraphAgent'):
    def should_continue(state: AgentState) -> Literal["tools", "finalize", "loop_detected"]:
        if state.get("iteration", 0) >= state.get("max_iterations", 20):
            return "finalize"
        if not state.get("tool_calls"):
            return "finalize"
        session = SessionContext()
        state_to_session(state.get("session_context", {}), session)
        if detect_loop(session):
            return "loop_detected"
        return "tools"
    return should_continue


# ============================================================================
# 主类
# ============================================================================
class LangGraphAgent:
    """基于 LangGraph 的 ReAct Agent，接口完全兼容原 QwenAgentFramework。"""

    def __init__(
        self,
        model_forward_fn: Callable,
        work_dir: Optional[str] = None,
        enable_bash: bool = False,
        max_iterations: int = 20,
        middlewares: Optional[List[AgentMiddleware]] = None,
        enable_memory: bool = True,
        enable_reflection: bool = True,
        enable_parallel: bool = True,   # 保留参数签名兼容，内部始终并发
        enable_tool_learning: bool = True,
        default_runtime_context: Optional[Dict] = None,
    ):
        self.model_forward_fn = model_forward_fn
        self.max_iterations = max_iterations
        self.middlewares = middlewares or []
        self.default_runtime_context = default_runtime_context or {}
        self.tool_executor = ToolExecutor(work_dir=work_dir, enable_bash=enable_bash)
        self.tool_parser = ToolParser()
        self.vector_memory = VectorMemory() if enable_memory else None
        # 兼容属性：部分外部代码通过 fw.memory 访问
        self.memory = self.vector_memory
        self.tool_learner = AdaptiveToolLearner() if enable_tool_learning else None
        self.reflection = DeepReflectionEngine() if enable_reflection else None
        if self.reflection and self.tool_learner:
            self.reflection.attach_tool_learner(self.tool_learner)
        self.system_prompt = self._build_system_prompt()

        # 检查点管理器（支持断点续跑）
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self.monitor = get_monitor_logger()

    def _build_system_prompt(self) -> str:
        work_dir_abs = str(self.tool_executor.work_dir.resolve())
        enable_bash = self.tool_executor.enable_bash
        tools = (
            "bash, read_file, write_file, edit_file, list_dir, execute_python"
            if enable_bash else
            "read_file, write_file, edit_file, list_dir, execute_python"
        )
        return (
            f"你是智能助手，使用 ReAct 模式工作。\n"
            f"当前工作目录：{work_dir_abs}\n"
            f"可用工具：{tools}\n"
            f"【规则】批量扫描用bash grep；execute_python连续失败立即改用bash；"
            f"相同工具参数不重复超2次。"
        )

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", AgentNode(self))
        builder.add_node("tools", ToolNode(self))
        builder.add_node("finalize", FinalizeNode())
        builder.add_node("loop_detected", LoopDetectedNode())
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", make_should_continue(self))
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        builder.add_edge("loop_detected", END)
        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # 阻塞式入口
    # ------------------------------------------------------------------
    async def run(
        self,
        user_input: str,
        session: SessionContext,
        history: Optional[List[Dict]] = None,
        runtime_context: Optional[Dict] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        if not isinstance(user_input, str):
            user_input = str(user_input) if user_input is not None else ""
        start = time.time()
        self.monitor.info(f"LangGraphAgent.run 开始，输入长度: {len(user_input)}")
        session.current_tool_chain_id = f"chain_{int(start)}_{uuid.uuid4().hex[:6]}"
        messages = (history or []) + [{"role": "user", "content": user_input}]
        runtime_ctx = {
            **self.default_runtime_context,
            **(runtime_context or {}),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        initial: AgentState = {
            "messages": messages,
            "user_input": user_input,
            "session_id": session.current_tool_chain_id,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "tool_calls": [],
            "tool_results": [],
            "final_response": None,
            "runtime_context": runtime_ctx,
            "session_context": session_to_state(session),
            "tool_call_history": [],
            "reflection_history": session.reflection_history,
        }
        config = {"configurable": {"thread_id": initial["session_id"]}}
        final = await self.graph.ainvoke(initial, config)
        state_to_session(final.get("session_context", {}), session)
        if self.vector_memory:
            self.vector_memory.save_to_disk()

        _duration = time.time() - start
        self.monitor.info(
            f"LangGraphAgent.run 完成，迭代: {final.get('iteration', 0)}，"
            f"工具调用数: {len(final.get('tool_results', []))}，耗时: {_duration:.2f}s"
        )

        return {
            "response": final.get("final_response", ""),
            "tool_calls": final.get("tool_results", []),
            "iterations": final.get("iteration", 0),
            "duration": time.time() - start,
        }

    # ------------------------------------------------------------------
    # 兼容接口：process_message
    # ------------------------------------------------------------------
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
        """将 (user, assistant) 元组历史转换后调用 run()。"""
        hist = []
        if chat_history:
            for u, a in chat_history:
                if u:
                    hist.append({"role": "user", "content": u})
                if a:
                    hist.append({"role": "assistant", "content": a})
        res = await self.run(user_input, session, hist, runtime_context,
                             temperature, top_p, max_tokens)
        exec_log = [
            {
                "iteration": i,
                "type": "tool_call",
                "tool": r.get("tool"),
                "success": r.get("success"),
            }
            for i, r in enumerate(res["tool_calls"])
        ]
        return res["response"], exec_log, runtime_context

    # ------------------------------------------------------------------
    # 流式接口：process_message_direct_stream
    # 注意：自定义 model_forward_fn 不触发 on_chat_model_stream 事件，
    # 因此此处用 run() 完成计算后一次性 yield，保证与所有模型兼容。
    # ------------------------------------------------------------------
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
        """流式输出接口（yield (response, meta)），兼容所有 model_forward_fn 类型。"""
        user_input = messages[-1]["content"] if messages else ""
        hist = messages[:-1]

        res = await self.run(
            user_input=user_input,
            session=session,
            history=hist,
            runtime_context=runtime_context,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        final_response = res.get("response", "")
        tool_results = res.get("tool_calls", [])

        # 逐条工具结果 yield 让调用方感知进度
        for tc in tool_results:
            yield "", {"type": "tool_result", "tool": tc.get("tool"), "success": tc.get("success")}

        yield final_response, {}