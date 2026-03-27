# -*- coding: utf-8 -*-
"""流式输出框架 - 支持 SSE 实时展示进度"""

import json
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from datetime import datetime


class StreamEvent:
    """流式事件"""

    def __init__(self, event_type: str, data: Any, timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def to_sse(self) -> str:
        """转换为 SSE 格式"""
        data_json = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event_type}\ndata: {data_json}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp
        }


class StreamingFramework:
    """
    流式输出框架 - 实时展示 Agent 执行进度

    事件类型：
    - start: 开始执行
    - thought: 思考过程
    - tool_call: 工具调用
    - tool_result: 工具结果
    - reflection: 反思
    - progress: 进度更新
    - complete: 完成
    - error: 错误
    """

    def __init__(
        self,
        framework,  # QwenAgentFramework 实例
        enable_sse: bool = True
    ):
        self.framework = framework
        self.enable_sse = enable_sse

    def run_stream(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[StreamEvent, None, None]:
        """流式运行 - 生成事件流"""

        # 发送开始事件
        yield StreamEvent("start", {
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        })

        messages = self.framework._build_messages(user_input, history)
        runtime_context = runtime_context or {}
        runtime_context["start_time"] = time.time()

        if self.framework.memory:
            self.framework.memory.add_context("current_task", user_input)

        tool_calls_log = []
        last_response = ""

        for iteration in range(self.framework.max_iterations):
            # 发送进度事件
            yield StreamEvent("progress", {
                "iteration": iteration + 1,
                "max_iterations": self.framework.max_iterations,
                "message": f"迭代 {iteration + 1}/{self.framework.max_iterations}"
            })

            # 检查是否应该继续
            if self.framework.reflection:
                should_continue, reason = self.framework.reflection.should_continue(
                    self.framework.reflection_history
                )
                if not should_continue:
                    yield StreamEvent("error", {
                        "reason": reason,
                        "iteration": iteration + 1
                    })
                    break

            # 上下文管理
            messages = self.framework._compress_context_smart(messages)
            messages = self.framework._inject_task_context(messages)
            messages = self.framework._inject_reflection(messages)

            # 中间件
            for mw in self.framework.middlewares:
                if hasattr(mw, "process_before_llm"):
                    messages = mw.process_before_llm(messages, runtime_context)

            # 发送思考事件
            yield StreamEvent("thought", {
                "iteration": iteration + 1,
                "message": "正在思考下一步..."
            })

            # 调用模型
            try:
                response = self.framework.model_forward_fn(messages, self.framework.system_prompt)
                last_response = response

                # 发送思考结果
                yield StreamEvent("thought", {
                    "iteration": iteration + 1,
                    "content": response[:200] + "..." if len(response) > 200 else response
                })

            except Exception as e:
                yield StreamEvent("error", {
                    "type": "model_error",
                    "message": str(e),
                    "iteration": iteration + 1
                })
                break

            # 解析工具
            tool_calls_raw = self.framework.tool_parser.parse_tool_calls(response)
            tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

            if not tool_calls:
                # 无工具调用，完成
                yield StreamEvent("complete", {
                    "response": response,
                    "iterations": iteration + 1,
                    "tool_calls": tool_calls_log
                })
                break

            # 检测可并行工具
            parallel_tools, sequential_tools = self.framework._detect_parallel_tools(tool_calls)

            # 执行工具
            results = []

            # 并行执行
            if parallel_tools:
                yield StreamEvent("tool_call", {
                    "mode": "parallel",
                    "tools": [tc["name"] for tc in parallel_tools],
                    "count": len(parallel_tools)
                })

                for tc in parallel_tools:
                    yield StreamEvent("tool_call", {
                        "tool": tc["name"],
                        "args": tc["args"],
                        "mode": "parallel"
                    })

                parallel_results = self.framework._execute_tools_parallel(parallel_tools)
                results.extend(parallel_results)

                for r in parallel_results:
                    yield StreamEvent("tool_result", {
                        "tool": r["tool"],
                        "success": not r["result"].get("error"),
                        "result": str(r["result"])[:200],
                        "mode": "parallel"
                    })

            # 串行执行
            for tc in sequential_tools:
                yield StreamEvent("tool_call", {
                    "tool": tc["name"],
                    "args": tc["args"],
                    "mode": "sequential"
                })

                result = self.framework._execute_single_tool(tc["name"], tc["args"])
                results.append({"tool": tc["name"], "result": result, "parallel": False})

                yield StreamEvent("tool_result", {
                    "tool": tc["name"],
                    "success": not result.get("error"),
                    "result": str(result)[:200],
                    "mode": "sequential"
                })

            # 记录日志
            for r in results:
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": r["tool"],
                    "success": not r["result"].get("error"),
                    "parallel": r.get("parallel", False)
                })

            # 发送反思事件
            if self.framework.reflection_history:
                last_reflection = self.framework.reflection_history[-1]
                yield StreamEvent("reflection", {
                    "tool": last_reflection["tool"],
                    "success": last_reflection["success"],
                    "analysis": last_reflection["analysis"],
                    "suggestions": last_reflection["suggestions"]
                })

            # 循环检测
            if self.framework._detect_loop():
                yield StreamEvent("error", {
                    "type": "loop_detected",
                    "message": "检测到循环，建议重新规划任务",
                    "iteration": iteration + 1
                })
                break

            # 回注结果
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": self.framework._format_results(results)})

        # 保存记忆
        if self.framework.memory:
            self.framework.memory.save_to_disk()

        # 发送最终完成事件
        yield StreamEvent("complete", {
            "response": last_response,
            "iterations": iteration + 1,
            "tool_calls": tool_calls_log,
            "duration": time.time() - runtime_context["start_time"]
        })

    def run_stream_sse(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """流式运行 - 返回 SSE 格式"""
        for event in self.run_stream(user_input, history, runtime_context):
            yield event.to_sse()


def create_streaming_wrapper(framework):
    """创建流式包装器"""
    return StreamingFramework(framework, enable_sse=True)
