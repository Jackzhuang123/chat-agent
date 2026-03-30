# -*- coding: utf-8 -*-
"""流式输出框架 - 基于 QwenAgentFramework._run_iter() 生成器，支持 SSE 实时展示进度

设计原则：
  StreamingFramework 不重复实现任何 ReAct 逻辑，
  完全消费 QwenAgentFramework._run_iter() 产出的事件，
  仅负责将事件翻译为 SSE / StreamEvent 格式输出给前端。
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional


class StreamEvent:
    def __init__(self, event_type: str, data: Any, timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def to_sse(self) -> str:
        data_json = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event_type}\ndata: {data_json}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event_type, "data": self.data, "timestamp": self.timestamp}


class StreamingFramework:
    """流式包装器——完全复用 QwenAgentFramework._run_iter()，零逻辑重复。

    职责：
      1. 初始化 _run_iter 所需的 messages / runtime_context
      2. 将生成器事件映射为 StreamEvent（SSE 友好格式）
      3. 通过 run_stream_sse() 对外暴露 SSE 字符串生成器
    """

    def __init__(self, framework, enable_sse: bool = True):
        self.framework = framework
        self.enable_sse = enable_sse

    # ------------------------------------------------------------------
    # 主要公共接口
    # ------------------------------------------------------------------
    def run_stream(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Generator[StreamEvent, None, None]:
        """消费 _run_iter 生成器，将内部事件转换为 StreamEvent 序列。"""
        fw = self.framework

        # ── 初始化（与 run() 保持一致） ──
        import uuid
        fw._current_tool_chain_id = f"chain_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        messages = fw._build_messages(user_input, history)
        runtime_context = runtime_context or {}
        runtime_context["start_time"] = time.time()
        runtime_context["tool_chain_id"] = fw._current_tool_chain_id

        if fw.memory:
            fw.memory.add_context("current_task", user_input)
            if hasattr(fw.memory, "add"):
                fw.memory.add(
                    content=f"User: {user_input}",
                    metadata={"role": "user", "type": "input"},
                    importance=0.8,
                    tool_chain_id=fw._current_tool_chain_id
                )

        fw.task_context["current_task"] = user_input

        yield StreamEvent("start", {
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        })

        tool_calls_log: List[Dict] = []
        last_response = ""
        final_response = ""

        # ── 消费 _run_iter ──
        for event in fw._run_iter(messages, user_input, runtime_context, temperature, top_p, max_tokens):
            evt = event.get("event")

            if evt == "progress":
                yield StreamEvent("progress", {
                    "iteration": event["iteration"],
                    "max_iterations": event["max_iterations"],
                    "message": f"迭代 {event['iteration']}/{event['max_iterations']}"
                })

            elif evt == "thought":
                last_response = event.get("content", "")
                snippet = last_response[:200] + ("..." if len(last_response) > 200 else "")
                yield StreamEvent("thought", {
                    "iteration": event["iteration"],
                    "content": snippet
                })

            elif evt == "tool_call":
                mode = event.get("mode", "sequential")
                if mode == "parallel":
                    yield StreamEvent("tool_call", {
                        "mode": "parallel",
                        "tools": event.get("tools", []),
                        "count": len(event.get("tools", []))
                    })
                else:
                    yield StreamEvent("tool_call", {
                        "mode": "sequential",
                        "tool": event.get("tool"),
                        "args": event.get("args", {})
                    })

            elif evt == "tool_result":
                result = event.get("result", {})
                result_preview = str(result)[:200] if result else ""
                yield StreamEvent("tool_result", {
                    "tool": event.get("tool"),
                    "success": event.get("success", False),
                    "result": result_preview,
                    "mode": event.get("mode", "sequential")
                })
                # 累积 tool_calls_log
                tool_calls_log.append({
                    "tool": event.get("tool"),
                    "success": event.get("success", False),
                    "mode": event.get("mode", "sequential")
                })

            elif evt == "reflection":
                refl = event.get("data", {})
                yield StreamEvent("reflection", {
                    "success": refl.get("success", False),
                    "level": refl.get("level", "surface"),
                    "analysis": refl.get("analysis", ""),
                    "suggestions": refl.get("suggestions", []),
                    "action": refl.get("action", "continue")
                })

            elif evt == "format_error":
                yield StreamEvent("progress", {
                    "iteration": event.get("iteration", 0),
                    "message": "格式纠错中，重新生成..."
                })

            elif evt == "done":
                final_response = event.get("final_response", "")
                tool_calls_log = event.get("tool_calls", tool_calls_log)
                yield StreamEvent("complete", {
                    "response": final_response,
                    "iterations": event["iterations"],
                    "tool_calls": tool_calls_log,
                    "duration": time.time() - runtime_context.get("start_time", time.time())
                })
                break

            elif evt == "interrupted":
                reason = event.get("reason", "")
                tool_calls_log = event.get("tool_calls", tool_calls_log)
                yield StreamEvent("error", {
                    "type": "interrupted",
                    "reason": reason,
                    "iterations": event["iterations"]
                })
                yield StreamEvent("complete", {
                    "response": f"⚠️ 任务中断: {reason}",
                    "iterations": event["iterations"],
                    "tool_calls": tool_calls_log,
                    "duration": time.time() - runtime_context.get("start_time", time.time()),
                    "interrupted": True
                })
                break

            elif evt == "loop_detected":
                tool_calls_log = event.get("tool_calls", tool_calls_log)
                yield StreamEvent("error", {
                    "type": "loop_detected",
                    "message": "检测到循环，建议重新规划任务",
                    "iteration": event.get("iterations", 0)
                })
                yield StreamEvent("complete", {
                    "response": "⚠️ 检测到循环。建议：1.换用其他工具 2.重新分析问题 3.调整参数",
                    "iterations": event.get("iterations", 0),
                    "tool_calls": tool_calls_log,
                    "duration": time.time() - runtime_context.get("start_time", time.time()),
                    "loop_detected": True
                })
                break

            elif evt == "llm_error":
                resp = event.get("fallback") or f"❌ 模型调用失败: {event.get('error', '')}"
                tool_calls_log = event.get("tool_calls", tool_calls_log)
                yield StreamEvent("error", {
                    "type": "model_error",
                    "message": event.get("error", ""),
                    "iteration": event.get("iterations", 0)
                })
                yield StreamEvent("complete", {
                    "response": resp,
                    "iterations": event.get("iterations", 0),
                    "tool_calls": tool_calls_log,
                    "duration": time.time() - runtime_context.get("start_time", time.time()),
                    "error": event.get("error")
                })
                break

            elif evt == "max_iter":
                final_response = event.get("final_response", "⚠️ 达到最大迭代次数")
                tool_calls_log = event.get("tool_calls", tool_calls_log)
                yield StreamEvent("complete", {
                    "response": final_response,
                    "iterations": event["iterations"],
                    "tool_calls": tool_calls_log,
                    "duration": time.time() - runtime_context.get("start_time", time.time()),
                    "max_iter_reached": True
                })
                break

        # 持久化记忆
        if fw.memory:
            fw.memory.save_to_disk()

    def run_stream_sse(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """直接输出 SSE 字符串，供 Flask/FastAPI 的 StreamingResponse 使用。"""
        for event in self.run_stream(user_input, history, runtime_context, temperature, top_p, max_tokens):
            yield event.to_sse()


def create_streaming_wrapper(framework) -> StreamingFramework:
    return StreamingFramework(framework, enable_sse=True)