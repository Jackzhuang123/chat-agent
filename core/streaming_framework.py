# -*- coding: utf-8 -*-
"""流式输出框架 - 基于 LangGraphAgent，支持 SSE 实时展示进度

设计原则：
  StreamingFramework 不重复实现任何 ReAct 逻辑，
  完全依赖 LangGraphAgent.run() 驱动，
  仅负责将结果翻译为 SSE / StreamEvent 格式输出给前端。
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


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
    """流式包装器——完全复用 LangGraphAgent.run()，零逻辑重复。

    职责：
      1. 调用 LangGraphAgent.run() 获得最终结果
      2. 在调用前后发出 start / complete / error 事件
      3. 通过 run_stream_sse() 对外暴露 SSE 字符串生成器
    """

    def __init__(self, framework, enable_sse: bool = True):
        self.framework = framework
        self.enable_sse = enable_sse

    # ------------------------------------------------------------------
    # 主要公共接口
    # ------------------------------------------------------------------
    from typing import AsyncGenerator

    async def run_stream(
            self,
            user_input: str,
            session: "SessionContext",
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> AsyncGenerator[StreamEvent, None]:
        """调用 LangGraphAgent.run()，将结果转换为 StreamEvent 序列输出。"""
        fw = self.framework
        runtime_context = runtime_context or {}
        start_time = time.time()

        # ── 记忆写入（与 legacy 保持一致）──
        if getattr(fw, "vector_memory", None):
            if hasattr(fw.vector_memory, "add_context"):
                fw.vector_memory.add_context("current_task", user_input)
            if hasattr(fw.vector_memory, "add"):
                fw.vector_memory.add(
                    content=f"User: {user_input}",
                    metadata={"role": "user", "type": "input"},
                    importance=0.8,
                    original_question=user_input,  # 新增参数
                )

        yield StreamEvent("start", {
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        })

        try:
            result = await fw.run(
                user_input=user_input,
                session=session,
                history=history,
                runtime_context=runtime_context,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            yield StreamEvent("error", {
                "type": "runtime_error",
                "message": str(e),
            })
            yield StreamEvent("complete", {
                "response": f"❌ 执行出错: {e}",
                "iterations": 0,
                "tool_calls": [],
                "duration": time.time() - start_time,
                "error": str(e),
            })
            return

        response = result.get("response", "")
        tool_calls = result.get("tool_calls", [])
        iterations = result.get("iterations", 0)
        duration = result.get("duration", time.time() - start_time)

        # 将工具调用记录转换为事件（让前端感知执行过程）
        for i, tc in enumerate(tool_calls):
            yield StreamEvent("tool_result", {
                "tool": tc.get("tool"),
                "success": tc.get("success", False),
                "result": str(tc.get("result", ""))[:200],
                "mode": "sequential",
            })

        yield StreamEvent("complete", {
            "response": response,
            "iterations": iterations,
            "tool_calls": [
                {"tool": tc.get("tool"), "success": tc.get("success", False)}
                for tc in tool_calls
            ],
            "duration": duration,
        })

    async def run_stream_sse(
            self,
            user_input: str,
            session: "SessionContext",
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> AsyncGenerator[str, None]:
        """直接输出 SSE 字符串，供 Flask/FastAPI 的 StreamingResponse 使用。"""
        async for event in self.run_stream(
                user_input, session, history, runtime_context,
                temperature, top_p, max_tokens):
            yield event.to_sse()


def create_streaming_wrapper(framework) -> StreamingFramework:
    return StreamingFramework(framework, enable_sse=True)