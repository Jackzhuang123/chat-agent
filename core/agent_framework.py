#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Agent 框架 - 为 Qwen 模型添加工具调用能力
包含 Agent 循环、工具执行和响应生成
"""

import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from .agent_tools import ToolExecutor, ToolParser


class QwenAgentFramework:
    """
    Qwen Agent 框架 - 为 Qwen 模型添加工具调用和任务规划能力

    核心循环:
    1. 用户输入 + 历史
    2. 模型生成响应(可能包含工具调用)
    3. 解析和执行工具
    4. 将结果返回给模型
    5. 重复直到任务完成
    """

    def __init__(
        self,
        model_forward_fn,  # 模型前向函数: fn(messages, system_prompt) -> str
        work_dir: Optional[str] = None,
        enable_bash: bool = False,
        max_iterations: int = 10,
        tools_in_system_prompt: bool = True
    ):
        """
        初始化 Agent 框架

        Args:
            model_forward_fn: 模型前向函数,接收 (messages, system_prompt) 返回生成文本
            work_dir: 工作目录 (用于工具执行)
            enable_bash: 是否启用 bash 工具 (需谨慎使用)
            max_iterations: 最大循环次数 (防止无限循环)
            tools_in_system_prompt: 是否在系统提示词中包含工具信息
        """
        self.model_forward_fn = model_forward_fn
        self.tool_executor = ToolExecutor(work_dir=work_dir, enable_bash=enable_bash)
        self.tool_parser = ToolParser()
        self.max_iterations = max_iterations
        self.tools_in_system_prompt = tools_in_system_prompt

        # 构建系统提示词
        self.system_prompt_template = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建包含工具信息的系统提示词"""
        if not self.tools_in_system_prompt:
            return ""

        tools_info = self._format_tools_info()
        return f"""你是一个能够使用工具的智能助手。

可用的工具:
{tools_info}

当你需要使用工具时,请按照以下格式输出:
<tool>tool_name</tool>
<input>{{"parameter": "value"}}</input>

然后我会执行这个工具并将结果返回给你。继续处理任务直到完成。

记住:
- 优先使用工具而不是猜测
- 在执行复杂操作前先探索文件结构
- 如果不确定路径,先使用 list_dir
"""

    def _format_tools_info(self) -> str:
        """格式化工具信息用于系统提示词"""
        tools = self.tool_executor.get_tools()
        info = []

        for tool in tools:
            info.append(f"- **{tool['name']}**: {tool['description']}")
            if "properties" in tool.get("input_schema", {}):
                props = tool["input_schema"]["properties"]
                for prop_name, prop_info in props.items():
                    desc = prop_info.get("description", "")
                    info.append(f"    - {prop_name}: {desc}")

        return "\n".join(info)

    def process_message(
        self,
        user_message: str,
        history: List[Tuple[str, str]],
        system_prompt_override: Optional[str] = None,
        **model_kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        处理用户消息并执行 Agent 循环

        Args:
            user_message: 用户输入
            history: 对话历史 [(user_msg, assistant_msg), ...]
            system_prompt_override: 覆盖系统提示词
            **model_kwargs: 传递给模型的其他参数

        Returns:
            (最终响应, 执行记录)
        """
        # 构建消息列表
        messages = self._build_messages(user_message, history)
        system_prompt = system_prompt_override or self.system_prompt_template

        # 执行记录
        execution_log = []

        # Agent 循环
        for iteration in range(self.max_iterations):
            # 1. 调用模型
            model_response = self.model_forward_fn(messages, system_prompt, **model_kwargs)

            execution_log.append({
                "iteration": iteration,
                "type": "model_response",
                "content": model_response
            })

            # 2. 尝试解析工具调用
            tool_calls = self.tool_parser.parse_tool_calls(model_response)

            if not tool_calls:
                # 没有工具调用,直接返回响应
                return model_response, execution_log

            # 3. 执行工具
            tool_results = []
            for tool_name, tool_input in tool_calls:
                result = self.tool_executor.execute_tool(tool_name, tool_input)
                tool_results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result
                })

                execution_log.append({
                    "iteration": iteration,
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result
                })

            # 4. 将工具结果加入消息历史,继续循环
            messages.append({"role": "assistant", "content": model_response})

            # 将工具结果组织成可读的格式
            result_text = self._format_tool_results(tool_results)
            messages.append({"role": "user", "content": result_text})

        # 达到最大迭代次数
        final_response = f"已达到最大迭代次数({self.max_iterations})。最后的模型响应:\n\n{model_response}"
        return final_response, execution_log

    def process_message_stream(
        self,
        user_message: str,
        history: List[Tuple[str, str]],
        system_prompt_override: Optional[str] = None,
        **model_kwargs
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        流式处理用户消息 (用于实时显示)

        Yields:
            (流式文本, 执行信息)
        """
        messages = self._build_messages(user_message, history)
        system_prompt = system_prompt_override or self.system_prompt_template

        for iteration in range(self.max_iterations):
            # 调用模型
            model_response = self.model_forward_fn(messages, system_prompt, **model_kwargs)

            yield model_response, {
                "iteration": iteration,
                "type": "model_response"
            }

            # 解析工具调用
            tool_calls = self.tool_parser.parse_tool_calls(model_response)

            if not tool_calls:
                # 任务完成
                return

            # 执行工具并显示进度
            tool_results = []
            for tool_name, tool_input in tool_calls:
                result = self.tool_executor.execute_tool(tool_name, tool_input)
                tool_results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result
                })

                yield f"[执行工具] {tool_name}\n", {
                    "iteration": iteration,
                    "type": "tool_execution",
                    "tool": tool_name
                }

            # 继续循环
            messages.append({"role": "assistant", "content": model_response})
            result_text = self._format_tool_results(tool_results)
            messages.append({"role": "user", "content": result_text})

    def _build_messages(self, user_message: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []

        # 添加历史消息
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """格式化工具结果用于返回给模型"""
        formatted = []

        for result_info in tool_results:
            tool_name = result_info["tool"]
            tool_input = result_info["input"]
            result_str = result_info["result"]

            try:
                result_data = json.loads(result_str)
            except json.JSONDecodeError:
                result_data = {"raw": result_str}

            formatted.append(f"工具 '{tool_name}' 的执行结果:")
            formatted.append(json.dumps(result_data, ensure_ascii=False, indent=2))
            formatted.append("")

        return "\n".join(formatted)

    def enable_tools_in_prompt(self, enable: bool):
        """动态启用/禁用工具信息在系统提示词中"""
        self.tools_in_system_prompt = enable


# ============================================================================
# 用于与 Qwen 模型集成的适配器
# ============================================================================

def create_qwen_model_forward(qwen_agent, system_prompt_base: str = ""):
    """
    创建 Qwen 模型前向函数

    Args:
        qwen_agent: QwenAgent 实例
        system_prompt_base: 基础系统提示词

    Returns:
        适配的前向函数
    """
    def forward(messages: List[Dict[str, str]], system_prompt: str = "", **kwargs) -> str:
        # 合并系统提示词
        combined_system_prompt = system_prompt_base
        if system_prompt:
            combined_system_prompt = f"{combined_system_prompt}\n\n{system_prompt}"

        # 构建消息列表
        full_messages = [{"role": "system", "content": combined_system_prompt}] + messages

        # 调用模型生成 (非流式)
        response_text = ""
        for token in qwen_agent.generate_stream_text(
            full_messages,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512)
        ):
            response_text = token

        return response_text

    return forward


if __name__ == "__main__":
    print("✅ Agent 框架模块加载成功")
    print("📚 可用类: QwenAgentFramework")

