# -*- coding: utf-8 -*-
"""模型前向函数工厂 - 将 qwen_agent / glm_agent 包装为统一的 model_forward_fn 接口"""

from typing import Dict, List


def create_qwen_model_forward(qwen_agent, system_prompt_base: str = ""):
    """
    创建模型前向函数，兼容 QwenAgent 和 GLMAgent。

    Args:
        qwen_agent: QwenAgent 或 GLMAgent 实例，需实现
                    generate_stream_text(messages, **kwargs) 或
                    generate_stream_with_messages(messages, **kwargs)。
        system_prompt_base: 基础系统提示词，会与调用时传入的 system_prompt 合并。

    Returns:
        forward(messages, system_prompt="", **kwargs) -> str
            - 消费流式生成器，返回最终完整字符串。
            - 支持 temperature / top_p / max_tokens 关键字参数。
    """

    def forward(messages: List[Dict[str, str]], system_prompt: str = "", **kwargs) -> str:
        # 合并系统提示词
        combined = system_prompt_base
        if system_prompt:
            combined = f"{combined}\n\n{system_prompt}" if combined else system_prompt

        # 构建消息列表（仅在 system_prompt 非空时才插入 system 消息）
        if combined and combined.strip():
            full_messages = [{"role": "system", "content": combined}] + list(messages)
        else:
            full_messages = list(messages)

        gen_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 512),
        }

        response_text = ""
        if hasattr(qwen_agent, "generate_stream_text"):
            for token in qwen_agent.generate_stream_text(full_messages, **gen_kwargs):
                response_text = token
        elif hasattr(qwen_agent, "generate_stream_with_messages"):
            for token in qwen_agent.generate_stream_with_messages(full_messages, **gen_kwargs):
                response_text = token
        else:
            raise AttributeError(
                f"{type(qwen_agent).__name__} 未实现 generate_stream_text 或 "
                f"generate_stream_with_messages 方法"
            )

        return response_text

    return forward

