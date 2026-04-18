# -*- coding: utf-8 -*-
"""模型前向函数工厂 - 将 qwen_agent / glm_agent 包装为统一的 model_forward_fn 接口"""

from typing import Dict, List


def _merge_system_messages(messages: List[Dict[str, str]], combined_system_prompt: str = "") -> List[Dict[str, str]]:
    """合并外部 system_prompt 与消息中的 system 消息，避免多 system 导致模型接口报错。"""
    merged_system_parts = []
    if combined_system_prompt and combined_system_prompt.strip():
        merged_system_parts.append(combined_system_prompt.strip())

    normalized_messages = []
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "system":
            merged_system_parts.append(content)
            continue
        normalized_messages.append({"role": role, "content": content})

    if merged_system_parts:
        merged_system = "\n\n".join(part for part in merged_system_parts if part)
        return [{"role": "system", "content": merged_system}] + normalized_messages
    return normalized_messages


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

        # 合并所有 system 消息，兼容仅支持单个 system 消息的模型接口（如 GLM）。
        full_messages = _merge_system_messages(list(messages), combined)

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
