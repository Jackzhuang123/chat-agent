# -*- coding: utf-8 -*-
"""格式纠错注入"""

from typing import Dict, List

_TOOL_INTENT_KEYWORDS = ("execute_python", "read_file", "write_file", "bash")


def should_retry_tool_format(response: str, has_successful_tool_result: bool = False) -> bool:
    """判断是否应对自然语言响应触发一次工具格式纠正。

    一旦当前链路已经有成功工具结果，后续包含工具名的自然语言更可能是在解释结果，
    不应再因为出现 bash/read_file 等词而重跑工具。
    """
    if has_successful_tool_result:
        return False
    if not isinstance(response, str) or not response.strip():
        return False
    return any(keyword in response for keyword in _TOOL_INTENT_KEYWORDS)


def inject_format_correction(messages: List[Dict], response: str, work_dir: str = "", custom_msg: str = None) -> List[Dict]:
    """
    向消息列表注入格式纠正提示。

    Args:
        messages: 当前消息列表
        response: 模型上一次的原始响应
        work_dir: 工作目录（用于生成默认提示）
        custom_msg: 自定义纠正消息，若为 None 则使用默认消息

    Returns:
        更新后的消息列表
    """
    if custom_msg is None:
        if work_dir:
            default_msg = (
                "⚠️ [系统提示] 未检测到有效的工具调用格式。\n"
                "✅ 正确格式：\n"
                "read_file\n"
                f'{{"path": "{work_dir}/文件名.py"}}\n\n'
                "请直接重新输出工具调用，不要添加多余解释。"
            )
        else:
            default_msg = (
                "⚠️ [系统提示] 未检测到有效的工具调用格式。\n"
                "✅ 正确格式：工具名独占一行，JSON 紧跟其后。\n"
                "read_file\n"
                '{"path": "文件路径"}\n\n'
                "请直接重新输出工具调用。"
            )
        custom_msg = default_msg

    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": custom_msg})
    return messages
