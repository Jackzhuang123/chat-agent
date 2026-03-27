#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具强制执行中间件 - 确保 tools 模式下模型必须调用工具

解决问题：
- 模型在 tools 模式下直接回答问题，不调用工具
- 模型输出格式不规范，导致工具解析失败

策略：
1. 检测模型输出是否包含工具调用
2. 若未检测到，注入强制提示并标记需要重试
3. 提供清晰的格式示例
"""

from typing import Dict, List
from .agent_middlewares import AgentMiddleware
from .agent_tools import ToolParser


class ToolEnforcementMiddleware(AgentMiddleware):
    """工具强制执行中间件 - 确保 tools 模式下模型必须调用工具"""

    def __init__(self, max_retries: int = 2):
        """
        Args:
            max_retries: 最大重试次数（默认 2 次）
        """
        self.max_retries = max_retries

    def before_llm_call(self, messages: List[Dict], context: Dict) -> List[Dict]:
        """在 LLM 调用前注入强制工具调用提示"""
        run_mode = context.get("run_mode", "chat")

        # 仅在 tools 模式下生效
        if run_mode != "tools":
            return messages

        # 检查是否已经重试过
        retry_count = context.get("_tool_enforcement_retry", 0)

        if retry_count > 0:
            # 重试时注入更强的提示
            enforcement_msg = {
                "role": "system",
                "content": (
                    "⚠️ 重要提醒：你必须使用工具来完成任务，不能直接回答。\n\n"
                    "【正确示例】\n"
                    "用户: 读取 core/agent_tools.py\n"
                    "助手: read_file\n"
                    '{"path": "core/agent_tools.py"}\n\n'
                    "用户: 列出 core 目录\n"
                    "助手: list_dir\n"
                    '{"path": "core"}\n\n'
                    "【错误示例】\n"
                    "助手: 我已经读取了文件，内容如下... ❌\n\n"
                    "请严格按照【工具名 + JSON参数】的格式输出。"
                )
            }
            messages.insert(0, enforcement_msg)

        return messages

    def after_llm_call(self, response: str, context: Dict) -> str:
        """检查模型输出是否包含工具调用"""
        run_mode = context.get("run_mode", "chat")

        # 仅在 tools 模式下生效
        if run_mode != "tools":
            return response

        # 解析工具调用
        tool_calls = ToolParser.parse_tool_calls(response)

        if not tool_calls:
            # 未检测到工具调用 → 注入提示并标记需要重试
            retry_count = context.get("_tool_enforcement_retry", 0)

            if retry_count < self.max_retries:
                # 标记需要重试
                context["_tool_enforcement_retry"] = retry_count + 1
                context["_needs_retry"] = True

                # 注入强制提示
                warning = (
                    "\n\n⚠️ 检测到你直接回答了问题，但当前处于工具模式。\n"
                    "请使用以下格式重新输出：\n\n"
                    "tool_name\n"
                    '{"param": "value"}\n\n'
                    "可用工具: read_file, write_file, edit_file, list_dir, bash"
                )

                return response + warning
            else:
                # 超过最大重试次数，记录错误
                error_msg = (
                    f"\n\n❌ 工具调用解析失败（已重试 {retry_count} 次）\n"
                    f"模型输出：\n{response[:200]}...\n\n"
                    "可能原因：\n"
                    "1. 模型未按工具格式输出\n"
                    "2. 工具调用格式不规范\n\n"
                    "建议：切换到 chat 模式或重新描述任务"
                )
                context["_tool_enforcement_failed"] = True
                return response + error_msg

        # 检测到工具调用，清除重试标记
        context["_tool_enforcement_retry"] = 0
        context["_needs_retry"] = False

        return response

    def get_name(self) -> str:
        """返回中间件名称"""
        return "ToolEnforcementMiddleware"


class DirectCommandDetector:
    """直接命令检测器 - 识别明确的工具调用指令"""

    # 直接命令模式（高置信度）
    DIRECT_PATTERNS = [
        (r'读取\s+[\w./\\-]+', 'read_file', 0.95),
        (r'写入\s+[\w./\\-]+', 'write_file', 0.95),
        (r'列出\s+[\w./\\-]+', 'list_dir', 0.95),
        (r'执行\s+.+', 'bash', 0.90),
        (r'调用工具', None, 0.95),  # "调用工具进行读取"
        (r'用工具', None, 0.95),  # "用工具读取"
        (r'使用工具', None, 0.95),  # "使用工具读取"
        (r'调用\s*[:：]\s*(\w+)', None, 0.90),  # "调用: read_file"
        (r'用\s*(\w+)\s*工具', None, 0.90),  # "用 read_file 工具"
        (r'使用\s*(\w+)\s*读取', 'read_file', 0.90),
    ]

    @staticmethod
    def detect(user_input: str) -> Dict:
        """
        检测用户输入是否为直接命令

        Returns:
            {
                "is_direct_command": bool,
                "tool_name": str | None,
                "confidence": float,
                "reason": str
            }
        """
        import re

        for pattern, tool, confidence in DirectCommandDetector.DIRECT_PATTERNS:
            match = re.search(pattern, user_input)
            if match:
                return {
                    "is_direct_command": True,
                    "tool_name": tool or (match.group(1) if match.groups() else None),
                    "confidence": confidence,
                    "reason": f"匹配直接命令模式: {pattern}"
                }

        return {
            "is_direct_command": False,
            "tool_name": None,
            "confidence": 0.0,
            "reason": "未匹配直接命令模式"
        }
