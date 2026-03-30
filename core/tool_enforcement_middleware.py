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

注意：对于混合任务（既有文件操作又有知识问答），知识问答子任务允许不调用工具直接回答。
"""

import re
from typing import Dict, List

from .agent_middlewares import AgentMiddleware
from .agent_tools import ToolParser


class ToolEnforcementMiddleware(AgentMiddleware):
    """工具强制执行中间件 - 确保 tools 模式下模型必须调用工具"""

    # 纯知识问答的特征信号（无需工具即可回答的子任务标记）
    _KNOWLEDGE_QA_PATTERNS = [
        r'最出名|最著名|最经典|最热门|十首|十大|推荐.*首',
        r'什么是|为什么|怎么理解|解释一下|介绍',
        r'歌曲|歌词|音乐|演唱|专辑',
        r'历史|文化|概念|原理|理论',
    ]

    def __init__(self, max_retries: int = 2):
        """
        Args:
            max_retries: 最大重试次数（默认 2 次）
        """
        self.max_retries = max_retries

    @staticmethod
    def _has_pending_tool_tasks(context: Dict) -> bool:
        """检查当前迭代是否还有未完成的工具任务（即工具调用历史中有失败或者任务刚开始）。"""
        # 如果已完成步骤不为空，说明工具任务已经执行过
        completed = context.get("completed_steps", [])
        failed = context.get("failed_attempts", [])
        return len(completed) == 0 and len(failed) == 0

    @staticmethod
    def _is_knowledge_qa_response(response: str, context: Dict) -> bool:
        """判断模型输出是否是对知识问答子任务的合理回答（无需工具）。"""
        # 响应中包含列举性内容（如：1. 2. 3. 或 - 条目）且内容较长
        has_list = bool(re.search(r'(?:^|\n)\s*(?:\d+[.。]|-|•)\s+\S', response))
        is_substantial = len(response.strip()) > 200
        # 包含知识性内容的标志词
        knowledge_markers = ("首歌", "专辑", "发行", "歌词", "含义", "概念", "原理", "解释", "介绍",
                             "总结", "如下", "以下是", "分别是", "包括")
        has_knowledge_content = any(m in response for m in knowledge_markers)
        return (has_list or has_knowledge_content) and is_substantial

    def process_before_llm(self, messages: List[Dict], context: Dict) -> List[Dict]:
        """在 LLM 调用前注入强制工具调用提示（仅在重试时注入，且跳过知识问答场景）。"""
        run_mode = context.get("run_mode", "chat")
        if run_mode != "tools":
            return messages

        # _knowledge_qa_detected 仅在当前迭代有效，下一轮开始时重置，
        # 防止该标志跨迭代持续跳过工具强制检查（污染后续需要工具调用的迭代）。
        current_iter = context.get("iteration", 0)
        qa_detected_iter = context.get("_knowledge_qa_detected_iter", -1)
        if context.get("_knowledge_qa_detected") and current_iter > qa_detected_iter:
            context["_knowledge_qa_detected"] = False

        # 若当前迭代刚检测到知识问答上下文，不注入强制提示
        if context.get("_knowledge_qa_detected"):
            return messages

        retry_count = context.get("_tool_enforcement_retry", 0)
        if retry_count > 0:
            enforcement_msg = {
                "role": "system",
                "content": (
                    "⚠️ 重要提醒：对于需要操作文件/目录的任务，必须使用工具，不能直接回答。\n\n"
                    "【正确示例】\n"
                    "用户: 读取 core/agent_tools.py\n"
                    "助手: read_file\n"
                    '{"path": "core/agent_tools.py"}\n\n'
                    "用户: 扫描 core 目录\n"
                    "助手: bash\n"
                    '{"command": "grep -Ern \'^class |^def \' core/"}\n\n'
                    "【注意】如果当前子任务是知识问答（无需文件操作），可以直接回答。\n"
                    "请严格按照【工具名 + JSON参数】的格式输出工具调用。"
                )
            }
            messages.insert(0, enforcement_msg)
        return messages

    def process_after_llm(self, response: str, context: Dict) -> str:
        """检查模型输出是否包含工具调用。

        注意：优先使用 ToolParser 的完整解析逻辑（含容错），仅在确实没有任何
        工具调用迹象时才触发重试，避免因格式检查过严误杀合法输出。
        """
        run_mode = context.get("run_mode", "chat")
        if run_mode != "tools":
            return response

        # 使用完整解析器检查（与框架主循环保持一致）
        tool_calls = ToolParser.parse_tool_calls(response)

        # 若解析到工具调用，重置重试计数器
        if tool_calls:
            context["_tool_enforcement_retry"] = 0
            context["_needs_retry"] = False
            return response

        # ---- 无工具调用 ----
        # 先判断是否已是"任务完成"状态的自然语言回答（不应强制重试）
        finish_signals = ("完成", "已完成", "总结", "综上", "结论", "以上", "如下", "好的", "以下是",
                          "done", "finished", "complete")
        is_likely_finished = any(s in response[:100] for s in finish_signals)

        # 若看起来是任务结束的总结，放行（交给主循环的 _looks_finished 处理）
        if is_likely_finished:
            context["_needs_retry"] = False
            return response

        # 检测是否为知识问答子任务的合理回答（混合任务场景）
        if self._is_knowledge_qa_response(response, context):
            context["_knowledge_qa_detected"] = True
            # 记录检测时的迭代号，使该标志仅在本迭代内有效
            context["_knowledge_qa_detected_iter"] = context.get("iteration", 0)
            context["_needs_retry"] = False
            return response

        retry_count = context.get("_tool_enforcement_retry", 0)
        if retry_count < self.max_retries:
            context["_tool_enforcement_retry"] = retry_count + 1
            context["_needs_retry"] = True
            warning = (
                "\n\n⚠️ [系统] 未检测到工具调用，请严格按以下格式输出（工具名单独一行，JSON 紧跟其后）：\n\n"
                "bash\n"
                '{"command": "grep -Ern \'^class |^def \' core/"}\n\n'
                "read_file\n"
                '{"path": "core/agent_tools.py"}\n\n'
                "可用工具: read_file, write_file, edit_file, list_dir, bash\n"
                "⚠️ 禁止在工具名前加任何描述性文字，工具名必须单独占一行。"
            )
            return response + warning
        else:
            # 已达最大重试次数，标记失败但不再追加错误提示（避免污染上下文）
            context["_tool_enforcement_failed"] = True
            context["_needs_retry"] = False
            return response



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
