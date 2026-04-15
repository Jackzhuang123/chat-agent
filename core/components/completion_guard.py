# -*- coding: utf-8 -*-
"""完成判定守卫系统"""

import re
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import SessionContext

def looks_finished(response: str, session: "SessionContext", runtime_context: Dict = None) -> bool:
    """综合判断任务是否完成"""
    # 如果响应中包含工具名关键词，但未成功解析，视为未完成
    tool_keywords = {"execute_python", "read_file", "write_file", "edit_file", "list_dir", "bash"}
    if any(kw in response for kw in tool_keywords):
        # 进一步检查是否有工具调用的典型特征（括号、引号、JSON等）
        if re.search(r'(?:execute_python|read_file|write_file|bash)\s*[\(\{\n]', response, re.IGNORECASE):
            # 明确意图是调用工具，但格式不规范，需要继续
            return False
    # 若有工具调用格式，未完成
    if re.search(r'(?:read_file|write_file|list_dir|bash|execute_python)\s*\n\s*\{', response):
        return False
    # 有完成信号
    if any(sig in response[:150] for sig in ("完成", "总结", "以上", "如下", "已写入", "已读取")):
        return True
    # 已完成步骤数 > 0 且响应足够长
    completed = session.task_context.get("completed_steps", [])
    return len(completed) > 0 and len(response) > 200