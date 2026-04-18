# -*- coding: utf-8 -*-
"""组件模块 - 从 agent_framework 拆分的独立功能"""

from .loop_detector import detect_loop
from .completion_guard import looks_finished
from .output_cleaner import clean_react_tags, strip_trailing_tool_call
from .task_injector import inject_task_context
from .format_corrector import inject_format_correction, should_retry_tool_format

__all__ = [
    "detect_loop",
    "looks_finished",
    "clean_react_tags",
    "strip_trailing_tool_call",
    "inject_task_context",
    "inject_format_correction",
    "should_retry_tool_format",
]
