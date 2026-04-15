# -*- coding: utf-8 -*-
"""组件模块 - 从 agent_framework 拆分的独立功能"""

from .deep_reflection import DeepReflectionEngine
from .context_compressor import compress_context_smart
from .loop_detector import detect_loop
from .completion_guard import looks_finished
from .output_cleaner import clean_react_tags, strip_trailing_tool_call
from .task_injector import inject_task_context
from .tool_recommender import inject_tool_recommendations, inject_efficient_sequences
from .format_corrector import inject_format_correction

__all__ = [
    "DeepReflectionEngine",
    "compress_context_smart",
    "detect_loop",
    "looks_finished",
    "clean_react_tags",
    "strip_trailing_tool_call",
    "inject_task_context",
    "inject_tool_recommendations",
    "inject_efficient_sequences",
    "inject_format_correction",
]