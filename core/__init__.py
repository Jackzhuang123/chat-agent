"""
core 模块 - Agent 核心系统
"""

# 兼容占位：READ_ONLY_TOOLS / register_read_only_tool / ParallelConfig /
# EnhancedParallelExecutor 保留供外部引用，但已无实际业务逻辑依赖。
from typing import Set as _Set

from .components import DeepReflectionEngine
# ── 从各自独立模块导入（不再依赖 agent_framework_legacy）──────────────────
from .langgraph_agent import LangGraphAgent as QwenAgentFramework  # 向后兼容别名
from .model_forward import create_qwen_model_forward

READ_ONLY_TOOLS: _Set[str] = {"read_file"}

def register_read_only_tool(tool_name: str) -> None:
    """注册只读工具名称（兼容接口，现已不影响 LangGraph 调度逻辑）。"""
    READ_ONLY_TOOLS.add(tool_name)

from dataclasses import dataclass as _dataclass

@_dataclass
class ParallelConfig:
    """兼容占位——LangGraphAgent 内部使用 asyncio.gather 并发，无需此配置。"""
    max_workers: int = 4

    def get_optimal_workers(self, task_count: int) -> int:
        return min(task_count, self.max_workers)

class EnhancedParallelExecutor:
    """兼容占位——并行逻辑已内联到 LangGraphAgent.ToolNode。"""
    READ_ONLY_TOOLS: _Set[str] = READ_ONLY_TOOLS

    def __init__(self, *args, **kwargs):
        self.stats = {"total": 0, "parallel": 0, "failed": 0}
from .agent_middlewares import (
    AgentMiddleware,
    RuntimeModeMiddleware,
    PlanModeMiddleware,
    SkillsContextMiddleware,
    UploadedFilesMiddleware,
    ToolResultGuardMiddleware,
    ConversationSummaryMiddleware,
    CompletenessMiddleware,
    AskUserQuestionMiddleware,
    CompletionStatusMiddleware,
    SearchBeforeBuildingMiddleware,
    RepoOwnershipMiddleware,
)
from .agent_skills import SkillManager, SkillInjector, create_example_skills
from .agent_tools import ToolExecutor, ToolParser, ToolRegistry, create_web_search_tool_placeholder
from .interaction import format_interaction_prompt, parse_interaction_response
from .interactive import format_interactive_prompt, parse_interactive_response as parse_interactive_response_raw
from .mode_router import PreciseModeRouter, IntentType, IntentResult
from .langgraph_agent import LangGraphAgent
from .multi_agent import (
    MultiAgentOrchestrator,
    ReActMultiAgentOrchestrator,
    PlannerAgent,
    ExecutorAgent,
    ReviewerAgent,
)
from .phase_runner import PhaseRunner
from .plugin_executor import execute_plugin, PluginResult
from .plugin_registry import PluginRegistry
from .state_manager import WorkflowStateManager
from .streaming_framework import StreamingFramework, create_streaming_wrapper
from .tool_enforcement_middleware import ToolEnforcementMiddleware
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory, LocalEmbeddingProvider
from .workflow_dag import WorkflowDAG, EXAMPLE_CONFIG
# 新增模块
from .workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    # Framework
    "QwenAgentFramework",
    "ParallelConfig",
    "EnhancedParallelExecutor",
    "DeepReflectionEngine",
    "READ_ONLY_TOOLS",
    "register_read_only_tool",
    "create_qwen_model_forward",
    # Middlewares
    "AgentMiddleware",
    "RuntimeModeMiddleware",
    "PlanModeMiddleware",
    "SkillsContextMiddleware",
    "UploadedFilesMiddleware",
    "ToolResultGuardMiddleware",
    "ConversationSummaryMiddleware",
    "CompletenessMiddleware",
    "AskUserQuestionMiddleware",
    "CompletionStatusMiddleware",
    "SearchBeforeBuildingMiddleware",
    "RepoOwnershipMiddleware",
    "ToolEnforcementMiddleware",
    # Skills
    "SkillManager",
    "SkillInjector",
    "create_example_skills",
    # Tools
    "ToolExecutor",
    "ToolParser",
    "ToolRegistry",
    "create_web_search_tool_placeholder",
    # Routing
    "PreciseModeRouter",
    "IntentType",
    "IntentResult",
    # Multi-Agent
    "MultiAgentOrchestrator",
    "ReActMultiAgentOrchestrator",
    "PlannerAgent",
    "ExecutorAgent",
    "ReviewerAgent",
    # Streaming
    "StreamingFramework",
    "create_streaming_wrapper",
    # Learning & Memory
    "AdaptiveToolLearner",
    "VectorMemory",
    "LocalEmbeddingProvider",
    # New workflow modules
    "WorkflowOrchestrator",
    "WorkflowStateManager",
    "format_interaction_prompt",
    "parse_interaction_response",
    "format_interactive_prompt",
    "parse_interactive_response_raw",
    "PluginRegistry",
    "execute_plugin",
    "PluginResult",
    "PhaseRunner",
    "WorkflowDAG",
    "EXAMPLE_CONFIG",
    "LangGraphAgent",
]