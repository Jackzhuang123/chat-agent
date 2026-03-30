"""
core 模块 - Agent 核心系统
"""

from .agent_framework import (
    QwenAgentFramework,
    ParallelConfig,
    EnhancedParallelExecutor,
    DeepReflectionEngine,
    READ_ONLY_TOOLS,
    register_read_only_tool,
    create_qwen_model_forward,
)
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
from .mode_router import PreciseModeRouter, IntentType, IntentResult
from .multi_agent import (
    MultiAgentOrchestrator,
    ReActMultiAgentOrchestrator,
    PlannerAgent,
    ExecutorAgent,
    ReviewerAgent,
)
from .streaming_framework import StreamingFramework, create_streaming_wrapper
from .tool_enforcement_middleware import ToolEnforcementMiddleware
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory, LocalEmbeddingProvider

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
]