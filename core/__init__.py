"""
core 模块 - Agent 核心系统

包含:
  - agent_framework  : Agent 主循环（QwenAgentFramework + 工具调用）
  - agent_middlewares: 中间件链（13 个中间件 + 基类）
  - agent_tools      : 工具系统（文件、命令等）
  - agent_skills     : Skills 系统（知识外置化）
  - todo_manager     : TODO 状态机（四态：pending/in_progress/completed/failed）
  - intent_router    : 意图路由（规则路由 + AI 语义路由）
  - task_planner     : 任务规划与执行（Orchestrator-Worker 架构）
  - memory_manager   : 跨会话持久化记忆系统

设计架构（借鉴 DeerFlow + gstack）：
  - MemoryManager          : 跨会话记忆 → 借鉴 DeerFlow MemoryUpdater
  - MemoryInjectionMiddleware : 记忆注入中间件 → 借鉴 DeerFlow MemoryMiddleware
  - ConversationSummaryMiddleware : 长对话摘要 → 借鉴 DeerFlow SummarizationMiddleware
  - make_todo_context_middleware  : 增强版 TodoContext 工厂（含上下文丢失检测）
  - EnhancedTodoContextMiddleware : 顶层类（可直接使用，无需工厂函数）
  - CompletenessMiddleware : 完整性原则 → 借鉴 gstack Boil the Lake
  - AskUserQuestionMiddleware : 结构化提问格式 → 借鉴 gstack SKILL.md
  - CompletionStatusMiddleware : 完成状态协议 → 借鉴 gstack ETHOS.md
  - SearchBeforeBuildingMiddleware : 搜索优先 → 借鉴 gstack ETHOS.md
  - RepoOwnershipMiddleware : 仓库所有权模式 → 借鉴 gstack ARCHITECTURE.md
"""

# ============================================================================
# 核心 Agent 框架（保留 QwenAgentFramework + create_qwen_model_forward）
# ============================================================================
from .agent_framework import (
    QwenAgentFramework,
    create_qwen_model_forward,
)

# ============================================================================
# 中间件（从 agent_middlewares 导入，包含更优化的实现）
# ============================================================================
from .agent_middlewares import (
    AgentMiddleware,
    RuntimeModeMiddleware,
    PlanModeMiddleware,
    SkillsContextMiddleware,
    UploadedFilesMiddleware,
    ToolResultGuardMiddleware,
    TodoContextMiddleware,
    EnhancedTodoContextMiddleware,
    make_todo_context_middleware,
    MemoryInjectionMiddleware,
    ConversationSummaryMiddleware,
    # gstack 架构移植中间件
    CompletenessMiddleware,
    AskUserQuestionMiddleware,
    CompletionStatusMiddleware,
    SearchBeforeBuildingMiddleware,
    RepoOwnershipMiddleware,
)

# ============================================================================
# TODO 状态管理（从 todo_manager 导入）
# ============================================================================
from .todo_manager import (
    TodoItem,
    TodoManager,
)

# ============================================================================
# 意图路由（从 intent_router 导入）
# ============================================================================
from .intent_router import (
    IntentRouter,
    AIIntentRouter,
)

# ============================================================================
# 任务规划与执行（从 task_planner 导入）
# ============================================================================
from .task_planner import (
    TaskPlanner,
    TaskExecutor,
)

# ============================================================================
# 记忆系统（从 memory_manager 导入）
# ============================================================================
from .memory_manager import MemoryManager

# ============================================================================
# 工具与技能系统
# ============================================================================
from .agent_skills import SkillManager, SkillInjector, create_example_skills
from .agent_tools import ToolExecutor, ToolParser, ToolRegistry, create_web_search_tool_placeholder

__all__ = [
    # ---- 框架核心 ----
    "QwenAgentFramework",
    "create_qwen_model_forward",

    # ---- 中间件基类 ----
    "AgentMiddleware",

    # ---- 中间件：上下文注入 ----
    "RuntimeModeMiddleware",
    "PlanModeMiddleware",
    "SkillsContextMiddleware",
    "UploadedFilesMiddleware",
    "ToolResultGuardMiddleware",

    # ---- 中间件：TODO 状态追踪 ----
    "TodoContextMiddleware",
    "EnhancedTodoContextMiddleware",
    "make_todo_context_middleware",

    # ---- 中间件：记忆与摘要（DeerFlow 架构） ----
    "MemoryInjectionMiddleware",
    "ConversationSummaryMiddleware",

    # ---- 中间件：gstack 架构移植 ----
    "CompletenessMiddleware",
    "AskUserQuestionMiddleware",
    "CompletionStatusMiddleware",
    "SearchBeforeBuildingMiddleware",
    "RepoOwnershipMiddleware",

    # ---- 记忆系统 ----
    "MemoryManager",

    # ---- TODO 管理 ----
    "TodoItem",
    "TodoManager",

    # ---- 意图路由 ----
    "IntentRouter",
    "AIIntentRouter",

    # ---- 任务规划 & 执行 ----
    "TaskPlanner",
    "TaskExecutor",

    # ---- 工具系统 ----
    "ToolExecutor",
    "ToolParser",
    "ToolRegistry",
    "create_web_search_tool_placeholder",

    # ---- 技能系统 ----
    "SkillManager",
    "SkillInjector",
    "create_example_skills",
]

