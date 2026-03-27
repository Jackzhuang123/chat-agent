"""
core 模块 - Agent 核心系统

包含:
  - agent_framework  : Agent 主循环（QwenAgentFramework - ReAct + 反思 + 并行执行 + 持久化记忆）
  - agent_middlewares: 中间件链（13 个中间件 + 基类）
  - agent_tools      : 工具系统（文件、命令等）
  - agent_skills     : Skills 系统（知识外置化）

核心特性：
  - ReAct 模式：Reasoning → Acting → Observation → Reflection
  - 并行工具执行：只读工具（read_file/list_dir）自动并发
  - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
  - 语义压缩：基于消息重要性的智能上下文压缩
  - 循环检测：3次相同失败自动中断
  - 智能重试：自动修复常见错误（grep转义、路径补全）
"""

# ============================================================================
# 核心 Agent 框架（精简版 201 行）
# ============================================================================
from .agent_framework import QwenAgentFramework
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
# 工具与技能系统
# ============================================================================
from .agent_skills import SkillManager, SkillInjector, create_example_skills
from .agent_tools import ToolExecutor, ToolParser, ToolRegistry, create_web_search_tool_placeholder
from .mode_router import ModeRouter, AutoModeMiddleware, create_auto_mode_framework
from .multi_agent import (
    PlannerAgent,
    ExecutorAgent,
    ReviewerAgent,
    MultiAgentOrchestrator,
)
from .streaming_framework import StreamingFramework, create_streaming_wrapper
from .tool_learner import ToolLearner
# ============================================================================
# 高级特性（新增）
# ============================================================================
from .vector_memory import VectorMemory


# 已移除未使用组件：TodoManager, TaskPlanner, IntentRouter, MemoryManager
# 记忆功能已集成到 agent_framework.SessionMemory


# ============================================================================
# 工具函数：将 Agent 实例包装为 QwenAgentFramework 所需的 forward_fn
# ============================================================================
def create_qwen_model_forward(agent):
    """工厂函数：将 QwenAgent / GLMAgent 包装为 QwenAgentFramework 所需的 model_forward_fn。

    返回一个同步调用函数 forward_fn(messages, system_prompt="", **kwargs) -> str，
    内部消耗 agent.generate_stream_with_messages 的全部 token，最终返回完整生成文本。

    用法：
        forward_fn = create_qwen_model_forward(my_agent)
        framework = QwenAgentFramework(model_forward_fn=forward_fn, ...)
    """
    def forward_fn(messages, system_prompt="", **kwargs):
        full_messages = list(messages)
        if system_prompt and system_prompt.strip():
            # 避免重复注入：若 messages 第一条已是内容相同的 system 消息则跳过
            _already_has_system = (
                full_messages
                and full_messages[0].get("role") == "system"
                and full_messages[0].get("content", "").strip() == system_prompt.strip()
            )
            if not _already_has_system:
                full_messages = [{"role": "system", "content": system_prompt}] + full_messages
        result = ""
        for chunk in agent.generate_stream_with_messages(
            full_messages,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 8192),
        ):
            result = chunk  # generate_stream_with_messages 每次 yield 累积文本
        return result
    return forward_fn

__all__ = [
    # ---- 框架核心 ----
    "QwenAgentFramework",

    # ---- 中间件基类 ----
    "AgentMiddleware",

    # ---- 中间件：上下文注入 ----
    "RuntimeModeMiddleware",
    "PlanModeMiddleware",
    "SkillsContextMiddleware",
    "UploadedFilesMiddleware",
    "ToolResultGuardMiddleware",

    # ---- 中间件：TODO 状态追踪（可选）----
    "TodoContextMiddleware",
    "EnhancedTodoContextMiddleware",
    "make_todo_context_middleware",

    # ---- 中间件：记忆与摘要（可选）----
    "MemoryInjectionMiddleware",
    "ConversationSummaryMiddleware",

    # ---- 中间件：gstack 架构移植（可选）----
    "CompletenessMiddleware",
    "AskUserQuestionMiddleware",
    "CompletionStatusMiddleware",
    "SearchBeforeBuildingMiddleware",
    "RepoOwnershipMiddleware",

    # ---- 工具系统 ----
    "ToolExecutor",
    "ToolParser",
    "ToolRegistry",
    "create_web_search_tool_placeholder",

    # ---- 技能系统 ----
    "SkillManager",
    "SkillInjector",
    "create_example_skills",

    # ---- 工具函数 ----
    "create_qwen_model_forward",

    # ---- 高级特性 ----
    "VectorMemory",
    "PlannerAgent",
    "ExecutorAgent",
    "ReviewerAgent",
    "MultiAgentOrchestrator",
    "ToolLearner",
    "StreamingFramework",
    "create_streaming_wrapper",
    "ModeRouter",
    "AutoModeMiddleware",
    "create_auto_mode_framework",
]

