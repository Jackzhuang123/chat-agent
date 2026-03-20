"""
core 模块 - Agent 核心系统

包含:
  - agent_framework: Agent 循环和规划
  - agent_tools: 工具系统 (文件、命令等)
  - agent_skills: Skills 系统 (知识外置化)
"""

from .agent_framework import (
    AgentMiddleware,
    PlanModeMiddleware,
    QwenAgentFramework,
    RuntimeModeMiddleware,
    SkillsContextMiddleware,
    ToolResultGuardMiddleware,
    UploadedFilesMiddleware,
    create_qwen_model_forward,
)
from .agent_skills import SkillManager, SkillInjector, create_example_skills
from .agent_tools import ToolExecutor, ToolParser

__all__ = [
    "AgentMiddleware",
    "QwenAgentFramework",
    "create_qwen_model_forward",
    "RuntimeModeMiddleware",
    "PlanModeMiddleware",
    "SkillsContextMiddleware",
    "UploadedFilesMiddleware",
    "ToolResultGuardMiddleware",
    "ToolExecutor",
    "ToolParser",
    "SkillManager",
    "SkillInjector",
    "create_example_skills",
]

