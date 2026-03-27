ui/glm_agent.py 文件定义了一个名为 GLMAgent 的类，它是一个智能 GLM API 封装类，用于与 QwenAgent 接口完全兼容。该类支持流式输出，可以无障碍地替换本地模型。

主要功能包括：
- 定义了可用的模型列表（AVAILABLE_MODELS）。
- 提供了一个初始化方法（__init__），用于创建 GLMAgent 实例。
- 支持不同的模型选项，包括免费和付费版本。
- 可选的日志记录器，用于记录操作和日志。

使用方式：
from glm_agent import GLMAgent
agent = GLMAgent(api_key="your_api_key")"""核心模块 - Agent 核心系统

包含:
  - agent_framework  : Agent 主框架（QwenAgentFramework - ReAct + 反思 + 并行执行 + 持久化记忆
  - agent_middlewares: 中间件链（13 个中间件 + 基类
  - agent_tools      : 工具系统（文件、命令等
  - agent_skills     : Skills 系统（知识外化

核心特性:
  - ReAct 模式：Reasoning → Acting → Observation → Reflection
  - 并行工具执行：只读工具（read_file/list_dir）自动并行
  - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
  - 语义压缩：基于消息重要性的智能上下文压缩
  - 循环检测：3 次相同失败自动中断
  - 智能重试：自动修复常见错误（grep 转换、路径补全等）"""class SessionMemory:
    """会话记忆 + 工具使用统计 + 持久化"""

    def __init__(self, memory_dir: str = ".agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.current_session = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 工具使用统计
        self.tool_stats = defaultdict(lambda: {"success": 0, "failed": 0, "avg_time": 0})

        # 跨会话记忆文件
        self.memory_file = self.memory_dir / "session_memory.pkl"
        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载历史记忆"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.tool_stats = data.get("tool_stats", self.tool_stats)
                    # #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 中间件模块 - 实现Middleware Chain 设计模式

所有中间件遵循 AgentMiddleware 接口，可任意组合添加到 QwenAgentFramework 中。
每个中间件负责：
  - before_model   : 模型调用前修改消息列表（添加上下文信息）
  - after_model    : 模型返回后后处理文本
  - after_tool_call: 工具调用后标准化结果
  - after_run      : 会话结束处理工具
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# TodoManager 已移除，相关中间件保留但标记为可选


def _inject_context_before_last_user(
    messages: List[Dict[str, str]],
    context_message: Dict[str, str],
) -> List[Dict[str, str]]:
    """将上下文信息插入到最新用户消息前，保持上下文信息尽可能可见。"""
    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated.insert(idx, context_message)
            return updated
    updated.append(context_message)
    return updated

# ui/glm_agent.py 文件定义了一个名为 GLMAgent 的类，它是一个智能 GLM API 封装类，用于与 QwenAgent 接口完全兼容。该类支持流式输出，可以无障碍地替换本地模型。

主要功能包括：
- 定义了可用的模型列表（AVAILABLE_MODELS）。
- 提供了一个初始化方法（__init__），用于创建 GLMAgent 实例。
- 支持不同的模型选项，包括免费和付费版本。
- 可选的日志记录器，用于记录操作和日志。

使用方法：
from glm_agent import GLMAgent
agent = GLMAgent(api_key="your_api_key")"""
核心模块 - Agent 核心系统

包含:
  - agent_framework  : Agent 主框架（QwenAgentFramework - ReAct + 反思 + 并行执行 + 持久化记忆
  - agent_middlewares: 中间件链（13 个中间件 + 基类
  - agent_tools      : 工具系统（文件、命令等
  - agent_skills     : Skills 系统知识外化

核心特性:
  - ReAct 模式：Reasoning → Acting → Observation → Reflection
  - 并行工具执行：只读工具（read_file/list_dir）自动并行执行
  - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
  - 语法：
    - ReAct 模式：Reasoning → Acting → Observation → Reflection
    - 并行工具执行：只读工具（read_file/list_dir）自动并行执行
    - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
    - 语法：
      - ReAct 模式：Reasoning → Acting → Observation → Reflection
      - 并行工具执行：只读工具（read_file/list_dir）自动并行执行
      - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
      - 语法：
        - ReAct 模式：Reasoning → Acting → Observation → Reflection
        - 并行工具执行：只读工具（read_file/list_dir）自动并行执行
        - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
        - 语法：
          - ReAct 模式：Reasoning → Acting → Observation → Reflection
          - 并行工具执行：只读工具（read_file/list_dir）自动并行执行
          - 持久化记忆：SessionMemory 跨会话保存工具统计和上下文core/__init__.py 文件定义了 Agent 核心模块，包含以下内容：

- agent_framework  : Agent 主框架（QwenAgentFramework - ReAct + 反思 + 并行执行 + 持久化记忆）
- agent_middlewares: 中间件连接（13 个中间件 + 基类）
- agent_tools      : 工具系统（文件、命令等）
- agent_skills     : Skills 系统外设化

核心特性：
- ReAct 模式：Reasoning → Acting → Observation → Reflection
- 并行工具执行：只读工具（read_file/list_dir）自动并行
- 持久化记忆：SessionMemory 跨会话保存工具统计和上下文
- 语义压缩：基于消息重要性的智能上下文压缩
- 环境监测：3 次相同失败自动中断
- 智能重试：自动修复常见错误（grep 转换、路径补全）

# ============================================================================
# 核心Agent框架（精简版 201 行）
# ============================================================================
from .agent_framework import QwenAgentFramework
# ============================================================================
# 中间件（从 agent_middlewares 导入，包含优化的实现）
# ============================================================================
from .agent_middlewares import (
    AgentMidd...core/agent_framework.py 文件定义了 SessionMemory 类，用于会话记忆、工具使用统计和持久化。

类 SessionMemory 的主要功能：
- 初始化：创建内存目录，设置当前会话和会话ID。
- 工具使用统计：记录每个工具的成功、失败次数和平均执行时间。
- 会话记忆：保存和加载会话历史数据。

代码示例：
from .agent_middlewares import AgentMiddleware
from .agent_tools import ToolExecutor, ToolParser
from .tool_learner import ToolLearner

class SessionMemory:
    def __init__(self, memory_dir: str = ".agent_memory”):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.current_session = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S”)...core/agent_middlewares.py 文件定义了 AgentMiddleware 类，用于实现 Middleware Chain 设计模式。

主要功能：
- 中间件遵循统一的 AgentMiddleware 接口，可自由组合以适应 QwenAgentFramework。
- 每个中间件负责处理特定的任务，包括模型调用前后的消息修改、工具调用后的结果标准化和会话结束后的处理。

代码示例：
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

class AgentMiddleware:
    def __init__(self):
        pass

# TodoManager 已移除，相关中间件保留但标记为可选

# 示例方法：
def _inject_context_before_last_user(
    messages: List[Dict[str, str]],
    context_message: Dict[str, str],
):
    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated.insert(idx, context_message)
            return updated...core/agent_skills.py 文件定义了 Agent Skills 系统用于知识外化。

主要功能：
- 核心功能：
  - 工具 = 模型能做什么 (bash, read_file, etc.)
  - 技能 = 模型知道如何做 (PDF处理, MCP开发, 代码审查等)

- 技能允许按需加载特定领域的知识，无需重新训练模型。
- 关键特性：
  1. 渐进式提示：元数据 -> 详细说明 -> 资源
  2. 上下文高效：技能作为工具结果加入，保留缓存
  3. 可编辑知识：任何人都可以编写 SKILL.md 文件

代码示例：
from pathlib import Path
from typing import Any, Dict, List, Optional

class SkillManager:
    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            self.skills_dir = Path.cwd() / 'skills'