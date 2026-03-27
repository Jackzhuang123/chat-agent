# 模式路由器文档

## 概述

**ModeRouter** 是智能模式路由器，可以根据用户输入自动识别意图并推荐合适的运行模式，无需手动指定 `runtime_context`。

## 支持的模式

| 模式 | 描述 | 适用场景 | 关键词示例 |
|------|------|---------|-----------|
| **chat** | 纯对话模式 | 闲聊、问答、解释概念 | 什么是、为什么、怎么样 |
| **tools** | 工具模式 | 文件操作、代码分析、命令执行 | 读取、写入、修改、列出、执行 |
| **plan** | 计划模式 | 复杂任务、多步骤操作 | 分析、重构、优化、设计、实现 |
| **multi_agent** | 多Agent模式 | 需要规划-执行-审查 | 规划并执行、审查、评估质量 |
| **streaming** | 流式模式 | 实时展示进度 | 实时、流式、进度、监控 |

## 快速开始

### 方式1：使用 `create_auto_mode_framework`（推荐）

```python
from core import create_auto_mode_framework, QwenAgentFramework

# 创建自动模式框架
framework = create_auto_mode_framework(
    QwenAgentFramework,
    model_forward_fn=model_forward,
    enable_parallel=True,
    enable_memory=True
)

# 直接使用，无需指定模式
result = framework.run("读取core/agent_framework.py")
# 🔍 自动检测模式: tools (置信度: 0.85)
# 💡 原因: 检测到文件操作关键词，推荐工具模式

result = framework.run("什么是 Python？")
# 🔍 自动检测模式: chat (置信度: 0.70)
# 💡 原因: 检测到问答关键词，推荐对话模式
```

### 方式2：手动使用 ModeRouter

```python
from core import ModeRouter, QwenAgentFramework

router = ModeRouter()

# 检测模式
detection = router.detect_mode("读取core/agent_framework.py")
print(detection)
# {
#   "recommended_mode": "tools",
#   "confidence": 0.85,
#   "complexity": "simple",
#   "reasoning": "检测到 1 个关键词，推荐工具模式"
# }

# 手动应用模式
framework = QwenAgentFramework(model_forward_fn=model_forward)
runtime_context = router.suggest_parameters(
    detection["recommended_mode"],
    "读取core/agent_framework.py"
)

result = framework.run(
    "读取core/agent_framework.py",
    runtime_context=runtime_context
)
```

### 方式3：使用 AutoModeMiddleware

```python
from core import AutoModeMiddleware, QwenAgentFramework

middleware = AutoModeMiddleware()
framework = QwenAgentFramework(model_forward_fn=model_forward)

# 在调用前处理
user_input = "读取文件"
runtime_context = {}
runtime_context = middleware.process_before_run(user_input, runtime_context)

result = framework.run(user_input, runtime_context=runtime_context)
```

## 检测规则

### 关键词匹配

每种模式都有一组关键词，匹配越多，置信度越高。

**chat 模式关键词**：
```python
["什么是", "为什么", "怎么样", "如何理解", "解释", "介绍", "聊天"]
```

**tools 模式关键词**：
```python
["读取", "写入", "修改", "删除", "列出", "扫描", "查找",
 "执行", "运行", "命令", "grep", "find", "ls", "cat"]
```

**plan 模式关键词**：
```python
["分析", "重构", "优化", "设计", "实现", "开发",
 "生成", "创建项目", "搭建", "部署"]
```

### 复杂度评估

根据关键词和输入长度自动评估任务复杂度：

| 复杂度 | 关键词 | 输入长度 |
|--------|--------|---------|
| simple | 读取、查看、列出 | <30 字符 |
| medium | 修改、编辑、查找 | 30-100 字符 |
| complex | 重构、优化、设计 | >100 字符 |

### 自动升级规则

1. **复杂任务自动启用计划模式**
   ```python
   输入: "分析并重构整个项目的代码结构"
   检测: tools (0.8) → 自动升级为 plan
   ```

2. **多步骤任务自动启用多Agent**
   ```python
   输入: "规划并执行一个完整的项目搭建流程"
   检测: plan (0.7) + "并" → 自动升级为 multi_agent
   ```

## 参数推荐

不同模式会自动设置不同的参数：

```python
# chat 模式
{
    "run_mode": "chat",
    "plan_mode": False,
    "max_iterations": 10
}

# tools 模式
{
    "run_mode": "tools",
    "plan_mode": False,
    "max_iterations": 5  # 简单任务
}

# plan 模式
{
    "run_mode": "plan",
    "plan_mode": True,
    "max_iterations": 15
}

# multi_agent 模式
{
    "run_mode": "multi_agent",
    "use_multi_agent": True,
    "max_iterations": 20
}

# streaming 模式
{
    "run_mode": "streaming",
    "enable_streaming": True,
    "max_iterations": 10
}
```

## 使用示例

### 示例1：自动对话模式

```python
framework = create_auto_mode_framework(QwenAgentFramework, model_forward)

result = framework.run("什么是 ReAct 模式？")
# 🔍 自动检测模式: chat (置信度: 0.70)
# 💡 原因: 检测到问答关键词，推荐对话模式
#
# 响应: ReAct 是 Reasoning + Acting 的缩写...
```

### 示例2：自动工具模式

```python
result = framework.run("读取core/agent_framework.py并分析代码结构")
# 🔍 自动检测模式: tools (置信度: 0.85)
# 💡 原因: 检测到文件操作关键词，推荐工具模式
#
# 工具调用: read_file
# 响应: 已读取文件，包含666行代码...
```

### 示例3：自动计划模式

```python
result = framework.run("重构整个项目的架构")
# 🔍 自动检测模式: plan (置信度: 0.90)
# 💡 原因: 检测到重构关键词，任务复杂度高，推荐计划模式
#
# 计划:
#   1. 分析当前架构
#   2. 设计新架构
#   3. 逐步重构
#   4. 测试验证
```

### 示例4：自动多Agent模式

```python
result = framework.run("规划并执行一个完整的Web应用开发流程")
# 🔍 自动检测模式: multi_agent (置信度: 0.85)
# 💡 原因: 检测到多步骤任务，推荐多Agent模式
#
# Planner: 生成开发计划
# Executor: 执行各步骤
# Reviewer: 审查代码质量
```

### 示例5：自动流式模式

```python
result = framework.run("实时展示文件扫描进度")
# 🔍 自动检测模式: streaming (置信度: 0.80)
# 💡 原因: 检测到实时关键词，推荐流式模式
#
# [start] 开始扫描...
# [progress] 已扫描 10 个文件...
# [progress] 已扫描 20 个文件...
# [complete] 扫描完成
```

## 手动切换模式

如果自动检测不准确，可以手动指定：

```python
# 方式1：在 runtime_context 中指定
result = framework.run(
    "读取文件",
    runtime_context={"run_mode": "tools"}
)

# 方式2：使用原始框架（不使用自动模式）
framework = QwenAgentFramework(model_forward_fn=model_forward)
result = framework.run(
    "读取文件",
    runtime_context={"run_mode": "tools"}
)
```

## 自定义模式

可以扩展 ModeRouter 添加自定义模式：

```python
from core import ModeRouter

router = ModeRouter()

# 添加自定义模式
router.mode_patterns["custom"] = {
    "keywords": ["自定义", "特殊任务"],
    "priority": 0.8,
    "description": "自定义模式"
}

# 使用
detection = router.detect_mode("执行自定义任务")
print(detection["recommended_mode"])  # "custom"
```

## 调试

### 查看检测结果

```python
router = ModeRouter()
detection = router.detect_mode("读取文件")

print(f"推荐模式: {detection['recommended_mode']}")
print(f"置信度: {detection['confidence']}")
print(f"复杂度: {detection['complexity']}")
print(f"原因: {detection['reasoning']}")
print(f"备选方案: {detection['alternatives']}")
```

### 调整置信度阈值

默认置信度阈值是 0.6，可以调整：

```python
# 在 auto_switch_mode 中
new_mode, reason = router.auto_switch_mode(
    user_input,
    current_mode,
    context
)

# 手动检查置信度
detection = router.detect_mode(user_input)
if detection["confidence"] > 0.8:  # 更高的阈值
    # 切换模式
    pass
```

## 性能

- **检测速度**：<1ms（纯Python字符串匹配）
- **内存占用**：<1KB（只存储规则）
- **准确率**：~85%（基于关键词匹配）

## 常见问题

### Q: 自动检测不准确怎么办？
A: 可以手动指定 `runtime_context={"run_mode": "tools"}`

### Q: 如何禁用自动检测？
A: 使用原始框架 `QwenAgentFramework`，不使用 `create_auto_mode_framework`

### Q: 能否同时使用多个模式？
A: 可以，例如 `plan` + `streaming` 组合

### Q: 如何添加新的关键词？
A: 修改 `router.mode_patterns` 添加关键词

### Q: 置信度低怎么办？
A: 添加更多关键词或降低阈值（默认0.6）

## 与中间件的关系

ModeRouter 自动设置 `runtime_context`，中间件根据 `runtime_context` 注入提示：

```
用户输入
    ↓
ModeRouter 检测模式
    ↓
设置 runtime_context["run_mode"]
    ↓
RuntimeModeMiddleware 注入模式提示
    ↓
模型执行
```

## 总结

- ✅ **零配置**：自动检测模式，无需手动指定
- ✅ **智能升级**：复杂任务自动启用高级模式
- ✅ **可定制**：支持自定义模式和关键词
- ✅ **高性能**：<1ms 检测速度
- ✅ **易调试**：详细的检测原因和置信度
