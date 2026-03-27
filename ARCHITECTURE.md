# 架构设计

## 总体架构

```
QwenAgentFramework (720行)
├── SessionMemory          # 持久化记忆 + 工具统计
├── ReflectionEngine       # 反思引擎（错误分类 + 建议）
├── OutputValidator        # 输出验证器（参数检查 + 清理）
├── ToolExecutor           # 工具执行器
├── ToolParser             # 工具解析器
└── Middlewares (可选)     # 中间件链
```

## 核心组件

### 1. SessionMemory - 持久化记忆

**功能**：
- 跨会话保存工具使用统计
- 提取关键信息（用户请求、文件、错误）
- 计算消息重要性（语义评分）
- 构建上下文摘要

**持久化**：
- 文件：`.agent_memory/session_memory.pkl`
- 内容：`tool_stats` + `recent_context`
- 加载：启动时自动加载
- 保存：每次 `run()` 结束时

**工具统计**：
```python
{
    "read_file": {
        "success": 10,
        "failed": 2,
        "avg_time": 0.5
    },
    "bash": {
        "success": 5,
        "failed": 3,
        "avg_time": 1.2
    }
}
```

**语义评分**：
```python
score = sum(1 for kw in ["error", "file", "tool", ...] if kw in content)
score *= 1.5  # 用户消息权重
score *= 1.3  # 工具结果权重
```

### 2. ReflectionEngine - 反思引擎

**功能**：
- 反思工具执行结果
- 错误分类（not found / permission / syntax / unknown）
- 提供针对性建议
- 判断是否应该继续（连续失败检测）

**错误分类**：
```python
if "not found" in error.lower():
    analysis = "文件/路径不存在"
    suggestions = ["使用 list_dir 确认路径", "检查文件名拼写"]
elif "permission" in error.lower():
    analysis = "权限不足"
    suggestions = ["检查文件权限", "使用其他路径"]
elif "syntax" in error.lower():
    analysis = "命令语法错误"
    suggestions = ["检查参数格式", "参考工具文档"]
```

**连续失败检测**：
```python
recent = history[-3:]
all_failed = all(not h["success"] for h in recent)
if all_failed:
    return False, "连续失败，建议重新规划任务"
```

### 3. OutputValidator - 输出验证器

**功能**：
- 验证工具调用参数
- 清理过长输出（>2000字符）

**参数验证**：
```python
required_params = {
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_content", "new_content"],
    "list_dir": [],
    "bash": ["command"]
}
```

### 4. QwenAgentFramework - 主循环

**ReAct 流程**：
```
1. Thought (思考) → 分析当前情况
2. Action (行动) → 调用工具
3. Observation (观察) → 查看结果
4. Reflection (反思) → 分析是否成功
```

**主循环**：
```python
for iteration in range(max_iterations):
    # 1. 检查是否应该继续（连续失败检测）
    should_continue, reason = reflection.should_continue()

    # 2. 上下文管理（压缩 + 注入）
    messages = _compress_context_smart(messages)
    messages = _inject_task_context(messages)
    messages = _inject_reflection(messages)

    # 3. 中间件处理
    for mw in middlewares:
        messages = mw.process_before_llm(messages, runtime_context)

    # 4. 调用模型
    response = model_forward_fn(messages, system_prompt)

    # 5. 解析工具
    tool_calls = tool_parser.parse_tool_calls(response)

    # 6. 并行执行工具
    parallel_tools, sequential_tools = _detect_parallel_tools(tool_calls)
    results = _execute_tools_parallel(parallel_tools)
    results += _execute_tools_sequential(sequential_tools)

    # 7. 循环检测
    if _detect_loop():
        break

    # 8. 回注结果
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": _format_results(results)})
```

## 并行执行机制

### 检测可并行工具

```python
def _detect_parallel_tools(tool_calls):
    parallel_tools = []
    sequential_tools = []

    for tc in tool_calls:
        if tc["name"] in ["read_file", "list_dir"]:
            parallel_tools.append(tc)  # 只读工具
        else:
            sequential_tools.append(tc)  # 写入工具

    return parallel_tools, sequential_tools
```

### 并行执行

```python
def _execute_tools_parallel(tool_calls):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_execute_single_tool, tc["name"], tc["args"]): tc
            for tc in tool_calls
        }

        results = []
        for future in as_completed(futures):
            tc = futures[future]
            result = future.result()
            results.append({
                "tool": tc["name"],
                "result": result,
                "parallel": True
            })

    return results
```

## 语义压缩机制

### 压缩触发条件

```python
total_chars = sum(len(json.dumps(m)) for m in messages)
if total_chars / 1.5 < 6000 * 0.75:  # 75% 阈值
    return messages  # 不压缩
```

### 压缩策略

```python
# 1. 保留最近6条
recent = user_assistant[-6:]

# 2. 计算历史消息重要性
scores = memory.compute_message_importance(old)

# 3. 保留 top-3 重要消息
scored_msgs = list(zip(old, scores))
scored_msgs.sort(key=lambda x: x[1], reverse=True)
important_msgs = [msg for msg, _ in scored_msgs[:3]]

# 4. 构建压缩后的消息
return system + [summary] + important_msgs + recent
```

## 循环检测机制

### 检测逻辑

```python
def _detect_loop(max_same=3):
    if len(tool_history) < max_same:
        return False

    recent = tool_history[-max_same:]

    # 检查是否全部失败
    if not all(not h["success"] for h in recent):
        return False

    # 检查是否相同工具+参数
    first = recent[0]
    return all(
        h["tool"] == first["tool"] and h["args"] == first["args"]
        for h in recent
    )
```

## 智能重试机制

### 自动修复

```python
def _try_fix(tool, args, error):
    # 1. grep 转义错误
    if tool == "bash" and "grep" in args.get("command", ""):
        cmd = args["command"]
        if "\\(" in cmd and "\\\\(" not in cmd:
            return {"command": cmd.replace("\\(", "(").replace("\\)", ")")}

    # 2. 路径补全
    if tool in ["read_file", "edit_file"] and "not found" in error.lower():
        path = args.get("path", "")
        if path and not path.startswith("/") and not path.startswith("."):
            return {"path": f"./{path}"}

    return None
```

## 性能优化

### 1. Token 优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 上下文压缩 | 语义评分 + top-K | 保留关键信息 |
| 工具输出 | 截断 >2000 字符 | 减少回注 token |
| 历史摘要 | 提取关键信息 | 压缩历史对话 |

### 2. 执行优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 并行执行 | ThreadPoolExecutor | 2-3x faster |
| 循环检测 | 3次中断 | 避免死循环 |
| 智能重试 | 自动修复常见错误 | 减少失败次数 |

### 3. 记忆优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 持久化 | pickle 序列化 | 跨会话学习 |
| 工具统计 | 成功率 + 平均耗时 | 智能推荐 |
| 上下文保留 | 最近3次会话 | 连续性 |

## 对比其他框架

### vs LangChain

| 维度 | Chat-Agent | LangChain |
|------|-----------|-----------|
| **代码量** | 720行 | 10000+ 行 |
| **并行执行** | ✅ 自动检测 | ❌ 需手动配置 |
| **持久化记忆** | ✅ 内置 | ❌ 需插件 |
| **语义压缩** | ✅ 自动 | ❌ 简单截断 |
| **循环检测** | ✅ 3次中断 | ❌ 无 |

### vs AutoGPT

| 维度 | Chat-Agent | AutoGPT |
|------|-----------|---------|
| **反思机制** | ✅ ReflectionEngine | ✅ 有 |
| **工具执行** | 并行 + 串行 | 串行 |
| **上下文管理** | 语义压缩 | 简单截断 |
| **记忆系统** | 持久化 | 向量数据库 |

## 扩展性

### 添加新工具

```python
# 1. 在 ToolExecutor 中添加工具实现
def execute_tool(self, tool_name, args):
    if tool_name == "my_tool":
        return self._my_tool(args)

# 2. 在 OutputValidator 中添加参数验证
required_params = {
    "my_tool": ["arg1", "arg2"]
}

# 3. 更新系统提示
tools = ["read_file", "write_file", "my_tool"]
```

### 添加新中间件

```python
from core import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def process_before_llm(self, messages, runtime_context):
        # 在模型调用前处理消息
        return messages

framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    middlewares=[MyMiddleware()]
)
```

## 未来优化方向

1. **向量记忆**：使用 embedding 存储历史对话
2. **多 Agent 协作**：Planner + Executor + Reviewer
3. **工具学习**：根据历史自动选择工具
4. **流式输出**：支持 SSE 流式响应
5. **分布式执行**：支持远程工具调用
