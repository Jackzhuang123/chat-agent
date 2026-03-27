# 快速开始

## 安装

```bash
pip install -r requirements.txt
```

## 基础使用

### 1. 创建 Agent

```python
from core import QwenAgentFramework

def model_forward(messages, system_prompt):
    # 调用你的 LLM API
    # 例如：OpenAI, GLM, Qwen 等
    return llm.chat(messages, system_prompt)

framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    enable_bash=True,       # 启用 bash 工具
    enable_parallel=True,   # 启用并行执行
    enable_memory=True,     # 启用持久化记忆
)
```

### 2. 运行任务

```python
result = framework.run(
    user_input="读取 core/agent_framework.py 并分析代码结构",
    history=[]
)

print(result["response"])
print(f"工具调用: {len(result['tool_calls'])} 次")
print(f"迭代: {result['iterations']} 轮")
```

### 3. 多轮对话

```python
history = []

# 第一轮
result1 = framework.run("列出 core 目录", history)
history.append({"role": "user", "content": "列出 core 目录"})
history.append({"role": "assistant", "content": result1["response"]})

# 第二轮（带上下文）
result2 = framework.run("读取其中的 agent_framework.py", history)
```

## 配置选项

```python
framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    work_dir=".",                # 工作目录
    enable_bash=True,            # 启用 bash 工具
    max_iterations=10,           # 最大迭代次数
    middlewares=[...],           # 中间件列表
    enable_memory=True,          # 启用持久化记忆
    enable_reflection=True,      # 启用反思引擎
    enable_parallel=True,        # 启用并行执行
)
```

## 工具列表

| 工具 | 描述 | 示例 | 可并行 |
|------|------|------|--------|
| `read_file` | 读取文件内容 | `read_file\n{"path": "test.py"}` | ✅ |
| `list_dir` | 列出目录内容 | `list_dir\n{"path": "core"}` | ✅ |
| `write_file` | 写入文件 | `write_file\n{"path": "out.txt", "content": "..."}` | ❌ |
| `edit_file` | 编辑文件 | `edit_file\n{"path": "test.py", "old_content": "...", "new_content": "..."}` | ❌ |
| `bash` | 执行命令 | `bash\n{"command": "ls -la"}` | ❌ |

## 使用中间件

```python
from core import (
    QwenAgentFramework,
    SkillsContextMiddleware,
    RuntimeModeMiddleware,
)

framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    middlewares=[
        RuntimeModeMiddleware(),
        SkillsContextMiddleware(skills_dir="skills"),
    ]
)
```

## 查看工具统计

```python
if framework.memory:
    stats = framework.memory.tool_stats
    for tool_name, stat in stats.items():
        print(f"{tool_name}:")
        print(f"  成功: {stat['success']}")
        print(f"  失败: {stat['failed']}")
        print(f"  平均耗时: {stat['avg_time']:.2f}s")
```

## 导出上下文

```python
result = framework.run("分析项目结构", [])
context = result["context"]

print(f"当前任务: {context['task']}")
print(f"已完成步骤: {context['completed_steps']}")
print(f"失败尝试: {context['failed_attempts']}")
print(f"工具统计: {context['tool_stats']}")
```

## 性能优化技巧

### 1. 减少迭代次数（简单任务）
```python
framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    max_iterations=5  # 默认10
)
```

### 2. 禁用并行执行（调试时）
```python
framework = QwenAgentFramework(
    model_forward_fn=model_forward,
    enable_parallel=False
)
```

### 3. 清理记忆（重新开始）
```bash
rm -rf .agent_memory/
```

## 测试

```bash
python test_simple.py
```

输出示例：
```
=== 测试：记忆和上下文管理 ===

第1轮:
  迭代: 2
  工具调用: 1
  上下文导出: {'task': None, 'completed_steps': ["list_dir(['path'])"], ...}

第2轮:
  迭代: 1
  响应: 我看到了任务进度，继续执行下一步

✅ 测试通过
```

## 常见问题

### Q: 工具执行失败怎么办？
A: 框架会自动反思并提供建议，例如：
```
❌ read_file: Error: File not found
   💡 建议: 使用 list_dir 确认路径
```

### Q: 如何避免循环？
A: 框架自动检测3次相同失败并中断：
```
⚠️ 检测到循环。
💡 建议：
1. 换用其他工具
2. 重新分析问题
3. 调整参数
```

### Q: 上下文太长怎么办？
A: 框架自动压缩，保留最近6条 + 历史top-3重要消息。

## 下一步

- 查看 [README.md](README.md) 了解架构设计
- 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解实现细节
- 查看 `core/agent_framework.py` 源码
