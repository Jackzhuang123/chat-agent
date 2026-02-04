# 🔧 API 参考

核心类、方法和集成示例。

---

## 目录

1. [Agent Framework](#agent-framework)
2. [Tool System](#tool-system)
3. [Skills System](#skills-system)
4. [Web UI](#web-ui-集成)
5. [完整示例](#完整集成示例)

---

## Agent Framework

### QwenAgentFramework 类

**位置**：`core/agent_framework.py`

Agent 循环的主类，管理模型调用、工具执行和结果处理。

#### 构造函数

```python
QwenAgentFramework(
    model_forward_fn,           # 模型前向函数
    work_dir=None,              # 工作目录
    enable_bash=False,          # 是否启用 bash
    max_iterations=10,          # 最大循环次数
    tools_in_system_prompt=True # 包含工具信息
)
```

**参数说明**：
- `model_forward_fn`: 函数 `fn(messages, system_prompt, **kwargs) -> str`
- `work_dir`: 文件操作的工作目录，默认当前目录
- `enable_bash`: bash 工具默认禁用（安全考虑）
- `max_iterations`: 防止无限循环，默认 10 次
- `tools_in_system_prompt`: 工具说明是否加入系统提示词

#### 方法

##### process_message()

处理用户消息，执行 Agent 循环。

```python
response, execution_log = agent.process_message(
    user_message: str,
    history: List[Tuple[str, str]],
    system_prompt_override: Optional[str] = None,
    **model_kwargs
)
```

**返回**：
- `response`: 最终响应文本
- `execution_log`: 执行日志列表

**示例**：
```python
from core import QwenAgentFramework, create_qwen_model_forward

agent = QwenAgentFramework(
    model_forward_fn=create_qwen_model_forward(qwen),
    enable_bash=False
)

response, log = agent.process_message(
    "列出当前目录的文件",
    [],
    temperature=0.7
)

print(response)
for step in log:
    print(f"[{step['iteration']}] {step['type']}")
```

##### process_message_stream()

流式处理消息，实时返回结果。

```python
for text, info in agent.process_message_stream(
    user_message, history, system_prompt_override, **kwargs
):
    print(text)
```

### create_qwen_model_forward()

为 Qwen 模型创建适配器函数。

```python
forward_fn = create_qwen_model_forward(
    qwen_agent,              # QwenAgent 实例
    system_prompt_base=""    # 基础系统提示词
)

agent = QwenAgentFramework(model_forward_fn=forward_fn)
```

---

## Tool System

### ToolExecutor 类

**位置**：`core/agent_tools.py`

工具执行器，管理和执行所有可用工具。

#### 构造函数

```python
ToolExecutor(
    work_dir=None,      # 工作目录
    enable_bash=True    # 是否启用 bash
)
```

#### 方法

##### get_tools()

获取所有工具的描述。

```python
tools = executor.get_tools()
# 返回: [{name, description, input_schema}, ...]
```

##### execute_tool()

执行指定工具。

```python
result = executor.execute_tool(tool_name, tool_input)
# 返回: JSON 字符串格式的结果
```

**工具列表**：

| 工具 | 输入 | 返回 |
|------|------|------|
| read_file | `{"path": str}` | 文件内容 |
| write_file | `{"path": str, "content": str}` | 写入大小 |
| edit_file | `{"path": str, "old_content": str, "new_content": str}` | 成功标志 |
| list_dir | `{"path": str}` | 文件列表 |
| bash | `{"command": str}` | stdout, stderr, returncode |

**示例**：
```python
from core import ToolExecutor

executor = ToolExecutor(enable_bash=False)

# 读取文件
result = executor.execute_tool("read_file", {"path": "main.py"})

# 创建文件
result = executor.execute_tool("write_file", {
    "path": "test.py",
    "content": "print('hello')"
})

# 编辑文件
result = executor.execute_tool("edit_file", {
    "path": "test.py",
    "old_content": "hello",
    "new_content": "world"
})

# 列表目录
result = executor.execute_tool("list_dir", {"path": "."})
```

### ToolParser 类

**位置**：`core/agent_tools.py`

从模型输出中解析工具调用。

#### 方法

##### parse_tool_calls()

解析模型输出中的工具调用。

```python
calls = ToolParser.parse_tool_calls(text)
# 返回: [(tool_name, tool_input), ...]
```

**支持的格式**：

1. JSON：`[{"tool": "name", "input": {...}}]`
2. 标记：`<tool>name</tool><input>{...}</input>`
3. 文本：`Tool: name, Input: {...}`

**示例**：
```python
from core import ToolParser

text = '<tool>write_file</tool><input>{"path": "test.py", "content": "hi"}</input>'
calls = ToolParser.parse_tool_calls(text)
# 输出: [("write_file", {"path": "test.py", "content": "hi"})]
```

---

## Skills System

### SkillManager 类

**位置**：`core/agent_skills.py`

技能管理器，发现、加载和管理技能。

#### 构造函数

```python
SkillManager(skills_dir=None)  # 默认 ./skills
```

#### 方法

##### get_skills_list()

获取所有技能的元数据。

```python
skills = manager.get_skills_list()
# 返回: [{"id", "name", "description", "tags", "preview"}, ...]
```

##### get_skill_detail()

获取技能的完整内容。

```python
content = manager.get_skill_detail(skill_id)
# 返回: SKILL.md 的完整内容或 None
```

##### get_skill_resources()

获取技能的资源文件。

```python
resources = manager.get_skill_resources(skill_id)
# 返回: {filename: content}
```

##### find_skills_for_task()

根据任务描述查找相关技能。

```python
matched = manager.find_skills_for_task(task_description)
# 返回: [{"id", "name", "score", "description"}, ...]
```

**示例**：
```python
from core import SkillManager

manager = SkillManager()

# 获取所有技能
all_skills = manager.get_skills_list()
print(f"发现 {len(all_skills)} 个技能")

# 查找相关技能
task = "审查一个 Python 代码"
matched = manager.find_skills_for_task(task)
for skill in matched:
    print(f"- {skill['name']} (匹配度: {skill['score']})")
    content = manager.get_skill_detail(skill['id'])
```

### SkillInjector 类

**位置**：`core/agent_skills.py`

技能注入器，将技能知识注入到上下文。

#### 构造函数

```python
SkillInjector(skill_manager)
```

#### 方法

##### inject_skills_to_context()

将技能注入到消息上下文。

```python
updated_messages = injector.inject_skills_to_context(
    messages: List[Dict],      # 消息列表
    relevant_skills: List[str], # 技能 ID 列表
    include_full_content=False  # 是否包含完整内容
)
```

**关键特性**：
- 技能作为工具结果注入（不修改系统提示词）
- 保留系统提示词缓存
- 成本节省 70-80%

**示例**：
```python
from core import SkillManager, SkillInjector

manager = SkillManager()
injector = SkillInjector(manager)

messages = [{"role": "user", "content": "审查代码"}]

# 注入元数据 (快速)
messages = injector.inject_skills_to_context(
    messages,
    ["code-review"],
    include_full_content=False
)

# 或注入完整内容 (详细)
messages = injector.inject_skills_to_context(
    messages,
    ["code-review"],
    include_full_content=True
)
```

### create_example_skills()

创建示例技能文件。

```python
from core import create_example_skills

create_example_skills()
# 创建: pdf/, code-review/, python-dev/ 技能
```

---

## Web UI 集成

### web_agent_with_skills.py

完整版 Web UI，集成所有功能。

**启动**：
```bash
python3 ui/web_agent_with_skills.py
```

**功能**：
- 工具调用
- Skills 系统
- 参数调整
- 执行日志

---

## 完整集成示例

### 基础示例

```python
from core import QwenAgentFramework, create_qwen_model_forward
from ui.web_agent_with_skills import QwenAgent

# 初始化
qwen = QwenAgent()
agent = QwenAgentFramework(
    model_forward_fn=create_qwen_model_forward(qwen),
    enable_bash=False
)

# 处理消息
response, log = agent.process_message(
    "列出当前目录",
    [],
    temperature=0.7
)

print(response)
```

### 完整示例（含 Skills）

```python
from core import (
    QwenAgentFramework,
    create_qwen_model_forward,
    SkillManager,
    SkillInjector,
    create_example_skills
)
from ui.web_agent_with_skills import QwenAgent

# 初始化
qwen = QwenAgent()
create_example_skills()

# Skills 系统
manager = SkillManager()
injector = SkillInjector(manager)

# Agent 框架
agent = QwenAgentFramework(
    model_forward_fn=create_qwen_model_forward(qwen),
    enable_bash=False,
    max_iterations=5
)

# 处理任务
user_task = "审查一个 Python 代码"

# 自动匹配技能
matched_skills = manager.find_skills_for_task(user_task)
skill_ids = [s["id"] for s in matched_skills[:3]]

# 构建消息
messages = [{"role": "user", "content": user_task}]

# 注入技能
messages = injector.inject_skills_to_context(
    messages,
    skill_ids,
    include_full_content=False
)

# 处理
response, log = agent.process_message(
    user_task,
    [],
    temperature=0.7
)

print("响应:", response)
print("执行步骤:", len(log))
```

---

## 常见用法模式

### 模式 1: 仅工具

```python
from core import ToolExecutor

executor = ToolExecutor()
result = executor.execute_tool("read_file", {"path": "main.py"})
```

### 模式 2: 工具 + Agent

```python
from core import QwenAgentFramework

agent = QwenAgentFramework(model_forward_fn)
response, log = agent.process_message("创建文件", [])
```

### 模式 3: 仅 Skills

```python
from core import SkillManager

manager = SkillManager()
matched = manager.find_skills_for_task("审查代码")
content = manager.get_skill_detail(matched[0]["id"])
```

### 模式 4: 完整系统

```python
# 见"完整集成示例"
```

---

## 错误处理

```python
# 工具不存在
try:
    result = executor.execute_tool("invalid", {})
except Exception as e:
    print(f"工具错误: {e}")

# 路径超出范围
result = executor.execute_tool("read_file",
    {"path": "../../../etc/passwd"}
)
# 结果: {"error": "路径超出工作目录范围"}

# 技能不存在
content = manager.get_skill_detail("invalid")
# 结果: None
```

---

## 性能优化

### 缓存

Skills 自动缓存已加载的技能：

```python
# 首次: 从磁盘读取
content1 = manager.get_skill_detail("pdf")

# 第二次: 从缓存返回
content2 = manager.get_skill_detail("pdf")
```

### 批量操作

```python
# ✅ 好: 一次匹配
matched = manager.find_skills_for_task(task)

# ❌ 不好: 循环多次
for keyword in keywords:
    matched = manager.find_skills_for_task(keyword)
```

### 调试

```python
response, log = agent.process_message(...)

for step in log:
    print(f"[{step['iteration']}] {step['type']}")
    if step['type'] == 'tool_call':
        print(f"  工具: {step['tool']}")
        print(f"  结果: {step['result']}")
```

---

**更多帮助**：查看 `GUIDE.md`

