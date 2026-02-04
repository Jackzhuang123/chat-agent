# 📖 使用指南

完整的使用教程和常见问题解答。

## 目录

1. [快速开始](#快速开始)
2. [三种工作模式](#三种工作模式)
3. [工具系统](#工具系统)
4. [Skills 系统](#skills-系统)
5. [常见问题](#常见问题)
6. [高级用法](#高级用法)

---

## 快速开始

### 安装

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
pip install -r requirements.txt
```

### 启动

```bash
# 推荐：完整版
python3 ui/web_agent_with_skills.py

# 其他版本
python3 ui/web_agent_with_tools.py    # 工具版
python3 ui/web_agent_advanced.py      # 基础版
```

### 访问

浏览器自动打开：`http://127.0.0.1:7860`

---

## 三种工作模式

### 模式 1: 普通模式

**用途**：简单对话和文本生成

- 无需启用工具
- 直接输入问题
- 快速响应 (1-3 秒)

**示例**：
```
用户: "写一首关于春天的诗"
模型: [直接生成诗歌]
```

### 模式 2: 工具模式

**用途**：文件操作和自动化任务

**启用步骤**：
1. 勾选"启用工具模式"
2. 输入任务
3. 启用"显示执行日志"查看过程

**可用工具**：
- `read_file` - 读取文件
- `write_file` - 创建/覆盖文件
- `edit_file` - 编辑文件
- `list_dir` - 列表目录
- `bash` - 执行命令 (默认禁用)

**示例**：
```
用户: "创建 test.py, 内容是 print('hello')"
系统:
  1. 模型调用 write_file 工具
  2. 工具执行创建文件
  3. 返回成功结果
  4. 模型确认完成
```

### 模式 3: Skills 模式

**用途**：注入领域知识完成专业任务

**启用步骤**：
1. 勾选"启用 Skills"
2. 勾选"自动匹配技能"
3. 输入任务

**工作流**：
```
任务 → 提取关键词 → 匹配技能 → 注入知识 → 模型处理 → 响应
```

**示例**：
```
用户: "审查一个 Python 代码"
系统:
  1. 自动匹配 "code-review" 技能
  2. 注入审查检查清单
  3. 模型按照清单审查
  4. 提供专业意见
```

---

## 工具系统

### 工具概览

| 工具 | 功能 | 示例 |
|------|------|------|
| read_file | 读取文件 | 查看代码 |
| write_file | 创建文件 | 生成脚本 |
| edit_file | 编辑文件 | 修改代码 |
| list_dir | 列表目录 | 探索结构 |
| bash | 执行命令 | 运行脚本 |

### 使用示例

**例 1: 创建文件**

```
Web UI 操作:
  1. 勾选"启用工具模式"
  2. 输入: "创建 test.txt, 内容是 hello world"
  3. 等待完成

结果: test.txt 被创建
```

**例 2: 编辑文件**

```
用户输入: "读取 main.py, 将所有 print 改成 return, 保存为 new.py"

工作流:
  1. 模型调用 read_file 读取 main.py
  2. 模型修改内容
  3. 模型调用 write_file 创建 new.py
  4. 完成
```

**例 3: 目录探索**

```
用户输入: "列出当前目录的 Python 文件"

工作流:
  1. 模型调用 list_dir 获取文件列表
  2. 过滤 .py 文件
  3. 返回结果
```

### 工具调用格式

系统支持多种格式：

**格式 1: JSON (推荐)**
```json
[
  {
    "tool": "write_file",
    "input": {
      "path": "test.py",
      "content": "print('hello')"
    }
  }
]
```

**格式 2: 标记格式**
```
<tool>write_file</tool>
<input>{"path": "test.py", "content": "print('hello')"}</input>
```

---

## Skills 系统

### 什么是 Skills

**知识外置化** - 将专业知识存储在可编辑文件中

**成本对比**：
- 重新训练模型: $10K-$1M+, 数周 ❌
- 编写 SKILL.md: $0, 数分钟 ✅

### Skills 三层架构

```
第 1 层 (始终加载): 元数据 (~100 tokens)
  └─ 名称、描述、标签 → 快速匹配

第 2 层 (按需加载): 详细内容 (~2000 tokens)
  └─ SKILL.md 完整内容 → 模型详细处理

第 3 层 (需要时): 资源文件 (无限)
  └─ scripts/, references/ → 代码示例等
```

### 创建技能 (3 步)

**第 1 步: 创建目录**

```bash
mkdir skills/my-skill
```

**第 2 步: 编写 SKILL.md**

```markdown
---
name: 技能名称
description: 简短描述
tags: [tag1, tag2, tag3]
---

# 技能标题

## 概述

说明此技能的用途和背景。

## 最佳实践

### 原则 1
描述...

### 原则 2
描述...

## 示例

```python
# 代码示例
example_code = "示例"
```

## 常见问题

### Q: 问题 1?
A: 答案 1

## 参考资源

- 资源 1
- 资源 2
```

**第 3 步: 重启应用**

```bash
python3 ui/web_agent_with_skills.py
# 技能自动加载!
```

### 预定义技能

**pdf/** - PDF 处理
- 工具选择 (pdftotext vs PyMuPDF)
- 最佳实践
- 常见问题

**code-review/** - 代码审查
- 检查清单
- 质量指标
- 审查模板

**python-dev/** - Python 开发
- 环境管理
- 设计模式
- 性能优化

### 技能标签设计

**好的标签**：
```
技能: ["python", "development", "best-practices"]
或: ["code-review", "quality", "checkpoints"]
```

**规则**：
- 主领域 + 子领域 + 特定技术
- 3-5 个标签
- 英文小写

### 智能匹配原理

```
用户任务: "我需要审查 Python 代码"

步骤:
  1. 提取关键词: ["审查", "python", "代码"]

  2. 扫描所有技能并计分:
     code-review: score += 3 (匹配"审查")
     python-dev: score += 3 (匹配"python")

  3. 选择前 3 个技能 (默认)

  4. 注入到上下文并处理
```

---

## 常见问题

### Q: 工具模式什么时候用?

A: 当你需要：
- 创建或修改文件
- 探索项目结构
- 执行代码分析
- 自动化文件操作

### Q: Skills 会很慢吗?

A: 首次加载可能 10-20 秒。由于缓存保留，成本只比不用 Skills 多 10-20%。

### Q: 如何禁用某个工具?

A: 编辑启动代码：
```python
tool_executor = ToolExecutor(enable_bash=False)
```

### Q: 能否使用远程 Skills?

A: 当前只支持本地文件。可自己扩展代码添加 HTTP 支持。

### Q: 怎样创建更复杂的 Skill?

A: 分三步：
1. 写好 SKILL.md (核心知识)
2. 创建 scripts/ 目录 (代码示例)
3. 创建 references/ 目录 (参考文档)

### Q: 如何调试工具调用?

A: 启用"显示执行日志"查看：
- 模型生成的工具调用
- 工具执行的结果
- 错误信息

### Q: 性能如何优化?

A: 几个技巧：
- 用工具模式而非 Skills (若不需知识)
- 减少 Max Tokens 限制
- 关闭"显示执行日志"
- 使用更明确的提示词

---

## 高级用法

### 1. 自定义系统提示词

在 Web UI 中修改"系统提示词"框：

```
你是一个资深 Python 开发者。
当需要文件操作时,使用提供的工具。
遵循 PEP 8 风格规范。
添加详细的代码注释。
```

### 2. 调整模型参数

```
Temperature (温度):
  • 0.3 → 保守、准确 (编程)
  • 0.7 → 平衡 (对话)
  • 1.2 → 创意、随机 (写作)

Top P (核采样):
  • 0.8 → 保守
  • 0.9 → 推荐
  • 0.95 → 多样化

Max Tokens (长度):
  • 256 → 简短
  • 512 → 中等 (默认)
  • 1024+ → 长篇
```

### 3. 在代码中使用

```python
from core import QwenAgentFramework, SkillManager

# 初始化
skill_manager = SkillManager()
agent = QwenAgentFramework(...)

# 查找技能
matched = skill_manager.find_skills_for_task("处理 PDF")

# 处理消息
response, log = agent.process_message(
    "你的任务",
    [],
    system_prompt_override="自定义提示词"
)
```

### 4. 添加自定义工具

编辑 `core/agent_tools.py` 的 `ToolExecutor` 类：

```python
def execute_tool(self, tool_name, tool_input):
    if tool_name == "custom_tool":
        return self._custom_tool(tool_input)

def _custom_tool(self, params):
    # 实现自定义逻辑
    pass
```

### 5. 创建自定义 Skill

```
skills/my-domain/
├── SKILL.md          # 必需
├── scripts/          # 可选
│   ├── script1.py
│   └── script2.sh
└── references/       # 可选
    ├── reference.md
    └── spec.pdf
```

---

## 故障排除

### 问题: 工具模式不执行工具

**检查**：
- [ ] 勾选了"启用工具模式"?
- [ ] 启用"显示执行日志"查看错误?
- [ ] 系统提示词中提到了工具?

**解决**：
在系统提示词中添加：
```
当需要操作时,使用提供的工具。
```

### 问题: Skills 自动匹配没有工作

**检查**：
- [ ] 勾选了"自动匹配技能"?
- [ ] 技能目录下有 SKILL.md?
- [ ] 技能标签是否与任务匹配?

**调试**：
启用"显示执行日志"查看匹配分数

### 问题: 模型加载失败

```bash
# 确保模型存在
ls -la model/qwen2.5-0.5b/

# 不存在则下载
python3 download_model.py
```

### 问题: 内存不足

**症状**：运行时崩溃或变慢

**解决**：
- 关闭其他应用
- 减少 Max Tokens
- 使用基础版而非完整版

---

## 总结

### 三种模式的选择

```
简单对话? → 普通模式 ⚡ 快速
文件操作? → 工具模式 🔧 功能全
专业任务? → Skills 模式 🎓 智能
```

### 基本工作流

```
1. 选择合适的模式
2. 输入清晰的任务描述
3. 查看执行日志理解过程
4. 根据结果调整参数
5. 重复直到满意
```

### 最佳实践

- ✅ 使用清晰具体的语言
- ✅ 启用执行日志调试
- ✅ 定期检查生成的文件
- ✅ 创建有用的 Skills
- ✅ 备份重要数据

---

**更多信息**：查看 `API_REFERENCE.md`

