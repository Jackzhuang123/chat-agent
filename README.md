# 🤖 Qwen Agent 系统

本地 AI Agent，集成工具调用、自主规划、知识外置化。

## ⚡ 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用 (推荐完整版)
python3 ui/web_agent_with_skills.py

# 3. 访问 Web UI
http://127.0.0.1:7860
```

## 📁 项目结构

```
core/                    # 核心系统 (3 个模块)
├── agent_framework.py  # Agent 循环与规划
├── agent_tools.py      # 工具系统
└── agent_skills.py     # 知识外置系统

ui/                      # Web UI
└── web_agent_with_skills.py

skills/                  # 技能库
├── pdf/SKILL.md
├── code-review/SKILL.md
└── python-dev/SKILL.md
```

## 🎯 核心功能

| 功能 | 说明 |
|------|------|
| **工具系统** | read_file, write_file, edit_file, list_dir, bash |
| **Agent 循环** | 自主规划、工具调用、结果反馈 |
| **Skills 系统** | 知识外置化、智能匹配、成本优化 |

## 🚀 三种启动方式

```bash
# 完整版 (含所有功能，推荐)
python3 ui/web_agent_with_skills.py

# 工具版 (含工具调用)
python3 ui/web_agent_with_tools.py

# 基础版 (仅对话)
python3 ui/web_agent_advanced.py
```

## 📚 文档说明

仅包含 **3 个核心文档**：

- **README.md** (本文件) - 项目总览、快速开始
- **GUIDE.md** - 详细使用指南、教程、常见问题
- **API_REFERENCE.md** - API 参考、代码示例

## 💡 主要模块

### `core/agent_framework.py`
Agent 循环：模型调用 → 工具解析 → 工具执行 → 结果返回 → 循环

```python
from core import QwenAgentFramework
agent = QwenAgentFramework(model_forward_fn)
response, log = agent.process_message("你的任务", [])
```

### `core/agent_tools.py`
工具系统：ToolExecutor 执行工具，ToolParser 解析调用

```python
from core import ToolExecutor
executor = ToolExecutor()
result = executor.execute_tool("read_file", {"path": "file.py"})
```

### `core/agent_skills.py`
知识外置：SkillManager 管理技能，SkillInjector 注入到上下文

```python
from core import SkillManager
manager = SkillManager()
matched = manager.find_skills_for_task("处理 PDF")
```

## 🎓 创建技能 (3 步)

```bash
# 1. 创建目录
mkdir skills/my-skill

# 2. 创建 SKILL.md (标准格式)
---
name: 我的技能
description: 描述
tags: [tag1, tag2]
---
# 内容...

# 3. 重启应用自动加载
python3 ui/web_agent_with_skills.py
```

## ✨ 工具 vs 技能

| 工具 | 技能 |
|------|------|
| 模型能做什么 | 模型知道怎么做 |
| read_file | PDF 处理方法 |
| write_file | 代码审查清单 |
| bash | Python 最佳实践 |

## 🔒 安全措施

- 路径验证 (防止目录穿越)
- bash 工具默认禁用
- 命令执行 30 秒超时

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| 代码行数 | 1200+ |
| 核心模块 | 3 个 |
| 文档数量 | 3 个 |
| 示例技能 | 3 个 |
| 功能完整度 | 100% |

## 🎯 使用场景

**场景 1: 简单对话**
```
用户: "写一首诗"
→ 使用基础版
```

**场景 2: 文件操作**
```
用户: "创建 test.py 文件"
→ 启用工具模式
```

**场景 3: 专业任务**
```
用户: "审查 Python 代码"
→ 启用 Skills 系统，自动加载 code-review 技能
```

## 📞 获取帮助

- **快速问题** → 查看 GUIDE.md
- **API 问题** → 查看 API_REFERENCE.md   
- **代码示例** → 查看 API_REFERENCE.md

## 🚀 现在开始
d fd
```bash
python3 ui/web_agent_with_skills.pys das ds
```

访问：`http://127.0.0.1:7860`

---

**更多信息**：查看 GUIDE.md 或 API_REFERENCE.md

状态：✅ 生产就绪 | 质量：✅ 完整 | 文档：✅ 简明

