# 🎯 根本原因确认

## 问题症状
日志 `20260327_115450_325.json` 显示：
- 47次模型调用
- 0次bash调用
- 触发429速率限制
- 模型使用read_file逐个读取文件

## 根本原因

### 1. **enable_bash=False** ❌
**文件**: `ui/web_agent_with_skills.py` 第457行
```python
enable_bash=False,  # ← bash工具完全未启用！
```

### 2. **bash工具顺序错误** ❌
**文件**: `core/agent_framework.py` 第286-288行
```python
tools = ["read_file", "write_file", "edit_file", "list_dir"]
if self.tool_executor.enable_bash:
    tools.append("bash")  # ← bash被追加到末尾，不是第一位
```

### 3. **旧system prompt被使用** ❌
`agent_framework.py` 的 `_build_system_prompt()` 方法生成的是旧版提示词，**完全没有使用** `core/prompts.py` 中优化的 `TOOLS_MODE_SYSTEM_PROMPT`。

## 实际发送给模型的提示词

```
可用工具：
- read_file
- write_file
- edit_file
- list_dir
```

**bash根本不在列表中！**

## 为什么之前的修复都失败了

| 修复尝试 | 位置 | 实际效果 |
|---------|------|---------|
| 优化TOOLS_MODE_SYSTEM_PROMPT | prompts.py | ❌ 未被使用 |
| bash移到第一位 | prompts.py | ❌ 未被使用 |
| 添加批量扫描策略 | prompts.py | ❌ 未被使用 |
| Few-Shot示例 | prompts.py | ❌ 未被使用 |
| ToolEnforcementMiddleware | middleware | ✅ 工作，但无法强制bash |

## 正确的修复方案

### 修复1: 启用bash工具 ✅
**文件**: `ui/web_agent_with_skills.py` 第457行
```python
enable_bash=True,  # 启用bash工具
```

### 修复2: bash优先顺序 ✅
**文件**: `core/agent_framework.py` 第286-288行
```python
# 修改前
tools = ["read_file", "write_file", "edit_file", "list_dir"]
if self.tool_executor.enable_bash:
    tools.append("bash")

# 修改后
if self.tool_executor.enable_bash:
    tools = ["bash", "read_file", "write_file", "edit_file", "list_dir"]
else:
    tools = ["read_file", "write_file", "edit_file", "list_dir"]
```

### 修复3: 使用优化的system prompt ✅
**文件**: `core/agent_framework.py`

选项A（简单）：直接替换 `_build_system_prompt()` 返回优化的提示词
选项B（彻底）：删除 `_build_system_prompt()`，在初始化时接收外部提示词

推荐选项A，保持向后兼容。

## 预期效果

修复后，相同输入应该：
```
用户: 帮我扫描整个 core 目录，找出所有类和方法，整理成文档写入 API.md

模型看到的工具列表:
- bash: 执行 shell 命令（grep, find, awk 等，批量处理时优先使用）
- read_file: 读取单个文件内容
- ...

模型响应:
bash
{"command": "grep -rn '^class \\|^def ' core/"}

[执行成功]

write_file
{"path": "API.md", "content": "...", "mode": "overwrite"}
```

---

**发现时间**: 2026-03-27 12:10
**状态**: ✅ 根本原因已确认
**影响**: 所有之前的prompt优化都因为未被使用而无效
