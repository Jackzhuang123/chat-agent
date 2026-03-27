# ✅ 最终解决方案

## 问题总结

分析日志 `20260327_115450_325.json` 后发现：
- 模型进行了47次调用，0次bash调用
- 最终触发429速率限制
- **根本原因**: bash工具根本未启用 (`enable_bash=False`)

## 已实施的修复

### 修复1: 启用bash工具 ✅
**文件**: `ui/web_agent_with_skills.py:457`
```python
# 修改前
enable_bash=False,

# 修改后
enable_bash=True,  # 启用bash工具用于批量扫描
```

### 修复2: bash工具优先级 ✅
**文件**: `core/agent_framework.py:286-291`
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

### 修复3: 增强system prompt ✅
**文件**: `core/agent_framework.py`

新增内容：
1. **工具选择策略**：明确指导何时使用bash
2. **bash工具格式**：添加bash调用示例
3. **禁止行为**：禁止list_dir + read_file逐个读取
4. **禁止放弃话术**：不允许说"无法使用工具"

## 修复前后对比

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| bash启用状态 | ❌ False | ✅ True |
| bash在工具列表位置 | ❌ 末尾（第5位） | ✅ 首位（第1位） |
| 工具选择策略 | ❌ 无 | ✅ 明确指导 |
| bash调用示例 | ❌ 无 | ✅ 有 |
| 禁止逐个读取 | ❌ 无 | ✅ 明确禁止 |

## 预期效果

修复后，相同任务应该：

**输入**:
```
帮我扫描整个 core 目录，找出所有类和方法，整理成文档写入 API.md
```

**模型行为**:
```
第1步: 调用 bash
{"command": "grep -rn '^class \\|^def ' core/"}

第2步: 调用 write_file
{"path": "API.md", "content": "...", "mode": "overwrite"}

完成 ✅
```

**性能提升**:
- 模型调用次数: 47次 → 2-3次
- 执行时间: 583秒 → 10秒内
- Token消耗: 389K → 5K以内
- 成功率: 0% (429错误) → 100%

## 为什么之前的修复无效

所有之前在 `core/prompts.py` 中的优化（bash优先、Few-Shot示例、策略指导）都没有被使用，因为：

1. `agent_framework.py` 有自己的 `_build_system_prompt()` 方法
2. `web_agent_with_skills.py` 没有调用 `get_system_prompt()` 函数
3. 即使调用了，`enable_bash=False` 也会导致bash不在工具列表中

## 验证步骤

1. 重启服务:
```bash
python ui/web_agent_with_skills.py
```

2. 测试相同输入:
```
帮我扫描整个 core 目录，找出所有类和方法，整理成文档写入 API.md
```

3. 检查日志:
- 应该看到 bash 工具调用
- 总调用次数应该 < 5次
- 不应该触发429错误

## 关键改进点

1. **架构层修复**: 在框架层启用bash，而不是依赖prompt工程
2. **工具顺序**: bash作为第一个工具，模型优先看到
3. **明确指导**: 在system prompt中明确批量任务必须用bash
4. **禁止模式**: 明确禁止低效的list_dir + read_file模式

---

**完成时间**: 2026-03-27 12:15
**状态**: ✅ 关键修复已完成
**下一步**: 实际测试验证效果
