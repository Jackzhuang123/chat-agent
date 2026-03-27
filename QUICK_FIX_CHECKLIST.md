# ⚡ 快速修复检查清单

> 30 分钟内完成核心修复，立即提升工具调用成功率

---

## 🎯 核心问题

- ❌ 模型在 tools 模式下不调用工具，直接输出自然语言
- ❌ 意图识别置信度低，频繁误判
- ❌ 工具调用格式解析失败

---

## ✅ 修复步骤（按顺序执行）

### Step 1: 集成强化 System Prompt（10 分钟）

**文件**: `ui/web_agent_with_skills.py`

```python
# 1. 在文件开头添加导入
from core.prompts import get_system_prompt, inject_few_shot_examples

# 2. 找到构建 system_prompt 的地方（约 200-300 行）
# 原代码类似：
# system_prompt = f"你是智能助手，使用 ReAct 模式工作..."

# 3. 替换为：
system_prompt = get_system_prompt(
    mode=run_mode,
    work_dir=work_dir,
    skills_context=skills_text if selected_skills else ""
)

# 4. 如果是 tools 模式，注入 Few-Shot 示例
if run_mode == "tools":
    messages = inject_few_shot_examples(messages, max_examples=2)
```

**验证**: 启动 UI，输入"读取 test.py"，检查模型是否输出工具格式

---

### Step 2: 增强工具解析（5 分钟）

**文件**: `core/agent_tools.py`

在 `ToolParser.parse_tool_calls` 方法末尾（返回空列表之前）添加：

```python
# ===== 新增：格式8 - 最宽松裸格式（兜底）=====
for tool in known_tools:
    if tool in stripped:
        tool_idx = stripped.find(tool)
        json_start = stripped.find('{', tool_idx)
        if json_start != -1:
            for json_end in range(len(stripped), json_start, -1):
                try:
                    candidate = stripped[json_start:json_end]
                    args = ToolParser._parse_input_payload(candidate)
                    if args and isinstance(args, dict):
                        return [(tool, args)]
                except Exception:
                    continue
# ===== 新增结束 =====

return calls
```

**验证**: 运行 `python test_fixes.py`，检查工具解析测试是否通过

---

### Step 3: 添加直接命令检测（5 分钟）

**文件**: `core/mode_router.py`

在 `_rule_based_detect` 方法开头添加：

```python
from .tool_enforcement_middleware import DirectCommandDetector

def _rule_based_detect(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    user_input_lower = user_input.lower()

    # ===== 新增：直接命令检测 =====
    direct_cmd = DirectCommandDetector.detect(user_input)
    if direct_cmd["is_direct_command"]:
        return {
            "recommended_mode": "tools",
            "confidence": direct_cmd["confidence"],
            "alternatives": [],
            "complexity": "simple",
            "reasoning": direct_cmd["reason"]
        }
    # ===== 新增结束 =====

    # 原有逻辑继续...
```

**验证**: 在 UI 中输入"调用工具进行读取"，检查是否路由到 tools 模式

---

### Step 4: 调整路由阈值（1 分钟）

**文件**: `core/mode_router.py`

```python
def __init__(self, llm_forward_fn: Optional[Callable] = None, llm_confidence_threshold: float = 0.70):
    # 从 0.55 提升到 0.70
```

---

### Step 5: 运行测试（5 分钟）

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
python test_fixes.py
```

**期望输出**:
```
✅ PASS | 输入: 读取 core/agent_tools.py
✅ PASS | 输入: 调用工具进行读取
✅ PASS | 输入: 列出 core 目录
...
📊 结果: X 通过, 0 失败
```

---

### Step 6: UI 回归测试（5 分钟）

启动 UI 并测试：

```bash
python ui/web_agent_with_skills.py
```

| 测试输入 | 预期行为 | 检查点 |
|---------|---------|--------|
| "读取 core/agent_tools.py" | 调用 read_file | 模型输出包含 `read_file\n{"path": ...}` |
| "调用工具进行读取" | 强制工具调用 | 意图识别显示 tools 模式，置信度 >0.9 |
| "列出 core 目录" | 调用 list_dir | 正确执行并返回目录列表 |

---

## 🔍 快速诊断

### 问题：模型仍然不调用工具

**检查**:
```bash
# 1. 检查 system prompt 是否正确
python -c "from core.prompts import get_system_prompt; print(get_system_prompt('tools')[:200])"

# 应该看到：【核心规则】你必须使用工具...
```

**修复**: 确认 Step 1 已正确执行

---

### 问题：工具解析失败

**检查**:
```bash
# 2. 测试工具解析
python -c "from core.agent_tools import ToolParser; print(ToolParser.parse_tool_calls('read_file\n{\"path\": \"test.py\"}'))"

# 应该输出：[('read_file', {'path': 'test.py'})]
```

**修复**: 确认 Step 2 已正确执行

---

### 问题：意图识别不准

**检查**:
```bash
# 3. 测试直接命令检测
python -c "from core.tool_enforcement_middleware import DirectCommandDetector; print(DirectCommandDetector.detect('读取 test.py'))"

# 应该输出：{'is_direct_command': True, 'confidence': 0.95, ...}
```

**修复**: 确认 Step 3 已正确执行

---

## 📊 成功指标

修复完成后，应该达到：

- ✅ 工具调用成功率 > 90%
- ✅ 直接命令识别准确率 = 100%
- ✅ 意图识别置信度 > 0.90（tools 模式）
- ✅ 工具解析支持 8 种格式

---

## 🚨 如果仍有问题

1. **查看完整诊断**: `cat DIAGNOSIS_AND_FIX_PLAN.md`
2. **查看详细实施指南**: `cat IMPLEMENTATION_GUIDE.md`
3. **查看会话日志**: `ls -lt session_logs/ | head -1`
4. **检查模型输出**: 在会话日志中查看 `model_calls` 字段

---

## 📞 进阶修复（可选）

如果基础修复后仍有问题，执行进阶修复：

### 集成工具强制执行中间件（20 分钟）

**文件**: `ui/web_agent_with_skills.py` 或 `core/agent_framework.py`

```python
from core.tool_enforcement_middleware import ToolEnforcementMiddleware

# 在初始化 Agent 时添加中间件
agent = QwenAgentFramework(
    llm=llm,
    work_dir=work_dir,
    middlewares=[
        RuntimeModeMiddleware(),
        ToolEnforcementMiddleware(max_retries=2),  # 新增
        # ... 其他中间件
    ]
)
```

详见 `IMPLEMENTATION_GUIDE.md` 第二步。

---

**最后更新**: 2026-03-27
**预计修复时间**: 30 分钟
**难度**: ⭐⭐☆☆☆（中等）
