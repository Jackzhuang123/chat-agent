# 修复实施指南

本文档提供分步骤的修复实施指南，包含所有必要的代码修改。

---

## 第一步：集成新的 System Prompts（15 分钟）

### 1.1 修改 `ui/web_agent_with_skills.py`

找到构建 system prompt 的位置，替换为新的 prompts 模块：

```python
# 在文件开头添加导入
from core.prompts import get_system_prompt, inject_few_shot_examples

# 找到构建 system_prompt 的代码（大约在 run_agent 函数中）
# 原代码类似：
# system_prompt = f"你是智能助手，使用 ReAct 模式工作..."

# 替换为：
system_prompt = get_system_prompt(
    mode=run_mode,
    work_dir=work_dir,
    skills_context=skills_text if selected_skills else "暂无激活的技能"
)

# 如果是 tools 模式，注入 Few-Shot 示例
if run_mode == "tools":
    messages = inject_few_shot_examples(messages, max_examples=2)
```

### 1.2 修改 `core/agent_framework.py`（如果使用）

如果你的框架在 agent_framework.py 中构建 system prompt：

```python
from .prompts import get_system_prompt

class QwenAgentFramework:
    def _build_system_prompt(self, context: Dict) -> str:
        """根据运行模式构建 system prompt"""
        run_mode = context.get("run_mode", "chat")

        return get_system_prompt(
            mode=run_mode,
            work_dir=str(self.work_dir),
            skills_context=context.get("skills_context", "")
        )
```

---

## 第二步：集成工具强制执行中间件（20 分钟）

### 2.1 注册中间件到 Agent 框架

在 `ui/web_agent_with_skills.py` 或 `core/agent_framework.py` 中：

```python
# 在文件开头添加导入
from core.tool_enforcement_middleware import ToolEnforcementMiddleware

# 在初始化 Agent 时添加中间件
# 原代码类似：
# agent = QwenAgentFramework(
#     llm=llm,
#     work_dir=work_dir,
#     middlewares=[...]
# )

# 修改为：
agent = QwenAgentFramework(
    llm=llm,
    work_dir=work_dir,
    middlewares=[
        RuntimeModeMiddleware(),
        PlanModeMiddleware(),
        SkillsContextMiddleware(),
        UploadedFilesMiddleware(),
        ToolEnforcementMiddleware(max_retries=2),  # 新增
        ToolResultGuardMiddleware()
    ]
)
```

### 2.2 处理重试逻辑

在 Agent 主循环中添加重试处理：

```python
def run(self, user_input: str, runtime_context: Dict) -> str:
    """主运行循环"""
    max_iterations = 10

    for iteration in range(max_iterations):
        # ... 原有逻辑 ...

        # 执行中间件链
        response = self._call_llm_with_middlewares(messages, runtime_context)

        # 检查是否需要重试
        if runtime_context.get("_needs_retry"):
            # 将模型输出添加到历史，继续下一轮
            messages.append({"role": "assistant", "content": response})
            continue

        # 检查是否强制执行失败
        if runtime_context.get("_tool_enforcement_failed"):
            return response + "\n\n💡 建议：切换到 chat 模式重新提问"

        # ... 继续原有逻辑 ...
```

---

## 第三步：改进意图识别（15 分钟）

### 3.1 修改 `core/mode_router.py`

在 `_rule_based_detect` 方法开头添加直接命令检测：

```python
from .tool_enforcement_middleware import DirectCommandDetector

def _rule_based_detect(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """纯规则路由（原有逻辑）"""
    user_input_lower = user_input.lower()

    # ===== 新增：直接命令检测 =====
    direct_cmd = DirectCommandDetector.detect(user_input)
    if direct_cmd["is_direct_command"]:
        return {
            "recommended_mode": "tools",
            "confidence": direct_cmd["confidence"],
            "alternatives": [],
            "complexity": "simple",
            "reasoning": direct_cmd["reason"],
            "suggested_tool": direct_cmd["tool_name"]  # 可选：提示具体工具
        }
    # ===== 新增结束 =====

    # 0. 路径强信号先行判断（保留原逻辑）
    if self._PATH_PATTERN.search(user_input):
        ...
```

### 3.2 调整 LLM 路由阈值

在 `ModeRouter.__init__` 中：

```python
def __init__(self, llm_forward_fn: Optional[Callable] = None, llm_confidence_threshold: float = 0.70):
    # 从 0.55 提升到 0.70，减少 LLM 路由触发频率
    self.llm_forward_fn = llm_forward_fn
    self.llm_confidence_threshold = llm_confidence_threshold
```

---

## 第四步：增强 ToolParser 鲁棒性（10 分钟）

### 4.1 修改 `core/agent_tools.py`

在 `ToolParser.parse_tool_calls` 方法的末尾（返回空列表之前）添加：

```python
@staticmethod
def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """从模型输出中解析工具调用"""
    calls: List[Tuple[str, Dict[str, Any]]] = []
    stripped = text.strip()
    known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}

    # ... 原有的 7 种格式解析逻辑 ...

    # ===== 新增：格式8 - 最宽松的裸格式（兜底解析）=====
    # 适用于模型输出不规范的情况，如：
    # "read_file 我要读取 core/agent_tools.py {\"path\": \"core/agent_tools.py\"}"
    for tool in known_tools:
        if tool in stripped:
            # 查找工具名后的第一个 JSON 对象
            tool_idx = stripped.find(tool)
            json_start = stripped.find('{', tool_idx)
            if json_start != -1:
                # 从 JSON 起始位置向后尝试解析
                for json_end in range(len(stripped), json_start, -1):
                    try:
                        candidate = stripped[json_start:json_end]
                        args = ToolParser._parse_input_payload(candidate)
                        if args and isinstance(args, dict):
                            # 成功解析，返回工具调用
                            return [(tool, args)]
                    except Exception:
                        continue
    # ===== 新增结束 =====

    return calls
```

---

## 第五步：添加诊断日志（10 分钟）

### 5.1 在 `ui/web_agent_with_skills.py` 中添加模式切换日志

```python
def run_agent(user_input, history, run_mode, plan_mode, ...):
    """运行 Agent"""

    # 记录模式切换
    if 'last_mode' not in globals():
        globals()['last_mode'] = None

    if globals()['last_mode'] != run_mode:
        print(f"🔄 模式切换: {globals()['last_mode']} → {run_mode}")
        globals()['last_mode'] = run_mode

    # ... 原有逻辑 ...
```

### 5.2 在 `core/agent_tools.py` 中添加解析失败日志

```python
@staticmethod
def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """从模型输出中解析工具调用"""
    calls: List[Tuple[str, Dict[str, Any]]] = []
    stripped = text.strip()

    # ... 所有解析逻辑 ...

    # 如果所有格式都解析失败，记录日志
    if not calls:
        print(f"⚠️ 工具解析失败")
        print(f"模型输出（前 200 字符）: {stripped[:200]}")
        print(f"可能原因：模型未按工具格式输出")

    return calls
```

---

## 第六步：验证测试（30 分钟）

### 6.1 创建测试脚本

创建 `test_fixes.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试修复效果"""

from core.mode_router import ModeRouter
from core.tool_enforcement_middleware import DirectCommandDetector
from core.agent_tools import ToolParser
from core.prompts import get_system_prompt

def test_direct_command_detection():
    """测试直接命令检测"""
    print("=== 测试直接命令检测 ===")

    test_cases = [
        "读取 core/agent_tools.py",
        "调用工具进行读取",
        "列出 core 目录",
        "执行 pytest 测试",
        "你好，今天天气怎么样？",  # 非直接命令
    ]

    for case in test_cases:
        result = DirectCommandDetector.detect(case)
        print(f"输入: {case}")
        print(f"结果: {result}")
        print()

def test_tool_parsing():
    """测试工具解析"""
    print("=== 测试工具解析 ===")

    test_cases = [
        'read_file\n{"path": "core/agent_tools.py"}',  # 标准格式
        '{"tool": "read_file", "input": {"path": "test.py"}}',  # JSON 格式
        'read_file {"path": "test.py"}',  # 裸格式
        '```json\n{"tool": "list_dir", "input": {"path": "core"}}\n```',  # Markdown
    ]

    for case in test_cases:
        result = ToolParser.parse_tool_calls(case)
        print(f"输入: {case[:50]}...")
        print(f"解析结果: {result}")
        print()

def test_system_prompts():
    """测试 system prompt 生成"""
    print("=== 测试 System Prompt ===")

    for mode in ["chat", "tools", "plan", "hybrid"]:
        prompt = get_system_prompt(mode, work_dir="/test/dir")
        print(f"模式: {mode}")
        print(f"Prompt 长度: {len(prompt)} 字符")
        print(f"前 100 字符: {prompt[:100]}...")
        print()

if __name__ == "__main__":
    test_direct_command_detection()
    test_tool_parsing()
    test_system_prompts()
```

运行测试：

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
python test_fixes.py
```

### 6.2 回归测试会话日志中的问题

在 Gradio UI 中测试以下场景：

1. **场景 1：直接命令**
   - 输入："读取 core/agent_tools.py"
   - 预期：直接路由到 tools 模式，模型输出工具调用格式

2. **场景 2：调用工具**
   - 输入："调用工具进行读取"
   - 预期：识别为直接命令，强制工具调用

3. **场景 3：列出目录**
   - 输入："列出 core 目录下的文件"
   - 预期：正确调用 list_dir 工具

4. **场景 4：普通对话**
   - 输入："你好，今天天气怎么样？"
   - 预期：路由到 chat 模式，自由对话

---

## 第七步：监控和调试（持续）

### 7.1 查看会话日志

修复后，重新运行问题场景，检查新的会话日志：

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
ls -lt session_logs/ | head -5
cat session_logs/最新日志.json | python3 -m json.tool | less
```

### 7.2 关键指标

监控以下指标：

- **意图识别准确率**：`runtime_context.intent_reasons` 中的路由类型
- **工具调用成功率**：`execution_log` 中 `type: "tool_call"` 的 `success` 字段
- **重试次数**：`_tool_enforcement_retry` 计数

---

## 常见问题排查

### Q1: 模型仍然不调用工具

**排查步骤**：
1. 检查 `get_system_prompt("tools")` 是否正确注入
2. 检查 `ToolEnforcementMiddleware` 是否在中间件链中
3. 检查日志中的 `_tool_enforcement_retry` 计数

**解决方案**：
- 增加 Few-Shot 示例数量（`max_examples=3`）
- 降低模型 temperature（0.3 → 0.1）
- 手动在 prompt 中添加更强的约束

### Q2: 工具解析仍然失败

**排查步骤**：
1. 查看 `⚠️ 工具解析失败` 日志中的模型输出
2. 检查模型输出是否包含工具名
3. 检查 JSON 格式是否合法

**解决方案**：
- 在 `parse_tool_calls` 中添加更多格式支持
- 使用正则表达式提取工具名和参数
- 考虑切换到更稳定的模型（如 GLM-4-Air）

### Q3: 意图识别置信度过低

**排查步骤**：
1. 检查 `DirectCommandDetector` 的模式列表
2. 检查 LLM 路由阈值是否过高

**解决方案**：
- 扩充直接命令模式列表
- 调整置信度计算公式
- 降低 LLM 路由阈值（0.70 → 0.65）

---

## 完成检查清单

- [ ] 已集成新的 System Prompts（`core/prompts.py`）
- [ ] 已注册 `ToolEnforcementMiddleware`
- [ ] 已修改 `mode_router.py` 添加直接命令检测
- [ ] 已增强 `ToolParser` 的兜底解析
- [ ] 已添加诊断日志
- [ ] 已运行 `test_fixes.py` 验证
- [ ] 已回归测试会话日志中的问题场景
- [ ] 工具调用成功率 > 90%
- [ ] 意图识别准确率 > 95%

---

## 后续优化方向

1. **模型微调**：收集失败案例，构建训练数据集
2. **工具学习**：激活 ToolLearner，根据历史推荐工具
3. **向量记忆**：存储成功的工具调用模式
4. **多模态支持**：支持图片、文件上传的工具调用
5. **流式输出**：实时展示工具执行进度
