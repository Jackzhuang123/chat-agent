# Chat-Agent 诊断与修复方案

## 问题诊断总结

基于对会话日志 (`20260327_100959_154.json`, `20260327_102006_476.json`) 和核心代码的分析，识别出以下三大核心问题：

### 1. **意图识别不准确**
**症状**：用户说"调用工具进行读取"，系统能正确识别为 tools 模式，但模型实际输出却没有调用工具

**根本原因**：
- ModeRouter 的意图识别与 Agent 框架的实际执行脱节
- 规则路由置信度阈值过低 (0.55)，导致频繁触发 LLM 路由，但 LLM 路由结果不稳定
- 缺少"工具调用强制模式"：即使识别为 tools 模式，模型仍可能选择直接回答

**证据**：
```python
# core/mode_router.py:38
llm_confidence_threshold: float = 0.55  # 阈值过低
```

### 2. **模式匹配不准确**
**症状**：系统识别意图后，未能正确路由到对应的执行模式

**根本原因**：
- 关键词匹配过于简单，容易误判
- "读取"、"调用工具"等明确指令未被正确识别为高置信度 tools 模式
- 缺少对"直接命令"的特殊处理（如"读取 xxx.py"应该 100% 路由到 tools）

**证据**：
```python
# core/mode_router.py:42-56
"tools": {
    "keywords": [
        "读取", "写入", "修改", "删除", "列出", "扫描", "查找",
        "执行", "运行", "命令", ...
    ],
    "priority": 0.8,  # 优先级不够高
}
```

### 3. **工具类执行失败（模型输出解析问题）**
**症状**：模型输出了工具调用格式，但 ToolParser 未能正确解析，导致工具不执行

**根本原因**：
- **GLM-4-Flash 模型输出格式不稳定**：
  - 会话日志显示模型直接输出自然语言："已读取并输出 Api.md 文件内容..."
  - **没有输出任何 JSON 工具调用格式**
  - ToolParser 支持 7 种格式，但模型根本没按任何一种格式输出

- **System Prompt 不够强制**：
  - 当前 prompt 允许模型"自由发挥"，没有强制要求工具调用
  - 缺少 Few-Shot 示例

**证据（会话日志）**：
```json
{
  "call_index": 1,
  "response": "read_file\n{\"path\": \"/Users/zhuangranxin/PyCharmProjects/chat-Agent/ui/session_analyzer.py\"}",
  // 第一次调用：模型输出了正确的工具格式 ✓
}
{
  "call_index": 2,
  "response": "已读取并输出 Api.md 文件内容：\n\n# API 文档...",
  // 第二次调用：模型直接输出自然语言，没有工具调用 ✗
}
```

---

## 修复方案

### 优先级 1：修复工具调用解析（核心问题）

#### 1.1 强化 System Prompt - 强制工具调用格式

**文件**：`core/agent_framework.py` 或 `ui/web_agent_with_skills.py`

**修改策略**：
```python
TOOLS_MODE_SYSTEM_PROMPT = """你是智能助手，使用 ReAct (Reasoning + Acting) 模式工作。

【关键规则】你必须使用工具来完成任务，不能直接回答。

当前工作目录（绝对路径）：{work_dir}

可用工具：
- read_file: 读取文件内容
- write_file: 写入文件
- edit_file: 编辑文件
- list_dir: 列出目录
- bash: 执行命令

【工具调用格式】（严格遵守，每次只调用一个工具）：
read_file
{{"path": "相对或绝对路径"}}

write_file
{{"path": "目标文件路径", "content": "要写入的内容", "mode": "overwrite"}}

【示例】
用户: 读取 core/agent_tools.py 的内容
助手: read_file
{{"path": "core/agent_tools.py"}}

用户: 列出 core 目录
助手: list_dir
{{"path": "core"}}

【禁止】直接输出文件内容或命令结果，必须先调用工具。
"""
```

#### 1.2 增强 ToolParser - 支持"裸格式"（最宽松）

**文件**：`core/agent_tools.py`

**当前问题**：ToolParser 已经支持 7 种格式，但模型输出的"裸格式"解析不够鲁棒

**改进方案**：
```python
# 在 parse_tool_calls 方法末尾添加最宽松的兜底解析
# ---- 格式8：最宽松的裸格式（工具名 + 任意 JSON）----
for tool in known_tools:
    if tool in stripped:
        # 查找工具名后的第一个 JSON 对象
        tool_idx = stripped.find(tool)
        json_start = stripped.find('{', tool_idx)
        if json_start != -1:
            # 尝试提取并解析 JSON
            for json_end in range(len(stripped), json_start, -1):
                try:
                    candidate = stripped[json_start:json_end]
                    args = ToolParser._parse_input_payload(candidate)
                    if args:
                        return [(tool, args)]
                except:
                    continue
```

#### 1.3 添加工具调用验证中间件

**新建文件**：`core/tool_enforcement_middleware.py`

```python
class ToolEnforcementMiddleware(AgentMiddleware):
    """工具强制执行中间件 - 确保 tools 模式下模型必须调用工具"""

    def after_llm_call(self, response: str, context: Dict) -> str:
        """检查模型输出是否包含工具调用"""
        if context.get("run_mode") != "tools":
            return response

        # 解析工具调用
        tool_calls = ToolParser.parse_tool_calls(response)

        if not tool_calls:
            # 未检测到工具调用 → 注入提示并重试
            warning = "\n⚠️ 检测到你直接回答了问题，但当前处于工具模式。请使用工具格式调用。"
            context["_tool_enforcement_retry"] = context.get("_tool_enforcement_retry", 0) + 1

            if context["_tool_enforcement_retry"] < 2:
                # 重新注入强制提示
                return response + warning + "\n\n请按以下格式重新输出：\ntool_name\n{\"param\": \"value\"}"

        return response
```

### 优先级 2：改进意图识别

#### 2.1 提升"直接命令"的识别置信度

**文件**：`core/mode_router.py`

```python
def _rule_based_detect(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """纯规则路由（原有逻辑）"""
    user_input_lower = user_input.lower()

    # 0. 直接命令强信号（新增）
    DIRECT_TOOL_COMMANDS = [
        r'读取\s*[\w./]+',  # "读取 xxx.py"
        r'写入\s*[\w./]+',
        r'列出\s*[\w./]+',
        r'执行\s*[\w./]+',
        r'调用工具',  # 明确说"调用工具"
        r'用工具',
    ]

    for pattern in DIRECT_TOOL_COMMANDS:
        if re.search(pattern, user_input):
            return {
                "recommended_mode": "tools",
                "confidence": 0.95,  # 极高置信度
                "alternatives": [],
                "complexity": "simple",
                "reasoning": "检测到直接工具调用指令"
            }

    # 1. 路径强信号（保留原逻辑）
    if self._PATH_PATTERN.search(user_input):
        ...
```

#### 2.2 调整 LLM 路由阈值

```python
def __init__(self, llm_forward_fn: Optional[Callable] = None, llm_confidence_threshold: float = 0.70):
    # 从 0.55 提升到 0.70，减少 LLM 路由触发频率
```

### 优先级 3：模式匹配优化

#### 3.1 为 tools 模式添加"强制工具调用"标记

**文件**：`core/agent_framework.py`

```python
def _build_system_prompt(self, context: Dict) -> str:
    """根据运行模式构建 system prompt"""
    run_mode = context.get("run_mode", "chat")

    if run_mode == "tools":
        # tools 模式：强制工具调用
        return TOOLS_MODE_SYSTEM_PROMPT.format(work_dir=self.work_dir)
    elif run_mode == "chat":
        # chat 模式：自由对话
        return "你是智能助手，请用简洁、准确的方式回答用户问题。"
    ...
```

#### 3.2 添加"模式切换"日志

```python
def run(self, user_input: str, runtime_context: Dict) -> str:
    """主运行循环"""
    run_mode = runtime_context.get("run_mode", "chat")

    # 记录模式切换
    if hasattr(self, '_last_mode') and self._last_mode != run_mode:
        print(f"🔄 模式切换: {self._last_mode} → {run_mode}")
    self._last_mode = run_mode

    ...
```

---

## 实施计划

### 阶段 1：紧急修复（立即执行）

1. **修改 System Prompt**（30 分钟）
   - 文件：`ui/web_agent_with_skills.py` 或 `core/agent_framework.py`
   - 添加强制工具调用格式的 prompt
   - 添加 Few-Shot 示例

2. **增强 ToolParser**（20 分钟）
   - 文件：`core/agent_tools.py`
   - 添加"格式8：最宽松裸格式"解析

3. **提升直接命令识别**（15 分钟）
   - 文件：`core/mode_router.py`
   - 添加直接命令正则匹配

### 阶段 2：架构改进（1-2 小时）

1. **实现 ToolEnforcementMiddleware**（40 分钟）
   - 新建文件：`core/tool_enforcement_middleware.py`
   - 集成到 agent_framework.py

2. **调整模式路由参数**（10 分钟）
   - 提升 LLM 路由阈值到 0.70
   - 调整 tools 模式关键词权重

3. **添加诊断日志**（20 分钟）
   - 模式切换日志
   - 工具解析失败日志
   - 意图识别置信度日志

### 阶段 3：验证测试（30 分钟）

1. **回归测试**
   - 重现会话日志中的问题场景
   - 验证工具调用是否正常执行

2. **边界测试**
   - 测试各种工具调用格式
   - 测试意图识别准确率

---

## 预期效果

修复后的系统应该能够：

1. ✅ **100% 识别直接工具调用指令**
   - "读取 xxx.py" → 直接路由到 tools 模式，置信度 0.95

2. ✅ **强制模型使用工具格式**
   - tools 模式下，模型必须输出工具调用，不能直接回答
   - 若未检测到工具调用，自动提示并重试

3. ✅ **鲁棒的工具解析**
   - 支持 8 种工具调用格式
   - 即使模型输出格式不完美，也能尽可能解析

4. ✅ **可观测性提升**
   - 清晰的模式切换日志
   - 工具解析失败时的详细错误信息

---

## 后续优化建议

1. **模型微调**：
   - 使用 GLM-4-Flash 的微调功能，训练固定的工具调用格式
   - 收集失败案例，构建训练数据集

2. **Plan 模式增强**：
   - 复杂任务自动启用 Plan 模式
   - 任务分解 → 逐步执行 → 结果汇总

3. **工具学习**：
   - 激活 ToolLearner 模块
   - 根据历史成功率推荐最佳工具

4. **向量记忆**：
   - 激活 VectorMemory 模块
   - 存储成功的工具调用模式，用于相似场景推荐
