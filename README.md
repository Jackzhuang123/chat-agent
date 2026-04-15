# QwenAgentFramework 技术文档

## 智能体系统的架构、算法与数据流全解析

---

## 1. 系统总览：智能体的"数字城市"

### 1.1 生动比喻

想象这是一个**高度自治的数字城市**：

- **ModeRouter** = 城市快递分拣中心（决定包裹去居民区还是工业区）
- **QwenAgentFramework** = 城市执行中心（ReAct 循环 = 指挥大厅）
- **VectorMemory** = 魔法图书馆（按语义而非字母排序书籍）
- **ToolLearner** = 老工匠的笔记本（记录最佳工作流程）
- **DeepReflection** = 质量控制局（故障诊断与策略调整）
- **Skills** = 专业学院（PDF处理学院、代码审查学院等）

### 1.2 架构拓扑图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (Gradio)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Model Engine│  │  Mode Router │  │   Skill Manager      │  │
│  │ (Qwen/GLM)   │  │(Intent Class)│  │  (Knowledge Base)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼──────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QwenAgentFramework (Core)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Middleware Chain (Pre-process)             │  │
│  │  RuntimeMode → PlanMode → Skills → UploadedFiles → ...  │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ReAct Loop Engine                        │  │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────────┐        │  │
│  │   │  Thought │ → │  Action  │ → │ Observation  │        │  │
│  │   │Generator │   │ (Tool LLM│   │  (Tool Exec) │        │  │
│  │   └──────────┘   └────┬─────┘   └──────┬───────┘        │  │
│  │                        │                 │               │  │
│  │   ┌────────────────────┴─────────────────┴──────────┐   │  │
│  │   │              DeepReflectionEngine                │   │  │
│  │   │  ┌──────────────┐    ┌──────────────────────┐   │   │  │
│  │   │  │ Failure      │    │ Success Pattern      │   │   │  │
│  │   │  │ Analysis     │    │ Recording            │   │   │  │
│  │   │  └──────────────┘    └──────────────────────┘   │   │  │
│  │   └─────────────────────────────────────────────────┘   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Tool Execution Layer                         │  │
│  │   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │  │
│  │   │Parallel │  │ Sequential│  │  Bash    │  │ File    │  │  │
│  │   │Executor │  │ Executor  │  │ Executor │  │ Ops     │  │  │
│  │   └─────────┘  └──────────┘  └──────────┘  └─────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ VectorMemory    │ │AdaptiveTool  │ │ MultiAgent       │
│ (Semantic Store)│ │Learner       │ │ Orchestrator     │
│                 │ │(Markov Chain)│ │ (Planner-Executor│
│ • Embedding     │ │              │ │  -Reviewer)      │
│ • Hierarchical  │ │ • Transition │ │                  │
│ • Time Decay    │ │   Matrix     │ │ • Plan Mode      │
└─────────────────┘ └──────────────┘ └──────────────────┘
```

---

## 2. 核心组件深度解析

### 2.1 主执行流程（`agent_framework.py`）

**入口函数 `run()` → 生成器 `_run_iter()` → 工具执行 → 反思 → 输出**

```python
# 伪代码表示核心流程
def run(user_input):
    # 1. 初始化上下文
    chain_id = generate_uuid()
    messages = build_messages(user_input)
    
    # 2. ReAct 迭代循环（最多50轮）
    for iteration in range(max_iterations):
        # 2.1 上下文压缩与注入
        messages = compress_context(messages)  # 6000字符限制
        messages = inject_task_context(messages)  # 进度追踪
        messages = inject_reflection(messages)    # 反思提示
        
        # 2.2 LLM 生成 Thought + Action
        response = llm_forward(messages, system_prompt)
        
        # 2.3 解析工具调用
        tool_calls = ToolParser.parse(response)
        
        if not tool_calls:
            # 检查是否任务完成
            if looks_finished(response):
                return final_response
            else:
                # 格式错误，注入纠正提示
                messages.append(format_correction)
                continue
        
        # 2.4 并行/串行工具执行
        parallel, sequential = detect_parallel_tools(tool_calls)
        results = execute_parallel(parallel) + execute_sequential(sequential)
        
        # 2.5 反思引擎处理
        for result in results:
            reflection = reflection_engine.reflect(tool_name, result)
            if reflection.level == "strategic":
                # 连续失败3次则中断
                if should_stop(reflection_history):
                    return "任务中断"
        
        # 2.6 更新记忆
        vector_memory.add(content=result, tool_chain_id=chain_id)
        tool_learner.record_usage(task_type, tool_name, success)
        
        # 2.7 构造 Observation 反馈给 LLM
        observation = format_results(results)
        messages.append({"role": "user", "content": observation})
```

### 2.2 数据处理流向图

```
用户输入
    │
    ▼
[ModeRouter] ──► 意图识别 (置信度校准)
    │
    ├──► Chat 模式 ──► 直接生成回复
    │
    ├──► Tools 模式 ──► 进入 ReAct 循环
    │       │
    │       ▼
    │   [Middleware Chain]
    │       │
    │       ├──► RuntimeModeMiddleware (注入模式提示)
    │       ├──► SkillsContextMiddleware (注入技能知识)
    │       └──► UploadedFilesMiddleware (处理PDF/文件)
    │       │
    │       ▼
    │   [QwenAgentFramework._run_iter]
    │       │
    │       ├──► 上下文压缩 (_compress_context_smart)
    │       │       └──► 余弦相似度计算 (向量检索)
    │       │
    │       ├──► LLM 前向传播 (generate_stream)
    │       │       └──► Transformer 解码 (自注意力计算)
    │       │
    │       ├──► 工具解析 (ToolParser.parse_tool_calls)
    │       │       └──► 正则匹配 + JSON 解析 + 容错修复
    │       │
    │       ├──► 并行检测 (_detect_parallel_tools)
    │       │       └──► 读写分离算法 (读操作可并行)
    │       │
    │       ├──► 工具执行 (_execute_single_tool)
    │       │       ├──► 前置中间件处理
    │       │       ├──► 沙箱执行 (bash/read_file)
    │       │       └──► 后置中间件 (ToolResultGuard)
    │       │
    │       ├──► 反思引擎 (DeepReflectionEngine)
    │       │       ├──► 错误模式匹配 (正则表达式)
    │       │       ├──► 转移矩阵更新 (马尔可夫链)
    │       │       └──► 成功率计算 (贝叶斯统计)
    │       │
    │       └──► 记忆更新 (VectorMemory)
    │               ├──► Embedding 计算 (SentenceTransformer)
    │               ├──► 余弦相似度索引
    │               └──► 时间衰减加权 (指数函数)
    │
    └──► Plan 模式 ──► [MultiAgentOrchestrator]
            │
            ├──► PlannerAgent (任务分解)
            │       └──► LLM 生成 JSON 计划
            │
            ├──► ExecutorAgent (逐步执行)
            │       └──► 每步调用 ReAct 框架
            │
            └──► ReviewerAgent (质量检查)
                    └──► 成功率评估 + 建议生成
```

### 2.2 ModeRouter：智能分拣员的决策算法

#### 概念与比喻

**ModeRouter** 是系统的"快递分拣中心"。当用户请求（包裹）到达时，它必须在 **10ms** 内决定送往哪个通道：

- **Chat 通道**：普通信件（闲聊、知识问答）
- **Tools 通道**：需要开箱检查的包裹（文件操作、命令执行）
- **Skills 通道**：专业设备处理（PDF处理、代码审查）
- **Plan 模式**：大件货物拆分（复杂任务分解）

#### 数学模型：贝叶斯意图识别

**输入示例**：`"读取 /tmp/data.txt 并分析其中的错误日志"`

**步骤 1：特征提取（TF-IDF 向量化）**

```python
# 词频向量构建（简化版）
词频向量 = {
    "读取": 1,      # 文件操作动词，权重 +3
    "/tmp/data.txt": 1,  # 绝对路径，权重 +2
    "分析": 1,      # 处理动词，权重 +1
    "错误日志": 1   # 领域词汇，权重 +1
}
```

**步骤 2：贝叶斯后验计算**

$$
P(\text{Intent}|\text{Evidence}) = \frac{P(\text{Intent}) \cdot \prod_{i} P(\text{feature}_i|\text{Intent})}{P(\text{Evidence})}
$$

具体计算：

```python
# 先验概率（历史数据统计）
P(Tools) = 0.4;  P(Chat) = 0.5;  P(Skills) = 0.1

# 似然计算
P(路径|Tools) = 0.95
P(读取|Tools) = 0.90
P(分析|Tools) = 0.60

# 联合概率
P(Tools|证据) ∝ 0.4 × 0.95 × 0.90 × 0.60 = 0.2052
P(Chat|证据) ∝ 0.5 × 0.05 × 0.10 × 0.30 = 0.00075
P(Skills|证据) ∝ 0.1 × 0.20 × 0.20 × 0.40 = 0.0016

# 归一化（Softmax）
总分 = 0.2052 + 0.00075 + 0.0016 = 0.20755
置信度(Tools) = 0.2052 / 0.20755 ≈ 0.988  # 98.8% 置信度
```

**步骤 3：温度校准（Temperature Calibration）**

<br/>

当系统运行一段时间后，发现路由器"过于自信"（预测90%但实际只有70%准确），启动校准：

$$
\text{Calibrated} = \frac{\text{Raw}}{T} + \text{bias}
$$

```python
# 实际运行数据
预测置信度 = [0.9, 0.8, 0.95, 0.7, 0.85]
实际准确率 = [0.7, 0.75, 0.60, 0.80, 0.70]  # 系统过自信

# 计算期望校准误差 (ECE)
ECE = Σ|预测_i - 实际_i| / n = 0.17

# 自适应调整温度
if ECE > 0.2:
    T *= 1.1  # 增加温度，平滑分布（降低自信）
elif ECE < 0.1:
    T *= 0.95  # 降低温度，锐化分布
```

**输出决策**：

```json
{
    "intent": "tools",
    "confidence": 0.988,
    "router_type": "rule",
    "reasoning": "匹配路径模式(/tmp/data.txt) + 操作动词(读取)",
    "suggested_params": {
        "temperature": 0.3,  # 低温度，确定性输出
        "max_tokens": 2048
    }
}
```

---

### 2.2 VectorMemory：魔法图书馆的索引系统

#### 概念与比喻

**VectorMemory** 是"魔法图书馆"。传统图书馆按字母排序（A-Z），而这里按**语义相似度**排列。当你想找"如何烤蛋糕"时，它不仅能找到《蛋糕烘焙指南》，还 能找到《甜点制作》——因为它们在 384 维空间中的"距离"只有 32 度夹角。

#### 数学模型：语义嵌入与混合检索

**输入文本**：`"使用 bash 命令读取文件内容"`

**步骤 1：嵌入计算（简化哈希版）**

```python
dimension = 384
vector = [0.0] * dimension
tokens = ["使用", "bash", "命令", "读取", "文件", "内容"]

# 多哈希位置映射
for i, token in enumerate(tokens):
    weight = 1.0 / (i + 1)  # 位置权重衰减
    
    for hash_idx in range(3):
        hash_val = int(md5(f"{token}_{hash_idx}".encode()).hexdigest(), 16)
        position = hash_val % 384
        vector[position] += weight * (1 if hash_idx % 2 == 0 else -1)

# L2 归一化（投影到单位超球面）
norm = sqrt(sum(x^2 for x in vector))
normalized_vector = [x / norm for x in vector]
```

**可视化理解**：

```
高维空间中的向量（简化到3维展示）：
查询向量 q = [0.5, 0.3, 0.8]
记忆A向量 a = [0.4, 0.35, 0.75]  # "用cat命令查看文件"
记忆B向量 b = [0.1, 0.9, 0.1]   # "Python的for循环"

夹角计算：
cos(θ_qa) = (0.5*0.4 + 0.3*0.35 + 0.8*0.75) / (1*1) = 0.85  → θ ≈ 32°  ✅ 相关
cos(θ_qb) = (0.5*0.1 + 0.3*0.9 + 0.8*0.1) / (1*1) = 0.12   → θ ≈ 83°  ❌ 无关
```

**步骤 2：混合评分函数**

$$
\text{Score} = w_{\text{sem}} \cdot \text{CosSim} + w_{\text{rec}} \cdot e^{-t/24} + w_{\text{imp}} \cdot I + w_{\text{acc}} \cdot \min(\frac{A}{10}, 1)
$$

**具体数据示例**：

```python
# 候选记忆1："bash命令行操作指南"（3小时前存储）
cos_sim = 0.92
time_diff = 3  # 小时
recency = exp(-3/24) = exp(-0.125) ≈ 0.882
importance = 0.8
access_count = 5

score = 0.5*0.92 + 0.3*0.882 + 0.2*0.8 + 0.1*0.5
      = 0.46 + 0.265 + 0.16 + 0.05
      = 0.935

# 候选记忆2："文件系统基础"（48小时前存储）
cos_sim = 0.85
time_diff = 48
recency = exp(-48/24) = exp(-2) ≈ 0.135

score = 0.5*0.85 + 0.3*0.135 + 0.2*0.7 + 0.1*0.2
      = 0.425 + 0.0405 + 0.14 + 0.02
      = 0.626  # 时间衰减导致分数降低
```

**步骤 3：HNSW 近似最近邻搜索（ANN）**

```python
# 不计算与所有记忆的距离（O(n)太慢），使用分层可导航小世界图
# 搜索复杂度：O(log n)

layer_0: 最粗粒度（聚类中心）
  ├── 聚类A（文件操作）→ 进入
  │     ├── 子聚类A1（读取操作）→ 进入
  │     │     ├── "bash读取文件" → 距离0.1
  │     │     └── "Python读取文件" → 距离0.3
  │     └── 子聚类A2（写入操作）
  └── 聚类B（网络操作）→ 剪枝（距离>0.8）
```

---

### 2.3 AdaptiveToolLearner：老工匠的经验笔记

#### 概念与比喻

**ToolLearner** 是"老工匠的笔记本"。他不只记录"锤子用来钉钉子"，而是记录**工作流程的统计学**：

- 敲钉子前，80% 概率要先**测量**（转移概率）
- 如果上次砸到手（失败），下次调整握法（负样本学习）

#### 数学模型：马尔可夫决策过程

**观测数据**（10次任务记录）：

```python
sessions = [
    ["list_dir", "read_file", "edit_file"],      # 任务1：查看→读取→修改
    ["list_dir", "read_file", "read_file"],      # 任务2：查看→读取→再读
    ["bash", "read_file", "write_file"],         # 任务3：搜索→读取→写入
    ["read_file", "edit_file"],                  # 任务4：直接读→改
    ["list_dir", "read_file", "edit_file"],      # 任务5
]
```

**构建转移矩阵**（条件概率 $P(T_{next} | T_{current})$）：

```python
# 统计转移频次
transition_counts = {
    "list_dir": {"read_file": 3},      # 3次 list_dir → read_file
    "read_file": {
        "edit_file": 3,                 # 3次 read_file → edit_file
        "write_file": 1,                # 1次 read_file → write_file
        "read_file": 1                  # 1次 read_file → read_file
    },
    "bash": {"read_file": 1}
}

# 归一化为概率（最大似然估计）
transition_prob = {
    "read_file": {
        "edit_file": 3/5 = 0.6,
        "write_file": 1/5 = 0.2,
        "read_file": 1/5 = 0.2
    }
}
```

**工具推荐计算**（当前状态：刚执行了 `read_file`）：

$$
\text{Score}(T_{next}) = P(T_{next}|T_{current}) \times \text{SuccessRate}(T_{next}) \times \text{ContextSim}
$$

```python
# 当前上下文：已执行 ["list_dir", "read_file"]，任务类型："代码修改"
current_tool = "read_file"
candidates = transition_prob[current_tool]  # {"edit_file": 0.6, "write_file": 0.2}

# 成功率（贝叶斯更新）
success_rates = {
    "edit_file": (9+1)/(10+2) = 0.83,  # 9成功1失败，拉普拉斯平滑
    "write_file": (17+1)/(20+2) = 0.82
}

# 上下文相似度（Jaccard指数）
task_keywords = {"代码", "修改", "文件"}
context_sim = {
    "edit_file": len({"代码", "修改"} ∩ task_keywords) / len({"代码", "修改", "文件"})
             = 2/3 ≈ 0.67,
    "write_file": 0.33
}

# 最终评分
score_edit = 0.6 * 0.83 * 0.67 ≈ 0.334
score_write = 0.2 * 0.82 * 0.33 ≈ 0.054

推荐结果：edit_file（置信度0.334）> write_file（置信度0.054）
```

---

### 2.4 DeepReflectionEngine：质量控制局的诊断算法

#### 概念与比喻

**DeepReflection** 是"工厂质量控制局"。当工人（工具）犯错时：

1. **分类**：是材料不对（参数错误）还是工具坏了（执行错误）
2. **模式识别**：这个错误5分钟内出现过吗？（重复故障检测）
3. **策略升级**：第3次同样错误→停工检修（中断循环）

#### 数学模型：故障检测与统计过程控制

**场景**：用户要求读取不存在的文件

**步骤 1：错误分类（正则模式匹配）**

```python
FAILURE_PATTERNS = {
    "file_not_found": {
        "patterns": [r"not found", r"不存在", r"找不到"],
        "category": "parameter_error",
        "confidence": 0.85
    },
    "permission_denied": {
        "patterns": [r"permission", r"权限"],
        "category": "auth_error", 
        "confidence": 0.90
    }
}

# 输入："文件不存在: /tmp/test.txt"
error_lower = "文件不存在: /tmp/test.txt"

# 匹配计算
for pattern in [r"不存在", r"找不到"]:
    if re.search(pattern, error_lower):
        match_score += 0.85
        
# 结果：分类为 file_not_found，置信度0.85
```

**步骤 2：重复故障检测（滑动窗口统计）**

```python
# 反思历史（时间序列）
reflection_history = [
    {"tool": "read_file", "category": "file_not_found", "timestamp": T-300s},
    {"tool": "read_file", "category": "file_not_found", "timestamp": T-120s},
    {"tool": "read_file", "category": "file_not_found", "timestamp": T-30s},  # 当前
]

# 统计检验（最近300秒窗口）
window = 300  # 5分钟
recent_failures = [h for h in history if (T - h.time) < window and h.category == "file_not_found"]

λ = len(recent_failures) / window  # 故障率（泊松过程强度）
# λ = 3/300 = 0.01 次/秒

# 连续3次同类错误判定
if len(recent_failures) >= 3:
    # 二项分布检验：P(3次失败|随机) < 0.05？
    p_value = binom_test(3, 3, p=0.1, alternative='greater')  # 假设基础失败率10%
    if p_value < 0.05:
        decision = "STRATEGIC_ESCALATION"  # 统计显著，升级处理
```

**步骤 3：自适应中断策略（基于失败率）**

```python
# 计算动态阈值（指数加权移动平均）
failure_rate_ewma = α * current_failure + (1-α) * previous_rate
# α = 0.3（平滑因子）

if failure_rate_ewma > 0.6:  # 60%失败率阈值
    decision = "INTERRUPT"  # 贝叶斯决策：继续的期望损失 > 中断损失
    action = "request_human"
else:
    decision = "RETRY_WITH_FIX"
    suggestion = random_choice(pattern.fixes)  # 从修复建议中随机选择
```

---

### 2.5 ReAct 循环：侦探的认知闭环

#### 概念与比喻

**ReAct 循环** = 侦探破案过程：

1. **Thought**："根据线索，凶手应该藏在阁楼"（推理）
2. **Action**："我去阁楼搜查"（工具调用）
3. **Observation**："阁楼空无一人，但有脚印"（工具返回）
4. **Reflection**："推理错误，重新分析脚印方向"（反思）
5. **循环**：直到找到凶手或确定无解

#### 完整数据流示例

**任务**：`"统计 core/ 目录下有多少个 Python 文件，并写入 report.txt"`

**Iteration 1：探索**

```python
# Thought生成（LLM解码过程）
input_tokens = ["统计", "core", "Python", "文件"]
attention_weights = softmax(Q·K^T / √d_k)  # Transformer自注意力

# LLM输出
thought = "用户需要统计 core/ 目录的 Python 文件数量。我应该先列出目录结构。"
action_text = """
list_dir
{"path": "core/"}
"""

# 解析（正则DFA）
tool_name, args = parse(action_text)  # ("list_dir", {"path": "core/"})

# 执行（系统调用）
fd = open("core/", O_RDONLY)  # 系统调用号 2
entries = getdents(fd)         # 读取目录项
result = ["agent.py", "tools.py", "utils/", "config.json"]

# 反思
reflection = {
    "success": True,
    "analysis": "成功获取目录列表，发现2个.py文件，但utils/子目录可能包含更多"
}
```

**Iteration 2：深入**

```python
# Observation注入上下文
observation = "✅ list_dir: 目录 core/ 包含：agent.py, tools.py, utils/, config.json"

# LLM新的Thought
thought = "有子目录utils/，需要递归统计。使用bash的find命令更高效。"
action = """
bash
{"command": "find core/ -name '*.py' | wc -l"}
"""

# 执行（子进程）
pid = fork()
if pid == 0:
    execve("/bin/bash", ["bash", "-c", "find core/ -name '*.py' | wc -l"])
stdout = 5  # 假设找到5个文件
```

**Iteration 3：写入**

```python
# Thought
thought = "已确认5个Python文件，现在需要写入report.txt。"
action = """
write_file
{"path": "report.txt", "content": "core/ 目录下共有 5 个 Python 文件", "mode": "overwrite"}
"""

# 执行（文件系统）
fd = open("report.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644)
write(fd, "core/ 目录下共有 5 个 Python 文件", 28)
close(fd)

# 完成检测（_looks_finished）
has_tool_mention = False  # 输出中无工具名
finish_signals = ["完成", "已写入"]  # 检测到完成词
is_finished = True  # 触发循环终止
```

---

## 3. 数据处理流向全景图

```
用户输入："读取config并修改端口"
│
├─► 字符编码：UTF-8解码 → Unicode码点序列
│   ["读", "取", "c", "o", "n", "f", "i", "g", ...]
│
├─► 分词（Tokenizer）：BPE算法
│   ["读取", "config", "并", "修改", "端口"]
│   Token IDs: [1234, 5678, 90, 4321, 8765]
│
├─► ModeRouter特征提取
│   路径特征：hash("config") % 1000 = 234
│   动词特征：hash("读取") % 1000 = 876
│   向量：[0,0,0,1,0,...,1,0] (稀疏向量)
│
├─► 贝叶斯分类
│   P(Tools) = 0.92 → 决策：Tools模式
│
├─► Middleware链
│   RuntimeModeMiddleware：插入"<runtime_mode>tools</runtime_mode>"
│   SkillsContextMiddleware：检索相关技能（余弦相似度>0.8的Skills）
│
├─► ReAct循环
│   │
│   ├─► LLM生成（Transformer解码）
│   │   输入维度：[batch=1, seq_len=20, hidden=512]
│   │   注意力计算：Q[20,64] × K^T[64,20] → Score[20,20]
│   │   Softmax归一化 → 加权求和V → 输出[20,512]
│   │   解码结果："read_file\n{\"path\":\"config.json\"}"
│   │
│   ├─► 工具解析（正则DFA）
│   │   状态转移：START → TOOL_NAME → NEWLINE → JSON → END
│   │   捕获组：tool="read_file", args={"path":"config.json"}
│   │
│   ├─► 并行检测（读写锁算法）
│   │   读集合 = {read_file}，写集合 = {}
│   │   无冲突 → 允许并行执行
│   │
│   ├─► 沙箱执行（安全约束）
│   │   路径解析：realpath("config.json") → "/home/user/project/config.json"
│   │   安全检查：startswith("/home/user/project") → True
│   │   系统调用：openat(AT_FDCWD, "config.json", O_RDONLY) = 3
│   │   读取：read(3, buf, 4096) → "{\n  \"port\": 8080\n}"
│   │
│   ├─► 结果封装（JSON序列化）
│   │   {"success": True, "content": "{\n  \"port\": 8080\n}", "size": 18}
│   │
│   ├─► 反思引擎
│   │   成功检测：无error字段 → success=True
│   │   转移矩阵更新：M[null][read_file] += 1
│   │
│   ├─► 记忆存储（VectorMemory）
│   │   文本："Tool: read_file, Args: config.json, Success: True"
│   │   嵌入计算：MD5哈希 → 384维向量 → L2归一化
│   │   HNSW插入：找到最近邻（相似度0.92的"读取配置文件"记忆）
│   │   存储到working_memory（容量限制：10条）
│   │
│   └─► Observation构造
│       "✅ read_file: {\n  \"port\": 8080\n}"
│
├─► 下一次迭代（修改端口）
│   LLM生成：edit_file操作（基于上一次Observation）
│   ...
│
└─► 最终输出清理
    移除ReAct标签：_clean_react_tags()
    输出："已将config.json中的端口从8080修改为3000"
    
    日志记录：
    ├─► 用户消息哈希：SHA256("读取config...")
    ├─► 执行时间：Δt = 2.34s
    ├─► Token计数：Input=45, Output=128
    └─► 向量存储：写入.memory/vector_memory_v2.json
```

### 3.2 长期记忆的形成机制（跨会话）

```
会话1：
用户："如何用Python读取PDF？"
技能匹配：pdf-processing
执行：bash("pip install PyPDF2")
结果：成功
记忆写入：VectorMemory.add(
    content="任务: PDF处理, 工具: bash, 命令: pip install PyPDF2",
    embedding=[0.23, -0.45, ...],
    importance=0.8,
    timestamp=2026-03-31T00:00:00
)

会话2（3天后）：
用户："我要处理一个PDF文件"
检索触发：VectorMemory.search("PDF处理")
计算相似度：
├── 候选1: "PDF处理技能" (相似度: 0.92)
├── 候选2: "文件读取方法" (相似度: 0.65)
└── 候选3: "Python教程" (相似度: 0.34)
时间衰减加权：
├── 候选1分数: 0.92 × exp(-72/24) = 0.92 × 0.05 = 0.046
└── 候选2分数: 0.65 × exp(-0/24) = 0.65 × 1.0 = 0.65
返回 top_k=3，注入到当前上下文作为 Few-shot 示例
```

---

## 4. 端到端完整案例：PDF处理任务

**用户请求**：`"读取uploaded.pdf，提取前3页内容保存到summary.txt"`

### 阶段 1：意图识别与路由（0-10ms）

```python
# 输入分析
keywords = {"读取", "pdf", "提取", "保存"}  # TF-IDF权重高
file_ext = ".pdf"  # 触发Skills匹配

# ModeRouter计算
P(Skills) = 0.85  # PDF是专业领域
P(Tools) = 0.90   # 涉及文件读写
决策：Hybrid模式（技能和工具结合）

# Skill注入
匹配技能："pdf-processing"（相似度0.94）
注入内容："使用pdftotext或PyMuPDF处理PDF文件..."
```

### 阶段 2：ReAct执行（10ms-2s）

**Round 1**：验证PDF存在

```python
Thought: "先确认PDF文件是否存在，使用list_dir查看上传目录"
Action: list_dir({"path": "./uploads"})
Observation: {"items": [{"name": "uploaded.pdf", "size": 2048000}]}
Reflection: 成功，文件存在，大小2MB
```

**Round 2**：提取内容（技能指导）

```python
Thought: "根据pdf-processing技能，使用pdftotext提取前3页"
Action: bash({"command": "pdftotext -f 1 -l 3 uploads/uploaded.pdf -"})
Observation: "Page 1\nContent...\nPage 2\nContent...\nPage 3\nContent..."
Reflection: 成功提取3页内容，约5000字符
```

**Round 3**：写入文件

```python
Thought: "将提取的内容写入summary.txt"
Action: write_file({
    "path": "summary.txt",
    "content": "Page 1\nContent...\nPage 2\nContent...\nPage 3\nContent...",
    "mode": "overwrite"
})
Observation: {"success": True, "size": 5120}
Reflection: 任务完成
```

### 阶段 3：学习与记忆（2s-2.1s）

```python
# ToolLearner更新
transition_matrix["bash"]["write_file"] += 1
success_rate["pdftotext"] = (9+1)/(10+2)  # 贝叶斯更新

# VectorMemory存储
entry = {
    "content": "任务:PDF提取, 工具序列:[list_dir,bash,write_file], 耗时:2s",
    "embedding": [0.12, -0.34, ..., 0.89],  # 384维
    "importance": 0.8,
    "timestamp": "2026-03-31T00:18:00"
}
insert_to_hnsw(entry)  # O(log n)复杂度
```

---

## 5. 数学原理汇总表

|组件|核心算法|数学公式|复杂度|
|--|--|--|--|
|**ModeRouter**|朴素贝叶斯 + 温度校准|$P(I\|E) = \frac{P(I)\prod P(f_i|I)}{P(E)}$，$C_{cal} = \frac{C_{raw}}{T} + b$|$O(n)$，$n$=特征数|
|**VectorMemory**|余弦相似度 + HNSW|$\text{Sim} = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}||\mathbf{b}|}$，$\text{Score} = \sum w_i \cdot f_i$|检索$O(\log n)$，插入$O(\log n)$|
|**ToolLearner**|马尔可夫链 + MLE|$P(T_{next}\|T_{curr}) = \frac{\text{count}(T_{curr} \to T_{next})}{\sum \text{count}}$|$O(1)$查找，$O(n)$训练|
|**DeepReflection**|正则匹配 + 泊松检验|$\lambda = \frac{k}{t}$，$P(X \geq k) = 1 - \sum_{i=0}^{k-1}\frac{e^{-\lambda t}(\lambda t)^i}{i!}$|$O(m)$，$m$=历史长度|
|**ReAct**|Transformer解码 + 有限状态机|$\text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$|$O(L^2 \cdot d)$，$L$=序列长|
|**并行执行**|读写锁（读者-写者问题）|约束：$R \cap W = \emptyset \Rightarrow \text{parallel}$，否则$\text{serial}$|$O(n)$检测，$n$=工具数|