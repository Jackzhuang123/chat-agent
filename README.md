# Chat-Agent - 先进 AI Agent 框架

基于 ReAct 模式的智能 Agent 框架，针对 glm-4-flash (8k context) 优化，实现并行执行、持久化记忆和语义压缩。

## 核心特性（基础版）

### 1. ReAct 模式
- **Thought**: 分析当前情况，思考下一步
- **Action**: 选择并执行工具
- **Observation**: 观察结果
- **Reflection**: 反思是否成功，需要调整吗

### 2. 并行工具执行
- 自动检测只读工具（`read_file`, `list_dir`）
- 使用 `ThreadPoolExecutor` 并发执行，最多3个并发
- 写入工具（`write_file`, `edit_file`, `bash`）保持串行

### 3. 持久化记忆
- `SessionMemory` 自动保存到 `.agent_memory/session_memory.pkl`
- 跨会话记录工具使用统计（成功率、平均耗时）
- 保留最近3次会话的关键上下文

### 4. 语义压缩
- 基于关键词密度计算消息重要性
- 用户消息权重 1.5x，工具结果权重 1.3x
- 保留最近6条 + 历史top-3重要消息

### 5. 循环检测与智能重试
- 3次相同失败自动中断
- 自动修复 grep 转义错误
- 自动补全相对路径

## 高级特性（新增）

### 1. 向量记忆（VectorMemory）
- 使用 TF-IDF embedding 存储历史对话
- 基于余弦相似度的语义检索
- 持久化到 `.agent_memory/vector_memory.pkl`
- 自动去重（相似度 >0.95）

### 2. 多Agent协作（MultiAgentOrchestrator）
- **Planner**：任务分解和规划
- **Executor**：执行具体步骤
- **Reviewer**：结果审查和质量控制
- 完整的 Plan → Execute → Review 循环

### 3. 工具学习（ToolLearner）
- 任务分类（文件操作、代码分析、系统命令等）
- 基于历史成功率推荐工具
- 持久化学习结果
- 自适应优化

### 4. 流式输出（StreamingFramework）
- SSE（Server-Sent Events）格式
- 实时展示思考、工具调用、结果
- 支持 Web 应用集成
- 8种事件类型（start, thought, tool_call, tool_result, reflection, progress, complete, error）

### 5. 智能模式路由（ModeRouter）
- **自动检测**：根据用户输入自动识别意图
- **5种模式**：chat（对话）、tools（工具）、plan（计划）、multi_agent（多Agent）、streaming（流式）
- **智能升级**：复杂任务自动启用高级模式
- **零配置**：无需手动指定 `runtime_context`

---

## 核心能力

- **Agent 循环**：模型输出 -> 解析工具 -> 执行工具 -> 结果回注 -> 继续迭代
- **工具系统**：`read_file`、`write_file`、`edit_file`、`list_dir`、`bash`
- **Skills 系统**：任务自动匹配技能并注入上下文
- **模式化运行**：`chat` / `tools` / `skills` / `hybrid`
- **计划模式**：可注入分步执行约束
- **可观测性**：记录 `runtime_context` 与 `execution_log`，支持回看执行链路
- **gstack 架构原则**：移植自 gstack 的五大哲学中间件（完整性、结构化提问、完成状态、搜索优先、仓库所有权）

---

## 最近关键更新

本次已完成以下增强并落地：

- 引入中间件链（`RuntimeModeMiddleware`、`PlanModeMiddleware`、`SkillsContextMiddleware`、`UploadedFilesMiddleware`、`ToolResultGuardMiddleware`）
- `ToolParser` 增强容错：缺失 `</input>`、轻微 JSON 不完整时仍可解析工具调用
- `read_file` 结果改为结构化摘要（标题、章节、行数字符统计、预览），避免原文大段回灌
- 新增防复述守卫（anti-repeat guard）：模型复述工具结果时自动触发重写
- UI 新增计划模式开关，并在会话日志中持久化运行上下文与执行日志

### gstack 架构移植（最新）

完整扫描并移植了 `gstack` 的核心设计哲学，以中间件形式集成到 Agent 框架中：

| 中间件 | 移植自 gstack | 功能说明 |
|--------|-------------|---------|
| `CompletenessMiddleware` | `ETHOS.md` Boil the Lake | 注入"完整性优先"提示，引导模型选择完整方案而非捷径 |
| `AskUserQuestionMiddleware` | `SKILL.md` AskUserQuestion | 结构化提问格式：Re-ground / Simplify / Recommend / Options |
| `CompletionStatusMiddleware` | `SKILL.md` Completion Status | 强制使用 DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT 汇报 |
| `SearchBeforeBuildingMiddleware` | `ETHOS.md` Search Before Building | 构建前先搜索，三层知识体系 + Eureka Moment 发现机制 |
| `RepoOwnershipMiddleware` | `SKILL.md` Repo Ownership Mode | 区分 solo / collaborative 模式，决定主动修复还是先询问 |

新增技能：

- **`web-qa/`**：移植自 gstack QA 工作流，提供 Playwright 浏览器自动化测试的系统方法论

> 代码级详细流程见：`DEER_FLOW_INTEGRATION_GUIDE.md`

---

## 快速开始

### 1) 安装依赖

```bash
cd /Users/zhuangranxin/PyCharmProjects/chat-Agent
pip install -r requirements.txt
```

### 2) 启动主应用

```bash
python3 ui/web_agent_with_skills.py
```

打开：`http://127.0.0.1:7860`

### 3) 一键启动多应用（可选）

```bash
bash ui/start_all_apps.sh
```

将同时启动：

- Chat Agent: `http://127.0.0.1:7860`
- Session Viewer: `http://127.0.0.1:7861`
- Session Analyzer: `http://127.0.0.1:7862`

停止：

```bash
bash ui/stop_all_apps.sh
```

---

## 运行模式

- **chat**：纯对话
- **tools**：优先工具执行
- **skills**：优先技能知识
- **hybrid**：技能 + 工具协同

可在 UI 侧边栏控制：

- 工具模式
- Skills 系统
- 计划模式
- 模型参数（temperature/top_p/max_tokens）

---

## 项目结构

```text
chat-Agent/
├── core/
│   ├── agent_framework.py         # Agent主循环、中间件、结果守卫
│   │                              #   内置中间件：RuntimeModeMiddleware、PlanModeMiddleware
│   │                              #               SkillsContextMiddleware、UploadedFilesMiddleware
│   │                              #               ToolResultGuardMiddleware
│   │                              #   gstack移植：CompletenessMiddleware、AskUserQuestionMiddleware
│   │                              #               CompletionStatusMiddleware、SearchBeforeBuildingMiddleware
│   │                              #               RepoOwnershipMiddleware
│   ├── agent_tools.py             # 工具执行器 + 工具调用解析器
│   └── agent_skills.py            # Skills管理与上下文注入
├── ui/
│   ├── web_agent_with_skills.py   # 主应用UI
│   ├── session_logger.py          # 会话日志落盘
│   ├── session_viewer.py          # 会话查看
│   ├── session_analyzer.py        # 会话分析
│   ├── start_all_apps.sh
│   └── stop_all_apps.sh
├── skills/
│   ├── pdf/                       # PDF处理技能
│   ├── code-review/               # 代码审查技能
│   ├── python-dev/                # Python开发技能
│   └── web-qa/                    # Web QA测试技能（移植自gstack）
│       ├── SKILL.md               #   Playwright自动化测试方法论
│       └── references/
│           └── qa-checklist.md    #   完整QA检查清单
├── session_logs/
├── GUIDE.md
├── API_REFERENCE.md
├── QUICK_START.md
└── DEER_FLOW_INTEGRATION_GUIDE.md
```

---

## 文档导航

- `README.md`：项目总览与入口（本文件）
- `GUIDE.md`：使用说明、模式介绍与常见问题
- `API_REFERENCE.md`：核心 API 与调用示例
- `QUICK_START.md`：启动与日志查看快捷指南
- `DEER_FLOW_INTEGRATION_GUIDE.md`：框架集成与代码细节流程总结

---

## 高层执行流程

1. UI 收集用户输入、模式开关、上传文件、技能选择
2. 构造 `runtime_context` 并进入 `QwenAgentFramework`
3. 中间件链在模型调用前注入模式/计划/技能/文件上下文
4. 模型输出工具调用，`ToolParser` 解析并容错修复
5. `ToolExecutor` 执行工具，`ToolResultGuardMiddleware` 统一结果结构
6. 工具结果格式化回注（`read_file` 走结构化摘要）
7. 模型生成最终答复；若命中复述模式触发 anti-repeat 重写
8. 写入 `execution_log` + `runtime_context` 到会话日志

---

## 安全与约束

- 路径访问限制在工作目录内，防止目录穿越
- `bash` 工具默认关闭（可按需启用）
- shell 执行默认 30 秒超时
- 通过最大迭代次数防止无限循环

---

## 开发验证

建议在改动后执行：

```bash
python3 -m compileall core ui
```

如果只想验证核心逻辑：

```bash
python3 -m compileall core
```
