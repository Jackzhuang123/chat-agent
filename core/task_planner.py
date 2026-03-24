#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务规划与执行模块 - Orchestrator + Worker 架构

包含：
  - TaskPlanner    : AI 任务规划器，将复杂需求拆解为有序 TODO 列表
  - TaskExecutor   : Orchestrator + Worker 架构的任务执行器

设计模式：
  - 策略模式：Worker 角色根据工具类型自动专化（Reader/Writer/Shell 等）
  - 模板方法：_run_step 作为核心模板，execute_todos 复用执行模板
  - Orchestrator-Worker 架构：主控调度 + 子任务并发执行
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .todo_manager import TodoManager


class TaskPlanner:
    """
    AI 任务规划器：将复杂用户需求拆解为有序 TODO 列表。

    参考 Claude Code 的任务管理理念：
    - 每个 TODO 是独立可执行的原子任务
    - 有明确的执行状态：pending / in_progress / completed / failed
    - 支持工具调用（read_file / write_file / bash 等）
    """

    PLANNER_SYSTEM_PROMPT = """\
你是一个任务规划助手。将用户的复杂需求拆解为具体的执行步骤，返回 JSON，不要输出其他内容。

返回格式（严格 JSON）：
{
  "title": "任务标题（一句话）",
  "todos": [
    {"id": 1, "task": "步骤描述", "tool": "read_file|write_file|bash|none", "status": "pending"},
    {"id": 2, "task": "步骤描述", "tool": "none", "status": "pending"}
  ]
}

规则：
- 步骤数量：3-7 个，每步清晰可执行
- tool 字段：如果该步骤需要工具调用，填写工具名；否则填 "none"
- 步骤之间有逻辑顺序
- 每步描述简洁但具体（20字以内），必须明确操作对象（如"用grep扫描.py文件中的类和方法"而非"扫描目录"）
- 严格禁止两个步骤做同一件事（如步骤1扫描文件，步骤2不能再扫描同类文件）
- 步骤2及后续步骤必须依赖上一步的结果做进一步处理，不能重复上一步的命令
- 本项目是 Python 项目：涉及代码扫描时，优先搜索 .py 文件，禁止搜索 .java/.class 文件
- 【合并规则】"整理成文档"和"写入文件"是同一个操作，必须合并为一个步骤（tool=write_file），禁止拆成两步
- 运行命令时使用 python3 而非 python（macOS 环境）"""

    def __init__(self, model_forward_fn):
        self.model_forward_fn = model_forward_fn

    @staticmethod
    def _merge_todos(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        后处理合并规则，减少冗余步骤。

        合并规则：
        1. 相邻的两个 bash 步骤，若描述高度相似（都是扫描/解析同类文件），合并为一步
        2. tool=none 的"纯内容整理"步骤 + 紧随的 write_file 步骤 → 合并为单个 write_file 步骤
           条件约束（防止误合并）：
             a. none 步骤必须是整个列表中唯一的非工具步骤（前面没有真实工具步骤积累结果）
             b. 任务描述必须同时含有"整理/格式化/转换"类动词 AND "写/输出/文件"类名词
        3. write_file 步骤 + 紧随的 write_file 步骤（重复写入同一文件）→ 合并
        4. 合并后重新编号 id
        """
        if not todos:
            return todos

        _verb_keywords = {"整理成", "格式化", "转换为", "归纳为", "汇总为", "构建文档", "合并内容"}
        _target_keywords = {"写入", "输出到文件", "生成文件", "保存到"}
        _scan_keywords = {"扫描", "解析", "搜索", "查找", "grep", "find", "分析"}
        _real_tool_set = {"bash", "read_file", "write_file", "edit_file", "list_dir",
                          "python-dev", "python_dev", "code-review"}

        merged: List[Dict[str, Any]] = []
        skip_indices: set = set()

        for i, todo in enumerate(todos):
            if i in skip_indices:
                continue

            tool = (todo.get("tool") or "none").strip()
            task = todo.get("task", "")

            # 规则1：相邻两个 bash 扫描步骤合并
            if tool == "bash" and i + 1 < len(todos):
                next_todo = todos[i + 1]
                next_tool = (next_todo.get("tool") or "none").strip()
                next_task = next_todo.get("task", "")
                if (next_tool in ("bash", "python-dev", "python_dev")
                        and any(kw in task for kw in _scan_keywords)
                        and any(kw in next_task for kw in _scan_keywords)):
                    skip_indices.add(i + 1)
                    merged.append(todo)
                    continue

            # 规则2：tool=none 纯内容整理步骤 + 紧随的 write_file 步骤 → 合并为 write_file
            if tool == "none":
                has_verb = any(kw in task for kw in _verb_keywords)
                has_target = any(kw in task for kw in _target_keywords)
                prior_has_real_tool = any(
                    (todos[j].get("tool") or "none").strip() in _real_tool_set
                    for j in range(i)
                    if j not in skip_indices
                )
                if has_verb and has_target and not prior_has_real_tool:
                    if i + 1 < len(todos):
                        next_todo = todos[i + 1]
                        next_tool = (next_todo.get("tool") or "none").strip()
                        if next_tool == "write_file":
                            skip_indices.add(i + 1)
                            merged.append(dict(next_todo))
                            continue

            # 规则3：write_file + write_file 合并
            if tool == "write_file" and i + 1 < len(todos):
                next_todo = todos[i + 1]
                next_tool = (next_todo.get("tool") or "none").strip()
                if next_tool == "write_file":
                    skip_indices.add(i + 1)
                    merged.append(todo)
                    continue

            merged.append(todo)

        # 重新编号
        for idx, t in enumerate(merged):
            t["id"] = idx + 1

        return merged

    def plan(self, user_message: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        为用户需求生成 TODO 列表。

        Returns:
            {"title": str, "todos": [{"id", "task", "tool", "status"}, ...]}
            或 None（失败时调用方处理）
        """
        prompt = f"需求：{user_message[:500]}"
        if context:
            prompt += f"\n\n上下文：{context[:200]}"

        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.model_forward_fn(
                messages,
                system_prompt=self.PLANNER_SYSTEM_PROMPT,
                temperature=0.3,
                top_p=0.9,
                max_tokens=400,
            )
            raw = raw.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                todos = data.get("todos", [])
                if isinstance(todos, list) and len(todos) >= 2:
                    for i, t in enumerate(todos):
                        t.setdefault("id", i + 1)
                        t.setdefault("tool", "none")
                        t.setdefault("status", "pending")
                    todos = self._merge_todos(todos)
                    return {"title": data.get("title", "任务执行"), "todos": todos}
        except Exception:
            pass
        return None


class TaskExecutor:
    """
    Orchestrator + Worker 架构的 SubAgent 任务执行器。

    架构设计：
    ─────────────────────────────────────────────────────────────────
    Orchestrator（主控）：
      - 负责任务拆解后的分发与监控
      - 当某步骤失败时，调用 LLM 进行动态决策（retry/skip/abort）
      - 汇总所有步骤结果，生成最终回复

    Worker（执行者）：
      - 每个步骤由一个独立的 QwenAgentFramework 实例执行
      - 根据步骤工具类型自动选择角色专化的系统提示词
      - 拥有完整的中间件链（RuntimeMode、TodoContext、ToolResultGuard）
    ─────────────────────────────────────────────────────────────────
    与 Claude Code 的对比：
      Claude  → 一个长上下文 Agent 循环，模型自己决定何时完成何时继续
      本项目  → 多步 Agent 循环，每步独立 + 结果通过上下文传递，适配小模型
    """

    # ── Worker 角色专化：根据步骤工具类型分配不同系统提示词前缀 ──────────
    WORKER_ROLES: Dict[str, str] = {
        "read_file": (
            "你是一个代码阅读专家（Worker: Reader）。"
            "专注于提取文件的关键结构、核心逻辑和重要信息，用简洁的要点格式输出，不复述原文。"
        ),
        "write_file": (
            "你是一个文件写入工具操作员（Worker: Writer）。\n"
            "【核心职责】调用 write_file 工具将指定内容写入文件。\n"
            "【首要规则】当任务中出现 ---BEGIN CONTENT--- 和 ---END CONTENT--- 标记时：\n"
            "  1. 只做一件事：调用 write_file 工具，将 BEGIN 和 END 之间的文本原样复制为 content 参数\n"
            "  2. 严禁为任何理由修改、重写、添加或删除任何内容\n"
            "  3. 严禁将内容放入 ``` 代码块围栏中\n"
            "  4. 一次 write_file 成功后即返回结果，不重复调用\n"
            "  5. 严禁调用 todo_write，任务状态由框架自动管理\n"
            "【文档生成规则】当任务提供了按文件分组的 grep 数据模板时：\n"
            "  1. 模板中每个 '=== 文件名 ===' 分隔符对应一个 .py 文件，写入 ## 文件名 二级标题\n"
            "  2. 每行格式为 'file.py:行号:内容'，按如下规则解析：\n"
            "     - 若内容以 'class ' 开头：生成 ### ClassName 三级标题，开始列其方法\n"
            "     - 若内容以 '    def ' 开头（有前缀空格）：当前 class 的成员方法，写入列表\n"
            "     - 若内容以 'def ' 开头（无前缀空格）：模块级函数，归入 '### 模块级函数' 小节\n"
            "  3. 方法签名去掉行号前缀 ('file.py:26:' 部分)，只保留方法名和参数\n"
            "  4. 对每个文件的数据必须全部写入，不得省略任何行\n"
            "注重内容完整性：模板有多少行数据，输出就要覆盖多少行，绝对不能只写一部分。"
        ),
        "edit_file": (
            "你是一个代码修改专家（Worker: Editor）。"
            "精确定位修改点，最小化改动范围，保持原有代码风格，修改前说明变更理由。"
        ),
        "list_dir": (
            "你是一个项目结构分析师（Worker: Explorer）。"
            "快速梳理目录结构，识别关键文件和模块职责，用层次化方式输出。"
        ),
        "bash": (
            "你是一个命令行专家（Worker: Shell）。"
            "安全执行命令，解读输出结果，对错误给出诊断建议，不执行破坏性操作。\n"
            "【职责边界】你只负责执行 shell 命令并返回结果，禁止调用 write_file、edit_file、todo_write 等工具。\n"
            "  - 如需扫描文件或目录，使用 bash 工具执行 grep/find/cat 等命令\n"
            "  - 严禁将扫描结果自行整理后写入任何文件（写文件是后续步骤的职责）\n"
            "  - 命令执行完成后直接返回结果，不要自行标记任务完成（任务状态由框架自动管理）\n"
            "【项目类型感知】本项目是 Python 项目，核心文件扩展名为 .py。\n"
            "  - 扫描源码类/方法时：使用 grep -rn 'class\\|def ' 或 find . -name '*.py'\n"
            "  - 禁止搜索 .java、.class、.kt 等非 Python 文件扩展名\n"
            "  - 若扫描结果为空，应主动调整搜索条件（如换扩展名或换目录）再重试一次\n"
            "【macOS 环境感知】本系统为 macOS，Python 命令规则：\n"
            "  - 始终使用 python3 而非 python（macOS 上 python 命令不存在）\n"
            "  - 始终使用 pip3 而非 pip\n"
            "  - 运行测试：python3 -m pytest 或 python3 -m doctest\n"
            "  - 若命令返回 'command not found'，立即尝试加 3 后缀重新执行（python→python3）\n"
            "【重要】调用 bash 工具时，必须使用 JSON 格式，例如：\n"
            'bash\n{"command": "ls -la"}\n'
            "绝对不能直接输出裸命令（如 bash\\nls -la），必须包含 {\"command\": ...} 的 JSON 结构。\n"
            "【重要】严格只做当前步骤要求的事，不要做额外的探索性命令（如 ls、pwd）。"
        ),
        "none": (
            "你是一个综合分析专家（Worker: Analyst）。"
            "基于已收集的信息进行深度推理和总结，给出结构化、有洞察力的分析结果。"
        ),
    }

    # ── Orchestrator 决策 Prompt ──────────────────────────────────────────
    ORCHESTRATOR_SYSTEM = """\
你是任务编排者（Orchestrator）。当前执行的某个步骤失败了，请决定如何继续。
返回严格 JSON，不输出任何其他内容：
{"action": "retry|skip|abort", "reason": "一句话说明原因", "modified_task": "重试时可选修改任务描述，skip/abort 时留空"}

决策规则：
- retry：错误是临时性的（如参数有误、路径不对），可通过修改任务描述修复，最多重试1次
- skip：步骤可跳过（如可选信息收集、非关键步骤），后续步骤不依赖此步骤结果
- abort：任务根本无法完成（如核心文件不存在、权限不足），继续执行没有意义"""

    def __init__(
        self,
        model_forward_fn,
        tool_executor=None,  # ToolExecutor 实例（避免循环导入用 Any 类型）
        work_dir: Optional[str] = None,
        max_iterations_per_step: int = 3,
    ):
        """
        Args:
            model_forward_fn: 模型前向函数（同时用于 Worker 执行和 Orchestrator 决策）
            tool_executor: 可复用已有的工具执行器（避免重复初始化）
            work_dir: 工具工作目录
            max_iterations_per_step: 每个步骤允许的最大 Agent 迭代次数
        """
        self.model_forward_fn = model_forward_fn
        self._external_tool_executor = tool_executor
        self.work_dir = work_dir
        self.max_iterations_per_step = max_iterations_per_step

    # ── Orchestrator：失败决策 ────────────────────────────────────────────

    def _orchestrator_decide(
        self,
        task_desc: str,
        error_msg: str,
        accumulated_context: str,
    ) -> Dict[str, Any]:
        """
        Orchestrator 决策：步骤失败时调用 LLM 决定 retry/skip/abort。

        Args:
            task_desc: 失败的步骤描述
            error_msg: 错误信息
            accumulated_context: 已完成步骤的累积上下文（用于辅助判断）

        Returns:
            {"action": "retry|skip|abort", "reason": str, "modified_task": str}
        """
        prompt = (
            f"失败步骤：{task_desc}\n"
            f"错误信息：{error_msg[:300]}\n"
            f"已完成的上下文摘要：{accumulated_context[-300:] if accumulated_context else '（无）'}"
        )
        try:
            raw = self.model_forward_fn(
                [{"role": "user", "content": prompt}],
                system_prompt=self.ORCHESTRATOR_SYSTEM,
                temperature=0.1,
                top_p=0.9,
                max_tokens=200,
            )
            raw = raw.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                decision = json.loads(m.group())
                action = decision.get("action", "skip")
                if action not in ("retry", "skip", "abort"):
                    action = "skip"
                return {
                    "action": action,
                    "reason": str(decision.get("reason", "")),
                    "modified_task": str(decision.get("modified_task", "")),
                }
        except Exception as decide_err:
            print(f"[Orchestrator] 决策调用失败，降级为 skip: {decide_err}")
        return {"action": "skip", "reason": "Orchestrator 决策失败，降级跳过", "modified_task": ""}

    # ── Worker：构建专化系统提示词 ────────────────────────────────────────

    def _worker_system_prompt(self, base_system_prompt: str, tool_hint: str) -> str:
        """为 Worker 构建角色专化的系统提示词。"""
        role = self.WORKER_ROLES.get(tool_hint or "none", self.WORKER_ROLES["none"])
        return f"{base_system_prompt}\n\n{role}"

    # ── Worker：执行单个步骤 ────────────────────────────────────────────

    def _run_step(
        self,
        step_id: int,
        task_desc: str,
        tool_hint: str,
        user_message: str,
        accumulated_context: str,
        todo_mgr: "TodoManager",
        system_prompt: str,
        safe_kwargs: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行单个 TODO 步骤，返回 (step_response, step_log)。

        由 execute_todos 和重试逻辑共同调用（模板方法复用）。
        """
        # 延迟导入避免循环依赖
        from .agent_framework import QwenAgentFramework
        from .agent_middlewares import RuntimeModeMiddleware, ToolResultGuardMiddleware, TodoContextMiddleware

        # 将非法/虚拟工具名映射到真实可执行工具
        _VIRTUAL_TOOLS = {"python-dev", "python_dev", "code-review", "pdf", "skills"}
        _REAL_TOOLS = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write", "none"}
        if tool_hint and tool_hint not in _REAL_TOOLS:
            tool_hint = "bash"

        # 构建步骤 prompt
        step_content_parts = [f"当前任务步骤 [{step_id}]：{task_desc}"]
        if accumulated_context:
            step_content_parts.append(f"\n已完成步骤的信息摘要：\n{accumulated_context}")
            _empty_signals = ["未找到任何", "没有找到", "no results", "0 results", "空结果", "找不到"]
            _wrong_ext_signals = [".java", ".class", ".kt", ".jar"]
            has_empty_result = any(sig in accumulated_context for sig in _empty_signals)
            has_wrong_ext = any(sig in accumulated_context for sig in _wrong_ext_signals)
            if has_empty_result or has_wrong_ext:
                step_content_parts.append(
                    "\n⚠️ 注意：前序步骤的搜索结果为空，可能使用了错误的搜索条件（如搜索了 .java 而非 .py）。"
                    "请不要基于前序步骤的空结果放弃，而应独立执行本步骤，使用正确的 Python 文件扩展名（.py）重新执行。"
                )

        step_content_parts.append(f"\n原始需求：{user_message[:200]}")

        if not tool_hint or tool_hint == "none":
            step_content_parts.append(
                "\n⚠️ 本步骤为整理/分析步骤，禁止调用任何工具。\n"
                "请直接基于上方「已完成步骤的信息摘要」中的原始数据，完成整理任务。\n"
                "要求：\n"
                "1. 输出完整的结构化文本（如 Markdown），覆盖全部原始数据，不得省略任何条目\n"
                "2. 直接输出整理结果正文，不要用「以下是整理结果」等前缀句包裹\n"
                "3. 不要描述后续步骤，整理完成后立即结束输出"
            )
        else:
            _extra_boundary = ""
            if tool_hint == "bash":
                _extra_boundary = (
                    "\n⚠️ 职责边界：你是 bash 执行者，只能调用 bash 工具，"
                    "严禁调用 write_file/edit_file/read_file/todo_write 等工具。"
                    "命令执行完成后直接返回结果，不要自行标记任务完成。"
                )
            elif tool_hint == "write_file":
                _extra_boundary = (
                    "\n⚠️ 职责边界：你是文件写入者，只需调用一次 write_file 工具，"
                    "根据前序步骤的结果整理内容后写入文件，严禁调用 todo_write。"
                )
            elif tool_hint in ("read_file", "list_dir"):
                _extra_boundary = (
                    "\n⚠️ 职责边界：你是文件读取/探索者，只能调用 read_file/list_dir 工具，"
                    "严禁调用 write_file/edit_file 等文件写入工具。"
                )
            step_content_parts.append(
                f"\n提示：本步骤需要使用 {tool_hint} 工具完成。"
                "请直接调用工具，不要猜测结果。"
                "完成后只需返回结果，不要描述后续步骤。\n"
                "⚠️ 严格约束：只执行本步骤要求的操作，禁止执行与本步骤无关的额外命令"
                "（如 ls、pwd、cat 等探索性命令），除非本步骤明确需要。"
                + _extra_boundary
            )

        step_message = "\n".join(step_content_parts)
        worker_prompt = self._worker_system_prompt(system_prompt, tool_hint)

        need_bash = (tool_hint == "bash")
        todo_middleware = TodoContextMiddleware(todo_mgr, worker_mode=True)
        step_framework = QwenAgentFramework(
            model_forward_fn=self.model_forward_fn,
            work_dir=self.work_dir,
            enable_bash=need_bash,
            max_iterations=self.max_iterations_per_step,
            middlewares=[
                RuntimeModeMiddleware(),
                todo_middleware,
                ToolResultGuardMiddleware(),
            ],
        )
        # 仅当不需要 bash 时才复用外部 executor
        if self._external_tool_executor is not None and not need_bash:
            step_framework.tool_executor = self._external_tool_executor

        run_mode = "tools" if (tool_hint and tool_hint not in ("none", "")) else "chat"
        step_response, step_log = step_framework.process_message(
            user_message=step_message,
            history=[],
            system_prompt_override=worker_prompt,
            runtime_context={
                "run_mode": run_mode,
                "plan_mode": False,
                "uploaded_files": [],
            },
            **safe_kwargs,
        )
        return step_response, step_log

    # ── 主执行入口 ────────────────────────────────────────────────────────

    def execute_todos(
        self,
        todos: List[Dict[str, Any]],
        user_message: str,
        system_prompt: str = "你是一个智能助手。",
        context_messages: Optional[List[Dict[str, str]]] = None,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        依次执行 TODO 列表（Orchestrator + Worker 架构）。

        核心流程：
          1. 初始化 TodoManager，加载 todos
          2. 对每个步骤（Worker 执行）：
             a. 标记 in_progress，用角色专化的 system_prompt 执行
             b. 成功 → 标记 completed，更新累积上下文
             c. 失败 → 调用 Orchestrator 决策（retry/skip/abort）
                retry → 用修改后的任务描述重试一次
                skip  → 标记 failed，继续下一步
                abort → 标记 failed，终止整个任务
          3. Orchestrator 汇总所有步骤结果，生成最终回复

        Returns:
            {
              "final_response": str,
              "todos": [...],
              "step_results": [...],
              "execution_log": [...],
              "todo_manager": TodoManager
            }
        """
        # ── 1. 初始化 ──────────────────────────────────────────────────────
        todo_mgr = TodoManager(title="任务执行计划")
        todo_mgr.load_from_todos_list(todos)

        step_results: List[Dict[str, Any]] = []
        execution_log: List[Dict[str, Any]] = []
        accumulated_context = ""
        base_messages = list(context_messages or [])
        _safe_kwargs = {k: v for k, v in model_kwargs.items() if k in ("temperature", "top_p", "max_tokens")}

        # ── 2. 逐步执行（Orchestrator 调度 Workers）────────────────────────
        should_abort = False
        for item in todo_mgr._items:
            if should_abort:
                todo_mgr.update(item.id, "cancelled")
                execution_log.append({
                    "type": "todo_cancelled",
                    "id": item.id,
                    "reason": "Orchestrator 已终止任务",
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                continue

            step_id = item.id
            task_desc = item.task
            tool_hint = item.tool

            todo_mgr.update(step_id, "in_progress")
            execution_log.append({
                "type": "todo_start",
                "id": step_id,
                "task": task_desc,
                "tool": tool_hint,
                "worker_role": tool_hint or "none",
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

            step_response = None
            step_log: List[Dict[str, Any]] = []
            first_error: Optional[str] = None

            try:
                step_response, step_log = self._run_step(
                    step_id=step_id,
                    task_desc=task_desc,
                    tool_hint=tool_hint,
                    user_message=user_message,
                    accumulated_context=accumulated_context,
                    todo_mgr=todo_mgr,
                    system_prompt=system_prompt,
                    safe_kwargs=_safe_kwargs,
                )
            except Exception as e:
                first_error = str(e)

            # ── Worker 成功 ──────────────────────────────────────────────
            if step_response is not None and first_error is None:
                for tc in step_log:
                    if tc.get("type") == "tool_call":
                        execution_log.append({
                            "type": "tool_call",
                            "todo_id": step_id,
                            "tool": tc.get("tool"),
                            "input": tc.get("input"),
                            "result_preview": str(tc.get("result", ""))[:200],
                            "timestamp": tc.get("timestamp", ""),
                        })
                execution_log.extend(
                    entry for entry in step_log if entry.get("type") != "tool_call"
                )

                # 从 tool_call 结果中提取真实 result_preview
                _tool_result_preview = ""
                for _tc in step_log:
                    if _tc.get("type") == "tool_call" and _tc.get("tool") in ("write_file", "edit_file", "bash", "read_file"):
                        _raw_result = _tc.get("result", "")
                        try:
                            _r = json.loads(_raw_result)
                            if isinstance(_r, dict) and _r.get("success"):
                                _tool_result_preview = json.dumps(_r, ensure_ascii=False)[:200]
                        except Exception:
                            _tool_result_preview = str(_raw_result)[:200]
                        break

                # 过滤模型输出中的 <todo_list> XML
                _clean_response = step_response
                if step_response and step_response.strip().startswith("<todo_list"):
                    if _tool_result_preview:
                        _clean_response = f"步骤 [{step_id}] 已完成：{_tool_result_preview}"
                    else:
                        _clean_response = f"步骤 [{step_id}] 已完成。"

                result_preview = (_tool_result_preview or str(_clean_response))[:300]
                todo_mgr.update(step_id, "completed", result_preview=result_preview)

                # 累积上下文：tool=none 整理步骤全量保留，其他步骤取前 600 字符
                _is_none_step = not tool_hint or (tool_hint or "").strip() == "none"
                step_context_snippet = str(_clean_response) if _is_none_step else str(_clean_response)[:600]
                accumulated_context += f"\n步骤{step_id}（{task_desc}）结果：{step_context_snippet}\n"

                # 追加 bash 工具原始 stdout
                for _tc in step_log:
                    if _tc.get("type") == "tool_call" and _tc.get("tool") == "bash":
                        try:
                            _bash_result = json.loads(_tc.get("result", "{}"))
                            _stdout = _bash_result.get("stdout", "")
                            if _stdout and len(_stdout) > 50:
                                accumulated_context += f"[步骤{step_id} bash原始输出]\n{_stdout[:30000]}\n"
                        except Exception:
                            pass

                step_results.append({"id": step_id, "task": task_desc, "result": _clean_response})
                execution_log.append({
                    "type": "todo_done",
                    "id": step_id,
                    "status": "completed",
                    "result_preview": result_preview,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                continue

            # ── Worker 失败 → Orchestrator 决策 ─────────────────────────
            error_msg = first_error or "步骤执行返回空结果"
            execution_log.append({
                "type": "todo_failed",
                "id": step_id,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

            decision = self._orchestrator_decide(task_desc, error_msg, accumulated_context)
            action = decision["action"]
            execution_log.append({
                "type": "orchestrator_decision",
                "step_id": step_id,
                "action": action,
                "reason": decision["reason"],
                "modified_task": decision.get("modified_task", ""),
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

            if action == "abort":
                todo_mgr.update(step_id, "failed", error=error_msg)
                step_results.append({"id": step_id, "task": task_desc, "result": f"[终止: {error_msg}]"})
                should_abort = True
                continue

            if action == "retry":
                modified_task = decision.get("modified_task") or task_desc
                execution_log.append({
                    "type": "worker_retry",
                    "step_id": step_id,
                    "original_task": task_desc,
                    "modified_task": modified_task,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                try:
                    step_response, step_log = self._run_step(
                        step_id=step_id,
                        task_desc=modified_task,
                        tool_hint=tool_hint,
                        user_message=user_message,
                        accumulated_context=accumulated_context,
                        todo_mgr=todo_mgr,
                        system_prompt=system_prompt,
                        safe_kwargs=_safe_kwargs,
                    )
                    # 重试成功
                    for tc in step_log:
                        if tc.get("type") == "tool_call":
                            execution_log.append({
                                "type": "tool_call",
                                "todo_id": step_id,
                                "tool": tc.get("tool"),
                                "input": tc.get("input"),
                                "result_preview": str(tc.get("result", ""))[:200],
                                "timestamp": tc.get("timestamp", ""),
                            })
                    execution_log.extend(
                        entry for entry in step_log if entry.get("type") != "tool_call"
                    )
                    result_preview = str(step_response)[:300]
                    todo_mgr.update(step_id, "completed", result_preview=result_preview)
                    step_context_snippet = str(step_response)[:600]
                    accumulated_context += f"\n步骤{step_id}（{modified_task}）结果：{step_context_snippet}\n"
                    for _tc in step_log:
                        if _tc.get("type") == "tool_call" and _tc.get("tool") == "bash":
                            try:
                                _bash_result = json.loads(_tc.get("result", "{}"))
                                _stdout = _bash_result.get("stdout", "")
                                if _stdout and len(_stdout) > 50:
                                    accumulated_context += f"[步骤{step_id} bash原始输出]\n{_stdout[:30000]}\n"
                            except Exception:
                                pass
                    step_results.append({"id": step_id, "task": modified_task, "result": step_response})
                    execution_log.append({
                        "type": "todo_done",
                        "id": step_id,
                        "status": "completed",
                        "retry": True,
                        "result_preview": result_preview,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    })
                except Exception as retry_err:
                    retry_error = str(retry_err)
                    todo_mgr.update(step_id, "failed", error=retry_error)
                    step_results.append({"id": step_id, "task": task_desc, "result": f"[重试失败: {retry_error}]"})
                    execution_log.append({
                        "type": "worker_retry_failed",
                        "step_id": step_id,
                        "error": retry_error,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    })
            else:
                # skip：标记失败，继续下一步
                todo_mgr.update(step_id, "failed", error=error_msg)
                step_results.append({"id": step_id, "task": task_desc, "result": f"[跳过: {error_msg}]"})

        # ── 3. Orchestrator 汇总最终回复 ────────────────────────────────────
        final_response = self._summarize(
            user_message=user_message,
            step_results=step_results,
            system_prompt=system_prompt,
            base_messages=base_messages,
            todo_mgr=todo_mgr,
            aborted=should_abort,
            **_safe_kwargs,
        )

        return {
            "final_response": final_response,
            "todos": [item.to_dict() for item in todo_mgr._items],
            "step_results": step_results,
            "execution_log": execution_log,
            "todo_manager": todo_mgr,
        }

    def _summarize(
        self,
        user_message: str,
        step_results: List[Dict[str, Any]],
        system_prompt: str,
        base_messages: List[Dict[str, str]],
        todo_mgr: Optional["TodoManager"] = None,
        aborted: bool = False,
        **model_kwargs,
    ) -> str:
        """Orchestrator 汇总：将所有步骤结果整合为最终回复。"""
        if not step_results:
            return "任务执行完成，但无具体结果。"

        # 单步骤直接返回
        if len(step_results) == 1:
            result = step_results[0].get("result", "")
            return result if result else "步骤已完成。"

        # 多步骤：统计完成率
        completion_info = ""
        if todo_mgr:
            total = len(todo_mgr._items)
            done = sum(1 for i in todo_mgr._items if i.status == "completed")
            cancelled = sum(1 for i in todo_mgr._items if i.status == "cancelled")
            rate = done / total if total > 0 else 0.0
            completion_info = f"\n\n（任务完成率：{done}/{total} = {rate:.0%}"
            if aborted:
                completion_info += "，任务被 Orchestrator 提前终止"
            if cancelled:
                completion_info += f"，{cancelled} 个步骤已取消"
            completion_info += "）"

        def _fmt_step_result(r: Dict[str, Any]) -> str:
            result_str = str(r.get("result", ""))
            task_str = r.get("task", "")
            step_id = r.get("id", "?")
            # 过滤 <todo_list> XML（模型调了 todo_write 后的回显，不含有效信息）
            if result_str.strip().startswith("<todo_list"):
                result_str = f"[步骤 {step_id} 已完成]"
            # 过滤 bash 原始扫描数据
            elif re.search(r'\S+\.py:\d+:', result_str) and len(result_str) > 400:
                lines = result_str.splitlines()
                head_lines = lines[:5]
                result_str = "\n".join(head_lines) + f"\n...（共 {len(lines)} 行扫描结果，已省略）"
            elif len(result_str) > 800:
                head = result_str[:500]
                tail = result_str[-200:]
                result_str = f"{head}\n...(已省略中间内容)...\n{tail}"
            return f"步骤{step_id}（{task_str}）：{result_str}"

        steps_text = "\n".join(_fmt_step_result(r) for r in step_results)
        abort_hint = "\n\n注意：任务因关键步骤失败被提前终止，以下是已完成部分的结果。" if aborted else ""
        summary_messages = base_messages + [
            {
                "role": "user",
                "content": (
                    f"原始需求：{user_message[:300]}\n\n"
                    f"各步骤执行结果：\n{steps_text}"
                    f"{abort_hint}\n\n"
                    "请基于以上步骤结果，给出完整、连贯的最终回答。不要逐步复述，直接给出结论和关键信息。"
                ),
            }
        ]
        try:
            result = self.model_forward_fn(
                summary_messages,
                system_prompt=system_prompt,
                **model_kwargs,
            )
            return result + completion_info
        except Exception:
            summary = "任务已完成：\n" + "\n".join(
                f"• {r['task']}：{str(r['result'])[:150]}" for r in step_results
            )
            return summary + completion_info

