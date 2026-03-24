#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 中间件模块 - 实现 Middleware Chain 设计模式

所有中间件遵循统一接口 AgentMiddleware，可任意组合注入 QwenAgentFramework。
每个中间件职责单一：
  - before_model   : 模型调用前修改消息列表（注入上下文）
  - after_model    : 模型返回后后处理文本
  - after_tool_call: 工具执行后标准化结果
  - after_run      : 会话结束钩子
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .todo_manager import TodoManager


def _inject_context_before_last_user(
    messages: List[Dict[str, str]],
    context_message: Dict[str, str],
) -> List[Dict[str, str]]:
    """将上下文消息插入到最后一个用户消息前，保持上下文就近可见。"""
    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated.insert(idx, context_message)
            return updated
    updated.append(context_message)
    return updated


# ============================================================================
# 中间件基类
# ============================================================================

class AgentMiddleware:
    """轻量中间件接口，借鉴 DeerFlow 的 middleware chain 设计。"""

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """模型调用前可修改消息列表。"""
        return messages

    def after_model(
        self,
        model_response: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """模型返回后可修正文本。"""
        return model_response

    def after_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """工具执行后可标准化结果。"""
        return tool_result

    def after_run(
        self,
        execution_log: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """会话结束后钩子。"""
        return None


# ============================================================================
# 基础上下文注入中间件（首轮一次性注入模式的公共基类）
# ============================================================================

class _OnceInjectMiddleware(AgentMiddleware):
    """
    模板基类：首轮注入固定提示词，之后不再重复。

    子类只需实现 _should_inject() 和 _build_message() 即可，
    无需重复编写"已注入则跳过/标记注入"的样板代码。
    """

    #: 在 runtime_context 中标记"已注入"的键名，子类可覆盖
    _inject_flag: str = "_once_injected"

    def _should_inject(self, iteration: int, runtime_context: Dict[str, Any]) -> bool:
        """子类可覆盖：决定本次是否注入（基类默认只在首轮注入）。"""
        return iteration == 0

    def _build_message(
        self,
        messages: List[Dict[str, str]],
        runtime_context: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """子类实现：返回要注入的消息 dict，返回 None 则不注入。"""
        raise NotImplementedError

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get(self._inject_flag):
            return messages
        if not self._should_inject(iteration, runtime_context):
            return messages

        msg = self._build_message(messages, runtime_context)
        if msg is None:
            return messages

        runtime_context[self._inject_flag] = True
        return _inject_context_before_last_user(messages, msg)


# ============================================================================
# 运行模式中间件
# ============================================================================

class RuntimeModeMiddleware(AgentMiddleware):
    """模式注入中间件：将运行模式信息注入上下文。"""

    MODE_HINTS = {
        "chat": "当前是纯对话模式：直接回答用户问题，不主动规划工具调用。",
        "tools": "当前是工具模式：优先通过工具收集事实，避免凭空猜测。",
        "skills": "当前是技能模式：优先遵循注入的技能知识完成任务。",
        "hybrid": "当前是混合模式：先利用技能制定方法，再通过工具执行和验证。",
    }

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_runtime_mode_injected"):
            return messages

        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        hint = self.MODE_HINTS.get(run_mode)
        if not hint:
            return messages

        runtime_context["_runtime_mode_injected"] = True
        mode_message = {
            "role": "user",
            "content": (
                f"<runtime_mode name=\"{run_mode}\">\n"
                f"{hint}\n"
                "请按该模式约束完成任务。\n"
                "</runtime_mode>"
            ),
        }
        return _inject_context_before_last_user(messages, mode_message)


# ============================================================================
# 计划模式中间件
# ============================================================================

class PlanModeMiddleware(AgentMiddleware):
    """计划模式中间件：在首轮注入计划约束。

    注意：当 run_mode 为 tools 或 hybrid 时，plan_mode 不注入"先写计划"提示。
    工具模式下模型应直接输出 <tool> 格式调用，而非先写自然语言计划再执行。
    """

    PLAN_HINT = (
        "<plan_mode>\n"
        "当前已开启计划模式。\n"
        "请先输出一个不少于 3 步的计划，每一步给出状态（pending/in_progress/completed）。\n"
        "如果需要工具调用，先写计划再执行，并在最终答复里同步状态。\n"
        "</plan_mode>"
    )

    TOOL_PLAN_HINT = (
        "<plan_mode>\n"
        "当前已开启计划模式，且处于工具模式。\n"
        "请通过工具逐步收集信息，每次工具调用前用一句话说明当前步骤。\n"
        "【重要】每次只调用一个工具，工具执行后根据结果决定：\n"
        "  - 如果信息已足够完成任务：直接输出最终总结，无需再调用工具。\n"
        "  - 如果还需要更多信息：继续调用下一个工具。\n"
        "工具调用必须使用标准格式：<tool>工具名</tool><input>{参数}</input>\n"
        "</plan_mode>"
    )

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if not runtime_context.get("plan_mode"):
            return messages
        if runtime_context.get("_plan_mode_injected"):
            return messages

        runtime_context["_plan_mode_injected"] = True

        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        hint = self.TOOL_PLAN_HINT if run_mode in ("tools", "hybrid") else self.PLAN_HINT

        plan_message = {"role": "user", "content": hint}
        return _inject_context_before_last_user(messages, plan_message)


# ============================================================================
# 技能上下文中间件
# ============================================================================

class SkillsContextMiddleware(AgentMiddleware):
    """技能上下文中间件：在工具模式中保留技能指令。"""

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_skills_context_injected"):
            return messages

        skill_contexts = runtime_context.get("skill_contexts") or []
        if not skill_contexts:
            return messages

        lines = ["<skills_context>", "当前任务可使用以下技能知识：", ""]
        for item in skill_contexts:
            skill_id = str(item.get("id", "unknown"))
            name = str(item.get("name", skill_id))
            desc = str(item.get("description", "")).strip()
            tags = item.get("tags", [])

            lines.append(f"- {name} ({skill_id})")
            if desc:
                lines.append(f"  描述: {desc}")
            if tags:
                tag_text = ", ".join(str(tag) for tag in tags if str(tag).strip())
                if tag_text:
                    lines.append(f"  标签: {tag_text}")
            lines.append("")

        lines.append("请优先参考上述技能步骤，再决定是否调用工具。")
        lines.append("</skills_context>")

        runtime_context["_skills_context_injected"] = True
        skills_message = {"role": "user", "content": "\n".join(lines)}
        return _inject_context_before_last_user(messages, skills_message)


# ============================================================================
# 上传文件中间件
# ============================================================================

class UploadedFilesMiddleware(AgentMiddleware):
    """上传文件中间件：把文件元数据注入上下文。"""

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        size_kb = size / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"
        return f"{size_kb / 1024:.1f} MB"

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_uploaded_files_injected"):
            return messages

        uploaded_files = runtime_context.get("uploaded_files") or []
        if not uploaded_files:
            return messages

        lines = ["<uploaded_files>", "以下文件可用于当前任务：", ""]
        for file_info in uploaded_files:
            filename = str(file_info.get("filename", "unknown"))
            raw_path = str(file_info.get("path", ""))
            path_text = raw_path if raw_path else filename
            size = int(file_info.get("size", 0) or 0)
            lines.append(f"- {filename} ({self._format_size(size)})")
            lines.append(f"  Path: {path_text}")
            lines.append("")

        lines.append("如需读取文件，请优先使用 read_file 或相关工具。")
        lines.append("</uploaded_files>")

        runtime_context["_uploaded_files_injected"] = True
        upload_message = {"role": "user", "content": "\n".join(lines)}
        return _inject_context_before_last_user(messages, upload_message)


# ============================================================================
# 工具结果守卫中间件
# ============================================================================

class ToolResultGuardMiddleware(AgentMiddleware):
    """工具结果守卫：统一工具结果结构，降低模型误判。"""

    def after_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = {
                "success": False,
                "error": "工具返回非 JSON 字符串",
                "raw_result": tool_result,
            }

        if isinstance(payload, dict):
            if "error" in payload:
                payload.setdefault("success", False)
            else:
                payload.setdefault("success", True)
            payload.setdefault("tool", tool_name)
            payload.setdefault("timestamp", timestamp)
            return json.dumps(payload, ensure_ascii=False)

        return json.dumps(
            {
                "success": True,
                "tool": tool_name,
                "timestamp": timestamp,
                "result": payload,
            },
            ensure_ascii=False,
        )


# ============================================================================
# TODO 上下文中间件
# ============================================================================

class TodoContextMiddleware(AgentMiddleware):
    """
    Claude Code 风格的 TODO 上下文中间件。

    每次 before_model 时将当前 TodoManager 的状态注入消息列表，
    让模型感知当前执行到哪一步。

    与 PlanModeMiddleware 的区别：
      PlanModeMiddleware  → 只注入"请先规划"的提示词，不跟踪状态
      TodoContextMiddleware → 注入实时 TODO 状态，形成真正的状态感知循环
    """

    def __init__(self, todo_manager: "TodoManager", worker_mode: bool = False):
        self.todo_manager = todo_manager
        # worker_mode=True：只显示 in_progress + completed，隐藏 pending
        self.worker_mode = worker_mode

    def _build_context_content(self) -> str:
        """构建注入内容（策略模式：worker/orchestrator 两种渲染路径）。"""
        if self.worker_mode:
            todo_text = self.todo_manager.render_for_context_worker()
            current_step_lines = [
                f"  ▶ [{item.id}] {item.task}"
                for item in self.todo_manager._items
                if item.status == "in_progress"
            ]
            in_progress_hint = (
                ("\n当前正在执行：\n" + "\n".join(current_step_lines))
                if current_step_lines
                else ""
            )
            return (
                f"{todo_text}\n"
                f"当前正在执行任务计划中的单个步骤。{in_progress_hint}\n"
                "请只完成上方标注 ▶ 的当前步骤，任务完成后直接返回结果，"
                "不要调用 todo_write，不要描述后续步骤。"
            )
        else:
            todo_text = self.todo_manager.render_for_context(show_completed=True)
            return (
                f"{todo_text}\n"
                "当前正在执行任务计划。请聚焦于下一个 in_progress 或 pending 步骤。\n"
                "完成某步骤后，可调用 todo_write 更新状态：\n"
                '<tool>todo_write</tool><input>{"action": "update", "id": <步骤ID>, "status": "completed"}</input>'
            )

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}

        if not self.todo_manager._items:
            return messages
        # 所有任务完成后，iteration > 0 时不再注入（iteration=0 仍注入，给模型最终状态感知）
        if self.todo_manager.all_done() and iteration > 0:
            return messages

        context_message = {"role": "user", "content": self._build_context_content()}
        return _inject_context_before_last_user(messages, context_message)

    def after_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """拦截 todo_write 工具调用，直接处理 TODO 状态更新。"""
        if tool_name == "todo_write":
            return self.todo_manager.apply_tool_write(tool_input)
        return tool_result


class EnhancedTodoContextMiddleware(TodoContextMiddleware):
    """
    增强版 TodoContextMiddleware，含上下文丢失检测。

    当对话被压缩/截断导致 TODO 状态消失时，自动重新注入恢复提示。
    （借鉴 DeerFlow TodoMiddleware 的上下文丢失检测机制）
    """

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}

        if not self.todo_manager._items:
            return messages
        if self.todo_manager.all_done() and iteration > 0:
            return messages

        todo_text = self.todo_manager.render_for_context(show_completed=True)

        # 检测消息历史中是否已有 TODO 上下文
        has_todo_in_context = any(
            "<todo_list" in msg.get("content", "") or "todo_write" in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        )

        # 已有 TODO 且非首轮 → 跳过（避免重复注入）
        if has_todo_in_context and iteration > 0:
            return messages

        # 上下文丢失（找不到 TODO）且非首轮 → 注入恢复提示
        if not has_todo_in_context and iteration > 0:
            context_content = (
                "<system_reminder>\n"
                "任务计划状态（上下文压缩后恢复）：\n"
                f"{todo_text}\n"
                "请继续追踪并更新任务状态。\n"
                "</system_reminder>"
            )
        else:
            # 首轮 → 标准注入
            context_content = (
                f"{todo_text}\n"
                "当前正在执行任务计划。请聚焦于下一个 in_progress 或 pending 步骤。\n"
                "完成某步骤后，可调用 todo_write 更新状态：\n"
                '<tool>todo_write</tool><input>{"action": "update", "id": <步骤ID>, "status": "completed"}</input>'
            )

        context_message = {"role": "user", "content": context_content}
        return _inject_context_before_last_user(messages, context_message)


def make_todo_context_middleware(todo_manager: "TodoManager") -> "TodoContextMiddleware":
    """
    创建增强版 TodoContextMiddleware（含上下文丢失检测）工厂函数。

    Returns:
        EnhancedTodoContextMiddleware 实例
    """
    return EnhancedTodoContextMiddleware(todo_manager)


# ============================================================================
# 记忆注入中间件
# ============================================================================

class MemoryInjectionMiddleware(AgentMiddleware):
    """
    记忆注入中间件（借鉴 DeerFlow MemoryMiddleware 架构思路）。

    工作原理：
      1. before_model 时将记忆注入上下文（仅首轮，避免重复）
      2. after_run 时可选触发记忆更新（auto_update=True）
      3. 记忆以 <memory> XML 标签包裹，与工具结果区分
    """

    def __init__(
        self,
        memory_manager,  # MemoryManager 实例（避免循环导入用 Any 类型）
        auto_update: bool = False,
        model_forward_fn=None,
    ):
        self.memory_manager = memory_manager
        self.auto_update = auto_update
        self.model_forward_fn = model_forward_fn

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_memory_injected"):
            return messages

        memory_text = self.memory_manager.format_for_injection()
        runtime_context["_memory_injected"] = True

        if not memory_text:
            return messages

        memory_message = {
            "role": "user",
            "content": (
                "<memory>\n"
                "以下是关于你和用户过往交互的记忆摘要，请参考但不要直接引用：\n\n"
                f"{memory_text}\n"
                "</memory>"
            ),
        }
        return _inject_context_before_last_user(messages, memory_message)

    def after_run(
        self,
        execution_log: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """会话结束时可选触发记忆更新。"""
        if not self.auto_update:
            return
        if runtime_context is None:
            runtime_context = {}
        conversation = runtime_context.get("_conversation_pairs", [])
        if not conversation:
            return

        session_id = runtime_context.get("execution_id")
        try:
            self.memory_manager.update_from_conversation(
                conversation=conversation,
                session_id=session_id,
                model_forward_fn=self.model_forward_fn,
            )
        except Exception as e:
            print(f"[MemoryInjectionMiddleware] 记忆更新失败: {e}")


# ============================================================================
# 对话摘要中间件
# ============================================================================

class ConversationSummaryMiddleware(AgentMiddleware):
    """
    对话摘要中间件：当历史消息超过阈值时压缩早期对话。

    设计目标（借鉴 DeerFlow SummarizationMiddleware）：
      1. 保留最近 N 轮原始对话（模型处理时序关系需要）
      2. 对超出阈值的早期对话生成摘要，插入上下文头部
      3. 支持 LLM 语义摘要和规则摘要两种模式
    """

    def __init__(
        self,
        max_history_pairs: int = 8,
        keep_recent_pairs: int = 4,
        model_forward_fn=None,
    ):
        self.max_history_pairs = max_history_pairs
        self.keep_recent_pairs = keep_recent_pairs
        self.model_forward_fn = model_forward_fn

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}

        # 提取消息中的用户-助手对
        pairs: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                if pending_user is not None:
                    pairs.append((pending_user, ""))
                pending_user = content
            elif role == "assistant":
                if pending_user is not None:
                    pairs.append((pending_user, content))
                    pending_user = None

        # 未超过阈值，无需压缩
        if len(pairs) <= self.max_history_pairs:
            return messages

        # 超出阈值：压缩早期对话
        old_pairs = pairs[: len(pairs) - self.keep_recent_pairs]
        recent_pairs = pairs[len(pairs) - self.keep_recent_pairs:]

        summary_text = self._summarize(old_pairs)

        # 重建消息列表：摘要 + 最近轮次 + 当前用户消息
        new_messages: List[Dict[str, str]] = []
        if summary_text:
            new_messages.append({
                "role": "user",
                "content": (
                    "<conversation_summary>\n"
                    f"以下是早期对话的摘要（共 {len(old_pairs)} 轮）：\n"
                    f"{summary_text}\n"
                    "</conversation_summary>"
                ),
            })
            new_messages.append({
                "role": "assistant",
                "content": "好的，我了解了之前对话的背景。",
            })

        for u, a in recent_pairs:
            new_messages.append({"role": "user", "content": u})
            if a:
                new_messages.append({"role": "assistant", "content": a})

        if pending_user is not None:
            new_messages.append({"role": "user", "content": pending_user})

        runtime_context["_summary_compressed_pairs"] = len(old_pairs)
        return new_messages

    def _summarize(self, pairs: List[Tuple[str, str]]) -> str:
        """生成早期对话摘要：优先 LLM，降级为规则摘要。"""
        if self.model_forward_fn is not None:
            return self._llm_summarize(pairs)
        return self._rule_summarize(pairs)

    def _llm_summarize(self, pairs: List[Tuple[str, str]]) -> str:
        """调用 LLM 生成语义摘要。"""
        SUMMARY_SYSTEM = (
            "你是一个对话摘要助手。将以下多轮对话总结为 3-5 句话的摘要，"
            "保留关键信息（用户需求、已完成的操作、重要结论）。只输出摘要，不加任何前缀。"
        )
        conv_lines: List[str] = []
        for u, a in pairs[-6:]:
            if u:
                conv_lines.append(f"用户: {u[:150]}")
            if a:
                conv_lines.append(f"助手: {a[:150]}")

        try:
            return self.model_forward_fn(
                [{"role": "user", "content": "\n".join(conv_lines)}],
                system_prompt=SUMMARY_SYSTEM,
                temperature=0.3,
                top_p=0.9,
                max_tokens=200,
            )
        except Exception:
            return self._rule_summarize(pairs)

    @staticmethod
    def _rule_summarize(pairs: List[Tuple[str, str]]) -> str:
        """规则摘要：提取关键词 + 轮次计数，无 LLM 开销。"""
        if not pairs:
            return ""
        topics: List[str] = []
        for u, _ in pairs[-4:]:
            if u:
                snippet = u.strip()[:50].replace("\n", " ")
                if snippet:
                    topics.append(snippet)
        summary = f"早期对话共 {len(pairs)} 轮。"
        if topics:
            summary += "主要话题包括：" + "；".join(topics[:3]) + "。"
        return summary


# ============================================================================
# gstack 架构移植中间件
# ============================================================================

class CompletenessMiddleware(_OnceInjectMiddleware):
    """
    完整性原则中间件（移植自 gstack Boil the Lake 哲学）

    AI 辅助编码使完整实现的边际成本趋近于零。
    当"完整方案 A"与"捷径方案 B"仅差几十行代码时，始终选择 A。
    """

    _inject_flag = "_completeness_injected"

    COMPLETENESS_HINT = (
        "<completeness_principle>\n"
        "【完整性原则 - Boil the Lake】\n"
        "AI 辅助编码使完整实现的边际成本趋近于零。在呈现选项时：\n\n"
        "✅ 如果方案 A 是完整实现（100% 覆盖 + 边界情况），方案 B 是跳过边界情况的捷径：\n"
        "   → 始终推荐 A。70行代码的差距在 AI 时代只需几秒。\n\n"
        "✅ 「湖」vs「海」区分：\n"
        "   - 「湖」是可以煮沸的：100% 测试覆盖、完整功能实现、所有错误路径 → 全部做完\n"
        "   - 「海」是煮不沸的：重写整个系统、修改你无法控制的依赖 → 标记为超出范围\n\n"
        "✅ 估算工作量时，同时给出两个维度：\n"
        "   人工团队时间 / AI辅助时间（如：3天人工 / 30分钟AI）\n\n"
        "❌ 反模式（禁止）：\n"
        "   - [选B——代码少10%但覆盖90%价值]（若A只多70行，选A）\n"
        "   - [暂时跳过边界情况以节省时间]（边界情况处理只需几分钟）\n"
        "   - [测试留到后续PR]（测试是最容易煮沸的湖）\n"
        "   - 只报人工时间：[这需要2周] → 应该说：[2周人工 / ~1小时AI辅助]\n"
        "</completeness_principle>"
    )

    def _should_inject(self, iteration: int, runtime_context: Dict[str, Any]) -> bool:
        # 仅工具/混合/技能模式的首轮注入
        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        return iteration == 0 and run_mode in ("tools", "hybrid", "skills")

    def _build_message(
        self,
        messages: List[Dict[str, str]],
        runtime_context: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        return {"role": "user", "content": self.COMPLETENESS_HINT}


class AskUserQuestionMiddleware(AgentMiddleware):
    """
    结构化提问中间件（移植自 gstack AskUserQuestion Format）

    检测模型提问意图，注入结构化提问格式要求（Re-ground / Simplify / Recommend / Options）。
    """

    QUESTION_SIGNALS = [
        "请问", "是否", "需要确认", "请确认", "您是否", "你是否",
        "需要我", "要不要", "是否需要", "请选择", "可以告诉我",
        "请提供", "需要澄清", "需要更多信息", "还不清楚",
        "应该", "还是", "哪种", "哪个方案",
    ]

    FORMAT_HINT = (
        "<ask_user_question_format>\n"
        "【结构化提问格式】当你需要向用户提问时，必须按以下结构组织提问：\n\n"
        "1. **Re-ground（背景重申）**：用1-2句话说明当前项目、上下文和计划任务\n"
        "   示例：'当前正在分析 /src/api.py，目标是重构错误处理逻辑。'\n\n"
        "2. **Simplify（简化说明）**：用简洁、无术语的语言解释遇到的问题\n"
        "   要求：不要出现函数名、内部术语，用具体例子和类比\n"
        "   说明它能做什么，而不是它叫什么\n\n"
        "3. **Recommend（推荐）**：`RECOMMENDATION: 选[X]，因为[一行原因]`\n"
        "   → 优先推荐完整方案（参考完整性原则）\n"
        "   → 每个选项标注完整度评分：`完整度: X/10`\n"
        "     校准参考：10=完整实现（所有边界），7=涵盖主路径但跳过部分边界，3=快捷方式\n\n"
        "4. **Options（字母选项）**：`A) ... B) ... C) ...`\n"
        "   → 涉及工作量时同时显示两个维度：`（人工: ~X / AI辅助: ~Y）`\n\n"
        "提示：假设用户在过去20分钟内没有看过这个窗口，没有打开代码。\n"
        "如果需要阅读源码才能理解你的解释，说明太复杂了，需要再简化。\n"
        "</ask_user_question_format>"
    )

    def after_model(
        self,
        model_response: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """检测提问意图，标记下一轮需要注入格式提示。"""
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_ask_format_injected"):
            return model_response

        text = (model_response or "").strip()
        has_question = any(signal in text for signal in self.QUESTION_SIGNALS)
        if has_question or text.endswith("？") or text.endswith("?"):
            runtime_context["_pending_ask_format_injection"] = True
        return model_response

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if not runtime_context.get("_pending_ask_format_injection"):
            return messages
        if runtime_context.get("_ask_format_injected"):
            return messages

        runtime_context["_ask_format_injected"] = True
        runtime_context["_pending_ask_format_injection"] = False

        format_message = {"role": "user", "content": self.FORMAT_HINT}
        return _inject_context_before_last_user(messages, format_message)


class CompletionStatusMiddleware(AgentMiddleware):
    """
    完成状态协议中间件（移植自 gstack Completion Status Protocol）

    强制要求 Agent 用 DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT 汇报结果。
    """

    STATUS_PROTOCOL_HINT = (
        "<completion_status_protocol>\n"
        "【完成状态报告格式】完成工作流时，必须用以下状态之一汇报：\n\n"
        "- **DONE**：所有步骤成功完成。每个声明都需要有依据（不能只说'完成了'）\n"
        "- **DONE_WITH_CONCERNS**：已完成，但有问题需告知用户。逐条列出每个关注点\n"
        "- **BLOCKED**：无法继续。说明什么在阻塞以及已尝试了什么\n"
        "- **NEEDS_CONTEXT**：缺少必要信息。精确说明需要什么才能继续\n\n"
        "上报格式（BLOCKED 或 NEEDS_CONTEXT 时使用）：\n"
        "```\n"
        "STATUS: BLOCKED | NEEDS_CONTEXT\n"
        "REASON: [1-2句话]\n"
        "ATTEMPTED: [尝试过的内容]\n"
        "RECOMMENDATION: [用户应该做什么]\n"
        "```\n\n"
        "升级规则：\n"
        "- 同一任务失败3次 → 停止并上报 BLOCKED\n"
        "- 对安全敏感变更不确定 → 停止并上报\n"
        "- 工作范围超出可验证范围 → 停止并上报\n"
        "坏的工作比没有工作更糟糕。上报不会受到惩罚。\n"
        "</completion_status_protocol>"
    )

    BLOCKED_SIGNALS = [
        "无法继续", "无法完成", "遇到问题", "权限不足", "文件不存在",
        "找不到", "缺少", "无法访问", "失败了", "错误", "异常",
        "cannot", "failed", "error", "blocked", "unable to",
        "不确定", "不清楚", "需要更多",
    ]

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if runtime_context is None:
            runtime_context = {}
        if runtime_context.get("_completion_status_injected"):
            return messages

        if iteration == 0:
            runtime_context["_completion_status_injected"] = True
            status_message = {"role": "user", "content": self.STATUS_PROTOCOL_HINT}
            return _inject_context_before_last_user(messages, status_message)

        return messages

    def after_model(
        self,
        model_response: str,
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """检测阻塞信号，统计失败次数。"""
        if runtime_context is None:
            runtime_context = {}
        text = (model_response or "").lower()
        if any(signal in text for signal in self.BLOCKED_SIGNALS):
            count = runtime_context.get("_blocked_attempt_count", 0) + 1
            runtime_context["_blocked_attempt_count"] = count
        return model_response


class SearchBeforeBuildingMiddleware(_OnceInjectMiddleware):
    """
    搜索优先中间件（移植自 gstack Search Before Building 哲学）

    在构建任何基础设施前先搜索，提供三层知识体系框架。
    """

    _inject_flag = "_search_first_injected"

    BUILD_SIGNALS = [
        "实现", "开发", "构建", "创建", "设计", "编写",
        "implement", "build", "create", "develop", "design",
        "框架", "库", "组件", "模块", "系统", "架构",
        "自定义", "从头开始", "重新实现", "重写",
    ]

    SEARCH_BEFORE_BUILDING_HINT = (
        "<search_before_building>\n"
        "【搜索优先原则】在构建任何基础设施或不熟悉的模式之前，先确认是否已有现成方案。\n\n"
        "三层知识体系：\n"
        "- **Layer 1（经典可靠）**：成熟的标准模式。检查成本为零。\n"
        "  偶尔质疑'理所当然'的方案，那里可能有突破点。\n"
        "- **Layer 2（新颖流行）**：当前最佳实践、博客、生态趋势。\n"
        "  搜索这些，但要批判性审视——搜索结果是你思考的输入，不是答案。\n"
        "- **Layer 3（第一性原理）**：通过对特定问题推理得出的原创观察。\n"
        "  这是最有价值的。当第一性原理揭示常规做法是错的，大声命名它：\n"
        '  "EUREKA: 大家因为[假设]做X。但[证据]显示这是错的，Y更好因为[推理]。"\n\n'
        "搜索不可用时：跳过搜索步骤，并注明：\n"
        '"搜索不可用 — 仅基于已知知识进行分析。"\n'
        "</search_before_building>"
    )

    def _should_inject(self, iteration: int, runtime_context: Dict[str, Any]) -> bool:
        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        return iteration == 0 and run_mode in ("tools", "hybrid")

    def _build_message(
        self,
        messages: List[Dict[str, str]],
        runtime_context: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        # 只在检测到"构建意图"时注入
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        if not any(signal in last_user_content for signal in self.BUILD_SIGNALS):
            return None

        return {"role": "user", "content": self.SEARCH_BEFORE_BUILDING_HINT}


class RepoOwnershipMiddleware(_OnceInjectMiddleware):
    """
    仓库所有权模式中间件（移植自 gstack Repo Ownership Mode）

    区分 solo / collaborative / unknown 三种模式，决定 Agent 主动程度。
    """

    _inject_flag = "_repo_mode_injected"

    MODE_HINTS = {
        "solo": (
            "<repo_ownership_mode mode=\"solo\">\n"
            "【仓库所有权 - 独立开发模式（Solo）】\n"
            "这是一个主要由你维护的仓库。当你在工作流步骤中发现当前修改范围之外的问题时：\n"
            "- 测试失败、废弃警告、安全建议、lint 错误、死代码、环境问题\n"
            "→ 主动调查并提议修复。这个独立开发者是唯一会修复它的人，不要等待。\n"
            "默认行动，而不是询问。\n\n"
            "【See Something, Say Something】无论在哪个步骤发现问题，都要简短说明：\n"
            "一句话：发现了什么 + 影响是什么 + '需要我来修复吗？'\n"
            "绝不让发现的问题悄悄溜走。\n"
            "</repo_ownership_mode>"
        ),
        "collaborative": (
            "<repo_ownership_mode mode=\"collaborative\">\n"
            "【仓库所有权 - 协作开发模式（Collaborative）】\n"
            "这个仓库有多个活跃贡献者。当你发现当前修改范围之外的问题时：\n"
            "→ 通过提问告知用户——那可能是其他人的职责。\n"
            "默认询问，而不是直接修改。\n\n"
            "【See Something, Say Something】无论在哪个步骤发现问题，都要简短说明：\n"
            "一句话：发现了什么 + 影响是什么。然后继续，不要主动修复。\n"
            "绝不让发现的问题悄悄溜走。\n"
            "</repo_ownership_mode>"
        ),
        "unknown": (
            "<repo_ownership_mode mode=\"unknown\">\n"
            "【仓库所有权 - 未知模式（Unknown）】\n"
            "仓库所有权未配置，按协作开发模式处理（更安全的默认值）。\n"
            "发现范围外问题时：通知用户，不要主动修复。\n"
            "可通过 runtime_context['repo_mode'] = 'solo' | 'collaborative' 配置。\n"
            "</repo_ownership_mode>"
        ),
    }

    def _build_message(
        self,
        messages: List[Dict[str, str]],
        runtime_context: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        repo_mode = str(runtime_context.get("repo_mode", "unknown")).lower()
        hint = self.MODE_HINTS.get(repo_mode, self.MODE_HINTS["unknown"])
        return {"role": "user", "content": hint}

