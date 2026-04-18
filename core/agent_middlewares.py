#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 中间件模块 - 实现 Middleware Chain 设计模式"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from core.monitor_logger import get_monitor_logger


class AgentMiddleware(ABC):
    @abstractmethod
    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        pass

    async def process_after_llm(self, response: str, context: Dict[str, Any]) -> str:
        return response

    async def process_before_tool(self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return tool_name, tool_args

    async def process_after_tool(self, tool_name: str, tool_args: Dict[str, Any], result: str, context: Dict[str, Any]) -> str:
        return result

    async def process_on_error(self, error: Exception, phase: str, context: Dict[str, Any]) -> Optional[str]:
        return None


def _inject_context_before_last_user(messages: List[Dict[str, str]], context_message: Dict[str, str]) -> List[Dict[str, str]]:
    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated.insert(idx, context_message) # 前面插入信息
            return updated
    updated.append(context_message)
    return updated


class _OnceInjectMiddleware(AgentMiddleware):
    _inject_flag: str = "_once_injected"

    def _should_inject(self, iteration: int, runtime_context: Dict[str, Any]) -> bool:
        return iteration == 0

    def _build_message(self, messages: List[Dict[str, str]], runtime_context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        raise NotImplementedError

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if context.get(self._inject_flag):
            return messages
        if not self._should_inject(context.get("iteration", 0), context):
            return messages
        msg = self._build_message(messages, context)
        if msg is None:
            return messages
        context[self._inject_flag] = True
        return _inject_context_before_last_user(messages, msg)


class RuntimeModeMiddleware(AgentMiddleware):
    MODE_HINTS = {
        "chat": "当前是纯对话模式：直接回答用户问题，不主动规划工具调用。",
        "tools": "当前是工具模式：优先通过工具收集事实，避免凭空猜测。",
        "skills": "当前是技能模式：优先遵循注入的技能知识完成任务。",
        "hybrid": "当前是混合模式：先利用技能制定方法，再通过工具执行和验证。",
    }

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if context.get("_runtime_mode_injected"):
            return messages
        run_mode = str(context.get("run_mode", "chat")).lower()
        hint = self.MODE_HINTS.get(run_mode)
        if not hint:
            return messages
        context["_runtime_mode_injected"] = True
        mode_message = {"role": "user", "content": f"<runtime_mode name=\"{run_mode}\">\n{hint}\n请按该模式约束完成任务。\n</runtime_mode>"}
        return _inject_context_before_last_user(messages, mode_message)


class PlanModeMiddleware(AgentMiddleware):
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

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if not context.get("plan_mode"):
            return messages
        if context.get("_plan_mode_injected"):
            return messages
        context["_plan_mode_injected"] = True
        run_mode = str(context.get("run_mode", "chat")).lower()
        hint = self.TOOL_PLAN_HINT if run_mode in ("tools", "hybrid") else self.PLAN_HINT
        plan_message = {"role": "user", "content": hint}
        return _inject_context_before_last_user(messages, plan_message)


class SkillsContextMiddleware(AgentMiddleware):
    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if context.get("_skills_context_injected"):
            return messages
        skill_contexts = context.get("skill_contexts") or []
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
        context["_skills_context_injected"] = True
        skills_message = {"role": "user", "content": "\n".join(lines)}
        return _inject_context_before_last_user(messages, skills_message)


class UploadedFilesMiddleware(AgentMiddleware):
    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        size_kb = size / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"
        return f"{size_kb / 1024:.1f} MB"

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if context.get("_uploaded_files_injected"):
            return messages
        uploaded_files = context.get("uploaded_files") or []
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
        context["_uploaded_files_injected"] = True
        upload_message = {"role": "user", "content": "\n".join(lines)}
        return _inject_context_before_last_user(messages, upload_message)


class ToolResultGuardMiddleware(AgentMiddleware):
    """工具结果守卫中间件。

    额外功能：
    1. 防止 write_file append 重复写入：追踪每个文件的 append 内容哈希，
       发现重复时拦截并返回警告，避免 API.md 等文件出现内容叠加。
    2. 标准化工具结果 JSON 格式（原有逻辑保持不变）。
    """

    def __init__(self):
        # key: 文件路径, value: set of content hash（追踪已 append 过的内容）
        self._append_history: Dict[str, set] = {}
        self.monitor = get_monitor_logger()

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        return messages

    async def process_before_tool(self, tool_name: str, tool_input: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """拦截重复 append 写入。"""
        if tool_name != "write_file":
            return tool_name, tool_input

        mode = tool_input.get("mode", "overwrite")
        path = tool_input.get("path", "")
        content = tool_input.get("content", "")

        if mode == "append" and path and content:
            content_hash = hash(content.strip())
            if path not in self._append_history:
                self._append_history[path] = set()
            if content_hash in self._append_history[path]:
                # 将工具名替换为 _noop 占位，并在结果中返回警告
                # 实际上通过修改 tool_input 注入一个空内容来跳过写入
                self.monitor.info(f"拦截重复 append 写入: {path} (内容哈希已存在)")
                tool_input = dict(tool_input)
                tool_input["_duplicate_append_blocked"] = True
                tool_input["content"] = ""  # 空内容写入，实际无副作用
            else:
                self._append_history[path].add(content_hash)
        elif mode == "overwrite" and path:
            # overwrite 重置该文件的 append 历史
            self._append_history.pop(path, None)

        return tool_name, tool_input

    async def process_after_tool(self, tool_name: str, tool_input: Dict[str, Any], tool_result: str, context: Dict[str, Any]) -> str:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # 拦截了重复 append 的情况：返回友好提示而非空写入结果
        if tool_name == "write_file" and tool_input.get("_duplicate_append_blocked"):
            self.monitor.debug(f"返回重复 append 拦截提示: {tool_input.get('path')}")
            return json.dumps({
                "success": True,
                "tool": tool_name,
                "timestamp": timestamp,
                "output": f"⚠️ 已跳过重复 append：文件 {tool_input.get('path', '')} 已包含相同内容，无需再次写入。"
            }, ensure_ascii=False)

        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = {"success": False, "error": "工具返回非 JSON 字符串", "raw_result": tool_result}
        if isinstance(payload, dict):
            if "error" in payload:
                payload.setdefault("success", False)
            else:
                payload.setdefault("success", True)
            payload.setdefault("tool", tool_name)
            payload.setdefault("timestamp", timestamp)
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps({"success": True, "tool": tool_name, "timestamp": timestamp, "result": payload}, ensure_ascii=False)


class ConversationSummaryMiddleware(AgentMiddleware):
    def __init__(self, max_history_pairs: int = 8, keep_recent_pairs: int = 4, model_forward_fn=None):
        self.max_history_pairs = max_history_pairs
        self.keep_recent_pairs = keep_recent_pairs
        self.model_forward_fn = model_forward_fn

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        pairs = []
        pending_user = None
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
        if len(pairs) <= self.max_history_pairs:
            return messages
        old_pairs = pairs[: len(pairs) - self.keep_recent_pairs]
        recent_pairs = pairs[len(pairs) - self.keep_recent_pairs:]
        summary_text = self._summarize(old_pairs)
        new_messages = []
        if summary_text:
            new_messages.append({"role": "user", "content": f"<conversation_summary>\n以下是早期对话的结构化摘要（共 {len(old_pairs)} 轮）：\n{summary_text}\n</conversation_summary>"})
            new_messages.append({"role": "assistant", "content": "好的，我了解了之前对话的背景。"})
        for u, a in recent_pairs:
            new_messages.append({"role": "user", "content": u})
            if a:
                new_messages.append({"role": "assistant", "content": a})
        if pending_user is not None:
            new_messages.append({"role": "user", "content": pending_user})
        context["_summary_compressed_pairs"] = len(old_pairs)
        return new_messages

    def _summarize(self, pairs: List[Tuple[str, str]]) -> str:
        if self.model_forward_fn is not None:
            return self._llm_summarize(pairs)
        return self._rule_summarize(pairs)

    def _llm_summarize(self, pairs: List[Tuple[str, str]]) -> str:
        SUMMARY_SYSTEM = (
            "你是一个对话摘要助手。请将以下多轮对话压缩为结构化 JSON，"
            "字段仅限 user_goals/completed_actions/failures/open_questions。"
            "每个字段都是字符串数组，只输出 JSON。"
        )
        conv_lines = []
        for u, a in pairs[-6:]:
            if u:
                conv_lines.append(f"用户: {u[:150]}")
            if a:
                conv_lines.append(f"助手: {a[:150]}")
        try:
            raw = self.model_forward_fn([{"role": "user", "content": "\n".join(conv_lines)}], system_prompt=SUMMARY_SYSTEM, temperature=0.3, top_p=0.9, max_tokens=240)
            json_start = raw.find("{")
            json_end = raw.rfind("}")
            if json_start != -1 and json_end > json_start:
                data = json.loads(raw[json_start:json_end + 1])
                return self._format_summary_dict(data, len(pairs))
        except Exception:
            pass
        return self._rule_summarize(pairs)

    @classmethod
    def _rule_summarize(cls, pairs: List[Tuple[str, str]]) -> str:
        if not pairs:
            return ""
        user_goals = []
        completed_actions = []
        failures = []
        open_questions = []

        for u, a in pairs[-6:]:
            if u:
                snippet = u.strip().replace("\n", " ")[:80]
                if snippet:
                    user_goals.append(snippet)
            if a:
                normalized = a.strip().replace("\n", " ")[:100]
                if any(k in normalized for k in ("失败", "错误", "超时", "不存在", "无法")):
                    failures.append(normalized)
                elif any(k in normalized for k in ("完成", "已", "成功", "找到", "读取")):
                    completed_actions.append(normalized)
                else:
                    open_questions.append(normalized)

        data = {
            "user_goals": user_goals[:3],
            "completed_actions": completed_actions[:3],
            "failures": failures[:3],
            "open_questions": open_questions[:3],
        }
        return cls._format_summary_dict(data, len(pairs))

    @staticmethod
    def _format_summary_dict(data: Dict[str, Any], pair_count: int) -> str:
        normalized = {
            "turn_count": pair_count,
            "user_goals": [str(x)[:120] for x in data.get("user_goals", []) if str(x).strip()],
            "completed_actions": [str(x)[:120] for x in data.get("completed_actions", []) if str(x).strip()],
            "failures": [str(x)[:120] for x in data.get("failures", []) if str(x).strip()],
            "open_questions": [str(x)[:120] for x in data.get("open_questions", []) if str(x).strip()],
        }
        return json.dumps(normalized, ensure_ascii=False, indent=2)


class ContextWindowMiddleware(AgentMiddleware):
    """按消息预算裁剪上下文，优先保留 system、最近对话和关键工具结果。"""

    def __init__(self, max_chars: int = 12000, max_messages: int = 16):
        self.max_chars = max_chars
        self.max_messages = max_messages
        self.monitor = get_monitor_logger()

    @staticmethod
    def _message_size(msg: Dict[str, str]) -> int:
        return len(msg.get("content", "")) + len(msg.get("role", "")) + 16

    @staticmethod
    def _is_critical_message(msg: Dict[str, str]) -> bool:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            return True
        if role == "user" and (
            content.startswith("✅ 工具执行成功")
            or content.startswith("❌")
            or "<conversation_summary>" in content
            or "<runtime_mode" in content
            or "<plan_mode>" in content
            or "<uploaded_files>" in content
            or "<skills_context>" in content
        ):
            return True
        return False

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if len(messages) <= self.max_messages:
            total_chars = sum(self._message_size(m) for m in messages)
            if total_chars <= self.max_chars:
                return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]
        critical_msgs = [m for m in non_system_msgs[:-6] if self._is_critical_message(m)]
        recent_msgs = non_system_msgs[-6:]

        trimmed: List[Dict[str, str]] = []
        seen_ids = set()
        for msg in system_msgs + critical_msgs + recent_msgs:
            msg_id = id(msg)
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)
            trimmed.append(msg)

        total_chars = sum(self._message_size(m) for m in trimmed)
        if total_chars > self.max_chars:
            # 从最早的非 system 消息开始裁剪，始终保留最近 3 条和所有 system。
            preserved_tail = trimmed[-3:] if len(trimmed) > 3 else list(trimmed)
            preserved_tail_ids = {id(m) for m in preserved_tail}
            compact = []
            for msg in trimmed:
                if msg.get("role") == "system" or id(msg) in preserved_tail_ids:
                    compact.append(msg)
            running_chars = sum(self._message_size(m) for m in compact)
            for msg in reversed(trimmed):
                if id(msg) in {id(m) for m in compact}:
                    continue
                msg_size = self._message_size(msg)
                if running_chars + msg_size > self.max_chars:
                    continue
                compact.insert(-len(preserved_tail) if preserved_tail else len(compact), msg)
                running_chars += msg_size
            trimmed = compact
            total_chars = running_chars

        if len(trimmed) < len(messages):
            context["_context_window_trimmed"] = {
                "before_messages": len(messages),
                "after_messages": len(trimmed),
                "before_chars": sum(self._message_size(m) for m in messages),
                "after_chars": total_chars,
            }
            self.monitor.debug(
                f"上下文窗口裁剪: {len(messages)} -> {len(trimmed)} 条消息, "
                f"{sum(self._message_size(m) for m in messages)} -> {total_chars} chars"
            )
        return trimmed


class CompletenessMiddleware(_OnceInjectMiddleware):
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
        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        return iteration == 0 and run_mode in ("tools", "hybrid", "skills")

    def _build_message(self, messages: List[Dict[str, str]], runtime_context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return {"role": "user", "content": self.COMPLETENESS_HINT}


class AskUserQuestionMiddleware(AgentMiddleware):
    QUESTION_SIGNALS = ["请问", "是否", "需要确认", "请确认", "您是否", "你是否", "需要我", "要不要", "是否需要", "请选择", "可以告诉我", "请提供", "需要澄清", "需要更多信息", "还不清楚", "应该", "还是", "哪种", "哪个方案"]
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

    async def process_after_llm(self, response: str, context: Dict[str, Any]) -> str:
        if context.get("_ask_format_injected"):
            return response
        text = (response or "").strip()
        has_question = any(signal in text for signal in self.QUESTION_SIGNALS)
        if has_question or text.endswith("？") or text.endswith("?"):
            context["_pending_ask_format_injection"] = True
        return response

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if not context.get("_pending_ask_format_injection"):
            return messages
        if context.get("_ask_format_injected"):
            return messages
        context["_ask_format_injected"] = True
        context["_pending_ask_format_injection"] = False
        format_message = {"role": "user", "content": self.FORMAT_HINT}
        return _inject_context_before_last_user(messages, format_message)


class CompletionStatusMiddleware(AgentMiddleware):
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
    BLOCKED_SIGNALS = ["无法继续", "无法完成", "遇到问题", "权限不足", "文件不存在", "找不到", "缺少", "无法访问", "失败了", "错误", "异常", "cannot", "failed", "error", "blocked", "unable to", "不确定", "不清楚", "需要更多"]

    async def process_before_llm(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> List[Dict[str, str]]:
        if context.get("_completion_status_injected"):
            return messages
        if context.get("iteration", 0) == 0:
            context["_completion_status_injected"] = True
            status_message = {"role": "user", "content": self.STATUS_PROTOCOL_HINT}
            return _inject_context_before_last_user(messages, status_message)
        return messages

    async def process_after_llm(self, response: str, context: Dict[str, Any]) -> str:
        text = (response or "").lower()
        if any(signal in text for signal in self.BLOCKED_SIGNALS):
            count = context.get("_blocked_attempt_count", 0) + 1
            context["_blocked_attempt_count"] = count
        return response


class SearchBeforeBuildingMiddleware(_OnceInjectMiddleware):
    _inject_flag = "_search_first_injected"
    BUILD_SIGNALS = ["实现", "开发", "构建", "创建", "设计", "编写", "implement", "build", "create", "develop", "design", "框架", "库", "组件", "模块", "系统", "架构", "自定义", "从头开始", "重新实现", "重写"]
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

    def _build_message(self, messages: List[Dict[str, str]], runtime_context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break
        if not any(signal in last_user_content for signal in self.BUILD_SIGNALS):
            return None
        return {"role": "user", "content": self.SEARCH_BEFORE_BUILDING_HINT}


class RepoOwnershipMiddleware(_OnceInjectMiddleware):
    _inject_flag = "_repo_mode_injected"
    MODE_HINTS = {
        "solo": ("<repo_ownership_mode mode=\"solo\">\n【仓库所有权 - 独立开发模式（Solo）】\n这是一个主要由你维护的仓库。当你在工作流步骤中发现当前修改范围之外的问题时：\n- 测试失败、废弃警告、安全建议、lint 错误、死代码、环境问题\n→ 主动调查并提议修复。这个独立开发者是唯一会修复它的人，不要等待。\n默认行动，而不是询问。\n\n【See Something, Say Something】无论在哪个步骤发现问题，都要简短说明：\n一句话：发现了什么 + 影响是什么 + '需要我来修复吗？'\n绝不让发现的问题悄悄溜走。\n</repo_ownership_mode>"),
        "collaborative": ("<repo_ownership_mode mode=\"collaborative\">\n【仓库所有权 - 协作开发模式（Collaborative）】\n这个仓库有多个活跃贡献者。当你发现当前修改范围之外的问题时：\n→ 通过提问告知用户——那可能是其他人的职责。\n默认询问，而不是直接修改。\n\n【See Something, Say Something】无论在哪个步骤发现问题，都要简短说明：\n一句话：发现了什么 + 影响是什么。然后继续，不要主动修复。\n绝不让发现的问题悄悄溜走。\n</repo_ownership_mode>"),
        "unknown": ("<repo_ownership_mode mode=\"unknown\">\n【仓库所有权 - 未知模式（Unknown）】\n仓库所有权未配置，按协作开发模式处理（更安全的默认值）。\n发现范围外问题时：通知用户，不要主动修复。\n可通过 runtime_context['repo_mode'] = 'solo' | 'collaborative' 配置。\n</repo_ownership_mode>"),
    }

    def _build_message(self, messages: List[Dict[str, str]], runtime_context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        repo_mode = str(runtime_context.get("repo_mode", "unknown")).lower()
        hint = self.MODE_HINTS.get(repo_mode, self.MODE_HINTS["unknown"])
        return {"role": "user", "content": hint}
