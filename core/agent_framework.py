#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Agent 框架 - 为 Qwen 模型添加工具调用能力
包含 Agent 循环、工具执行和响应生成
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from .agent_tools import ToolExecutor, ToolParser


def _inject_context_before_last_user(messages: List[Dict[str, str]], context_message: Dict[str, str]) -> List[Dict[str, str]]:
    """将上下文消息插入到最后一个用户消息前，保持上下文就近可见。"""
    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated.insert(idx, context_message)
            return updated
    updated.append(context_message)
    return updated


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
        runtime_context = runtime_context or {}
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


class PlanModeMiddleware(AgentMiddleware):
    """计划模式中间件：在首轮注入计划约束。"""

    PLAN_HINT = (
        "<plan_mode>\n"
        "当前已开启计划模式。\n"
        "请先输出一个不少于 3 步的计划，每一步给出状态（pending/in_progress/completed）。\n"
        "如果需要工具调用，先写计划再执行，并在最终答复里同步状态。\n"
        "</plan_mode>"
    )

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        runtime_context = runtime_context or {}
        if not runtime_context.get("plan_mode"):
            return messages
        if runtime_context.get("_plan_mode_injected"):
            return messages

        runtime_context["_plan_mode_injected"] = True
        plan_message = {"role": "user", "content": self.PLAN_HINT}
        return _inject_context_before_last_user(messages, plan_message)


class SkillsContextMiddleware(AgentMiddleware):
    """技能上下文中间件：在工具模式中保留技能指令。"""

    def before_model(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        runtime_context = runtime_context or {}
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
        runtime_context = runtime_context or {}
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


class QwenAgentFramework:
    """
    Qwen Agent 框架 - 为 Qwen 模型添加工具调用和任务规划能力

    核心循环:
    1. 用户输入 + 历史
    2. 模型生成响应(可能包含工具调用)
    3. 解析和执行工具
    4. 将结果返回给模型
    5. 重复直到任务完成
    """

    def __init__(
        self,
        model_forward_fn,  # 模型前向函数: fn(messages, system_prompt) -> str
        work_dir: Optional[str] = None,
        enable_bash: bool = False,
        max_iterations: int = 10,
        tools_in_system_prompt: bool = True,
        middlewares: Optional[List[AgentMiddleware]] = None,
        default_runtime_context: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 Agent 框架

        Args:
            model_forward_fn: 模型前向函数,接收 (messages, system_prompt) 返回生成文本
            work_dir: 工作目录 (用于工具执行)
            enable_bash: 是否启用 bash 工具 (需谨慎使用)
            max_iterations: 最大循环次数 (防止无限循环)
            tools_in_system_prompt: 是否在系统提示词中包含工具信息
            middlewares: 可选中间件链（DeerFlow 风格）
            default_runtime_context: 默认运行时上下文
        """
        self.model_forward_fn = model_forward_fn
        self.tool_executor = ToolExecutor(work_dir=work_dir, enable_bash=enable_bash)
        self.tool_parser = ToolParser()
        self.max_iterations = max_iterations
        self.tools_in_system_prompt = tools_in_system_prompt
        self.middlewares = list(middlewares) if middlewares else []
        self.default_runtime_context = deepcopy(default_runtime_context) if default_runtime_context else {}

        # 构建系统提示词
        self.system_prompt_template = self._build_system_prompt()

    def set_middlewares(self, middlewares: List[AgentMiddleware]):
        """设置完整中间件链。"""
        self.middlewares = list(middlewares)

    def add_middleware(self, middleware: AgentMiddleware):
        """追加一个中间件。"""
        self.middlewares.append(middleware)

    def get_middlewares(self) -> List[AgentMiddleware]:
        """获取当前中间件列表。"""
        return list(self.middlewares)

    def _build_runtime_context(self, runtime_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建当前请求的运行时上下文。"""
        merged = deepcopy(self.default_runtime_context)
        if runtime_context:
            for key, value in runtime_context.items():
                merged[key] = deepcopy(value)

        merged.setdefault("run_mode", "chat")
        merged.setdefault("plan_mode", False)
        merged.setdefault("uploaded_files", [])
        merged.setdefault("middleware_errors", [])
        merged.setdefault("execution_id", datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))
        return merged

    def _build_system_prompt(self) -> str:
        """构建包含工具信息的系统提示词"""
        if not self.tools_in_system_prompt:
            return ""

        tools_info = self._format_tools_info()
        return f"""你是一个能够使用工具的智能助手。

可用的工具:
{tools_info}

当你需要使用工具时,请按照以下格式输出:
<tool>tool_name</tool>
<input>{{"parameter": "value"}}</input>

然后我会执行这个工具并将结果返回给你。继续处理任务直到完成。

记住:
- 优先使用工具而不是猜测
- 在执行复杂操作前先探索文件结构
- 如果不确定路径,先使用 list_dir
"""

    def _format_tools_info(self) -> str:
        """格式化工具信息用于系统提示词"""
        tools = self.tool_executor.get_tools()
        info = []

        for tool in tools:
            info.append(f"- **{tool['name']}**: {tool['description']}")
            if "properties" in tool.get("input_schema", {}):
                props = tool["input_schema"]["properties"]
                for prop_name, prop_info in props.items():
                    desc = prop_info.get("description", "")
                    info.append(f"    - {prop_name}: {desc}")

        return "\n".join(info)

    def _resolve_system_prompt(
        self,
        system_prompt_override: Optional[str],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """解析最终系统提示词，避免覆盖工具指令。"""
        override = (system_prompt_override or "").strip()
        if not override:
            return self.system_prompt_template

        if not self.system_prompt_template:
            return override

        run_mode = str((runtime_context or {}).get("run_mode", "chat")).lower()
        if run_mode in {"tools", "hybrid"}:
            return f"{self.system_prompt_template}\n\n{override}"

        return override

    @staticmethod
    def _clean_path_candidate(raw_value: str) -> str:
        """清理路径候选值，去掉常见包裹符号。"""
        return raw_value.strip().strip("`\"'").rstrip("，。；：,.!?)]}>\n\r\t ")

    def _extract_path_candidate(self, text: str) -> Optional[str]:
        """从用户文本中提取最可能的文件路径。"""
        if not text:
            return None

        path_patterns = [
            r"`([^`]+)`",
            r"(/[^\s\"'`<>]+)",
            r"((?:\./|\.\./)?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9]{1,10})",
        ]
        excluded_tokens = {"read_file", "write_file", "edit_file", "list_dir", "bash"}

        for pattern in path_patterns:
            for match in re.findall(pattern, text):
                candidate = self._clean_path_candidate(match)
                if not candidate or " " in candidate:
                    continue
                if candidate.lower() in excluded_tokens:
                    continue
                return candidate

        return None

    def _infer_fallback_tool_calls(
        self,
        user_message: str,
        runtime_context: Dict[str, Any],
        iteration: int,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """当模型未按格式调用工具时，尝试基于用户意图兜底。"""
        if iteration != 0:
            return []

        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        if run_mode not in {"tools", "hybrid"}:
            return []

        message = (user_message or "").strip()
        if not message:
            return []

        path_candidate = self._extract_path_candidate(message)
        if not path_candidate:
            return []

        is_path_only = self._clean_path_candidate(message) == path_candidate
        has_read_intent = (
            "read_file" in message.lower()
            or "读取" in message
            or "阅读" in message
            or ("read" in message.lower() and "file" in message.lower())
        )

        if not (is_path_only or has_read_intent):
            return []

        return [("read_file", {"path": path_candidate})]

    def _apply_before_model_middlewares(
        self,
        messages: List[Dict[str, str]],
        iteration: int,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """按顺序执行 before_model 钩子。"""
        updated_messages = messages
        for middleware in self.middlewares:
            try:
                updated_messages = middleware.before_model(updated_messages, iteration, runtime_context)
            except Exception as error:  # pragma: no cover - 容错兜底
                runtime_context.setdefault("middleware_errors", []).append(
                    {
                        "middleware": middleware.__class__.__name__,
                        "hook": "before_model",
                        "error": str(error),
                    }
                )
        return updated_messages

    def _apply_after_model_middlewares(
        self,
        model_response: str,
        iteration: int,
        runtime_context: Dict[str, Any],
    ) -> str:
        """按顺序执行 after_model 钩子。"""
        updated_response = model_response
        for middleware in self.middlewares:
            try:
                updated_response = middleware.after_model(updated_response, iteration, runtime_context)
            except Exception as error:  # pragma: no cover - 容错兜底
                runtime_context.setdefault("middleware_errors", []).append(
                    {
                        "middleware": middleware.__class__.__name__,
                        "hook": "after_model",
                        "error": str(error),
                    }
                )
        return updated_response

    @staticmethod
    def _normalize_space(text: str) -> str:
        """压缩空白字符，便于检测重复文本。"""
        return re.sub(r"\s+", " ", text or "").strip()

    def _should_trigger_anti_repeat_guard(self, model_response: str, runtime_context: Dict[str, Any]) -> bool:
        """判断是否命中复述工具结果的模式。"""
        if runtime_context.get("_anti_repeat_guard_used"):
            return False

        summary = runtime_context.get("_last_read_file_summary")
        if not isinstance(summary, dict):
            return False

        text = (model_response or "").strip()
        if not text:
            return False

        repetition_markers = [
            "工具执行结果如下",
            "工具 'read_file' 的执行结果",
            "工具 `read_file` 的执行结果",
            "\"content\"",
            "```json",
            "...[内容已截断",
        ]
        if any(marker in text for marker in repetition_markers):
            return True

        if text.count("```") >= 2 and len(text) > 240:
            return True

        preview = str(summary.get("preview", "")).strip()
        if preview:
            normalized_preview = self._normalize_space(preview)
            normalized_text = self._normalize_space(text)
            probe = normalized_preview[:120]
            if len(probe) >= 40 and probe in normalized_text:
                return True

        return False

    def _rewrite_with_anti_repeat_guard(
        self,
        messages: List[Dict[str, str]],
        model_response: str,
        system_prompt: str,
        iteration: int,
        runtime_context: Dict[str, Any],
        **model_kwargs,
    ) -> str:
        """触发一次强约束重写，避免输出复述工具原文。"""
        summary = runtime_context.get("_last_read_file_summary") or {}
        sections = summary.get("section_headings") or []
        sections_text = ", ".join(str(item) for item in sections[:6]) if sections else "未识别章节"

        guard_message = {
            "role": "user",
            "content": (
                "<anti_repeat_guard>\n"
                "你刚才复述了工具执行结果，请重写最终回答。\n"
                "硬性要求：\n"
                "1) 禁止出现‘工具执行结果如下’等表述。\n"
                "2) 禁止输出 JSON、代码块、目录树或大段原文。\n"
                "3) 仅输出：核心结论（1-2句）+ 关键要点（最多3条）+ 后续建议（可选1条）。\n"
                "4) 若信息不足，请明确指出还需读取的章节。\n"
                f"当前文档标题：{summary.get('title', '未知')}\n"
                f"可参考章节：{sections_text}\n"
                "</anti_repeat_guard>"
            ),
        }

        rewrite_messages = [dict(message) for message in messages]
        rewrite_messages.append({"role": "assistant", "content": model_response})
        rewrite_messages.append(guard_message)

        rewritten = self.model_forward_fn(rewrite_messages, system_prompt, **model_kwargs)
        rewritten = self._apply_after_model_middlewares(rewritten, iteration, runtime_context)
        runtime_context["_anti_repeat_guard_used"] = True
        return rewritten

    def _apply_after_tool_call_middlewares(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: str,
        iteration: int,
        runtime_context: Dict[str, Any],
    ) -> str:
        """按顺序执行 after_tool_call 钩子。"""
        updated_result = tool_result
        for middleware in self.middlewares:
            try:
                updated_result = middleware.after_tool_call(
                    tool_name,
                    tool_input,
                    updated_result,
                    iteration,
                    runtime_context,
                )
            except Exception as error:  # pragma: no cover - 容错兜底
                runtime_context.setdefault("middleware_errors", []).append(
                    {
                        "middleware": middleware.__class__.__name__,
                        "hook": "after_tool_call",
                        "error": str(error),
                        "tool": tool_name,
                    }
                )
        return updated_result

    def _finalize_middlewares(self, execution_log: List[Dict[str, Any]], runtime_context: Dict[str, Any]) -> None:
        """会话结束时执行 after_run 钩子。"""
        for middleware in self.middlewares:
            try:
                middleware.after_run(execution_log, runtime_context)
            except Exception as error:  # pragma: no cover - 容错兜底
                runtime_context.setdefault("middleware_errors", []).append(
                    {
                        "middleware": middleware.__class__.__name__,
                        "hook": "after_run",
                        "error": str(error),
                    }
                )

    def process_message(
        self,
        user_message: str,
        history: List[Tuple[str, str]],
        system_prompt_override: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        return_runtime_context: bool = False,
        **model_kwargs,
    ) -> Union[Tuple[str, List[Dict[str, Any]]], Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        处理用户消息并执行 Agent 循环

        Args:
            user_message: 用户输入
            history: 对话历史 [(user_msg, assistant_msg), ...]
            system_prompt_override: 覆盖系统提示词
            runtime_context: 本次执行的运行时上下文
            return_runtime_context: 是否额外返回运行时上下文
            **model_kwargs: 传递给模型的其他参数

        Returns:
            默认返回 (最终响应, 执行记录)
            当 return_runtime_context=True 时返回 (最终响应, 执行记录, 运行时上下文)
        """
        messages = self._build_messages(user_message, history)
        runtime_ctx = self._build_runtime_context(runtime_context)
        system_prompt = self._resolve_system_prompt(system_prompt_override, runtime_ctx)

        execution_log: List[Dict[str, Any]] = []
        model_response = ""

        for iteration in range(self.max_iterations):
            messages = self._apply_before_model_middlewares(messages, iteration, runtime_ctx)

            model_response = self.model_forward_fn(messages, system_prompt, **model_kwargs)
            model_response = self._apply_after_model_middlewares(model_response, iteration, runtime_ctx)

            if self._should_trigger_anti_repeat_guard(model_response, runtime_ctx):
                original_response = model_response
                model_response = self._rewrite_with_anti_repeat_guard(
                    messages,
                    model_response,
                    system_prompt,
                    iteration,
                    runtime_ctx,
                    **model_kwargs,
                )
                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "anti_repeat_rewrite",
                        "original_preview": original_response[:240],
                        "rewritten_preview": model_response[:240],
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                )

            execution_log.append(
                {
                    "iteration": iteration,
                    "type": "model_response",
                    "content": model_response,
                    "run_mode": runtime_ctx.get("run_mode", "chat"),
                    "plan_mode": bool(runtime_ctx.get("plan_mode", False)),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
            )

            tool_calls = self.tool_parser.parse_tool_calls(model_response)
            if not tool_calls:
                tool_calls = self._infer_fallback_tool_calls(user_message, runtime_ctx, iteration)
                if tool_calls:
                    execution_log.append(
                        {
                            "iteration": iteration,
                            "type": "tool_fallback",
                            "tool_calls": [
                                {"tool": name, "input": payload} for name, payload in tool_calls
                            ],
                            "reason": "模型未输出可解析工具格式，触发文件读取兜底",
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }
                    )
                else:
                    self._finalize_middlewares(execution_log, runtime_ctx)
                    if return_runtime_context:
                        return model_response, execution_log, runtime_ctx
                    return model_response, execution_log

            tool_results = []
            for tool_name, tool_input in tool_calls:
                raw_result = self.tool_executor.execute_tool(tool_name, tool_input)
                guarded_result = self._apply_after_tool_call_middlewares(
                    tool_name,
                    tool_input,
                    raw_result,
                    iteration,
                    runtime_ctx,
                )
                tool_results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": guarded_result,
                })

                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                        "result": guarded_result,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                )

            messages.append({"role": "assistant", "content": model_response})
            result_text = self._format_tool_results(tool_results, runtime_ctx)
            messages.append({"role": "user", "content": result_text})

        final_response = f"已达到最大迭代次数({self.max_iterations})。最后的模型响应:\n\n{model_response}"
        self._finalize_middlewares(execution_log, runtime_ctx)
        if return_runtime_context:
            return final_response, execution_log, runtime_ctx
        return final_response, execution_log

    def process_message_stream(
        self,
        user_message: str,
        history: List[Tuple[str, str]],
        system_prompt_override: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        流式处理用户消息 (用于实时显示)

        Yields:
            (流式文本, 执行信息)
        """
        messages = self._build_messages(user_message, history)
        runtime_ctx = self._build_runtime_context(runtime_context)
        system_prompt = self._resolve_system_prompt(system_prompt_override, runtime_ctx)
        execution_log: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            messages = self._apply_before_model_middlewares(messages, iteration, runtime_ctx)

            model_response = self.model_forward_fn(messages, system_prompt, **model_kwargs)
            model_response = self._apply_after_model_middlewares(model_response, iteration, runtime_ctx)

            if self._should_trigger_anti_repeat_guard(model_response, runtime_ctx):
                original_response = model_response
                model_response = self._rewrite_with_anti_repeat_guard(
                    messages,
                    model_response,
                    system_prompt,
                    iteration,
                    runtime_ctx,
                    **model_kwargs,
                )
                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "anti_repeat_rewrite",
                        "original_preview": original_response[:240],
                        "rewritten_preview": model_response[:240],
                    }
                )

            execution_log.append(
                {
                    "iteration": iteration,
                    "type": "model_response",
                    "content": model_response,
                }
            )
            yield model_response, {"iteration": iteration, "type": "model_response"}

            tool_calls = self.tool_parser.parse_tool_calls(model_response)
            if not tool_calls:
                tool_calls = self._infer_fallback_tool_calls(user_message, runtime_ctx, iteration)
                if tool_calls:
                    execution_log.append(
                        {
                            "iteration": iteration,
                            "type": "tool_fallback",
                            "tool_calls": [
                                {"tool": name, "input": payload} for name, payload in tool_calls
                            ],
                            "reason": "模型未输出可解析工具格式，触发文件读取兜底",
                        }
                    )
                else:
                    self._finalize_middlewares(execution_log, runtime_ctx)
                    return

            tool_results = []
            for tool_name, tool_input in tool_calls:
                raw_result = self.tool_executor.execute_tool(tool_name, tool_input)
                guarded_result = self._apply_after_tool_call_middlewares(
                    tool_name,
                    tool_input,
                    raw_result,
                    iteration,
                    runtime_ctx,
                )

                tool_results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": guarded_result,
                })
                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                        "result": guarded_result,
                    }
                )

                yield f"[执行工具] {tool_name}\n", {
                    "iteration": iteration,
                    "type": "tool_execution",
                    "tool": tool_name,
                }

            messages.append({"role": "assistant", "content": model_response})
            result_text = self._format_tool_results(tool_results, runtime_ctx)
            messages.append({"role": "user", "content": result_text})

        self._finalize_middlewares(execution_log, runtime_ctx)

    def _build_messages(self, user_message: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []

        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 1800) -> Tuple[str, bool]:
        """按字符截断文本，保留头尾上下文。"""
        if len(text) <= max_chars:
            return text, False

        head_chars = int(max_chars * 0.75)
        tail_chars = max_chars - head_chars
        omitted_chars = len(text) - max_chars
        truncated = (
            text[:head_chars]
            + f"\n\n...[内容已截断，省略 {omitted_chars} 个字符]...\n\n"
            + text[-tail_chars:]
        )
        return truncated, True

    @staticmethod
    def _extract_markdown_headings(content: str, limit: int = 8) -> List[str]:
        """从 Markdown 文本中提取章节标题。"""
        headings: List[str] = []
        for raw_line in content.splitlines():
            match = re.match(r"^#{1,6}\s+(.*)$", raw_line.strip())
            if not match:
                continue

            heading = match.group(1).strip()
            if not heading:
                continue

            headings.append(heading)
            if len(headings) >= limit:
                break

        return headings

    @staticmethod
    def _first_non_empty_line(content: str) -> str:
        """返回首个非空行。"""
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line:
                return line
        return ""

    def _build_read_file_summary(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """将 read_file 原文转换为结构化摘要。"""
        path = str(result_data.get("path", ""))
        content = result_data.get("content")
        if not isinstance(content, str):
            content = ""

        section_headings = self._extract_markdown_headings(content, limit=8)
        title = section_headings[0] if section_headings else self._first_non_empty_line(content)
        if not title:
            title = path or "untitled"

        normalized_content = content.replace("\r\n", "\n").strip()
        preview_source = normalized_content or "[空文件或无可读文本]"
        preview, _ = self._truncate_text(preview_source, max_chars=260)

        return {
            "title": title[:120],
            "section_headings": section_headings,
            "line_count": content.count("\n") + (1 if content else 0),
            "char_count": len(content),
            "preview": preview,
        }

    def _prepare_tool_result_for_model(self, tool_name: str, result_data: Any) -> Tuple[Any, bool]:
        """压缩工具结果，避免把超长原文直接回灌给模型。"""
        if tool_name == "read_file" and isinstance(result_data, dict):
            summary = self._build_read_file_summary(result_data)
            structured_result = {
                "success": bool(result_data.get("success", True)),
                "path": str(result_data.get("path", "")),
                "summary": summary,
                "note": "read_file 结果已转为结构化摘要，未注入原文全文。",
            }
            return structured_result, True

        try:
            serialized = json.dumps(result_data, ensure_ascii=False)
        except TypeError:
            serialized = str(result_data)

        preview, truncated = self._truncate_text(serialized, max_chars=1200)
        if truncated:
            return {
                "truncated": True,
                "preview": preview,
                "note": "结果过长，已截断。",
            }, True

        return result_data, False

    def _format_tool_results(
        self,
        tool_results: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """格式化工具结果用于返回给模型。"""
        formatted: List[str] = []
        has_read_file = False
        read_file_structured = False

        for result_info in tool_results:
            tool_name = result_info["tool"]
            result_str = result_info["result"]

            try:
                result_data = json.loads(result_str)
            except json.JSONDecodeError:
                result_data = {"raw": result_str}

            prepared_result, was_structured = self._prepare_tool_result_for_model(tool_name, result_data)

            formatted.append(f"工具 '{tool_name}' 的执行结果:")
            formatted.append(json.dumps(prepared_result, ensure_ascii=False, indent=2))
            formatted.append("")

            if tool_name == "read_file":
                has_read_file = True
                read_file_structured = read_file_structured or was_structured
                if runtime_context is not None and isinstance(prepared_result, dict):
                    summary_payload = prepared_result.get("summary")
                    if isinstance(summary_payload, dict):
                        runtime_context["_last_read_file_summary"] = dict(summary_payload)
                        runtime_context["_anti_repeat_guard_used"] = False

        if has_read_file:
            formatted.append("请基于上面的结构化摘要直接给出结论。")
            formatted.append("禁止复述工具执行过程、JSON 字段或大段原文。")
            formatted.append("输出格式：核心结论(1-2句) + 关键要点(最多3条)。")
            if read_file_structured:
                formatted.append("若摘要信息不足，请明确指出需要继续读取的章节标题。")
            formatted.append("")

        return "\n".join(formatted)

    def enable_tools_in_prompt(self, enable: bool):
        """动态启用/禁用工具信息在系统提示词中"""
        self.tools_in_system_prompt = enable


# ============================================================================
# 用于与 Qwen 模型集成的适配器
# ============================================================================

def create_qwen_model_forward(qwen_agent, system_prompt_base: str = ""):
    """
    创建 Qwen 模型前向函数

    Args:
        qwen_agent: QwenAgent 实例
        system_prompt_base: 基础系统提示词

    Returns:
        适配的前向函数
    """

    def _consume_stream(stream) -> str:
        response_text = ""
        for token in stream:
            response_text = token
        return response_text

    def forward(messages: List[Dict[str, str]], system_prompt: str = "", **kwargs) -> str:
        combined_system_prompt = system_prompt_base
        if system_prompt:
            combined_system_prompt = f"{combined_system_prompt}\n\n{system_prompt}"

        full_messages = [{"role": "system", "content": combined_system_prompt}] + messages

        if hasattr(qwen_agent, "generate_stream_with_messages"):
            stream = qwen_agent.generate_stream_with_messages(
                full_messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return _consume_stream(stream)

        if hasattr(qwen_agent, "generate_stream_text"):
            stream = qwen_agent.generate_stream_text(
                full_messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return _consume_stream(stream)

        if hasattr(qwen_agent, "generate_stream"):
            chat_messages = [m for m in full_messages if m.get("role") != "system"]
            if not chat_messages:
                return ""

            latest_user = ""
            if chat_messages[-1].get("role") == "user":
                latest_user = chat_messages[-1].get("content", "")
                history_messages = chat_messages[:-1]
            else:
                history_messages = chat_messages

            history_pairs: List[Tuple[str, str]] = []
            pending_user = None
            for message in history_messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    if pending_user is not None:
                        history_pairs.append((pending_user, ""))
                    pending_user = content
                elif role == "assistant":
                    if pending_user is None:
                        history_pairs.append(("", content))
                    else:
                        history_pairs.append((pending_user, content))
                        pending_user = None

            if pending_user is not None:
                history_pairs.append((pending_user, ""))

            stream = qwen_agent.generate_stream(
                latest_user,
                history_pairs,
                system_prompt=combined_system_prompt,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return _consume_stream(stream)

        raise AttributeError(
            "qwen_agent 缺少可用的流式接口，请实现 generate_stream_with_messages、"
            "generate_stream_text 或 generate_stream。"
        )

    return forward


if __name__ == "__main__":
    print("✅ Agent 框架模块加载成功")
    print("📚 可用类: QwenAgentFramework, AgentMiddleware")

