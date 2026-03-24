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

# ============================================================================
# 子模块导入 - 向后兼容重导出（已迁移至独立子模块）
# ============================================================================
from .agent_middlewares import (
    AgentMiddleware,
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
        # 用于检测重复工具调用；存储已执行过的 (tool_name, input_hash)
        merged.setdefault("_executed_tool_calls", set())
        return merged

    def _build_system_prompt(self) -> str:
        """构建包含工具信息和 gstack 架构原则的系统提示词。

        集成了 gstack ETHOS.md / SKILL.md 的核心设计哲学：
        - 完整性原则（Boil the Lake）
        - 完成状态协议（DONE/BLOCKED/NEEDS_CONTEXT）
        - 搜索优先（Search Before Building）
        - 结构化提问（AskUserQuestion Format）
        这些原则通过中间件（Middleware）按需动态注入，此处系统提示词仅保留
        工具调用规则和基础身份定义，保持简洁以节省 Token。
        """
        if not self.tools_in_system_prompt:
            return ""

        tools_info = self._format_tools_info()
        return f"""你是一个能够使用工具的智能助手。设计原则来源于 gstack 架构。

可用的工具:
{tools_info}

【重要】调用工具时，必须严格使用以下 XML 标签格式，不得使用任何其他格式：

✅ 正确格式（必须遵守）:
<tool>read_file</tool>
<input>{{"path": "README.md"}}</input>

✅ 多参数示例:
<tool>write_file</tool>
<input>{{"path": "output.txt", "content": "hello"}}</input>

❌ 错误格式（禁止使用）:
read_file
{{"path": "README.md"}}

❌ 错误格式（禁止使用）:
```json
{{"tool": "read_file", "path": "README.md"}}
```

工具调用规则:
1. 需要调用工具时，直接输出 <tool>...</tool><input>...</input>，可以在调用前用一句话说明当前步骤
2. <input> 标签内必须是合法的 JSON 对象
3. 工具执行完后，我会将结果返回给你，你再决定是继续调用工具还是给出最终回答
4. 【重要】只有在信息已经足够完成任务时，才给出最终回答；否则继续调用所需工具
5. 优先使用工具而不是猜测或假设文件内容
6. 【重要】读取文件前，若不确定文件的完整路径，必须先调用 list_dir 探索目录结构，再调用 read_file，不得直接猜测路径
7. 【重要】list_dir 只显示一层目录。若目标文件不在当前层，必须对返回结果中每个 type=dir 的子目录继续调用 list_dir，逐层深入，直到找到目标文件
8. 【重要】当 list_dir 返回 hint 字段时，说明还有未展开的子目录，必须继续探索，不得直接放弃
9. 当 read_file 返回 file_not_found=true 时，说明文件不存在（而非空文件），必须先调用 list_dir 查找正确路径后再重试
10. 空文件（文件存在但内容为空）会返回 warning 字段，而非 file_not_found，两者不可混淆
11. 【重要】工具结果中的 _sys_note 字段是系统内部注释，仅供参考，不要在最终回答中直接引用或提及该字段的内容

核心行为准则（来自 gstack 架构）:
A. 【完整性】AI 辅助使完整实现成本趋近于零。选择完整方案而非捷径。估算工作量时同时给出人工时间和 AI 辅助时间。
B. 【状态汇报】完成任务时用 DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT 之一汇报。每个声明都要有证据。
C. 【搜索优先】构建任何功能前，先确认是否已有现成方案（Layer1：标准模式；Layer2：流行方案；Layer3：第一性原理）。
D. 【主动沟通】发现问题时（See Something, Say Something）立即简短说明：一句话描述发现了什么和影响是什么。
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
        model_response: str = "",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """当模型未按格式调用工具时，尝试基于用户意图兜底。

        注意：fallback 仅在以下条件同时满足时触发：
          1. 用户消息中包含可识别的独立文件路径（无自然语言附加）
          2. 用户明确表达了读取/列目录意图

        这样可避免把"读取 /path/to/file 内容并总结"中的附加文字
        也拼入路径参数，导致路径错误。
        """
        if iteration != 0:
            return []

        run_mode = str(runtime_context.get("run_mode", "chat")).lower()
        if run_mode not in {"tools", "hybrid"}:
            return []

        message = (user_message or "").strip()
        if not message:
            return []

        # 从用户消息中提取路径
        path_candidate = self._extract_path_candidate(message)
        if not path_candidate:
            return []

        # 严格校验：路径后不能跟随自然语言（如"内容并总结"）
        # 找到路径在消息中的位置，检查路径之后是否有非路径文字
        path_pos = message.find(path_candidate)
        if path_pos >= 0:
            after_path = message[path_pos + len(path_candidate):].strip()
            # 如果路径之后还有汉字或有意义的单词，说明用户写的是自然语言描述，不是纯路径指令
            if after_path and re.search(r'[\u4e00-\u9fff]|[a-zA-Z]{2,}', after_path):
                return []

        # 必须有明确的读取意图关键词
        has_read_intent = (
            "read_file" in message.lower()
            or "读取" in message
            or "阅读" in message
            or "查看" in message
            or ("read" in message.lower() and "file" in message.lower())
        )
        has_list_intent = (
            "list_dir" in message.lower()
            or "列目录" in message
            or "列出" in message
            or "目录结构" in message
        )

        if has_list_intent:
            return [("list_dir", {"path": path_candidate})]

        if has_read_intent:
            # 防止 fallback 重复触发同一文件（已在上下文中读取过）
            already_read = runtime_context.get("_read_file_paths", set())
            if path_candidate in already_read:
                return []
            return [("read_file", {"path": path_candidate})]

        # 如果消息本身就只是路径（无其他文字），也触发 read_file
        if self._clean_path_candidate(message) == path_candidate:
            already_read = runtime_context.get("_read_file_paths", set())
            if path_candidate in already_read:
                return []
            return [("read_file", {"path": path_candidate})]

        return []

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
        """判断是否命中复述工具结果的模式。

        排除条件：
        - guard 已使用过
        - 无读文件摘要上下文
        - 模型输出是工具调用（应让框架去执行，不是复述）
        """
        if runtime_context.get("_anti_repeat_guard_used"):
            return False

        summary = runtime_context.get("_last_read_file_summary")
        if not isinstance(summary, dict):
            return False

        text = (model_response or "").strip()
        if not text:
            return False

        # 关键排除：模型输出包含工具调用格式时，说明模型在调用工具而非复述内容，不应触发 guard
        if self.tool_parser.parse_tool_calls(text):
            return False

        repetition_markers = [
            "工具执行结果如下",
            "工具 'read_file' 的执行结果",
            "工具 `read_file` 的执行结果",
            "...[内容已截断",
        ]
        if any(marker in text for marker in repetition_markers):
            return True

        preview = str(summary.get("preview", "")).strip()
        if preview:
            normalized_preview = self._normalize_space(preview)
            normalized_text = self._normalize_space(text)
            # 跳过通用文件头（shebang / 编码声明），避免误判正常总结
            # 这些内容在 Python 文件中极为常见，模型引用也属正常
            GENERIC_PREFIXES = (
                "#!/usr/bin/env python",
                "#!/usr/bin/python",
                "# -*- coding:",
                "#!",
            )
            effective_preview = normalized_preview
            for pfx in GENERIC_PREFIXES:
                if effective_preview.startswith(pfx):
                    # 跳过到第一个实质性内容（非注释行）
                    first_real = re.search(r"(?:^| )(?!#)[^\s]", effective_preview)
                    if first_real:
                        effective_preview = effective_preview[first_real.start():].strip()
                    break

            # probe 取实质内容的前 120 个字符，且要足够有区分度（≥60 字符）
            probe = effective_preview[:120]
            if len(probe) >= 60 and probe in normalized_text:
                return True

        # 代码块复述检测：仅当代码块内容与文件 preview 高度重叠时才触发
        # （避免误伤正常的代码示例或分析总结）
        if text.count("```") >= 2 and len(text) > 500 and preview:
            # 提取代码块内容
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
            combined_code = " ".join(code_blocks)
            if combined_code and len(combined_code) > 200:
                # 代码块占响应比例超过 60%，且与文件内容高度相似，认为是复述
                code_ratio = len(combined_code) / len(text)
                if code_ratio > 0.6:
                    normalized_code = self._normalize_space(combined_code)
                    code_probe = normalized_preview[:80]
                    if len(code_probe) >= 40 and code_probe in normalized_code:
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
                "你刚才直接复述了工具执行的原始结果，请重写回答。\n"
                "硬性要求：\n"
                "1) 禁止出现'工具执行结果如下'等表述。\n"
                "2) 禁止输出 JSON、原始代码块、目录树或大段原文。\n"
                "3) 根据已获取的信息，判断任务是否完成：\n"
                "   - 若信息已足够：输出核心结论（1-2句）+ 关键要点（最多3条），直接完成任务。\n"
                "   - 若还需更多信息（如调用链路分析需读取其他文件）：\n"
                "     使用工具格式继续收集：<tool>工具名</tool><input>{参数}</input>\n"
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

    def _force_summarize_on_duplicate(
        self,
        messages: List[Dict[str, str]],
        model_response: str,
        system_prompt: str,
        iteration: int,
        runtime_context: Dict[str, Any],
        **model_kwargs,
    ) -> str:
        """当检测到模型陷入重复工具调用死循环时，强制执行一次总结调用。

        原理：将当前对话历史（含工具结果）追加强约束指令，
        让模型基于已有上下文直接给出最终回答，不再调用工具。
        """
        force_message = {
            "role": "user",
            "content": (
                "<force_summarize>\n"
                "⚠️ 系统检测到你持续重复调用相同工具，已强制终止工具循环。\n"
                "上下文中已包含工具执行结果，请立即基于这些已有信息完成用户的任务。\n"
                "硬性要求：\n"
                "1) 禁止输出任何 <tool> 工具调用标签。\n"
                "2) 直接给出最终回答，无需再收集信息。\n"
                "3) 若内容已截断，请基于已有部分作出尽力回答，并说明局限性。\n"
                "</force_summarize>"
            ),
        }

        force_messages = [dict(m) for m in messages]
        force_messages.append({"role": "assistant", "content": model_response})
        force_messages.append(force_message)

        final = self.model_forward_fn(force_messages, system_prompt, **model_kwargs)
        final = self._apply_after_model_middlewares(final, iteration, runtime_context)
        return final

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

    @staticmethod
    def _extract_system_from_messages(
        messages: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], str]:
        """从消息列表中剥离开头的 system 消息，返回 (非system消息列表, system内容)。

        create_qwen_model_forward 会自行在头部追加 system 消息，
        因此传入 model_forward_fn 的 messages 不应再包含 system 角色，
        避免重复拼接导致模型收到两条 system 消息。
        """
        if messages and messages[0].get("role") == "system":
            return messages[1:], messages[0].get("content", "")
        return messages, ""

    def process_message_direct(
        self,
        messages: List[Dict[str, str]],
        system_prompt_override: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        return_runtime_context: bool = False,
        **model_kwargs,
    ) -> Union[Tuple[str, List[Dict[str, Any]]], Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        单次调用模型（无工具循环），但完整经过中间件链。

        适用于 chat / skills 模式：不需要工具调用能力，但需要中间件对消息进行
        上下文注入（PlanModeMiddleware、RuntimeModeMiddleware、SkillsContextMiddleware、
        UploadedFilesMiddleware 等）。

        与 process_message 的区别：
        - process_message：带工具循环（max_iterations 次）
        - process_message_direct：单次调用，走完 before_model → model → after_model 中间件链

        Args:
            messages: 已构建好的消息列表（可含 system 消息，会自动剥离处理）
            system_prompt_override: 覆盖系统提示词（优先级高于 messages 中的 system）
            runtime_context: 本次执行的运行时上下文（含 run_mode、plan_mode 等）
            return_runtime_context: 是否额外返回运行时上下文
            **model_kwargs: 传递给模型的参数

        Returns:
            默认返回 (最终响应, 执行记录)
            当 return_runtime_context=True 时返回 (最终响应, 执行记录, 运行时上下文)
        """
        runtime_ctx = self._build_runtime_context(runtime_context)
        execution_log: List[Dict[str, Any]] = []

        # 剥离 messages 中的 system 消息，避免 model_forward_fn 重复追加
        non_system_messages, embedded_system = self._extract_system_from_messages(messages)
        effective_override = system_prompt_override or embedded_system or None
        system_prompt = self._resolve_system_prompt(effective_override, runtime_ctx)

        # 经过完整中间件链（before_model 钩子：RuntimeMode / PlanMode / Skills / Files 等）
        processed_messages = self._apply_before_model_middlewares(non_system_messages, 0, runtime_ctx)

        model_response = self.model_forward_fn(processed_messages, system_prompt, **model_kwargs)
        model_response = self._apply_after_model_middlewares(model_response, 0, runtime_ctx)

        execution_log.append({
            "iteration": 0,
            "type": "model_response",
            "content": model_response,
            "run_mode": runtime_ctx.get("run_mode", "chat"),
            "plan_mode": bool(runtime_ctx.get("plan_mode", False)),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

        self._finalize_middlewares(execution_log, runtime_ctx)
        if return_runtime_context:
            return model_response, execution_log, runtime_ctx
        return model_response, execution_log

    def process_message_direct_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt_override: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        stream_forward_fn=None,
        **model_kwargs,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        单次调用模型的流式版本（无工具循环），完整经过中间件链。

        适用于 chat / skills 模式的流式输出场景。
        messages 可包含开头的 system 消息，会自动剥离并正确传递给 model_forward_fn。

        架构说明：
          - model_forward_fn 返回完整字符串（用于工具循环中解析工具调用）
          - stream_forward_fn（可选）若提供，则为真流式生成器，每次 yield 累积文本
            优先使用 stream_forward_fn 以获得逐 token 输出体验
          - 若均不可用，则退化为单次 yield 完整响应

        Args:
            messages: 消息列表（可含 system 消息，会自动剥离）
            system_prompt_override: 覆盖系统提示词
            runtime_context: 运行时上下文
            stream_forward_fn: 可选真流式前向函数，签名: fn(messages, system_prompt, **kwargs) -> Generator[str, None, None]
            **model_kwargs: 传递给模型的参数

        Yields:
            (累积响应文本, 执行信息字典)
        """
        runtime_ctx = self._build_runtime_context(runtime_context)
        execution_log: List[Dict[str, Any]] = []

        # 剥离 messages 中的 system 消息，避免 model_forward_fn 重复追加
        non_system_messages, embedded_system = self._extract_system_from_messages(messages)
        effective_override = system_prompt_override or embedded_system or None
        system_prompt = self._resolve_system_prompt(effective_override, runtime_ctx)

        # 经过完整中间件链（before_model 钩子：RuntimeMode / PlanMode / Skills / Files 等）
        processed_messages = self._apply_before_model_middlewares(non_system_messages, 0, runtime_ctx)

        model_response = ""

        if stream_forward_fn is not None:
            # 真流式路径：逐 token yield，给 UI 提供实时输出体验
            try:
                for partial in stream_forward_fn(processed_messages, system_prompt, **model_kwargs):
                    model_response = partial  # 累积文本（生成器每次 yield 已累积）
                    yield model_response, {"iteration": 0, "type": "model_response_stream"}
            except Exception:
                # 降级到非流式
                model_response = self.model_forward_fn(processed_messages, system_prompt, **model_kwargs)
        else:
            # 非流式路径（model_forward_fn 返回完整字符串）
            model_response = self.model_forward_fn(processed_messages, system_prompt, **model_kwargs)

        model_response = self._apply_after_model_middlewares(model_response, 0, runtime_ctx)

        execution_log.append({
            "iteration": 0,
            "type": "model_response",
            "content": model_response,
            "run_mode": runtime_ctx.get("run_mode", "chat"),
            "plan_mode": bool(runtime_ctx.get("plan_mode", False)),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

        # 最终完整响应（非流式时首次 yield，流式时最后一次 yield 确保 after_model 后内容正确）
        yield model_response, {"iteration": 0, "type": "model_response"}
        self._finalize_middlewares(execution_log, runtime_ctx)

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
                tool_calls = self._infer_fallback_tool_calls(user_message, runtime_ctx, iteration, model_response)
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
            has_duplicate = False
            new_tool_executed = False
            executed_calls: set = runtime_ctx.setdefault("_executed_tool_calls", set())

            for tool_name, tool_input in tool_calls:
                call_key = (tool_name, json.dumps(tool_input, sort_keys=True, ensure_ascii=False))
                if call_key in executed_calls:
                    # 检测到重复工具调用，跳过执行，注入强约束提示
                    has_duplicate = True
                    duplicate_hint = json.dumps(
                        {
                            "duplicate_call": True,
                            "tool": tool_name,
                            "message": (
                                f"⚠️ 你已经调用过 {tool_name}（参数相同），结果已在上下文中。"
                                "请勿重复调用相同工具。请直接基于已获取的内容完成任务，给出最终回答。"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": duplicate_hint,
                    })
                    execution_log.append(
                        {
                            "iteration": iteration,
                            "type": "duplicate_tool_call_blocked",
                            "tool": tool_name,
                            "input": tool_input,
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }
                    )
                    continue

                raw_result = self.tool_executor.execute_tool(tool_name, tool_input)
                guarded_result = self._apply_after_tool_call_middlewares(
                    tool_name,
                    tool_input,
                    raw_result,
                    iteration,
                    runtime_ctx,
                )
                # Fix-3: bash 命令失败时（returncode != 0）不加入 executed_calls，
                # 允许安装依赖后重新执行相同命令（如 pip3 install 后再次 python3 -m pytest）
                _bash_failed = False
                if tool_name == "bash":
                    try:
                        _result_data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                        if isinstance(_result_data, dict) and _result_data.get("returncode", 0) != 0:
                            _bash_failed = True
                    except Exception:
                        pass
                if not _bash_failed:
                    executed_calls.add(call_key)
                new_tool_executed = True
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

            # 当所有工具调用都是重复的（无新工具执行），立即强制总结并退出循环，
            # 避免模型陷入"重复调用 → 警告提示 → 再次重复调用"的死循环
            if has_duplicate and not new_tool_executed:
                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "force_summarize",
                        "reason": "所有工具调用均为重复，触发强制总结退出",
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                )
                messages.append({"role": "assistant", "content": model_response})
                result_text = self._format_tool_results(tool_results, runtime_ctx, has_duplicate=True)
                messages.append({"role": "user", "content": result_text})
                final_response = self._force_summarize_on_duplicate(
                    messages, model_response, system_prompt, iteration, runtime_ctx, **model_kwargs
                )
                self._finalize_middlewares(execution_log, runtime_ctx)
                if return_runtime_context:
                    return final_response, execution_log, runtime_ctx
                return final_response, execution_log

            messages.append({"role": "assistant", "content": model_response})
            result_text = self._format_tool_results(tool_results, runtime_ctx, has_duplicate=has_duplicate)
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
            yield model_response, {"iteration": iteration, "type": "model_response"}

            tool_calls = self.tool_parser.parse_tool_calls(model_response)
            if not tool_calls:
                tool_calls = self._infer_fallback_tool_calls(user_message, runtime_ctx, iteration, model_response)
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
                    return

            tool_results = []
            has_duplicate = False
            new_tool_executed = False
            executed_calls: set = runtime_ctx.setdefault("_executed_tool_calls", set())

            for tool_name, tool_input in tool_calls:
                call_key = (tool_name, json.dumps(tool_input, sort_keys=True, ensure_ascii=False))
                if call_key in executed_calls:
                    # 检测到重复工具调用，跳过执行，注入强约束提示
                    has_duplicate = True
                    duplicate_hint = json.dumps(
                        {
                            "duplicate_call": True,
                            "tool": tool_name,
                            "message": (
                                f"⚠️ 你已经调用过 {tool_name}（参数相同），结果已在上下文中。"
                                "请勿重复调用相同工具。请直接基于已获取的内容完成任务，给出最终回答。"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": duplicate_hint,
                    })
                    execution_log.append(
                        {
                            "iteration": iteration,
                            "type": "duplicate_tool_call_blocked",
                            "tool": tool_name,
                            "input": tool_input,
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }
                    )
                    yield f"[重复调用已阻断] {tool_name}\n", {
                        "iteration": iteration,
                        "type": "duplicate_tool_call_blocked",
                        "tool": tool_name,
                    }
                    continue

                raw_result = self.tool_executor.execute_tool(tool_name, tool_input)
                guarded_result = self._apply_after_tool_call_middlewares(
                    tool_name,
                    tool_input,
                    raw_result,
                    iteration,
                    runtime_ctx,
                )
                # Fix: bash 命令失败时（returncode != 0）不加入 executed_calls，
                # 允许安装依赖后重新执行相同命令（如 pip install 后再次 pytest）
                _bash_failed = False
                if tool_name == "bash":
                    try:
                        _result_data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                        if isinstance(_result_data, dict) and _result_data.get("returncode", 0) != 0:
                            _bash_failed = True
                    except Exception:
                        pass
                if not _bash_failed:
                    executed_calls.add(call_key)
                new_tool_executed = True

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

                yield f"[执行工具] {tool_name}\n", {
                    "iteration": iteration,
                    "type": "tool_execution",
                    "tool": tool_name,
                }

            # 当所有工具调用都是重复的（无新工具执行），立即强制总结并退出循环，
            # 避免模型陷入"重复调用 → 警告提示 → 再次重复调用"的死循环
            if has_duplicate and not new_tool_executed:
                execution_log.append(
                    {
                        "iteration": iteration,
                        "type": "force_summarize",
                        "reason": "所有工具调用均为重复，触发强制总结退出",
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                )
                messages.append({"role": "assistant", "content": model_response})
                result_text = self._format_tool_results(tool_results, runtime_ctx, has_duplicate=True)
                messages.append({"role": "user", "content": result_text})
                final_response = self._force_summarize_on_duplicate(
                    messages, model_response, system_prompt, iteration, runtime_ctx, **model_kwargs
                )
                yield final_response, {"iteration": iteration, "type": "force_summarize"}
                self._finalize_middlewares(execution_log, runtime_ctx)
                return

            messages.append({"role": "assistant", "content": model_response})
            result_text = self._format_tool_results(tool_results, runtime_ctx, has_duplicate=has_duplicate)
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

    # 文件内容直接透传的字符数上限（超过此限制才截断）
    _READ_FILE_FULL_CONTENT_LIMIT = 12000
    # bash stdout 直接透传上限（与 read_file 保持一致，确保 grep 等完整输出不被截断）
    _BASH_STDOUT_FULL_LIMIT = 12000

    def _prepare_tool_result_for_model(self, tool_name: str, result_data: Any) -> Tuple[Any, bool]:
        """处理工具结果后回传给模型。

        read_file 策略:
          - 内容 <= 12000 字符：直接传完整内容，让模型看到真实代码/文本
          - 内容 >  12000 字符：截断保留头尾上下文，并附带结构化摘要辅助导航
        bash 策略:
          - stdout <= 12000 字符：直接透传完整 stdout，让模型看到完整命令输出
          - stdout >  12000 字符：截断保留头尾，附带省略提示
        其他工具结果超过 1200 字符时截断。
        """
        if tool_name == "bash" and isinstance(result_data, dict):
            stdout = result_data.get("stdout", "")
            if not isinstance(stdout, str):
                stdout = ""
            stdout_len = len(stdout)
            if stdout_len > self._BASH_STDOUT_FULL_LIMIT:
                truncated_stdout, _ = self._truncate_text(stdout, max_chars=self._BASH_STDOUT_FULL_LIMIT)
                result_data = dict(result_data)
                result_data["stdout"] = truncated_stdout
                result_data["_sys_note"] = f"stdout 共 {stdout_len} 字符，已截断显示（保留头尾）。"
            return result_data, stdout_len > self._BASH_STDOUT_FULL_LIMIT

        if tool_name == "read_file" and isinstance(result_data, dict):
            content = result_data.get("content", "")
            if not isinstance(content, str):
                content = ""
            char_count = len(content)

            if char_count <= self._READ_FILE_FULL_CONTENT_LIMIT:
                # 内容较短：直接透传完整内容，不做压缩
                return {
                    "success": bool(result_data.get("success", True)),
                    "path": str(result_data.get("path", "")),
                    "content": content,
                    "line_count": content.count("\n") + (1 if content else 0),
                    "char_count": char_count,
                }, False
            else:
                # 内容超长：截断保留头尾 + 附带结构化摘要
                truncated_content, _ = self._truncate_text(
                    content, max_chars=self._READ_FILE_FULL_CONTENT_LIMIT
                )
                summary = self._build_read_file_summary(result_data)
                return {
                    "success": bool(result_data.get("success", True)),
                    "path": str(result_data.get("path", "")),
                    "content": truncated_content,
                    "summary": summary,
                    "char_count": char_count,
                    # _sys_note 是内部系统注释，模型不应在回答中提及
                    "_sys_note": f"文件共 {char_count} 字符，已截断显示（保留头尾），如需查看其余部分可继续调用 read_file。",
                }, True

        try:
            serialized = json.dumps(result_data, ensure_ascii=False)
        except TypeError:
            serialized = str(result_data)

        preview, truncated = self._truncate_text(serialized, max_chars=1200)
        if truncated:
            return {
                "truncated": True,
                "preview": preview,
                # _sys_note 是内部系统注释，模型不应在回答中提及
                "_sys_note": "结果过长，已截断显示。",
            }, True

        return result_data, False

    def _format_tool_results(
        self,
        tool_results: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
        has_duplicate: bool = False,
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
                    # 优先使用 summary 字段（超长截断时存在），否则从完整内容即时构建摘要
                    summary_payload = prepared_result.get("summary")
                    if not isinstance(summary_payload, dict):
                        # 短文件直接透传了 content，现场生成摘要供 anti-repeat guard 使用
                        summary_payload = self._build_read_file_summary(
                            {"path": prepared_result.get("path", ""), "content": prepared_result.get("content", "")}
                        )
                    runtime_context["_last_read_file_summary"] = dict(summary_payload)
                    # 注意：不重置 _anti_repeat_guard_used。
                    # 仅在尚未设置时初始化为 False（首次读文件），避免每轮重置导致 guard 每轮都触发。
                    if "_anti_repeat_guard_used" not in runtime_context:
                        runtime_context["_anti_repeat_guard_used"] = False
                    # 记录已读文件路径集合，用于检测重复读取同一文件
                    read_path = str(prepared_result.get("path", ""))
                    if read_path:
                        runtime_context.setdefault("_read_file_paths", set()).add(read_path)

        if has_duplicate:
            formatted.append("⚠️【重要】你刚才重复调用了已执行过的工具（参数完全相同）。")
            formatted.append("框架已阻断重复执行。文件内容已在上下文历史中，请勿再次调用相同工具。")
            formatted.append("**请立即基于上下文中已有的文件内容，直接完成用户的任务并给出最终回答。**")
            formatted.append("禁止输出任何新的工具调用。")
            formatted.append("")
        elif has_read_file:
            if read_file_structured:
                # 文件过长被截断：提示还可以继续读，但要明确说明截断情况
                formatted.append("文件内容已读取（内容较长，已保留头尾关键部分）。")
                formatted.append("【重要】该文件已完整读取过，上下文中已有内容。请直接基于已有内容完成任务。")
                formatted.append("- 如果已有内容已足够：直接给出最终回答。")
                formatted.append("- 如果确实需要查看文件其他部分：说明原因并使用 read_file 继续读取。")
                formatted.append("- 【禁止】重复读取相同路径的文件，上下文中已有该文件内容。")
                formatted.append("- 【注意】回答时不要提及文件截断相关的技术细节（如字符数、截断位置等），直接基于已有内容作答即可。")
            else:
                # 文件内容已完整读取：明确禁止重复读同一文件
                formatted.append("文件内容已完整读取。")
                formatted.append("【重要】上下文中已包含该文件的完整内容，无需再次读取同一文件。")
                formatted.append("请立即基于已读取的文件内容完成用户任务：")
                formatted.append("- 如果任务只需要这一个文件：直接给出最终回答。")
                formatted.append("- 如果任务还需要读取其他文件（不是同一个文件）：调用 read_file 读取其他路径。")
            formatted.append("")

        return "\n".join(formatted)

    def enable_tools_in_prompt(self, enable: bool):
        """动态启用/禁用工具信息在系统提示词中"""
        self.tools_in_system_prompt = enable


# ============================================================================
# 意图路由器 - 自动分析用户意图，无需手动选择模式
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
                max_tokens=kwargs.get("max_tokens", 8192),
            )
            return _consume_stream(stream)

        if hasattr(qwen_agent, "generate_stream_text"):
            stream = qwen_agent.generate_stream_text(
                full_messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 8192),
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
                max_tokens=kwargs.get("max_tokens", 8192),
            )
            return _consume_stream(stream)

        raise AttributeError(
            "qwen_agent 缺少可用的流式接口，请实现 generate_stream_with_messages、"
            "generate_stream_text 或 generate_stream。"
        )

    return forward


# ============================================================================
# 持久化记忆系统 - 借鉴 DeerFlow MemoryUpdater，适配轻量本地/API 模型
# ============================================================================



if __name__ == "__main__":
    print("\u2705 Agent 框架模块加载成功")
    print("ℹ️  本文件只包含核心类：")
    print("  - QwenAgentFramework        Agent 主循环（工具调用 + 中间件链）")
    print("  - create_qwen_model_forward 适配 Qwen 模型的前向函数工厂")
    print("ℹ️  下列类已迁移至独立子模块，在此仅保留导入（向后兼容）：")
    print("  - AgentMiddleware / 中间件链   → agent_middlewares.py")
    print("  - TodoItem / TodoManager    → todo_manager.py")
    print("  - IntentRouter / AIIntentRouter → intent_router.py")
    print("  - TaskPlanner / TaskExecutor    → task_planner.py")
    print("  - MemoryManager                 → memory_manager.py")
