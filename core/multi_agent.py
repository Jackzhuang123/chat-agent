# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构（深度优化版）

最终修复：
- 强制 knowledge 步骤的前置 read_file 依赖
- write_file 步骤强制使用前置 knowledge 步骤的输出
- 最终报告按用户问题分组，去重，显示完成/失败/待协助
- 所有切片异常已防御
"""
import asyncio
import copy
import inspect
import json
import logging
import re
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, AsyncGenerator

from core.clarification import ClarificationManager
from core.components.output_cleaner import clean_react_tags
from core.monitor_logger import get_monitor_logger, log_event
from core.state_manager import SessionContext

_AVAILABLE_TOOLS = {"bash", "read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"}
_FILE_REF_EXTENSIONS = ("py", "md", "json", "txt", "log", "yaml", "yml", "csv", "pdf", "doc", "docx")
_FILE_REF_PATTERN = re.compile(
    rf'(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]+\.(?:{"|".join(_FILE_REF_EXTENSIONS)}))(?![A-Za-z0-9_./-])'
)


def safe_fstr(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def extract_file_references(text: str) -> List[str]:
    """Extract file-like references while avoiding Chinese prefixes joining the filename."""
    seen = set()
    matches = []
    for match in _FILE_REF_PATTERN.finditer(text or ""):
        candidate = match.group(1)
        if candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)
    return matches


def _escape_control_chars_in_json_strings(text: str) -> str:
    result = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                result.append(ch)
                escaped = False
            elif ch == "\\":
                result.append(ch)
                escaped = True
            elif ch == '"':
                result.append(ch)
                in_string = False
            elif ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_string = True
            result.append(ch)
    return "".join(result)


def _extract_balanced_json_block(text: str) -> Optional[str]:
    start = next((idx for idx, ch in enumerate(text or "") if ch in "{["), -1)
    if start < 0:
        return None

    stack = []
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                return None
            left = stack.pop()
            if (left, ch) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                return text[start:idx + 1]
    return None


def _repair_json_like_payload(text: str) -> str:
    cleaned = re.sub(r"```(?:json|javascript|js|python|text|plaintext)?\s*", "", text or "", flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    balanced = _extract_balanced_json_block(cleaned)
    if balanced:
        cleaned = balanced
    cleaned = _escape_control_chars_in_json_strings(cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:', r' "\1":', cleaned)
    cleaned = re.sub(r"(?<=\{|,)\s*'([^'\n]+)'\s*:", lambda m: f' "{m.group(1)}":', cleaned)
    cleaned = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', cleaned)
    return cleaned.strip()


def _parse_json_like_payload(text: str) -> Tuple[Optional[Any], str]:
    candidates = []
    cleaned = (text or "").strip()
    if cleaned:
        candidates.append(cleaned)
    repaired = _repair_json_like_payload(text or "")
    if repaired and repaired not in candidates:
        candidates.append(repaired)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate), candidate
        except json.JSONDecodeError:
            pass
        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
                return obj, candidate[idx:]
            except json.JSONDecodeError:
                continue
    return None, repaired


class StepEvidenceResolver:
    """从多 Agent 步骤结果中提取可复用证据，避免知识步骤脱离前置事实自由发挥。"""

    FILE_TASK_KEYWORDS = ("总结", "解释", "说明", "分析", "类名", "方法", "函数", "代码", "结构")

    @staticmethod
    def get_dependency_ids(step: Dict[str, Any]) -> List[int]:
        deps = step.get("depends_on", [])
        return deps if isinstance(deps, list) else []

    @staticmethod
    def get_dependency_results(step: Dict[str, Any], accumulated_context: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
        step_outputs = accumulated_context.get("step_outputs", {}) or {}
        items = []
        for dep_id in StepEvidenceResolver.get_dependency_ids(step):
            dep_result = step_outputs.get(dep_id)
            if isinstance(dep_result, dict):
                items.append((dep_id, dep_result))
        return items

    @staticmethod
    def extract_file_evidence(step: Dict[str, Any], accumulated_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence = []
        for dep_id, dep_result in StepEvidenceResolver.get_dependency_results(step, accumulated_context):
            for tool_call in dep_result.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("tool") != "read_file" or not tool_call.get("success"):
                    continue
                result_obj = tool_call.get("result", {}) or {}
                if not isinstance(result_obj, dict):
                    continue
                evidence.append({
                    "step_id": dep_id,
                    "path": result_obj.get("path", ""),
                    "content": result_obj.get("content", ""),
                    "file_facts": result_obj.get("file_facts", {}) or {},
                    "note": result_obj.get("resolution_note") or result_obj.get("note", ""),
                })
        return evidence

    @staticmethod
    def extract_dependency_response(step: Dict[str, Any], accumulated_context: Dict[str, Any]) -> Tuple[Optional[int], str]:
        for dep_id, dep_result in reversed(StepEvidenceResolver.get_dependency_results(step, accumulated_context)):
            response = (dep_result.get("response", "") or "").strip()
            if response:
                return dep_id, response
        return None, ""

    @staticmethod
    def should_ground_file_task(action: str, file_evidence: List[Dict[str, Any]]) -> bool:
        return bool(file_evidence) and any(keyword in (action or "") for keyword in StepEvidenceResolver.FILE_TASK_KEYWORDS)

    @staticmethod
    def build_grounding_block(file_evidence: List[Dict[str, Any]], max_content_chars: int = 1800) -> str:
        blocks = []
        for item in file_evidence:
            facts = item.get("file_facts", {}) or {}
            chunks = facts.get("chunk_summaries", [])[:4]
            chunk_lines = []
            for chunk in chunks:
                signals = " | ".join(chunk.get("signals", [])[:2]) if isinstance(chunk, dict) else ""
                if signals:
                    chunk_lines.append(f"- {chunk.get('line_range', '?')}: {signals}")
            content = (item.get("content", "") or "")[:max_content_chars]
            blocks.append(
                "\n".join(filter(None, [
                    f"文件：{item.get('path', '')}",
                    f"定位说明：{item.get('note', '')}" if item.get("note") else "",
                    f"结构摘要：{facts.get('summary', '')}",
                    f"类：{', '.join(facts.get('classes', [])) or '无'}",
                    f"函数：{', '.join(facts.get('functions', [])) or '无'}",
                    f"导入：{', '.join(facts.get('imports', [])[:6]) or '无'}",
                    "关键代码信号：\n" + "\n".join(chunk_lines) if chunk_lines else "",
                    "真实文件内容片段：\n"
                    "```python\n"
                    f"{content}\n"
                    "```",
                ]))
            )
        return "\n\n".join(blocks)

    @staticmethod
    def build_grounded_file_response(action: str, file_evidence: List[Dict[str, Any]]) -> str:
        sections = [f"以下内容严格基于已读取到的真实文件内容：{action}"]
        for item in file_evidence:
            facts = item.get("file_facts", {}) or {}
            classes = facts.get("classes", []) or []
            functions = facts.get("functions", []) or []
            imports = facts.get("imports", []) or []
            chunk_summaries = facts.get("chunk_summaries", [])[:5]
            sections.append(f"\n### {item.get('path', '未知文件')}")
            if item.get("note"):
                sections.append(f"- 文件定位：{item['note']}")
            sections.append(f"- 代码规模：{facts.get('line_count', '未知')} 行")
            sections.append(f"- 类名：{', '.join(classes) if classes else '未检测到'}")
            sections.append(f"- 方法/函数：{', '.join(functions) if functions else '未检测到'}")
            if imports:
                sections.append(f"- 主要导入：{', '.join(imports[:8])}")
            if chunk_summaries:
                sections.append("- 代码结构信号：")
                for chunk in chunk_summaries:
                    signals = "；".join(chunk.get("signals", [])[:3]) if isinstance(chunk, dict) else ""
                    classes_in_chunk = ", ".join(chunk.get("classes", [])) if isinstance(chunk, dict) else ""
                    funcs_in_chunk = ", ".join(chunk.get("functions", [])) if isinstance(chunk, dict) else ""
                    detail = []
                    if classes_in_chunk:
                        detail.append(f"类 {classes_in_chunk}")
                    if funcs_in_chunk:
                        detail.append(f"函数 {funcs_in_chunk}")
                    if signals:
                        detail.append(f"信号 {signals}")
                    sections.append(f"  - {chunk.get('line_range', '?')}: {'；'.join(detail) if detail else '无明显结构信号'}")
            sections.append("- 说明：以上结论来自文件静态结构提取；未在文件中直接出现的行为细节不做推断。")
        return "\n".join(sections).strip()


def _safe_model_call(model_forward_fn: Callable, messages: list, system_prompt: str = "", **kwargs) -> str:
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        return f"模型调用失败: {e}"

    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        full_text = ""
        try:
            for chunk in result:
                if isinstance(chunk, str):
                    if not full_text:
                        full_text = chunk
                    elif chunk.startswith(full_text):
                        # 某些流式实现返回“累计文本”，直接保留最新完整结果
                        full_text = chunk
                    elif len(chunk) >= max(16, len(full_text) // 2) and chunk[:1] in "{[`\"":
                        # 也有实现返回“重写后的完整快照”，不是严格前缀关系
                        full_text = chunk
                    else:
                        # 其他实现返回增量 token，则按增量拼接
                        full_text += chunk
        except Exception:
            pass
        return full_text

    if not isinstance(result, str):
        return str(result) if result is not None else ""
    return result


class PlannerAgent:
    def __init__(self, model_forward_fn: Callable, available_tools: Optional[set] = None):
        self.model_forward_fn = model_forward_fn
        self.available_tools = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)
        self.monitor = get_monitor_logger()

    @staticmethod
    def _is_simple_file_request(user_input: str) -> bool:
        if not user_input:
            return False
        stripped = user_input.strip()
        has_path = bool(re.search(r'[/\\]', stripped))
        has_extension = bool(re.search(r'\.\w{2,}$', stripped))
        is_plain_name = bool(re.match(r'^[a-zA-Z0-9_\-\.]+$', stripped))
        operation_keywords = ['读取', '解析', '分析', '查看', '打开', '写入', '修改', '删除', '运行', '执行']
        has_operation = any(kw in stripped for kw in operation_keywords)
        return (has_path or has_extension or is_plain_name) and not has_operation

    @staticmethod
    def _is_high_risk_knowledge_action(user_input: str) -> bool:
        high_risk_keywords = ["歌词", "片段", "十首", "top", "排名", "最出名", "最有名", "排行榜",
                              "具体数据", "年份", "日期", "价格", "名单", "列出", "给出"]
        return any(kw in user_input for kw in high_risk_keywords)

    def _extract_first_json(self, text: str):
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] not in '{[':
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end_idx = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
                elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    return obj[0]
                idx += end_idx
            except json.JSONDecodeError:
                idx += 1
        return None

    @staticmethod
    def _clean_json(text: str) -> str:
        return _repair_json_like_payload(text)

    def _sanitize_plan_steps(self, plan: dict) -> dict:
        steps = plan.get("steps", [])
        for step in steps:
            tool = step.get("tool", "none")
            if tool not in self.available_tools:
                self.monitor.warning(f"检测到非法工具名 {tool}，已回退为 none")
                step["tool"] = "none"
                step["task_type"] = "knowledge"
                step["tool_input"] = {}
            step.setdefault("tool_input", {})
            step.setdefault("task_type", "knowledge" if step.get("tool") == "none" else "tool")
        return plan

    def _ensure_write_step_if_needed(self, user_input: str, plan: dict) -> dict:
        write_keywords = ["写入", "保存", "生成", "创建.*文件", "输出到", "写.*md", "写.*txt"]
        if not any(re.search(kw, user_input) for kw in write_keywords):
            return plan

        steps = plan.get("steps", [])
        if any(s.get("tool") == "write_file" for s in steps):
            return plan

        last_knowledge = None
        for s in steps:
            if s.get("task_type") == "knowledge" or s.get("tool") == "none":
                last_knowledge = s

        if not last_knowledge:
            return plan

        file_match = re.search(r'[\w\-]+\.(md|txt|json|log)', user_input)
        filename = file_match.group(0) if file_match else "summary.md"

        new_step = {
            "id": max(s["id"] for s in steps) + 1,
            "action": f"将上述总结写入文件 {filename}",
            "tool": "write_file",
            "tool_input": {
                "path": filename,
                "content": "【请使用前置步骤的总结内容替换此处】",
                "mode": "overwrite"
            },
            "task_type": "tool",
            "depends_on": [last_knowledge["id"]]
        }
        steps.append(new_step)
        return plan

    def _sanitize_high_risk_knowledge_plan(self, user_input: str, plan: dict) -> dict:
        if not self._is_high_risk_knowledge_action(user_input):
            return plan

        for step in plan.get("steps", []):
            if step.get("task_type") != "knowledge" or step.get("tool") != "none":
                continue
            action = step.get("action", "")
            if extract_file_references(action):
                continue
            step.setdefault("original_action", action)
            if any(token in action for token in ("片段", "原文", "歌词", "引用", "逐字")):
                step["high_risk_policy"] = "no_verbatim"
            elif any(token in action for token in ("十首", "top", "排名", "最出名", "最有名", "排行榜", "名单", "列出", "给出")):
                step["high_risk_policy"] = "cautious_list"
            else:
                step["high_risk_policy"] = "cautious_summary"
        return plan

    @staticmethod
    def _trim_plan_to_budget(plan: dict, max_steps: int = 8) -> dict:
        steps = plan.get("steps", [])
        if len(steps) <= max_steps:
            return plan

        trimmed = []
        overflow_actions = []
        for step in steps:
            if len(trimmed) < max_steps:
                trimmed.append(copy.deepcopy(step))
            else:
                overflow_actions.append(step.get("action", ""))

        if overflow_actions:
            for step in reversed(trimmed):
                if step.get("task_type") == "knowledge" or step.get("tool") == "none":
                    merged = "；".join([a for a in overflow_actions if a])
                    if merged:
                        step["action"] = f"{step.get('action', '')}；并合并处理：{merged}"
                    break

        id_map = {}
        for idx, step in enumerate(trimmed, start=1):
            id_map[step["id"]] = idx
            step["id"] = idx
        for step in trimmed:
            deps = step.get("depends_on", [])
            if not isinstance(deps, list):
                continue
            remapped = [id_map[d] for d in deps if d in id_map and id_map[d] != step["id"]]
            if remapped:
                step["depends_on"] = remapped
            else:
                step.pop("depends_on", None)
        plan["steps"] = trimmed
        raw_estimated = str(plan.get("estimated_time", "300"))
        matched_seconds = re.search(r'\d+', raw_estimated)
        estimated_seconds = int(matched_seconds.group()) if matched_seconds else 300
        plan["estimated_time"] = str(min(estimated_seconds, max_steps * 45))
        return plan

    def _enforce_read_before_summarize(self, plan: dict) -> dict:
        """确保任何要求总结/解释具体文件的 knowledge 步骤都有前置 read_file 步骤"""
        steps = plan.get("steps", [])
        if not steps:
            return plan

        new_steps = []
        old_to_new = {}
        for step in steps:
            step_copy = copy.deepcopy(step)
            prepended_dep_ids = []
            if step_copy.get("task_type") == "knowledge" and step_copy.get("tool") == "none":
                action = step_copy.get("action", "")
                referenced_files = extract_file_references(action)
                existing_deps = list(step_copy.get("depends_on", []))
                has_read_dep = any(
                    d for d in existing_deps
                    if any(s for s in steps if s["id"] == d and s.get("tool") == "read_file")
                )
                if referenced_files and not has_read_dep:
                    for target_file in referenced_files:
                        read_step = {
                            "id": len(new_steps) + 1,
                            "action": f"读取文件 {target_file}",
                            "tool": "read_file",
                            "tool_input": {"path": target_file},
                            "task_type": "tool"
                        }
                        new_steps.append(read_step)
                        prepended_dep_ids.append(read_step["id"])
            step_copy["_old_depends_on"] = list(step.get("depends_on", []))
            step_copy["_prepended_dep_ids"] = prepended_dep_ids
            step_copy["id"] = len(new_steps) + 1
            old_to_new[step["id"]] = step_copy["id"]
            new_steps.append(step_copy)

        for step in new_steps:
            old_deps = step.pop("_old_depends_on", [])
            prepended_dep_ids = step.pop("_prepended_dep_ids", [])
            remapped_old_deps = [old_to_new.get(dep, dep) for dep in old_deps]
            merged_deps = []
            for dep in prepended_dep_ids + remapped_old_deps:
                if dep != step["id"] and dep not in merged_deps:
                    merged_deps.append(dep)
            if merged_deps:
                step["depends_on"] = merged_deps
            else:
                step.pop("depends_on", None)
        return {
            "steps": new_steps,
            "complexity": plan.get("complexity", "medium"),
            "estimated_time": plan.get("estimated_time", "300")
        }
        return plan

    def plan(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        if self._is_simple_file_request(user_input):
            self.monitor.info(f"检测到简单文件请求: {safe_fstr(user_input)}，直接生成单步计划")
            return {
                "success": True,
                "plan": {
                    "complexity": "simple",
                    "steps": [
                        {
                            "id": 1,
                            "action": f"读取文件 {safe_fstr(user_input)} 并展示内容",
                            "tool": "read_file",
                            "tool_input": {"path": user_input},
                            "task_type": "tool"
                        }
                    ],
                    "estimated_time": "10"
                }
            }

        tools_list = " / ".join(sorted(self.available_tools))
        system_prompt = f"""你是任务规划助手。将用户需求分解为可执行步骤。

        可用工具（仅限以下工具）：
        {tools_list}

        【绝对强制 - 输出格式】
        只允许输出一个有效 JSON，包含 complexity、steps、estimated_time 三个字段。
        禁止输出任何非 JSON 文字，禁止输出工具调用格式（如 read_file、bash 等）。
        如果用户输入看起来像直接命令，也必须将其封装在 JSON 的 steps 中。
        对于仅提供文件名/路径的请求，请只生成一个步骤（读取文件），不要过度分解。

        【步骤依赖规则】
        1. 如果某个步骤是 “write_file” 或 “edit_file”，它必须依赖于一个前置的 “read_file” 或 “knowledge” 步骤。
        2. 如果某个步骤是 knowledge 类型，且其 action 中明确提到了一个具体文件（如 xxx.py），则你必须先生成一个 read_file 步骤来读取该文件，并将该 knowledge 步骤的 depends_on 设为该 read_file 步骤的 id。
        3. 在输出 JSON 时，为每个有依赖的步骤添加 "depends_on": [前置步骤id] 字段。

        正确示例（输出纯 JSON）：
        {{
          "complexity": "medium",
          "steps": [
            {{"id": 1, "action": "读取文件 clarification.py", "tool": "read_file", "tool_input": {{"path": "clarification.py"}}, "task_type": "tool"}},
            {{"id": 2, "action": "总结clarification.py中的方法，类名", "tool": "none", "task_type": "knowledge", "depends_on": [1]}},
            {{"id": 3, "action": "写入总结到md文件", "tool": "write_file", "tool_input": {{"path": "summary.md", "content": "..."}}, "task_type": "tool", "depends_on": [2]}}
          ],
          "estimated_time": "120"
        }}

        返回格式（严格 JSON，不要输出其他内容，不要添加任何解释、标点或多余字符）：
        {{
          "complexity": "simple|medium|complex",
          "steps": [
            {{"id": 1, "action": "步骤描述（必须含具体文件名/目录名）", "tool": "工具名", "tool_input": {{"参数名": "参数值"}}, "task_type": "tool|knowledge"}},
            {{"id": 2, "action": "步骤描述", "tool": "none", "tool_input": {{}}, "task_type": "knowledge"}}
          ],
          "estimated_time": "预计耗时（秒）"
        }}
        """

        if self._is_high_risk_knowledge_action(user_input):
            system_prompt += (
                "\n\n【⚠️ 高风险知识任务】\n"
                "当前任务要求提供歌词片段、具体排名等无法通过本地工具核实的信息。\n"
                "你**绝对不得**编造任何歌词原文、榜单名次、具体数字。\n"
                "只需给出歌曲名称和概括性的主题描述，并一律标注“（待核实）”。\n"
                "如果无法确定，请直接说“无法提供准确信息”。"
            )

        messages = [{"role": "user", "content": user_input}]
        if context:
            ctx_parts = []
            if context.get("completed_steps"):
                ctx_parts.append(f"已完成步骤：{safe_fstr(', '.join(context['completed_steps']))}")
            if context.get("previous_task"):
                ctx_parts.append(f"上一轮任务：{safe_fstr(context['previous_task'])}")
            if context.get("files_touched"):
                ctx_parts.append(f"已操作文件：{safe_fstr(', '.join(context['files_touched'][-5:]))}")
            if context.get("current_task"):
                ctx_parts.append(f"当前任务：{safe_fstr(context['current_task'])}")
            if ctx_parts:
                context_str = "\n".join(ctx_parts)
                messages.insert(0, {"role": "system", "content": f"任务上下文：\n{safe_fstr(context_str)}"})

        max_parse_attempts = 2
        plan = None
        raw_response = ""

        for attempt in range(max_parse_attempts):
            if attempt == 0:
                response = _safe_model_call(self.model_forward_fn, messages, system_prompt, temperature=0.1)
            else:
                second_prompt = "你刚才的输出不符合 JSON 格式要求。请严格输出 JSON 计划，禁止任何额外文本。"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": second_prompt})
                response = _safe_model_call(self.model_forward_fn, messages, system_prompt, temperature=0.1)

            raw_response = response
            plan, parsed_payload = _parse_json_like_payload(response)
            if not isinstance(plan, dict):
                self.monitor.warning(f"第 {attempt + 1} 次未提取到 JSON")
                continue

            try:
                if "steps" not in plan or not isinstance(plan.get("steps"), list) or len(plan["steps"]) == 0:
                    raise ValueError("计划中缺少有效步骤")
                plan = self._sanitize_plan_steps(plan)
                plan = self._sanitize_high_risk_knowledge_plan(user_input, plan)
                # 自动添加 read_file 依赖
                plan = self._enforce_read_before_summarize(plan)
                for step in plan["steps"]:
                    if step.get("tool") == "read_file" and "tool_input" in step:
                        ti = step["tool_input"]
                        if "file_path" in ti and "path" not in ti:
                            ti["path"] = ti.pop("file_path")
                    if step.get("task_type") == "knowledge" and any(
                            kw in step.get("action", "") for kw in ["解释", "说明", "分析"]):
                        for read_step in plan["steps"]:
                            if read_step.get("tool") == "read_file":
                                file_path = read_step.get("tool_input", {}).get("path", "")
                                if file_path and file_path in step.get("action", ""):
                                    step.setdefault("depends_on", []).append(read_step["id"])
                plan = self._ensure_write_step_if_needed(user_input, plan)
                plan = self._trim_plan_to_budget(plan, max_steps=8)
                return {"success": True, "plan": plan, "raw_response": raw_response}

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                snippet = safe_fstr((parsed_payload or response or "")[:400])
                self.monitor.warning(f"第 {attempt + 1} 次 JSON 内容无效: {e} | 响应片段: {snippet}")
                if attempt == max_parse_attempts - 1:
                    return {"success": False, "error": f"计划内容无效: {e}", "raw_response": raw_response}
                continue

        return {"success": False, "error": "无法从响应中提取有效 JSON", "raw_response": raw_response}


class ExecutorAgent:
    # 省略，与之前一致，无需修改
    def __init__(self, tool_executor, available_tools: Optional[set] = None, model_forward_fn: Optional[Callable] = None):
        self.tool_executor = tool_executor
        self.available_tools = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)
        self.model_forward_fn = model_forward_fn

    def execute_step(self, step: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        tool = step.get("tool", "none")
        action = step.get("action", "")
        task_type = step.get("task_type", "tool")
        step_id = step.get("id")

        if task_type == "knowledge" or tool == "none":
            if self.model_forward_fn and self._is_reasoning_task(action):
                try:
                    context_str = ""
                    if context and context.get("previous_results"):
                        context_str = "\n\n已有信息：\n" + "\n".join([f"- {r.get('action', '')}: {self._extract_content_summary(r.get('result', ''))}" for r in context["previous_results"] if r.get("success") and r.get("result")])
                    prompt = f"请完成以下任务：{action}{context_str}"
                    messages = [{"role": "user", "content": prompt}]
                    response = _safe_model_call(self.model_forward_fn, messages, "你是一个智能助手，请直接回答用户的问题，不要添加额外的解释或格式。", temperature=0.7, top_p=0.9, max_tokens=1024)
                    return {"success": True, "step_id": step_id, "action": action, "result": response, "reasoning_task": True}
                except Exception as e:
                    return {"success": False, "step_id": step_id, "action": action, "error": f"推理任务执行失败: {str(e)}"}
            return {"success": True, "step_id": step_id, "action": action, "result": "步骤完成（无需工具）"}

        if tool not in self.available_tools:
            return {"success": False, "step_id": step_id, "action": action, "tool": tool, "error": f"工具 '{tool}' 不在可用列表中，跳过此步骤"}

        args = step.get("tool_input", {})
        if not isinstance(args, dict):
            args = {}

        try:
            result = self.tool_executor.execute_tool(tool, args)
            try:
                result_obj = json.loads(result)
                success = not result_obj.get("error")
            except (json.JSONDecodeError, AttributeError):
                success = not result.startswith("Error:")
            return {"success": success, "step_id": step_id, "action": action, "tool": tool, "result": result}
        except Exception as e:
            return {"success": False, "step_id": step_id, "action": action, "tool": tool, "error": str(e)}

    def _is_reasoning_task(self, action: str) -> bool:
        reasoning_keywords = ["列出", "给出", "解释", "说明", "分析", "总结", "描述", "比较", "评价", "推荐", "建议", "预测", "判断", "截取", "片段", "含义", "阅读", "理解"]
        return any(kw in action for kw in reasoning_keywords)

    def _extract_content_summary(self, result: str, max_len: int = 300) -> str:
        if not result:
            return ""
        try:
            result_obj = json.loads(result)
            if isinstance(result_obj, dict):
                content = result_obj.get("content", result_obj.get("stdout", str(result_obj)))
                if isinstance(content, str):
                    return content[:max_len] + ("..." if len(content) > max_len else "")
            return str(result_obj)[:max_len]
        except (json.JSONDecodeError, TypeError):
            return result[:max_len] + ("..." if len(result) > max_len else "")


class ReviewerAgent:
    def __init__(self, model_forward_fn: Callable):
        self.model_forward_fn = model_forward_fn

    def review(self, user_input: str, plan: Dict[str, Any], execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = """你是结果审查助手。评估任务完成情况。

返回格式（严格 JSON）：
{
  "completed": true|false,
  "quality": "excellent|good|fair|poor",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}

评估标准：
- 是否完成所有步骤
- 是否有错误
- 结果是否符合预期
- 是否需要改进"""
        context = f"""用户需求：{user_input}

执行计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

执行结果：
{json.dumps(execution_results, ensure_ascii=False, indent=2)}"""
        messages = [{"role": "user", "content": context}]
        try:
            response = _safe_model_call(self.model_forward_fn, messages, system_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                review = json.loads(json_match.group())
                return {"success": True, "review": review, "raw_response": response}
            else:
                all_success = all(r.get("success", False) for r in execution_results)
                return {"success": True, "review": {"completed": all_success, "quality": "good" if all_success else "fair", "issues": [] if all_success else ["部分步骤执行失败"], "suggestions": []}, "raw_response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}


class MultiAgentOrchestrator:
    # 省略，与之前一致
    def __init__(self, model_forward_fn: Callable, tool_executor, max_retries: int = 1, enable_bash: bool = False):
        available_tools = set(_AVAILABLE_TOOLS)
        if enable_bash:
            available_tools.add("bash")
        self.planner = PlannerAgent(model_forward_fn, available_tools=available_tools)
        self.executor = ExecutorAgent(tool_executor, available_tools=available_tools, model_forward_fn=model_forward_fn)
        self.reviewer = ReviewerAgent(model_forward_fn)
        self.max_retries = max_retries

    def run(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        start_time = datetime.now()
        plan_context = {}
        if context:
            plan_context["completed_steps"] = context.get("completed_steps", [])
            plan_context["previous_task"] = context.get("previous_task") or context.get("current_task", "")
            plan_context["files_touched"] = context.get("files_touched", [])
            plan_context["current_task"] = context.get("current_task", "")
            plan_context["current_topic"] = context.get("current_topic", "")
        plan_result = self.planner.plan(user_input, plan_context if any(plan_context.values()) else None)
        if not plan_result["success"]:
            return {"success": False, "stage": "planning", "error": plan_result.get("error", "规划失败")}
        plan = plan_result["plan"]
        execution_results = []
        for step in plan.get("steps", []):
            step_context = {"previous_results": execution_results}
            result = self.executor.execute_step(step, context=step_context)
            execution_results.append(result)
            if not result["success"] and step.get("critical", False):
                break
        review_result = self.reviewer.review(user_input, plan, execution_results)
        completed = False
        if review_result["success"]:
            completed = review_result["review"].get("completed", False)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return {"success": True, "completed": completed, "plan": plan, "execution_results": execution_results, "review": review_result.get("review", {}), "duration": duration, "timestamp": end_time.isoformat()}

    def run_and_generate_response(self, user_input: str, model_forward_fn: Callable, context: Optional[Dict] = None, system_prompt: str = "你是一个智能助手。请根据执行结果，完整详细地回答用户的问题。对于列表、解释、分析等内容，请展示完整结果，不要省略。", temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048) -> Dict[str, Any]:
        result = self.run(user_input, context)
        execution_summary = self._format_execution_summary(result.get("execution_results", []))
        final_prompt = f"""用户问题：{user_input}

执行过程：
{execution_summary}

请根据以上执行结果回答用户。
- 如果执行中有错误，请明确指出失败的原因（例如文件不存在、工具不可用等）
- 如果执行成功，请整合各步骤的结果，给出完整详细的回答
- 对于推理任务的结果（如列表、解释、分析等），请直接展示完整内容，不要省略或概括"""
        messages = [{"role": "user", "content": final_prompt}]
        try:
            final_response = _safe_model_call(model_forward_fn, messages, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        except Exception as e:
            final_response = f"生成最终回答时出错: {e}"
        result["final_response"] = final_response
        return result

    @classmethod
    def from_react_framework(cls, react_framework, max_retries: int = 1) -> "MultiAgentOrchestrator":
        return cls(
            model_forward_fn=react_framework.model_forward_fn,
            tool_executor=react_framework.tool_executor,
            max_retries=max_retries,
            enable_bash=react_framework.tool_executor.enable_bash,
        )

    def _format_execution_summary(self, execution_results: List[Dict[str, Any]]) -> str:
        lines = []
        for step in execution_results:
            step_id = step.get("step_id")
            action = step.get("action")
            tool = step.get("tool", "none")
            success = step.get("success", False)
            result = step.get("result", "")
            error = step.get("error", "")
            if tool == "none":
                lines.append(f"步骤 {step_id}: {safe_fstr(action)} (无需工具) - {'成功' if success else '失败'}")
                if success and result and step.get("reasoning_task"):
                    lines.append(f"  结果: {safe_fstr(result)}")
                elif not success and error:
                    lines.append(f"  错误: {safe_fstr(error[:200])}")
            else:
                lines.append(f"步骤 {step_id}: 调用工具 {tool} ({safe_fstr(action)}) - {'成功' if success else '失败'}")
                if success:
                    try:
                        if result:
                            result_obj = json.loads(result)
                            if isinstance(result_obj, dict):
                                if "error" in result_obj:
                                    lines.append(f"  错误: {safe_fstr(result_obj['error'][:200])}")
                                elif "content" in result_obj:
                                    content = result_obj["content"]
                                    summary = content[:200] + ("..." if len(content) > 200 else "")
                                    lines.append(f"  内容: {safe_fstr(summary)}")
                                elif "stdout" in result_obj:
                                    stdout = result_obj["stdout"][:200]
                                    lines.append(f"  输出: {safe_fstr(stdout)}")
                                else:
                                    lines.append(f"  结果: {safe_fstr(result[:200])}")
                            else:
                                lines.append(f"  结果: {safe_fstr(result[:200])}")
                        else:
                            lines.append("  结果: 无返回内容")
                    except (json.JSONDecodeError, TypeError):
                        lines.append(f"  结果: {safe_fstr(result[:200])}")
                else:
                    lines.append(f"  错误: {safe_fstr(error[:200])}")
            lines.append("")
        return "\n".join(lines)


class WorkflowPlanState:
    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan
        self.steps_status: Dict[int, str] = {}
        self.results: Dict[int, Dict] = {}
        self.pending_clarifications: List[str] = []
        for step in plan.get("steps", []):
            self.steps_status[step["id"]] = "pending"

    def mark_step(self, step_id: int, status: str, result: Dict = None):
        self.steps_status[step_id] = status
        if result is not None:
            self.results[step_id] = result
        else:
            self.results.setdefault(step_id, {})

    def get_next_ready_steps(self, steps: List[Dict]) -> List[Dict]:
        ready = []
        for step in steps:
            if self.steps_status.get(step["id"]) != "pending":
                continue
            if not self._deps_met(step):
                continue
            ready.append(step)
        return ready

    def _deps_met(self, step: Dict) -> bool:
        deps = step.get("depends_on", [])
        for dep_id in deps:
            if self.steps_status.get(dep_id) != "completed":
                return False
        return True

    def is_all_done(self):
        return all(status == "completed" for status in self.steps_status.values())

    def has_blocked(self):
        return any(status == "blocked" for status in self.steps_status.values())


class ReActMultiAgentOrchestrator:
    MAX_STEP_RETRIES = 2

    def __init__(self, react_framework, max_plan_steps: int = 4, max_retries: int = 1,
                 clarification_mgr: ClarificationManager = None):
        self.react_framework = react_framework
        self.max_plan_steps = max_plan_steps
        self.max_retries = max_retries
        self.enable_auto_rescue = False
        self._sqlite_lock = asyncio.Lock()

        self.planner = PlannerAgent(
            react_framework.model_forward_fn,
            available_tools=set(
                (["bash"] if react_framework.tool_executor.enable_bash else []) +
                ["read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"]
            )
        )
        self.reviewer = ReviewerAgent(react_framework.model_forward_fn)
        self.monitor = get_monitor_logger()
        self.clarification_mgr = clarification_mgr or ClarificationManager()
        self.workflow_states: Dict[str, WorkflowPlanState] = {}
        self._current_plan = None

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    def _normalize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        # 先修正 read_file 参数名
        for step in steps:
            if step.get("tool") == "read_file" and "tool_input" in step:
                ti = step["tool_input"]
                if "file_path" in ti and "path" not in ti:
                    ti["path"] = ti.pop("file_path")
        # 自动为 write_file 添加依赖
        for step in steps:
            if step.get("tool") == "write_file" and "depends_on" not in step:
                prev = [
                    s for s in steps
                    if s["id"] < step["id"] and (
                            s.get("tool") == "read_file" or
                            (s.get("tool") == "none" and s.get("task_type") == "knowledge")
                    )
                ]
                if prev:
                    step["depends_on"] = [prev[-1]["id"]]
        plan = self._auto_add_file_deps(plan)  # 补充计划
        plan = self._dedupe_read_file_steps(plan)
        plan = self._apply_plan_budget(plan)
        return plan

    def _apply_plan_budget(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if len(steps) <= self.max_plan_steps:
            return plan

        kept_steps = [copy.deepcopy(step) for step in steps[:self.max_plan_steps]]
        overflow_steps = steps[self.max_plan_steps:]
        overflow_actions = [step.get("action", "") for step in overflow_steps if step.get("action")]

        if overflow_actions:
            for step in reversed(kept_steps):
                if step.get("task_type") == "knowledge" or step.get("tool") == "none":
                    step["action"] = f"{step.get('action', '')}；并尽量合并处理：{'；'.join(overflow_actions)}"
                    break

        id_map = {}
        for idx, step in enumerate(kept_steps, start=1):
            id_map[step["id"]] = idx
            step["id"] = idx

        for step in kept_steps:
            deps = step.get("depends_on", [])
            if not isinstance(deps, list):
                continue
            remapped = [id_map[d] for d in deps if d in id_map and id_map[d] != step["id"]]
            if remapped:
                step["depends_on"] = remapped
            else:
                step.pop("depends_on", None)

        plan["steps"] = kept_steps
        self.monitor.info(f"_apply_plan_budget: 步骤数从 {len(steps)} 压缩到 {len(kept_steps)}")
        return plan

    def _dedupe_read_file_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if not steps:
            return plan

        canonical_read_ids: Dict[str, int] = {}
        duplicate_to_canonical: Dict[int, int] = {}
        kept_steps: List[Dict[str, Any]] = []

        for step in steps:
            if step.get("tool") != "read_file":
                kept_steps.append(copy.deepcopy(step))
                continue

            tool_input = step.get("tool_input", {}) or {}
            path = (tool_input.get("path") or "").strip()
            if not path:
                kept_steps.append(copy.deepcopy(step))
                continue

            canonical_id = canonical_read_ids.get(path)
            if canonical_id is None:
                canonical_read_ids[path] = step["id"]
                kept_steps.append(copy.deepcopy(step))
            else:
                duplicate_to_canonical[step["id"]] = canonical_id
                self.monitor.info(
                    f"_dedupe_read_file_steps: 合并重复读取步骤 {step['id']} -> {canonical_id} ({path})"
                )

        if not duplicate_to_canonical:
            plan["steps"] = kept_steps
            return plan

        for step in kept_steps:
            deps = step.get("depends_on", [])
            if not isinstance(deps, list):
                continue
            remapped = []
            for dep in deps:
                mapped_dep = duplicate_to_canonical.get(dep, dep)
                if mapped_dep not in remapped and mapped_dep != step.get("id"):
                    remapped.append(mapped_dep)
            if remapped:
                step["depends_on"] = remapped
            else:
                step.pop("depends_on", None)

        old_to_new: Dict[int, int] = {}
        renumbered_steps: List[Dict[str, Any]] = []
        for idx, step in enumerate(kept_steps, start=1):
            old_to_new[step["id"]] = idx
            step["id"] = idx
            renumbered_steps.append(step)

        for step in renumbered_steps:
            deps = step.get("depends_on", [])
            if not isinstance(deps, list):
                continue
            step["depends_on"] = [old_to_new.get(dep, dep) for dep in deps if old_to_new.get(dep, dep) != step["id"]]
            if not step["depends_on"]:
                step.pop("depends_on", None)

        plan["steps"] = renumbered_steps
        return plan

    def _auto_add_file_deps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if not steps:
            return plan
        self.monitor.info(f"_auto_add_file_deps: 正在检查 {len(steps)} 个步骤...")
        new_steps = []
        old_to_new = {}
        for step in steps:
            step_copy = copy.deepcopy(step)
            prepended_dep_ids = []
            if step_copy.get("task_type") == "knowledge" and step_copy.get("tool") == "none":
                action = step_copy.get("action", "")
                self.monitor.debug(f"检查 knowledge 步骤: {action}")
                referenced_files = extract_file_references(action)
                existing_deps = list(step_copy.get("depends_on", []))
                has_read_dep = any(
                    d for d in existing_deps
                    if any(s for s in steps if s["id"] == d and s.get("tool") == "read_file")
                )
                if referenced_files and not has_read_dep:
                    for filename in referenced_files:
                        self.monitor.info(f"→ 为步骤 {step_copy.get('id', '?')} 自动添加前置读取: {filename}")
                        read_step = {
                            "id": len(new_steps) + 1,
                            "action": f"读取文件 {filename}",
                            "tool": "read_file",
                            "tool_input": {"path": filename},
                            "task_type": "tool"
                        }
                        new_steps.append(read_step)
                        prepended_dep_ids.append(read_step["id"])
            step_copy["_old_depends_on"] = list(step.get("depends_on", []))
            step_copy["_prepended_dep_ids"] = prepended_dep_ids
            step_copy["id"] = len(new_steps) + 1
            old_to_new[step["id"]] = step_copy["id"]
            new_steps.append(step_copy)

        for step in new_steps:
            old_deps = step.pop("_old_depends_on", [])
            prepended_dep_ids = step.pop("_prepended_dep_ids", [])
            remapped_old_deps = [old_to_new.get(dep, dep) for dep in old_deps]
            merged_deps = []
            for dep in prepended_dep_ids + remapped_old_deps:
                if dep != step["id"] and dep not in merged_deps:
                    merged_deps.append(dep)
            if merged_deps:
                step["depends_on"] = merged_deps
            else:
                step.pop("depends_on", None)

        plan["steps"] = new_steps
        self.monitor.info(f"_auto_add_file_deps: 完成，最终 {len(new_steps)} 个步骤")
        return plan

    @staticmethod
    def _extract_missing_from_status(text: str) -> List[str]:
        reasons = []
        for line in text.split('\n'):
            if line.startswith("REASON:") or line.startswith("RECOMMENDATION:"):
                reasons.append(line.split(":", 1)[1].strip())
        return reasons if reasons else ["需要更多信息"]

    @staticmethod
    def _is_high_risk_knowledge_action(action: str) -> bool:
        high_risk_keywords = [
            "歌词", "片段", "引用", "原文", "逐字", "名言", "十首", "top", "排名", "最出名",
            "最有名", "排行榜", "具体数据", "年份", "日期", "价格", "名单", "列出", "给出",
        ]
        return any(kw in action for kw in high_risk_keywords)

    def _analyze_step_result(self, react_result: Dict, task_type: str,
                             tool_hint: str, action: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        if task_type == "tool":
            successful_calls = [tc for tc in tool_calls if tc.get("success")]
            if not successful_calls:
                if not tool_calls:
                    response = react_result.get("response", "")
                    if any(t in response for t in ["execute_python", "read_file", "write_file", "bash"]):
                        if tool_hint == "execute_python":
                            return False, {
                                "type": "execute_python_parse_failed",
                                "message": "模型输出了 execute_python 调用但格式未被系统识别",
                                "hint": "放弃使用 execute_python，请改用 bash 命令完成相同任务。"
                            }
                        return False, {
                            "type": "parse_failure",
                            "message": "模型输出了工具调用但格式未被系统识别",
                            "hint": f"请严格遵循格式：工具名独占一行，紧接着 JSON 参数。例如：\n{self._escape_braces(tool_hint)}\n{{\"参数名\": \"参数值\"}}"
                        }
                    else:
                        return False, {
                            "type": "no_tool_call",
                            "message": "模型未调用任何工具",
                            "hint": f"你必须调用工具 {self._escape_braces(tool_hint)} 来完成任务。请立即输出工具调用格式。"
                        }
                else:
                    failed = tool_calls[0]
                    error_msg = failed.get("result", {}).get("error", "未知错误")
                    stderr = failed.get("result", {}).get("stderr", "")
                    stdout = failed.get("result", {}).get("stdout", "")
                    if tool_hint == "execute_python":
                        if "SyntaxError" in error_msg or "SyntaxError" in stderr:
                            return False, {
                                "type": "python_syntax_error",
                                "message": f"代码存在语法错误: {self._escape_braces(error_msg or stderr[:200])}",
                                "hint": "请检查代码中的引号、缩进、括号是否匹配。或者改用 bash 命令完成任务。"
                            }
                        if not stdout.strip() and "open(" not in failed.get("args", {}).get("code", ""):
                            return False, {
                                "type": "execute_python_empty_output",
                                "message": "代码执行成功但未产生任何输出",
                                "hint": "请确保代码中有 print() 语句输出结果。或者改用 bash 命令。"
                            }
                    if "not found" in error_msg.lower() or "不存在" in error_msg:
                        filename = tool_calls[0].get("args", {}).get("path", "")
                        return False, {
                            "type": "file_not_found",
                            "message": self._escape_braces(error_msg),
                            "filename": self._escape_braces(filename),
                            "hint": f"文件 {self._escape_braces(filename)} 未找到，请使用 bash find 命令搜索正确路径，或检查文件名拼写。"
                        }
                    elif "permission" in error_msg.lower():
                        return False, {
                            "type": "permission_denied",
                            "message": self._escape_braces(error_msg),
                            "hint": "权限不足，请检查文件权限或更换路径。"
                        }
                    else:
                        return False, {
                            "type": "tool_error",
                            "message": self._escape_braces(error_msg),
                            "hint": "请检查工具参数是否正确，或尝试其他方法。"
                        }
            for tc in successful_calls:
                if tc.get("tool") == "execute_python":
                    result = tc.get("result", {})
                    stdout = result.get("stdout", "")
                    code = tc.get("args", {}).get("code", "")
                    if not stdout.strip() and "open(" not in code and "write" not in code:
                        return False, {
                            "type": "execute_python_empty_output",
                            "message": "代码执行成功但未产生任何输出",
                            "hint": "请确保代码中有 print() 语句输出结果，或改用 bash 命令完成任务。"
                        }
            return True, {}

        if task_type == "knowledge":
            response = react_result.get("response", "")
            if any(marker in response for marker in ("待核实", "无法获取", "无法提供", "无可靠数据")):
                return True, {}
            if len(response) < 50:
                return False, {
                    "type": "knowledge_incomplete",
                    "message": "知识回答不完整或质量不足",
                    "hint": "请提供更详细、准确的回答，并标注不确定部分。"
                }
            if self._is_high_risk_knowledge_action(action):
                if self._knowledge_response_looks_overconfident(action, response):
                    return False, {
                        "type": "knowledge_unverified",
                        "message": "回答过于具体且无证据",
                        "hint": "不要编造细节，所有不确定内容均需标注“待核实”。"
                    }
            return True, {}
        return True, {}

    @staticmethod
    def _knowledge_response_looks_overconfident(action: str, response: str) -> bool:
        caution_markers = ["待核实", "无法核实", "无法确认", "不保证", "可能", "建议核对", "我不能确认", "未提供证据"]
        numbered_items = len(re.findall(r'^\s*\d+\.', response, re.MULTILINE))
        quote_count = response.count("“") + response.count("\"")
        exactness_markers = ["含义：", "片段：", "第1", "第2", "Top", "TOP"]
        looks_exact = numbered_items >= 5 or quote_count >= 6 or any(marker in response for marker in exactness_markers)
        asks_exact_list = any(token in action for token in ["十首", "top", "排名", "片段", "原文", "歌词", "引用"])
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        item_lines = [line for line in lines if re.match(r'^\d+\.', line) or "片段：" in line or "含义：" in line]
        per_item_cautious = bool(item_lines) and all(
            any(marker in line for marker in caution_markers) for line in item_lines)
        summary_only = numbered_items == 0 and quote_count < 2
        if summary_only:
            return False
        return asks_exact_list and looks_exact and not per_item_cautious

    @staticmethod
    def _annotate_high_risk_knowledge_response(action: str, response: str) -> str:
        text = (response or "").strip()
        if not text:
            return text

        caution_markers = ("待核实", "无法核实", "无法确认", "不保证", "建议核对", "未提供证据")
        if not any(marker in text for marker in caution_markers):
            lines = text.splitlines()
            annotated_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    annotated_lines.append(line)
                    continue
                if re.match(r'^\s*(\d+\.\s+|[-*]\s+)', line):
                    if not any(marker in stripped for marker in caution_markers):
                        if "：" in line:
                            prefix, suffix = line.split("：", 1)
                            line = f"{prefix}（待核实）：{suffix}"
                        elif ":" in line:
                            prefix, suffix = line.split(":", 1)
                            line = f"{prefix}（待核实）:{suffix}"
                        else:
                            line = f"{line}（待核实）"
                annotated_lines.append(line)
            text = "\n".join(annotated_lines).strip()

        if any(token in action for token in ("歌词", "片段", "原文", "引用", "逐字")):
            prefix = (
                "## 风险提示\n"
                "以下内容基于模型生成与常见知识归纳，仅供参考；涉及逐字片段、原文或精确引述的部分均需单独核实。\n\n"
            )
        elif any(token in action for token in ("十首", "top", "排名", "最出名", "最有名", "排行榜", "名单", "列出", "给出")):
            prefix = (
                "## 风险提示\n"
                "以下内容基于模型生成与常见公众印象整理，不代表权威榜单或精确排序；名单、顺序和细节均需核实。\n\n"
            )
        else:
            prefix = (
                "## 风险提示\n"
                "以下内容基于模型生成，未附带可核验证据来源；其中具体事实与细节请结合外部权威资料核实。\n\n"
            )
        return prefix + text

    @staticmethod
    def _step_display_action(step: Dict[str, Any]) -> str:
        if not isinstance(step, dict):
            return ""
        return step.get("original_action") or step.get("action", "")

    def _build_correction_prompt(self, action, tool_hint, error_info, retry_count, step_id, total_steps) -> str:
        base = f"【纠错重试 - 第{retry_count}次】之前的执行失败了。\n"
        base += f"原始任务：{self._escape_braces(action)}\n"
        base += f"失败原因：{self._escape_braces(error_info.get('message', ''))}\n"
        base += f"纠错建议：{self._escape_braces(error_info.get('hint', ''))}\n\n"

        if error_info["type"] == "file_not_found":
            filename = error_info.get("filename", "")
            find_cmd = f"find . -name '{filename}' 2>/dev/null" if filename else "find . -name '*' 2>/dev/null"
            base += (
                f"【立即执行】文件未找到，请使用以下命令搜索，然后读取正确的路径：\n"
                f"bash\n{{\"command\": \"{self._escape_braces(find_cmd)}\"}}\n"
                "如果仍然找不到，请输出 STATUS: BLOCKED 并说明文件不存在。"
            )
        elif error_info["type"] == "execute_python_parse_failed":
            base += (
                "⚠️ **放弃使用 execute_python**，因为你编写的 Python 代码格式无法被系统解析。\n"
                "请**立即改用 bash 命令**完成该任务，例如：\n"
                "bash\n{\"command\": \"grep -Ern '^class |^def ' core/ > API.md\"}\n"
                "**不要**再尝试输出任何 execute_python 调用！\n"
            )
        elif error_info["type"] == "parse_failure":
            base += (
                "⚠️ 特别注意：工具调用格式必须为：\n"
                f"{self._escape_braces(tool_hint)}\n"
                '{"参数名": "参数值"}\n'
                "不要使用任何函数调用风格！"
            )
        elif error_info["type"] == "python_syntax_error":
            base += (
                "⚠️ 代码存在语法错误，请仔细检查。若无法修复，**请改用 bash 命令**完成相同任务。\n"
                "bash\n{\"command\": \"grep ... > output.md\"}\n"
            )
        elif error_info["type"] == "execute_python_empty_output":
            base += (
                "⚠️ 代码执行了但没有 print 输出。请添加 print 语句，或直接改用 bash 命令。\n"
                "bash\n{\"command\": \"grep ... > output.md\"}\n"
            )
        elif error_info["type"] == "no_tool_call":
            base += (
                f"你必须调用工具 {self._escape_braces(tool_hint)} 来完成任务。"
                "请立即输出工具调用格式。"
            )
        elif error_info["type"] == "knowledge_incomplete":
            base += "请输出完整、准确的回答，不要含糊其辞。"
        elif error_info["type"] == "knowledge_unverified":
            base += (
                "⚠️ 这是高风险知识题，当前没有受信证据支撑具体细节。\n"
                "不要再给出逐字歌词、名言、Top-N 精确排序或具体片段。\n"
                "请改为谨慎回答：\n"
                "1. 明确说明缺少可核验证据；\n"
                "2. 只给高层概括；\n"
                "3. 所有不确定细节必须标注“待核实”。"
            )
        base += f"\n现在请重新执行步骤 {step_id}/{total_steps}。"
        return base

    def _build_step_prompt(self, action, tool_hint, accumulated_context, step_id, total_steps,
                           task_type="tool", full_plan_steps=None) -> str:
        parts = []
        current_step = next((s for s in (full_plan_steps or []) if s.get("id") == step_id), None)
        file_evidence = StepEvidenceResolver.extract_file_evidence(current_step or {}, accumulated_context) if current_step else []
        if full_plan_steps and total_steps > 1:
            plan_overview = [f"📋 完整计划（共{total_steps}步）："]
            for s in full_plan_steps:
                sid = s.get("id", "?")
                sa = s.get("action", "")
                marker = "▶ 【当前】" if sid == step_id else (
                    "  ✅ 已完成" if f"步骤{sid}" in " ".join(accumulated_context.get("completed_steps", []))
                    else f"  ⏳ 步骤{sid}")
                plan_overview.append(f"  {marker}: {self._escape_braces(sa)}")
            parts.append("\n".join(plan_overview))
            parts.append("")

        parts.append(f"【计划执行 步骤 {step_id}/{total_steps}】{self._escape_braces(action)}")

        if task_type == "knowledge":
            if file_evidence:
                parts.append("📄 当前步骤依赖的真实文件证据：")
                parts.append(self._escape_braces(StepEvidenceResolver.build_grounding_block(file_evidence)))
                parts.append("⚠️ 你必须只依据上述真实文件内容与结构摘要回答，不得写“根据文件名推测”之类内容。")
            else:
                parts.append("（无前置文件内容，请基于自身知识回答，并标注不确定信息。）")
            parts.append("📝 此步骤为知识问答，不需要调用工具，直接输出完整回答。")
            parts.append("⚠️ 对于可能不准确的内容，请标注「（待核实）」。")
            if self._is_high_risk_knowledge_action(action):
                parts.append("🚨 高风险知识题：只做概括，不确定细节必须标注「（待核实）」。")
            parts.append("✅ 成功标准：给出谨慎回答，无证据不编造。")

        elif tool_hint and tool_hint != "none":
            parts.append("")
            parts.append("⚠️ 工具调用强制格式要求：")
            parts.append(f"{self._escape_braces(tool_hint)}")
            if tool_hint == "execute_python":
                parts.append('{"code": "你的Python代码"}')
            elif tool_hint == "read_file":
                parts.append('{"path": "文件路径"}')
            elif tool_hint == "write_file":
                parts.append('{"path": "文件路径", "content": "内容", "mode": "overwrite"}')
                # 如果 write_file 依赖前置 knowledge，则注入其输出
                dep_id, dep_text = StepEvidenceResolver.extract_dependency_response(current_step or {}, accumulated_context)
                if dep_id and dep_text:
                    parts.append(
                        f"\n⚠️ **必须将以下前置步骤{dep_id}的总结内容作为 content 写入文件，不得修改或补写：**\n"
                        f"{self._escape_braces(dep_text[:1500])}"
                    )
            elif tool_hint == "bash":
                parts.append('{"command": "shell命令"}')
            elif tool_hint == "list_dir":
                parts.append('{"path": "目录路径"}')
            elif tool_hint == "edit_file":
                parts.append('{"path": "文件路径", "old_content": "旧内容", "new_content": "新内容"}')
            else:
                parts.append('{"param": "value"}')
            parts.append("严禁使用其他参数名或命令行风格！")
            parts.append("✅ 成功标准：执行完毕后输出结果摘要。")

        completed = accumulated_context.get("completed_steps", [])
        if not isinstance(completed, list):
            completed = []
        if completed:
            parts.append("✅ 已完成的前置步骤：" + "；".join([self._escape_braces(c) for c in completed[-5:]]))

        step_outputs = accumulated_context.get("step_outputs", {})
        if step_outputs and task_type != "knowledge":
            parts.append("📄 前置步骤执行结果摘要：")
            for sid in sorted(step_outputs.keys()):
                if sid < step_id:
                    out = step_outputs[sid]
                    if isinstance(out, dict):
                        out_str = out.get("response", "") or json.dumps(out, ensure_ascii=False)
                    else:
                        out_str = str(out)
                    if out_str and "文件内容" not in out_str:
                        parts.append(f"  步骤{sid}输出: {self._escape_braces(out_str[:200])}")

        if accumulated_context.get("failed_steps"):
            parts.append("⚠️ 之前失败的步骤：" + "；".join([self._escape_braces(f) for f in accumulated_context["failed_steps"]]))

        parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _looks_like_placeholder_content(content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return True
        placeholder_markers = ("...", "待补充", "替换此处", "完整复制上方输出", "请使用前置步骤")
        return any(marker in text for marker in placeholder_markers)

    def _try_execute_grounded_knowledge_step(self, step: Dict[str, Any], accumulated_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = self._step_display_action(step)
        file_evidence = StepEvidenceResolver.extract_file_evidence(step, accumulated_context)
        if not StepEvidenceResolver.should_ground_file_task(action, file_evidence):
            return None
        self.monitor.info(
            f"步骤 {step.get('id')} 命中文件证据直执行策略，使用 {len(file_evidence)} 个前置 read_file 结果生成回答"
        )
        response = StepEvidenceResolver.build_grounded_file_response(action, file_evidence)
        return {
            "step": step,
            "success": True,
            "result": {
                "response": response,
                "tool_calls": [],
                "grounded_by_files": [item.get("path", "") for item in file_evidence],
            },
            "error_info": None,
            "blocked": False,
        }

    def _try_execute_write_file_step(self, step: Dict[str, Any], accumulated_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if step.get("tool") != "write_file":
            return None
        tool_input = copy.deepcopy(step.get("tool_input", {}) or {})
        existing_content = tool_input.get("content", "")
        dep_id, dep_text = StepEvidenceResolver.extract_dependency_response(step, accumulated_context)
        if dep_id is None or not dep_text or not self._looks_like_placeholder_content(existing_content):
            return None
        tool_input["content"] = dep_text
        self.monitor.info(
            f"步骤 {step.get('id')} 命中文件写入直执行策略，直接写入依赖步骤 {dep_id} 的总结内容"
        )
        result_str = self.react_framework.tool_executor.execute_tool("write_file", tool_input)
        try:
            result_obj = json.loads(result_str)
        except json.JSONDecodeError:
            result_obj = {"error": result_str}
        success = not result_obj.get("error")
        response = (
            f"已将步骤{dep_id}的总结内容写入文件 {tool_input.get('path', '')}"
            if success else
            f"写入文件失败：{result_obj.get('error', '未知错误')}"
        )
        return {
            "step": step,
            "success": success,
            "result": {
                "response": response,
                "tool_calls": [{
                    "tool": "write_file",
                    "args": tool_input,
                    "result": result_obj,
                    "success": success,
                }],
                "written_from_step": dep_id,
            },
            "error_info": None if success else {"message": result_obj.get("error", "写入失败")},
            "blocked": False,
        }

    def _validate_grounded_knowledge_response(self, step: Dict[str, Any], accumulated_context: Dict[str, Any],
                                              response_text: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        file_evidence = StepEvidenceResolver.extract_file_evidence(step, accumulated_context)
        if not file_evidence:
            return True, None
        suspicious_markers = (
            "根据提供的文件名",
            "可能包含",
            "可能是一个主要的类",
            "实际内容可能有所不同",
            "由于没有文件的具体内容",
            "仅供参考",
        )
        if any(marker in (response_text or "") for marker in suspicious_markers):
            return False, {
                "type": "knowledge_ignored_file_evidence",
                "message": "当前回答忽略了前置文件证据，仍在按文件名推测内容",
                "hint": "你已经拿到真实文件内容，必须基于已读文件中的类、函数、导入和代码片段作答，不得再写“可能包含”之类推测句。",
            }
        return True, None

    async def _execute_step_async(self, step: Dict, session: SessionContext,
                                  runtime_context: Dict, temperature, top_p, max_tokens,
                                  trace_id: str, accumulated_context: Dict) -> Dict:
        step_id = step.get("id")
        action = step.get("action", "")
        tool_hint = step.get("tool", "none")
        task_type = step.get("task_type", "tool")

        try:
            self.monitor.info(f"并行执行步骤 {step_id}: {action}")

            if task_type == "knowledge":
                direct_result = self._try_execute_grounded_knowledge_step(step, accumulated_context)
                if direct_result:
                    return direct_result
            elif tool_hint == "write_file":
                direct_result = self._try_execute_write_file_step(step, accumulated_context)
                if direct_result:
                    return direct_result

            plan_steps = self._current_plan.get("steps", []) if self._current_plan else []
            effective_max_tokens = min(max_tokens, 1536 if len(plan_steps) >= 6 else max_tokens)
            sub_task = self._build_step_prompt(action, tool_hint, accumulated_context,
                                               step_id, len(plan_steps),
                                               task_type=task_type, full_plan_steps=plan_steps)

            step_session = SessionContext()
            step_session.task_context["completed_steps"] = copy.copy(session.task_context.get("completed_steps", []))
            step_session.task_context["current_task"] = sub_task

            retry_count = 0
            last_error_info = None
            step_success = False
            while retry_count <= self.MAX_STEP_RETRIES and not step_success:
                if retry_count == 1 and last_error_info and last_error_info.get("type") == "file_not_found":
                    filename = last_error_info.get("filename", "")
                    correction = (
                        f"【强制指令】文件 {self._escape_braces(filename)} 未找到。你现在必须使用 bash find 搜索该文件，不要再次尝试 read_file。\n"
                        f"bash\n{{\"command\": \"find . -name '{self._escape_braces(filename)}' 2>/dev/null\"}}\n"
                        "找到正确路径后，再用 read_file 读取。"
                    )
                    current_task = sub_task + "\n\n" + correction
                elif retry_count > 0 and last_error_info:
                    correction = self._build_correction_prompt(
                        action, tool_hint, last_error_info, retry_count, step_id, len(plan_steps)
                    )
                    current_task = sub_task + "\n\n" + correction
                else:
                    current_task = sub_task

                step_session.task_context["current_task"] = current_task

                try:
                    step_thread_id = f"{trace_id}_step{step_id}_t{retry_count}"
                    async with self._sqlite_lock:
                        react_result = await self.react_framework.run(
                            user_input=current_task,
                            session=step_session,
                            history=None,
                            runtime_context={**(runtime_context or {}), "run_mode": "tools"},
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=effective_max_tokens,
                            thread_id=step_thread_id
                        )
                except Exception as e:
                    self.monitor.error(f"步骤 {step_id} 执行异常: {e}")
                    return {"step": step, "success": False, "result": None,
                            "error_info": {"message": str(e)}, "blocked": False}

                response_text = react_result.get("response", "")
                tool_calls = react_result.get("tool_calls", [])
                if (
                    task_type == "knowledge"
                    and self._is_high_risk_knowledge_action(action)
                    and not StepEvidenceResolver.extract_file_evidence(step, accumulated_context)
                ):
                    response_text = self._annotate_high_risk_knowledge_response(action, response_text)
                    react_result["response"] = response_text
                    react_result["cautious_high_risk"] = True

                if re.search(r'STATUS:\s*(BLOCKED|NEEDS_CONTEXT)', response_text, re.IGNORECASE):
                    return {
                        "step": step,
                        "success": False,
                        "result": react_result,
                        "error_info": {"message": response_text},
                        "blocked": True
                    }

                step_success, error_info = self._analyze_step_result(
                    react_result, task_type, tool_hint, action, tool_calls
                )
                if step_success and task_type == "knowledge":
                    grounded_ok, grounded_error = self._validate_grounded_knowledge_response(
                        step, accumulated_context, response_text
                    )
                    if not grounded_ok:
                        step_success = False
                        error_info = grounded_error
                if step_success:
                    return {
                        "step": step,
                        "success": True,
                        "result": react_result,
                        "error_info": None,
                        "blocked": False
                    }
                else:
                    last_error_info = error_info
                    retry_count += 1

            return {
                "step": step,
                "success": False,
                "result": None,
                "error_info": last_error_info,
                "blocked": False
            }
        except Exception as e:
            self.monitor.error(f"步骤 {step_id} 未预期的异常: {e}\n{traceback.format_exc()}")
            return {
                "step": step,
                "success": False,
                "result": None,
                "error_info": {"message": str(e)},
                "blocked": False
            }

    async def run_stream(
            self,
            user_input: str,
            session: SessionContext,
            context: Optional[Dict] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
            resume: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()
        trace_id = (runtime_context or {}).get("trace_id") or str(time.time())

        # 获取/恢复计划
        if resume and trace_id in self.workflow_states:
            state = self.workflow_states[trace_id]
            plan = state.plan
        else:
            plan_context = {
                "current_task": session.task_context.get("current_task", ""),
                "previous_task": session.task_context.get("previous_task", ""),
                "current_topic": session.task_context.get("current_topic", ""),
                "completed_steps": session.task_context.get("completed_steps", []),
            }
            plan_result = self.planner.plan(user_input, plan_context if any(plan_context.values()) else None)
            if not plan_result.get("success"):
                yield {"type": "final", "response": f"规划失败：{plan_result.get('error')}", "is_final": True}
                return
            plan = plan_result["plan"]
            plan = self._normalize_plan(plan)
            state = WorkflowPlanState(plan)
            self.workflow_states[trace_id] = state

        self._current_plan = plan
        all_steps = plan.get("steps", [])

        completed_actions = []
        partial_results = []
        final_response = ""

        try:
            while True:
                ready_steps = state.get_next_ready_steps(all_steps)
                if not ready_steps:
                    # 所有步骤均达到终态（completed/failed/blocked）
                    if all(s in ('completed', 'failed', 'blocked') for s in state.steps_status.values()):
                        break
                    await asyncio.sleep(0.5)
                    continue

                accumulated_context = {
                    "completed_steps": completed_actions,
                    "step_outputs": state.results,
                    "current_task": user_input
                }

                tasks = []
                for step in ready_steps:
                    state.mark_step(step["id"], "running")
                    tasks.append(asyncio.create_task(
                        self._execute_step_async(step, session, runtime_context,
                                                 temperature, top_p, max_tokens,
                                                 trace_id, accumulated_context)
                    ))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        self.monitor.error(f"步骤执行异常: {res}")
                        if i < len(ready_steps):
                            step = ready_steps[i]
                            state.mark_step(step["id"], "failed")
                            yield {"type": "step_failed", "step_id": step.get("id"),
                                   "error": str(res), "is_final": False}
                        continue

                    step = res.get("step")
                    if not step:
                        continue
                    step_id = step["id"]
                    if res.get("blocked"):
                        state.mark_step(step_id, "blocked")
                        yield {"type": "step_blocked", "step_id": step_id,
                               "message": res['error_info']['message'], "is_final": False}
                    elif res.get("success"):
                        state.mark_step(step_id, "completed", res["result"])
                        completed_actions.append(f"步骤{step_id}: {self._escape_braces(step.get('action'))}")
                        resp_text = (res["result"].get("response", "")[:500]
                                     if isinstance(res["result"], dict) else "")
                        partial_results.append({"step_id": step_id, "action": step.get("action"),
                                                "result": resp_text})
                        yield {"type": "step_complete", "step_id": step_id,
                               "action": step.get("action"), "result": resp_text, "is_final": False}
                    else:
                        state.mark_step(step_id, "failed", res.get("result"))
                        yield {"type": "step_failed", "step_id": step_id,
                               "error": res['error_info'].get('message', '未知错误'), "is_final": False}

            completed_steps = [s for s in all_steps if state.steps_status.get(s["id"]) == "completed"]
            blocked_steps = [s for s in all_steps if state.steps_status.get(s["id"]) == "blocked"]
            failed_steps = [s for s in all_steps if state.steps_status.get(s["id"]) == "failed"]

            # 收集每个步骤的完整响应文本（不截断）
            step_summaries = []
            for s in completed_steps:
                is_read_file = s.get("tool") == "read_file"
                consumed_by_knowledge = any(
                    s["id"] in (other.get("depends_on", []) or [])
                    and other.get("task_type") == "knowledge"
                    for other in all_steps
                )
                if is_read_file and consumed_by_knowledge:
                    continue
                result = state.results.get(s["id"], {})
                resp = (result.get("response", "") if isinstance(result, dict) else "").strip()
                resp = clean_react_tags(resp)
                if resp:
                    step_summaries.append(
                        f"### 步骤{s['id']}：{s.get('action', '')}\n{resp}"
                    )
            for s in blocked_steps:
                step_summaries.append(
                    f"### 步骤{s['id']}（阻塞）：{s.get('action', '')}\n该步骤未完成，需要用户提供更多信息。"
                )
            for s in failed_steps:
                step_summaries.append(
                    f"### 步骤{s['id']}（失败）：{s.get('action', '')}\n执行过程中出现错误。"
                )

            if not step_summaries:
                final_response = "所有任务已完成，但未产生文本内容。"
            else:
                final_response = self._build_evidence_layered_response(user_input, all_steps, state)

        except Exception as e:
            self.monitor.error(f"多 Agent 流程错误: {e}\n{traceback.format_exc()}")
            final_response = f"系统内部错误：{str(e)}"
        finally:
            duration = time.time() - start_time
            self.monitor.info(f"多 Agent 并行流程完成，耗时 {duration:.2f}s")
            yield {"type": "final", "response": final_response, "partial_results": partial_results,
                   "duration": duration, "is_final": True}


    async def run(self, user_input: str, session: SessionContext, context=None, runtime_context=None,
                  temperature=0.7, top_p=0.9, max_tokens=8192, resume=False) -> Dict[str, Any]:
        final_event = None
        async for event in self.run_stream(user_input, session, context, runtime_context,
                                           temperature, top_p, max_tokens, resume):
            if event.get("is_final"):
                final_event = event
                break
        if final_event:
            return {
                "success": True,
                "final_response": final_event.get("response", ""),
                "duration": final_event.get("duration", 0),
                "plan": getattr(self, "_current_plan", None),
                "workflow_state": None
            }
        return {"success": False, "final_response": "未收到任何结果"}

    def _format_final_from_completed(self, completed_steps: List[Dict], state: WorkflowPlanState) -> str:
        from core.components.output_cleaner import clean_react_tags
        parts = []
        previous_content = ""
        for s in completed_steps:
            result = state.results.get(s["id"])
            if not isinstance(result, dict):
                continue
            content = result.get("response", "")
            content = clean_react_tags(content)
            if previous_content and content:
                sim = self._jaccard_similarity(previous_content, content)
                if sim > 0.85:
                    continue
            parts.append(f"### ✅ {self._escape_braces(s.get('action'))}")
            parts.append(self._escape_braces(content[:500]))
            previous_content = content
        return "\n\n".join(parts)

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 800) -> str:
        content = (text or "").strip()
        if len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + "..."

    def _build_evidence_layered_response(self, user_input: str, all_steps: List[Dict[str, Any]],
                                         state: WorkflowPlanState) -> str:
        file_sections = []
        tool_sections = []
        cautious_sections = []
        unresolved_sections = []

        for step in all_steps:
            step_id = step.get("id")
            status = state.steps_status.get(step_id)
            result = state.results.get(step_id, {}) if isinstance(state.results.get(step_id), dict) else {}
            response = self._truncate_text(result.get("response", ""))
            title = f"步骤{step_id}: {self._step_display_action(step)}"

            if status == "completed":
                if result.get("grounded_by_files"):
                    file_sections.append(f"### {title}\n{response}")
                elif result.get("cautious_high_risk"):
                    cautious_sections.append(f"### {title}\n{response}")
                elif result.get("tool_calls"):
                    tool_sections.append(f"### {title}\n{response}")
                elif response:
                    cautious_sections.append(f"### {title}\n{response}")
            elif status == "blocked":
                unresolved_sections.append(f"### {title}\n需要更多上下文或参数。")
            elif status == "failed":
                unresolved_sections.append(f"### {title}\n执行失败，当前未得到可靠结果。")

        sections = [f"## 任务结果\n原始需求：{user_input.strip()}"]
        if file_sections:
            sections.append("## 基于文件证据")
            sections.extend(file_sections)
        if tool_sections:
            sections.append("## 基于工具执行")
            sections.extend(tool_sections)
        if cautious_sections:
            sections.append("## 谨慎结论 / 待核实")
            sections.extend(cautious_sections)
        if unresolved_sections:
            sections.append("## 未完成部分")
            sections.extend(unresolved_sections)
        if len(sections) == 1:
            sections.append("未产生可用结果。")
        return "\n\n".join(sections).strip()

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        def get_2grams(t):
            t = re.sub(r'\s+', '', t)
            return set(t[i:i + 2] for i in range(len(t) - 1))

        set1, set2 = get_2grams(text1), get_2grams(text2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / max(len(set1 | set2), 1)
