# -*- coding: utf-8 -*-
"""Support policies for the multi-agent execution pipeline.

This module intentionally holds reusable helper classes so the main orchestrator
can stay focused on planning, scheduling, and step lifecycle transitions.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.state_manager import SessionContext


class StepEvidenceResolver:
    """Extract reusable evidence from prior step outputs for grounded answering."""

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
                    "真实文件内容片段：\n```python\n" + f"{content}\n```",
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


class ExecutionContextFactory:
    """Create step-scoped runtime data for the orchestrator."""

    @staticmethod
    def build_accumulated_context(user_input: str, completed_actions: List[str],
                                  step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "completed_steps": completed_actions,
            "step_outputs": step_outputs,
            "current_task": user_input,
        }

    @staticmethod
    def prepare_step_session(session: SessionContext, current_task: str) -> SessionContext:
        step_session = SessionContext()
        step_session.task_context["completed_steps"] = copy.copy(
            session.task_context.get("completed_steps", [])
        )
        step_session.task_context["current_task"] = current_task
        return step_session

    @staticmethod
    def build_step_runtime_context(runtime_context: Optional[Dict[str, Any]], step: Dict[str, Any],
                                   step_run_mode: str, retry_count: int) -> Dict[str, Any]:
        base = dict(runtime_context or {})
        base.update({
            "run_mode": step_run_mode,
            "step_id": step.get("id"),
            "step_action": step.get("action", ""),
            "step_task_type": step.get("task_type", "tool"),
            "step_retry": retry_count,
        })
        return base


class FinalResponseComposer:
    """Group completed step outputs into stable user-facing sections."""

    def __init__(self, truncate_text: Callable[[str], str], step_display_action: Callable[[Dict[str, Any]], str]):
        self.truncate_text = truncate_text
        self.step_display_action = step_display_action

    def classify_step_section(self, step: Dict[str, Any], status: str,
                              result: Dict[str, Any]) -> Tuple[str, str]:
        step_id = step.get("id")
        response = self.truncate_text(result.get("response", ""))
        title = f"步骤{step_id}: {self.step_display_action(step)}"
        if status == "completed":
            if result.get("grounded_by_files"):
                return "file_sections", f"### {title}\n{response}"
            if result.get("tool_calls"):
                return "tool_sections", f"### {title}\n{response}"
            return "cautious_sections", f"### {title}\n{response}"
        if status == "blocked":
            return "unresolved_sections", f"### {title}\n需要更多上下文或参数。"
        if status == "failed":
            return "unresolved_sections", f"### {title}\n执行失败，当前未得到可靠结果。"
        return "ignored", ""
