# -*- coding: utf-8 -*-
"""Legacy multi-agent compatibility layer.

These classes are kept for backward compatibility and tests. New production
requests should prefer ReActMultiAgentOrchestrator in core.multi_agent.
"""

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.multi_agent import PlannerAgent, _AVAILABLE_TOOLS, _safe_model_call, safe_fstr


class ExecutorAgent:
    """Legacy sequential step executor kept for compatibility."""

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
                        context_str = "\n\n已有信息：\n" + "\n".join([
                            f"- {r.get('action', '')}: {self._extract_content_summary(r.get('result', ''))}"
                            for r in context["previous_results"]
                            if r.get("success") and r.get("result")
                        ])
                    prompt = f"请完成以下任务：{action}{context_str}"
                    messages = [{"role": "user", "content": prompt}]
                    response = _safe_model_call(
                        self.model_forward_fn,
                        messages,
                        "你是一个智能助手，请直接回答用户的问题，不要添加额外的解释或格式。",
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=1024,
                    )
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

    @staticmethod
    def _is_reasoning_task(action: str) -> bool:
        reasoning_keywords = ["列出", "给出", "解释", "说明", "分析", "总结", "描述", "比较", "评价", "推荐", "建议", "预测", "判断", "截取", "片段", "含义", "阅读", "理解"]
        return any(kw in action for kw in reasoning_keywords)

    @staticmethod
    def _extract_content_summary(result: str, max_len: int = 300) -> str:
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
    """Legacy final review helper kept for compatibility."""

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
            all_success = all(r.get("success", False) for r in execution_results)
            return {
                "success": True,
                "review": {
                    "completed": all_success,
                    "quality": "good" if all_success else "fair",
                    "issues": [] if all_success else ["部分步骤执行失败"],
                    "suggestions": [],
                },
                "raw_response": response,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class MultiAgentOrchestrator:
    """Legacy sequential orchestrator kept for backward compatibility."""

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
        return {
            "success": True,
            "completed": completed,
            "plan": plan,
            "execution_results": execution_results,
            "review": review_result.get("review", {}),
            "duration": duration,
            "timestamp": end_time.isoformat(),
        }

    def run_and_generate_response(self, user_input: str, model_forward_fn: Callable, context: Optional[Dict] = None,
                                  system_prompt: str = "你是一个智能助手。请根据执行结果，完整详细地回答用户的问题。对于列表、解释、分析等内容，请展示完整结果，不要省略。",
                                  temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048) -> Dict[str, Any]:
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
            final_response = _safe_model_call(
                model_forward_fn,
                messages,
                system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
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

    @staticmethod
    def _format_execution_summary(execution_results: List[Dict[str, Any]]) -> str:
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
