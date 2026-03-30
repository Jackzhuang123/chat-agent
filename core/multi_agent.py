# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构"""

import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

_AVAILABLE_TOOLS = {"read_file", "write_file", "edit_file", "list_dir", "none"}


class PlannerAgent:
    def __init__(self, model_forward_fn: Callable, available_tools: Optional[set] = None):
        self.model_forward_fn = model_forward_fn
        self.available_tools = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)

    def plan(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        tools_list = " / ".join(sorted(self.available_tools))
        system_prompt = f"""你是任务规划助手。将用户需求分解为2-4个具体步骤。

可用工具（仅限以下工具，禁止使用未列出的工具）：
{tools_list}

返回格式（严格 JSON）：
{{
  "complexity": "simple|medium|complex",
  "steps": [
    {{"id": 1, "action": "步骤描述", "tool": "上方可用工具之一"}},
    {{"id": 2, "action": "步骤描述", "tool": "none"}}
  ],
  "estimated_time": "预计耗时（秒）"
}}

规则：
- 步骤数量：2-4个，简洁可执行
- tool 字段只能用上方可用工具，不可用 bash 等未列出工具
- 步骤之间有逻辑顺序
- 每步描述简洁（20字以内）
- 禁止重复步骤
- 如果上下文中已经有完成过的步骤，不要重复规划这些步骤"""
        messages = [{"role": "user", "content": user_input}]
        if context:
            ctx_parts = []
            if context.get("completed_steps"):
                ctx_parts.append(f"已完成步骤：{', '.join(context['completed_steps'])}")
            if context.get("previous_task"):
                ctx_parts.append(f"上一轮任务：{context['previous_task']}")
            if context.get("files_touched"):
                ctx_parts.append(f"已操作文件：{', '.join(context['files_touched'][-5:])}")
            if context.get("current_task"):
                ctx_parts.append(f"当前任务：{context['current_task']}")
            if ctx_parts:
                context_str = "任务上下文：\n" + "\n".join(ctx_parts)
            else:
                context_str = f"上下文：{json.dumps(context, ensure_ascii=False)}"
            messages.insert(0, {"role": "system", "content": context_str})
        try:
            response = self.model_forward_fn(messages, system_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan = json.loads(json_match.group())
                for step in plan.get("steps", []):
                    if step.get("tool", "none") not in self.available_tools:
                        step["tool"] = "none"
                return {"success": True, "plan": plan, "raw_response": response}
            else:
                return {"success": False, "error": "无法解析计划", "raw_response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ExecutorAgent:
    def __init__(self, tool_executor, available_tools: Optional[set] = None, model_forward_fn: Optional[Callable] = None):
        self.tool_executor = tool_executor
        self.available_tools = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)
        self.model_forward_fn = model_forward_fn

    def execute_step(self, step: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        tool = step.get("tool", "none")
        action = step.get("action", "")
        if tool == "none":
            if self.model_forward_fn and self._is_reasoning_task(action):
                try:
                    context_str = ""
                    if context and context.get("previous_results"):
                        context_str = "\n\n已有信息：\n" + "\n".join([f"- {r.get('action', '')}: {self._extract_content_summary(r.get('result', ''))}" for r in context["previous_results"] if r.get("success") and r.get("result")])
                    prompt = f"请完成以下任务：{action}{context_str}"
                    messages = [{"role": "user", "content": prompt}]
                    response = self.model_forward_fn(messages, system_prompt="你是一个智能助手，请直接回答用户的问题，不要添加额外的解释或格式。", temperature=0.7, top_p=0.9, max_tokens=1024)
                    return {"success": True, "step_id": step.get("id"), "action": action, "result": response, "reasoning_task": True}
                except Exception as e:
                    return {"success": False, "step_id": step.get("id"), "action": action, "error": f"推理任务执行失败: {str(e)}"}
            return {"success": True, "step_id": step.get("id"), "action": action, "result": "步骤完成（无需工具）"}
        if tool not in self.available_tools:
            return {"success": False, "step_id": step.get("id"), "action": action, "tool": tool, "error": f"工具 '{tool}' 不在可用列表中，跳过此步骤"}
        args = self._extract_tool_args(action, tool)
        if isinstance(args, dict) and "error" in args:
            return {"success": False, "step_id": step.get("id"), "action": action, "tool": tool, "error": args["error"], "result": json.dumps({"success": False, "error": args["error"]}, ensure_ascii=False)}
        try:
            result = self.tool_executor.execute_tool(tool, args)
            try:
                result_obj = json.loads(result)
                success = not result_obj.get("error")
            except (json.JSONDecodeError, AttributeError):
                success = not result.startswith("Error:")
            return {"success": success, "step_id": step.get("id"), "action": action, "tool": tool, "result": result}
        except Exception as e:
            return {"success": False, "step_id": step.get("id"), "action": action, "tool": tool, "error": str(e)}

    def _extract_tool_args(self, action: str, tool: str) -> Dict[str, str]:
        if tool in ["read_file", "list_dir"]:
            ext_match = re.search(r'([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)', action)
            if ext_match:
                return {"path": ext_match.group(1)}
            quoted = re.search(r'["\']([^"\']+)["\']', action)
            if quoted:
                return {"path": quoted.group(1).strip()}
            for kw in ["文件", "目录", "路径", "file", "dir", "path"]:
                match = re.search(rf'{kw}[：:\s]*([^\s"\'，。]+)', action, re.I)
                if match:
                    candidate = match.group(1).strip()
                    candidate = re.sub(r'(内容|代码|数据|信息)$', '', candidate)
                    if candidate and not candidate.isdigit() and len(candidate) > 1:
                        return {"path": candidate}
            rel_match = re.search(r'(?:\.{0,2}/)?[\w/-]+(?:/[\w/-]+)*', action)
            if rel_match:
                candidate = rel_match.group(0)
                if len(candidate) > 2 and not candidate.isdigit():
                    return {"path": candidate}
            cn_match = re.search(r'(?:读取|查看|打开|列出|扫描|阅读)\s*([^\s，。]+)', action)
            if cn_match:
                candidate = cn_match.group(1).strip()
                candidate = re.sub(r'(内容|代码|数据|信息|文件)$', '', candidate)
                if candidate and not candidate.isdigit() and len(candidate) > 1:
                    return {"path": candidate}
            return {"error": f"无法从步骤描述中提取路径: {action}"}
        elif tool == "write_file":
            path_match = re.search(r'["\']?([^\s"\']+\.[a-zA-Z0-9]+)["\']?', action)
            if not path_match:
                path_match = re.search(r'["\']?([\w/-]+)["\']?', action)
            path = path_match.group(1) if path_match else "output.txt"
            content_match = re.search(r'["\']([^"\']+)["\']', action)
            if content_match:
                content = content_match.group(1)
            else:
                content = f"[需要提供实际内容，当前步骤描述: {action}]"
            return {"path": path, "content": content}
        elif tool == "bash":
            cmd_match = re.search(r'命令[：:]?\s*(.+)', action)
            if cmd_match:
                return {"command": cmd_match.group(1).strip()}
            return {"command": action}
        elif tool == "edit_file":
            path_match = re.search(r'["\']?([^\s"\']+\.[a-zA-Z0-9]+)["\']?', action)
            if path_match:
                return {"path": path_match.group(1), "old_content": "", "new_content": ""}
            return {"error": f"无法从步骤描述中提取文件路径: {action}"}
        return {}

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
            response = self.model_forward_fn(messages, system_prompt)
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
            final_response = model_forward_fn(messages, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        except Exception as e:
            final_response = f"生成最终回答时出错: {e}"
        result["final_response"] = final_response
        return result

    @classmethod
    def from_react_framework(cls, react_framework, max_retries: int = 1) -> "MultiAgentOrchestrator":
        """从现有的 QwenAgentFramework 创建 Orchestrator（共享同一个模型和工具执行器）。

        这样 plan 模式也能复用 ReAct 的反思引擎、记忆和工具学习器。
        """
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
                lines.append(f"步骤 {step_id}: {action} (无需工具) - {'成功' if success else '失败'}")
                if success and result and step.get("reasoning_task"):
                    lines.append(f"  结果: {result}")
                elif not success and error:
                    lines.append(f"  错误: {error[:200]}")
            else:
                lines.append(f"步骤 {step_id}: 调用工具 {tool} ({action}) - {'成功' if success else '失败'}")
                if success:
                    try:
                        if result:
                            result_obj = json.loads(result)
                            if isinstance(result_obj, dict):
                                if "error" in result_obj:
                                    lines.append(f"  错误: {result_obj['error'][:200]}")
                                elif "content" in result_obj:
                                    content = result_obj["content"]
                                    summary = content[:200] + ("..." if len(content) > 200 else "")
                                    lines.append(f"  内容: {summary}")
                                elif "stdout" in result_obj:
                                    stdout = result_obj["stdout"][:200]
                                    lines.append(f"  输出: {stdout}")
                                else:
                                    lines.append(f"  结果: {result[:200]}")
                            else:
                                lines.append(f"  结果: {result[:200]}")
                        else:
                            lines.append("  结果: 无返回内容")
                    except (json.JSONDecodeError, TypeError):
                        lines.append(f"  结果: {result[:200]}")
                else:
                    lines.append(f"  错误: {error[:200]}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReActMultiAgentOrchestrator
# 融合版：Planner 生成步骤 → 每步交给 QwenAgentFramework（ReAct）执行
# 充分利用：反思引擎 + 工具学习器 + 向量记忆 + 并行工具
# ---------------------------------------------------------------------------
class ReActMultiAgentOrchestrator:
    """Planner + ReAct Executor + Reviewer 融合架构。

    相比旧 MultiAgentOrchestrator：
      • Executor 换成 QwenAgentFramework，享有：
        - DeepReflectionEngine（成功/失败模式学习）
        - AdaptiveToolLearner（序列推荐）
        - VectorMemory（跨步骤记忆）
        - 并行只读工具执行
      • 计划中的每个步骤以自然语言传递给 ReAct，
        ReAct 内部自行选择合适工具完成该步骤
      • Reviewer 评估最终结果，可将建议写回 ReAct 的 task_context

    使用示例：
        orchestrator = ReActMultiAgentOrchestrator(react_framework)
        result = orchestrator.run("分析并修复 core/ 下所有 bug")
    """

    def __init__(self, react_framework, max_plan_steps: int = 6, max_retries: int = 1):
        """
        Args:
            react_framework: QwenAgentFramework 实例（所有子模块共享）
            max_plan_steps: 计划步骤上限，防止 Planner 生成过长计划
            max_retries: 单步失败后的重试次数
        """
        self.react_framework = react_framework
        self.max_plan_steps = max_plan_steps
        self.max_retries = max_retries

        # Planner 复用 ReAct 的模型
        self.planner = PlannerAgent(
            react_framework.model_forward_fn,
            available_tools=set(
                (["bash"] if react_framework.tool_executor.enable_bash else []) +
                ["read_file", "write_file", "edit_file", "list_dir", "none"]
            )
        )
        # Reviewer 复用 ReAct 的模型
        self.reviewer = ReviewerAgent(react_framework.model_forward_fn)

    def run(
            self,
            user_input: str,
            context: Optional[Dict] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """执行完整的 Plan → ReAct Execute → Review 流水线。

        Returns:
            {
              success, plan, step_results, review,
              final_response, reflection_summary,
              tool_chain_ids, duration
            }
        """
        start_time = time.time()
        fw = self.react_framework

        # ── 1. 规划阶段 ──
        plan_ctx: Dict = {}
        if context:
            plan_ctx = {
                "completed_steps": context.get("completed_steps", []),
                "previous_task": context.get("previous_task", ""),
                "files_touched": context.get("files_touched", []),
                "current_task": context.get("current_task", user_input),
            }
        plan_result = self.planner.plan(user_input, plan_ctx if any(plan_ctx.values()) else None)
        if not plan_result["success"]:
            return {
                "success": False,
                "stage": "planning",
                "error": plan_result.get("error", "规划失败"),
                "duration": time.time() - start_time,
            }

        plan = plan_result["plan"]
        steps = plan.get("steps", [])[:self.max_plan_steps]

        # ── 2. ReAct 执行阶段（每步独立调用 ReAct）──
        step_results: List[Dict] = []
        tool_chain_ids: List[str] = []
        accumulated_context = {
            "completed_steps": [],
            "failed_steps": [],
            "task": user_input,
        }

        for step in steps:
            step_id = step.get("id", len(step_results) + 1)
            action = step.get("action", "")
            tool_hint = step.get("tool", "none")

            # 构建给 ReAct 的子任务描述（附带上下文和工具提示）
            sub_task = self._build_step_prompt(
                action, tool_hint, accumulated_context, step_id, len(steps)
            )

            # 构建子任务的 runtime_context（继承父级）
            step_runtime_ctx = dict(runtime_context or {})
            step_runtime_ctx.update({
                "run_mode": "tools",
                "task": user_input,
                "plan_step": step_id,
                "plan_total": len(steps),
            })

            # ReAct 执行（最多 max_retries+1 次）
            react_result: Optional[Dict] = None
            for attempt in range(self.max_retries + 1):
                react_result = fw.run(
                    user_input=sub_task,
                    history=None,
                    runtime_context=step_runtime_ctx,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                # 判断是否成功（无错误 && 未被中断且未检测到循环）
                if not react_result.get("error") and not react_result.get("interrupted") and not react_result.get("loop_detected"):
                    break
                if attempt < self.max_retries:
                    # 将失败信息注入下次重试的 context
                    step_runtime_ctx["_retry_reason"] = react_result.get("response", "")

            step_success = (
                react_result is not None and
                not react_result.get("error") and
                not react_result.get("interrupted")
            )

            step_record = {
                "step_id": step_id,
                "action": action,
                "tool_hint": tool_hint,
                "success": step_success,
                "response": react_result.get("response", "") if react_result else "",
                "tool_calls": react_result.get("tool_calls", []) if react_result else [],
                "iterations": react_result.get("iterations", 0) if react_result else 0,
                "reflection_summary": react_result.get("reflection_summary", "") if react_result else "",
            }
            step_results.append(step_record)

            if react_result and fw._current_tool_chain_id:
                tool_chain_ids.append(fw._current_tool_chain_id)

            # 更新累积上下文供后续步骤使用
            if step_success:
                accumulated_context["completed_steps"].append(f"步骤{step_id}: {action}")
            else:
                accumulated_context["failed_steps"].append(f"步骤{step_id}: {action}")

            # 若步骤被标记为 critical 且失败，中断
            if not step_success and step.get("critical", False):
                break

        # ── 3. 评审阶段 ──
        review_result = self.reviewer.review(
            user_input,
            plan,
            [{"action": r["action"], "success": r["success"], "result": r["response"][:300]} for r in step_results]
        )

        # ── 4. 生成最终回答（复用 ReAct 的模型）──
        final_summary = self._generate_final_response(
            user_input, step_results, review_result, fw.model_forward_fn,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        duration = time.time() - start_time
        return {
            "success": True,
            "plan": plan,
            "step_results": step_results,
            "review": review_result.get("review", {}),
            "final_response": final_summary,
            "reflection_summary": fw.reflection.get_reflection_summary() if fw.reflection else "",
            "tool_chain_ids": tool_chain_ids,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }

    def _build_step_prompt(
            self,
            action: str,
            tool_hint: str,
            accumulated_context: Dict,
            step_id: int,
            total_steps: int,
    ) -> str:
        """将计划步骤转换为 ReAct 可理解的子任务 prompt。"""
        parts = [f"【计划执行 步骤 {step_id}/{total_steps}】{action}"]
        if tool_hint and tool_hint != "none":
            parts.append(f"提示：可能需要使用 {tool_hint} 工具。")
        if accumulated_context.get("completed_steps"):
            parts.append("已完成的前置步骤：" + "；".join(accumulated_context["completed_steps"][-3:]))
        if accumulated_context.get("failed_steps"):
            parts.append("⚠️ 以下步骤之前失败：" + "；".join(accumulated_context["failed_steps"]))
        parts.append("请直接执行此步骤，完成后输出结果摘要。")
        return "\n".join(parts)

    def _generate_final_response(
            self,
            user_input: str,
            step_results: List[Dict],
            review_result: Dict,
            model_forward_fn,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 2048,
    ) -> str:
        """整合所有步骤结果，生成面向用户的最终回答。"""
        lines = [f"用户需求：{user_input}", "", "各步骤执行结果："]
        for r in step_results:
            status = "✅" if r["success"] else "❌"
            lines.append(f"{status} 步骤{r['step_id']}: {r['action']}")
            if r["response"]:
                lines.append(f"   {r['response'][:300]}")

        review = review_result.get("review", {})
        if review.get("issues"):
            lines.append("\n⚠️ 问题：" + "；".join(review["issues"][:3]))
        if review.get("suggestions"):
            lines.append("💡 建议：" + "；".join(review["suggestions"][:3]))

        prompt = "\n".join(lines)
        prompt += "\n\n请根据以上执行结果，给出完整详细的回答，不要省略重要信息。"

        try:
            return model_forward_fn(
                [{"role": "user", "content": prompt}],
                system_prompt="你是智能助手，请整合执行结果回答用户。",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # 降级：直接拼接步骤结果
            return "\n".join([
                f"{'✅' if r['success'] else '❌'} {r['action']}: {r['response'][:200]}"
                for r in step_results
            ]) or f"执行完成（生成回答时出错: {e}）"
