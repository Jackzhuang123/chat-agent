# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构（完整优化版）

优化点：
- Planner 要求 LLM 直接输出 tool_input JSON，Executor 直接使用，避免正则解析错误。
- 保留原有的步骤级重试、纠错提示、证据提取、最终补救等完整功能。
"""

import inspect
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.monitor_logger import get_monitor_logger, log_event
from core.state_manager import SessionContext

_AVAILABLE_TOOLS = {"bash", "read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"}


def _safe_model_call(model_forward_fn: Callable, messages: list, system_prompt: str = "", **kwargs) -> str:
    """安全调用 model_forward_fn，兼容同步返回 str 和流式 generator 两种情况。"""
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        return f"模型调用失败: {e}"

    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        chunks = []
        try:
            for chunk in result:
                if isinstance(chunk, str):
                    chunks.append(chunk)
        except Exception:
            pass
        return chunks[-1] if chunks else ""

    if not isinstance(result, str):
        return str(result) if result is not None else ""

    return result


class PlannerAgent:
    def __init__(self, model_forward_fn: Callable, available_tools: Optional[set] = None):
        self.model_forward_fn = model_forward_fn
        self.available_tools = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)

    def plan(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        tools_list = " / ".join(sorted(self.available_tools))
        system_prompt = f"""你是任务规划助手。将用户需求分解为可执行步骤。

        可用工具（仅限以下工具）：
        {tools_list}

        返回格式（严格 JSON，不要输出其他内容）：
        {{
          "complexity": "simple|medium|complex",
          "steps": [
            {{"id": 1, "action": "步骤描述（必须含具体文件名/目录名）", "tool": "工具名", "tool_input": {{"参数名": "参数值"}}, "task_type": "tool|knowledge"}},
            {{"id": 2, "action": "步骤描述", "tool": "none", "tool_input": {{}}, "task_type": "knowledge"}}
          ],
          "estimated_time": "预计耗时（秒）"
        }}

        核心规则（违反将导致执行失败）：
        1. 步骤数量：严格限制 2-4 个，多个用户子任务合并为操作类别，不要每个子任务单独一步。
        2. action 描述必须具体：若涉及文件操作，必须写出具体文件名（如 "用 bash 扫描 core/ 目录提取类和方法并重定向写入 API.md" 而非 "读取源文件"）。
        3. task_type 字段：需要工具的步骤填 "tool"；纯知识问答（无文件操作）填 "knowledge"，tool 填 "none"。
        4. tool_input 字段：对于工具步骤，必须提供可直接用于工具调用的完整 JSON 对象，包含所有必需参数。对于知识步骤，tool_input 为空对象 {{}}。
        5. 工具选择策略：
           - 批量搜索/统计多文件 → bash（grep/find/wc）
           - 读单文件 → read_file
           - 写文件 → write_file
           - 搜索+格式化+写入 → 优先用 bash 重定向
           - 尽量避免 execute_python，除非任务需要复杂数据计算或转换。
        6. 禁止生成幻觉文件名（如 "源文件.py"），必须使用用户明确提到的真实文件名或合理推测的路径。
        7. 如果有已完成步骤，不要重复规划。

        示例（用户有3个子任务时如何合并为2-3步）：
        用户: "扫描core目录找出所有类和方法整理成文档写入API.md，列出周杰伦10首歌，阅读session_analyzer.py解释"
        正确规划:
        - 步骤1: "用bash扫描core/目录的所有.py文件，提取类和方法名，重定向写入API.md" tool=bash task_type=tool tool_input={{"command": "grep -E '^class |^def ' core/*.py > API.md"}}
        - 步骤2: "列出周杰伦最出名十首歌并解释" tool=none task_type=knowledge tool_input={{}}
        - 步骤3: "用read_file读取ui/session_analyzer.py并解释代码" tool=read_file task_type=tool tool_input={{"path": "ui/session_analyzer.py"}}
        """
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
            response = _safe_model_call(self.model_forward_fn, messages, system_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan = json.loads(json_match.group())
                for step in plan.get("steps", []):
                    if step.get("tool", "none") not in self.available_tools:
                        step["tool"] = "none"
                    # 确保 tool_input 存在
                    if "tool_input" not in step:
                        step["tool_input"] = {}
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
        task_type = step.get("task_type", "tool")
        step_id = step.get("id")

        if task_type == "knowledge" or tool == "none":
            # 知识问答步骤
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

        # 直接使用规划阶段生成的 tool_input
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
# ReActMultiAgentOrchestrator（增强版：步骤级自动纠错与重试）
# ---------------------------------------------------------------------------
class ReActMultiAgentOrchestrator:
    """Planner + ReAct Executor + Reviewer 融合架构（增强版）。

    新增功能：
      - 每步支持智能重试（最多 MAX_STEP_RETRIES 次）
      - 根据错误类型生成针对性纠错提示（包括 execute_python 代码语法错误、空输出等）
      - 提取工具执行证据供后续步骤和最终回答使用
      - 最终回答前自动检测并补救未完成的关键任务
    """

    MAX_STEP_RETRIES = 2

    def __init__(self, react_framework, max_plan_steps: int = 4, max_retries: int = 1):
        self.react_framework = react_framework
        self.max_plan_steps = max_plan_steps
        self.max_retries = max_retries

        self.planner = PlannerAgent(
            react_framework.model_forward_fn,
            available_tools=set(
                (["bash"] if react_framework.tool_executor.enable_bash else []) +
                ["read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"]
            )
        )
        self.reviewer = ReviewerAgent(react_framework.model_forward_fn)
        self.monitor = get_monitor_logger()

    async def run(
            self,
            user_input: str,
            session: "SessionContext",
            context: Optional[Dict] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        import traceback
        start_time = time.time()
        fw = self.react_framework
        trace_id = (runtime_context or {}).get("trace_id") or (context or {}).get("trace_id") or "-"
        self.monitor.info(f"多 Agent 流程开始，trace_id={trace_id}, 输入长度: {len(user_input)}")
        log_event(
            "multi_agent_started",
            "多 Agent 流程开始",
            trace_id=trace_id,
            input_len=len(user_input),
        )

        session.task_context["_last_failed_hallucination"] = None
        if "step_outputs" in session.task_context:
            completed = session.task_context.get("completed_steps", [])
            session.task_context["step_outputs"] = {
                k: v for k, v in session.task_context["step_outputs"].items()
                if f"步骤{k}" in " ".join(completed)
            }

        # 1. 规划
        plan_ctx: Dict = {}
        if context:
            plan_ctx = {
                "completed_steps": context.get("completed_steps", []),
                "previous_task": context.get("previous_task", ""),
                "files_touched": context.get("files_touched", []),
                "current_task": context.get("current_task", user_input),
            }
        try:
            plan_result = self.planner.plan(user_input, plan_ctx if any(plan_ctx.values()) else None)
        except Exception as e:
            self.monitor.error(f"规划阶段异常: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "stage": "planning",
                "error": f"规划失败: {e}",
                "duration": time.time() - start_time,
            }

        if not plan_result["success"]:
            return {
                "success": False,
                "stage": "planning",
                "error": plan_result.get("error", "规划失败"),
                "duration": time.time() - start_time,
            }

        plan = plan_result["plan"]
        steps = plan.get("steps", [])[:self.max_plan_steps]
        self.monitor.info(f"规划完成，步骤数: {len(steps)}，复杂度: {plan.get('complexity', 'unknown')}")
        log_event(
            "multi_agent_plan_ready",
            "多 Agent 规划完成",
            trace_id=trace_id,
            complexity=plan.get("complexity", "unknown"),
            steps=len(steps),
        )

        # 2. 执行（带智能重试）
        step_results: List[Dict] = []
        tool_chain_ids: List[str] = []
        accumulated_context = {
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {},
            "task": user_input,
        }

        for step in steps:
            step_id = step.get("id", len(step_results) + 1)
            action = step.get("action", "")
            tool_hint = step.get("tool", "none")
            task_type = step.get("task_type", "tool")
            # 获取预生成的 tool_input，供后续可能使用
            pre_tool_input = step.get("tool_input", {})

            if not action:
                self.monitor.warning(f"步骤 {step_id} 缺少 action 描述，跳过")
                continue

            retry_count = 0
            step_success = False
            last_error_info: Optional[Dict] = None
            step_response = ""
            tool_calls = []
            step_iterations = 0
            step_reflection_summary = ""

            while retry_count <= self.MAX_STEP_RETRIES and not step_success:
                if retry_count > 0 and last_error_info:
                    sub_task = self._build_correction_prompt(
                        action, tool_hint, last_error_info, retry_count, step_id, len(steps)
                    )
                    self.monitor.info(f"步骤 {step_id} 重试 {retry_count}/{self.MAX_STEP_RETRIES}，错误类型: {last_error_info.get('type')}")
                else:
                    sub_task = self._build_step_prompt(
                        action, tool_hint, accumulated_context, step_id, len(steps),
                        task_type=task_type, full_plan_steps=steps
                    )

                session.task_context["completed_steps"] = []
                session.task_context["failed_attempts"] = []
                session.task_context["subtask_status"] = {}
                session.task_context["_plan_step_task_type"] = task_type
                session.task_context["_plan_step_id"] = step_id
                session.task_context["_plan_step_total"] = len(steps)
                session.task_context["_plan_step_action"] = action
                session.task_context["current_task"] = sub_task

                step_runtime_ctx = dict(runtime_context or {})
                step_runtime_ctx.update({
                    "run_mode": "tools",
                    "task": user_input,
                    "plan_step": step_id,
                    "plan_total": len(steps),
                    "max_timeout_seconds": min(len(steps) * 90, 300),
                    "_retry_attempt": retry_count,
                    # 传递预生成的工具参数，供 Agent 内部使用（可选）
                    "_pre_tool_input": pre_tool_input,
                })

                try:
                    step_thread_id = f"{trace_id or 'multi'}_step{step_id}_try{retry_count}"
                    react_result = await fw.run(
                        user_input=sub_task,
                        session=session,
                        history=None,
                        runtime_context=step_runtime_ctx,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        thread_id=step_thread_id,
                    )
                except Exception as e:
                    self.monitor.error(f"步骤 {step_id} 执行异常: {e}")
                    react_result = {"response": f"执行异常: {e}", "error": str(e)}

                tool_calls = react_result.get("tool_calls", [])
                step_success, error_info = self._analyze_step_result(
                    react_result, task_type, tool_hint, action, tool_calls
                )

                if step_success:
                    step_response = react_result.get("response", "")
                    step_iterations = react_result.get("iterations", 0)
                    step_reflection_summary = react_result.get("reflection_summary", "")
                    self.monitor.debug(f"步骤 {step_id} 成功（尝试 {retry_count + 1} 次）")
                    log_event(
                        "multi_agent_step_completed",
                        "步骤执行成功",
                        trace_id=trace_id,
                        step_id=step_id,
                        tool_hint=tool_hint,
                        retries=retry_count,
                        tool_calls=len(tool_calls),
                    )
                else:
                    last_error_info = error_info
                    retry_count += 1
                    self.monitor.warning(
                        f"步骤 {step_id} 执行失败 (尝试 {retry_count}/{self.MAX_STEP_RETRIES})，"
                        f"错误类型: {error_info.get('type')}，消息: {error_info.get('message')[:100]}"
                    )
                    log_event(
                        "multi_agent_step_failed",
                        "步骤执行失败",
                        trace_id=trace_id,
                        step_id=step_id,
                        tool_hint=tool_hint,
                        retries=retry_count,
                        error_type=error_info.get("type"),
                    )

            evidence = ""
            if step_success:
                evidence = self._extract_evidence_from_tool_calls(tool_calls)
                accumulated_context["step_outputs"][step_id] = evidence or step_response[:400]
                accumulated_context["completed_steps"].append(f"步骤{step_id}: {action}")
            else:
                accumulated_context["failed_steps"].append(f"步骤{step_id}: {action}")

            step_record = {
                "step_id": step_id,
                "action": action,
                "tool_hint": tool_hint,
                "task_type": task_type,
                "success": step_success,
                "response": step_response if step_success else "",
                "tool_calls": tool_calls,
                "evidence_summary": evidence,
                "iterations": step_iterations,
                "reflection_summary": step_reflection_summary,
                "retry_count": retry_count,
                "error_info": last_error_info if not step_success else None,
            }
            step_results.append(step_record)

            if react_result and session.current_tool_chain_id:
                tool_chain_ids.append(session.current_tool_chain_id)

            if not step_success and step.get("critical", False):
                self.monitor.warning(f"关键步骤 {step_id} 失败，终止执行")
                break

        # 3. 评审
        _review_steps = []
        for r in step_results:
            sid = r["step_id"]
            _step_out = accumulated_context.get("step_outputs", {}).get(sid, "")
            _evidence = r.get("evidence_summary", "")
            _result_text = _evidence or _step_out or r["response"]
            if not r["success"]:
                _result_text = f"[执行失败] {r['response'][:200]}"
            _review_steps.append({
                "action": r["action"],
                "success": r["success"],
                "result": _result_text[:400],
                "tool_calls_count": len(r.get("tool_calls", [])),
            })
        review_result = self.reviewer.review(user_input, plan, _review_steps)

        # 4. 最终回答（含自动补救）
        final_artifact = await self._generate_final_response_with_rescue(
            user_input, step_results, review_result, fw.model_forward_fn,
            session, accumulated_context,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        final_response = final_artifact.get("final_response", "")

        duration = time.time() - start_time
        self.monitor.info(
            f"多 Agent 流程完成，成功步骤: {sum(1 for r in step_results if r['success'])}/{len(step_results)}，"
            f"总耗时: {duration:.2f}s"
        )
        log_event(
            "multi_agent_completed",
            "多 Agent 流程完成",
            trace_id=trace_id,
            steps_total=len(step_results),
            steps_success=sum(1 for r in step_results if r["success"]),
            final_facts=len(final_artifact.get("final_facts", [])),
            unresolved=len(final_artifact.get("unresolved_issues", [])),
            duration_s=f"{duration:.2f}",
        )

        return {
            "success": True,
            "plan": plan,
            "step_results": step_results,
            "review": review_result.get("review", {}),
            "final_response": final_response,
            "final_artifact": final_artifact,
            "reflection_summary": fw.reflection.get_reflection_summary(session.reflection_history) if fw.reflection else "",
            "reflection_report": fw.reflection.get_reflection_report(session.reflection_history) if fw.reflection else "",
            "tool_chain_ids": tool_chain_ids,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }

    def _analyze_step_result(
            self,
            react_result: Dict,
            task_type: str,
            tool_hint: str,
            action: str,
            tool_calls: List[Dict]
    ) -> tuple:
        """分析步骤执行结果，返回 (是否成功, 错误信息字典)。"""
        if task_type == "tool":
            successful_calls = [tc for tc in tool_calls if tc.get("success")]
            if not successful_calls:
                if not tool_calls:
                    response = react_result.get("response", "")
                    # 检测到意图但解析失败
                    if any(t in response for t in ["execute_python", "read_file", "write_file", "bash"]):
                        # 特殊处理 execute_python 的解析失败：强制降级到 bash
                        if tool_hint == "execute_python":
                            return False, {
                                "type": "execute_python_parse_failed",
                                "message": "模型输出了 execute_python 调用但格式未被系统识别（可能因 JSON 中换行未转义）",
                                "hint": "放弃使用 execute_python，请改用 bash 命令完成相同任务。"
                            }
                        return False, {
                            "type": "parse_failure",
                            "message": "模型输出了工具调用但格式未被系统识别",
                            "hint": f"请严格遵循格式：工具名独占一行，紧接着 JSON 参数。例如：\n{tool_hint}\n{{\"参数名\": \"参数值\"}}"
                        }
                    else:
                        return False, {
                            "type": "no_tool_call",
                            "message": "模型未调用任何工具",
                            "hint": f"你必须调用工具 {tool_hint} 来完成任务。请立即输出工具调用格式。"
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
                                "message": f"代码存在语法错误: {error_msg or stderr[:200]}",
                                "hint": "请检查代码中的引号、缩进、括号是否匹配。或者改用 bash 命令完成任务。"
                            }
                        if not stdout.strip() and "open(" not in failed.get("args", {}).get("code", ""):
                            return False, {
                                "type": "execute_python_empty_output",
                                "message": "代码执行成功但未产生任何输出",
                                "hint": "请确保代码中有 print() 语句输出结果。或者改用 bash 命令。"
                            }
                    if "not found" in error_msg.lower() or "不存在" in error_msg:
                        return False, {
                            "type": "file_not_found",
                            "message": error_msg,
                            "hint": "文件路径错误，请使用 bash find 命令搜索正确路径，或检查文件名拼写。"
                        }
                    elif "permission" in error_msg.lower():
                        return False, {
                            "type": "permission_denied",
                            "message": error_msg,
                            "hint": "权限不足，请检查文件权限或更换路径。"
                        }
                    else:
                        return False, {
                            "type": "tool_error",
                            "message": error_msg,
                            "hint": "请检查工具参数是否正确，或尝试其他方法。"
                        }
            # 有成功调用，但针对 execute_python 进一步检查输出
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
            if len(response) < 50:
                return False, {
                    "type": "knowledge_incomplete",
                    "message": "知识回答不完整或质量不足",
                    "hint": "请提供更详细、准确的回答，不要使用模糊表述。"
                }
            if self._is_high_risk_knowledge_action(action):
                if self._knowledge_response_looks_overconfident(action, response):
                    return False, {
                        "type": "knowledge_unverified",
                        "message": "这是高风险知识题，但回答在无证据情况下给出了过于具体的结论",
                        "hint": "不要编造歌词、引文、排行榜、具体数量或细节。若无证据，请明确标注“待核实”，避免逐条给出确定性结论。"
                    }
            return True, {}

        return True, {}

    @staticmethod
    def _is_high_risk_knowledge_action(action: str) -> bool:
        high_risk_keywords = [
            "歌词", "片段", "引用", "原文", "逐字", "名言", "十首", "top", "排名", "最出名",
            "最有名", "排行榜", "具体数据", "年份", "日期", "价格", "名单", "列出", "给出",
        ]
        lowered = action.lower()
        return any(keyword in action or keyword in lowered for keyword in high_risk_keywords)

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
        per_item_cautious = bool(item_lines) and all(any(marker in line for marker in caution_markers) for line in item_lines)
        summary_only = numbered_items == 0 and quote_count < 2
        if summary_only:
            return False
        return asks_exact_list and looks_exact and not per_item_cautious

    def _build_correction_prompt(
            self,
            action: str,
            tool_hint: str,
            error_info: Dict,
            retry_count: int,
            step_id: int,
            total_steps: int
    ) -> str:
        """构建纠错重试提示。"""
        base = f"【纠错重试 - 第{retry_count}次】之前的执行失败了。\n"
        base += f"原始任务：{action}\n"
        base += f"失败原因：{error_info.get('message')}\n"
        base += f"纠错建议：{error_info.get('hint')}\n\n"

        if error_info["type"] == "execute_python_parse_failed":
            base += (
                "⚠️ **放弃使用 execute_python**，因为你编写的 Python 代码格式无法被系统解析。\n"
                "请**立即改用 bash 命令**完成该任务，例如：\n"
                "bash\n"
                "{\"command\": \"grep -Ern '^class |^def ' core/ > API.md\"}\n"
                "**不要**再尝试输出任何 execute_python 调用！\n"
            )
        elif error_info["type"] == "parse_failure":
            base += (
                "⚠️ 特别注意：工具调用格式必须为：\n"
                f"{tool_hint}\n"
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
        elif error_info["type"] == "file_not_found":
            base += (
                "请先用 bash 命令搜索正确路径：\n"
                'bash\n{"command": "find . -name \\"目标文件名\\" 2>/dev/null"}\n'
                "找到后再用 read_file 读取。"
            )
        elif error_info["type"] == "no_tool_call":
            base += (
                f"你必须调用工具 {tool_hint} 来完成任务。"
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

    def _extract_evidence_from_tool_calls(self, tool_calls: List[Dict]) -> str:
        evidence_parts = []
        for tc in tool_calls:
            if not tc.get("success"):
                continue
            tool = tc.get("tool")
            result = tc.get("result", {})
            if tool == "read_file" and "content" in result:
                content = result["content"]
                evidence_parts.append(f"[文件内容开始]\n{content[:2000]}\n[文件内容结束]")
            elif tool == "execute_python" and "stdout" in result:
                evidence_parts.append(f"[Python执行输出]\n{result['stdout'][:1000]}")
            elif tool == "bash" and "stdout" in result:
                evidence_parts.append(f"[命令输出]\n{result['stdout'][:1000]}")
        return "\n\n".join(evidence_parts)

    @staticmethod
    def _stringify_confirmed_fact(fact: Any) -> str:
        if isinstance(fact, dict):
            kind = fact.get("kind")
            if kind == "file_read":
                return f"已读取文件: {fact.get('path', '')}"
            if kind == "command_executed":
                command = str(fact.get("command", "")).strip()
                output = str(fact.get("output", "")).strip()
                return f"已执行命令: {command}" + (f" -> {output}" if output else "")
            tool = fact.get("tool")
            if tool:
                return f"工具成功: {tool}"
        return str(fact).strip()

    @staticmethod
    def _stringify_file_fact(file_fact: Any) -> str:
        if isinstance(file_fact, dict):
            path = str(file_fact.get("path", "")).strip()
            line_count = file_fact.get("line_count")
            classes = ", ".join(file_fact.get("classes", [])[:4])
            functions = ", ".join(file_fact.get("functions", [])[:4])
            parts = [f"文件: {path}" if path else "文件证据"]
            if line_count:
                parts.append(f"行数 {line_count}")
            if classes:
                parts.append(f"类 {classes}")
            if functions:
                parts.append(f"函数 {functions}")
            summary = str(file_fact.get("summary", "")).strip()
            if summary:
                parts.append(summary[:120])
            return "；".join(parts)
        return str(file_fact).strip()

    @staticmethod
    def _stringify_failed_action(item: Any) -> str:
        if isinstance(item, dict):
            tool = str(item.get("tool", "")).strip()
            error = str(item.get("error", "")).strip()
            args = item.get("args", {})
            target = ""
            if isinstance(args, dict):
                target = str(args.get("path") or args.get("command") or "")[:120]
            pieces = [p for p in [tool, target, error] if p]
            return " 失败: ".join([pieces[0], " | ".join(pieces[1:])]) if pieces else "失败步骤"
        return str(item).strip()

    def _build_trusted_evidence_bundle(
            self,
            session: SessionContext,
            step_results: List[Dict],
            review_result: Dict,
    ) -> Dict[str, Any]:
        ledger = (session.task_context or {}).get("facts_ledger", {}) or {}
        confirmed_facts = [
            self._stringify_confirmed_fact(item)
            for item in ledger.get("confirmed_facts", [])
            if self._stringify_confirmed_fact(item)
        ]
        file_facts_raw = ledger.get("file_facts", []) or []
        file_evidence = [
            self._stringify_file_fact(item)
            for item in file_facts_raw
            if self._stringify_file_fact(item)
        ]
        failed_attempts = [
            self._stringify_failed_action(item)
            for item in ledger.get("failed_actions", [])
            if self._stringify_failed_action(item)
        ]
        unresolved_questions = [str(item).strip() for item in ledger.get("open_questions", []) if str(item).strip()]

        successful_tools = []
        citations = []
        latest_successful_read_file = ""
        latest_successful_read_fact = ""
        for step in step_results:
            for tc in step.get("tool_calls", []):
                result = tc.get("result", {}) or {}
                if tc.get("success"):
                    tool = tc.get("tool")
                    summary = ""
                    citation = ""
                    if tool == "read_file":
                        path = str(result.get("path", "")).strip()
                        latest_successful_read_file = path or latest_successful_read_file
                        summary = f"read_file 成功: {path}" if path else "read_file 成功"
                        citation = path
                        file_fact = result.get("file_facts")
                        if file_fact:
                            latest_successful_read_fact = self._stringify_file_fact(file_fact)
                    elif tool == "bash":
                        command = str(tc.get("args", {}).get("command", "")).strip()
                        summary = f"bash 成功: {command}" if command else "bash 成功"
                        citation = command
                    else:
                        summary = f"{tool} 成功"
                    successful_tools.append({
                        "tool": tool,
                        "summary": summary,
                        "citation": citation,
                    })
                    if citation:
                        citations.append(citation)

        review = review_result.get("review", {}) if isinstance(review_result, dict) else {}
        if isinstance(review, dict):
            unresolved_questions.extend(
                str(item).strip() for item in review.get("open_questions", []) if str(item).strip()
            )

        return {
            "confirmed_facts": confirmed_facts[:12],
            "file_facts": file_facts_raw[:8],
            "file_evidence": file_evidence[:8],
            "failed_attempts": failed_attempts[:10],
            "unresolved_questions": unresolved_questions[:10],
            "successful_tools": successful_tools[:12],
            "citations": list(dict.fromkeys(citations))[:12],
            "latest_successful_read_file": latest_successful_read_file,
            "latest_successful_read_fact": latest_successful_read_fact,
        }

    @staticmethod
    def _shorten_text(text: str, limit: int = 260) -> str:
        text = re.sub(r'\s+', ' ', str(text or '')).strip()
        if not text:
            return ""
        return text[:limit] + ("..." if len(text) > limit else "")

    def _summarize_step_result_for_template(self, step: Dict[str, Any]) -> str:
        response = self._shorten_text(step.get("response", ""), limit=320)
        if step.get("success"):
            tool_calls = step.get("tool_calls", []) or []
            successful_calls = [tc for tc in tool_calls if tc.get("success")]
            if successful_calls:
                tc = successful_calls[-1]
                tool = tc.get("tool")
                result = tc.get("result", {}) or {}
                if tool == "bash":
                    command = str(tc.get("args", {}).get("command", "")).strip()
                    redirect_match = re.search(r'>\s*([^\s]+)', command)
                    if redirect_match:
                        return f"结果已写入 {redirect_match.group(1)}。"
                    stdout = self._shorten_text(result.get("stdout", ""), limit=220)
                    return stdout or "命令已执行完成。"
                if tool == "read_file":
                    path = str(result.get("path", "")).strip()
                    return response or (f"已读取并分析文件 {path}。" if path else "文件已读取完成。")
                return response or f"工具 {tool} 已执行成功。"
            return response or "任务已执行完成。"

        error_info = step.get("error_info") or {}
        message = str(error_info.get("message", "") or step.get("response", "")).strip()
        hint = str(error_info.get("hint", "")).strip()
        combined = "；".join(part for part in [message, hint] if part)
        return self._shorten_text(combined or "任务执行失败。", limit=320)

    def _build_task_results(self, step_results: List[Dict]) -> List[str]:
        task_results = []
        for idx, step in enumerate(step_results, start=1):
            action = str(step.get("action", "")).strip() or f"步骤{idx}"
            result_summary = self._summarize_step_result_for_template(step)
            item = f"{idx}. {action}"
            if result_summary:
                item += f" -> {result_summary}"
            task_results.append(item)
        return task_results[:12]

    @staticmethod
    def _render_final_response_template(artifact: Dict[str, Any]) -> str:
        sections = []
        answer = str(artifact.get("answer", "")).strip()
        if answer:
            sections.append(answer)

        task_results = [str(item).strip() for item in artifact.get("task_results", []) if str(item).strip()]
        if task_results:
            sections.append("任务结果：\n" + "\n".join(f"- {item}" for item in task_results))

        mapping = [
            ("已确认", artifact.get("confirmed_facts", [])),
            ("文件证据", artifact.get("file_evidence", [])),
            ("失败记录", artifact.get("failed_attempts", [])),
            ("未解决问题", artifact.get("unresolved_questions", [])),
            ("引用", artifact.get("citations", [])),
        ]
        for title, items in mapping:
            normalized = [str(item).strip() for item in items if str(item).strip()]
            if not normalized:
                continue
            body = "\n".join(f"- {item}" for item in normalized)
            sections.append(f"{title}：\n{body}")

        return "\n\n".join(sections).strip() or "执行完成，但未生成可展示的最终回答。"

    @staticmethod
    def _normalize_text_list(value: Any, limit: int = 8, item_limit: int = 300) -> List[str]:
        if isinstance(value, list):
            items = value
        elif isinstance(value, str) and value.strip():
            items = [value]
        else:
            items = []
        return [str(item).strip()[:item_limit] for item in items if str(item).strip()][:limit]

    def _build_fallback_structured_artifact(
            self,
            step_results: List[Dict],
            trusted_bundle: Dict[str, Any],
            error: str = "",
    ) -> Dict[str, Any]:
        answer_parts = []
        successful = [step.get("action", "") for step in step_results if step.get("success") and step.get("action")]
        failed = [step.get("action", "") for step in step_results if not step.get("success") and step.get("action")]
        if successful:
            answer_parts.append("已完成：" + "；".join(successful[:4]))
        if failed:
            answer_parts.append("失败：" + "；".join(failed[:4]))
        if error:
            answer_parts.append(f"最终结构化总结回退：{error[:160]}")

        artifact = {
            "answer": "。".join(answer_parts) if answer_parts else "执行完成，但仅能返回回退结果。",
            "task_results": self._build_task_results(step_results),
            "confirmed_facts": trusted_bundle.get("confirmed_facts", [])[:5],
            "file_evidence": trusted_bundle.get("file_evidence", [])[:5],
            "failed_attempts": trusted_bundle.get("failed_attempts", [])[:5],
            "unresolved_questions": trusted_bundle.get("unresolved_questions", [])[:5],
            "citations": trusted_bundle.get("citations", [])[:5],
        }
        return self._finalize_artifact(artifact)

    def _finalize_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        finalized = {
            "answer": str(artifact.get("answer", "")).strip(),
            "task_results": self._normalize_text_list(artifact.get("task_results", []), limit=12, item_limit=500),
            "confirmed_facts": self._normalize_text_list(artifact.get("confirmed_facts", []), limit=8),
            "file_evidence": self._normalize_text_list(artifact.get("file_evidence", []), limit=8),
            "failed_attempts": self._normalize_text_list(artifact.get("failed_attempts", []), limit=8),
            "unresolved_questions": self._normalize_text_list(artifact.get("unresolved_questions", []), limit=8),
            "citations": self._normalize_text_list(artifact.get("citations", []), limit=8),
        }
        if not finalized["answer"]:
            finalized["answer"] = "执行完成，但模型未返回结构化 answer。"
        finalized["final_response"] = self._render_final_response_template(finalized)
        finalized["final_facts"] = finalized["confirmed_facts"]
        finalized["unresolved_issues"] = finalized["unresolved_questions"]
        finalized["evidence_used"] = list(dict.fromkeys(
            finalized["file_evidence"] + finalized["citations"]
        ))[:8]
        return finalized

    @staticmethod
    def _filter_allowed_items(items: List[str], allowed: List[str]) -> List[str]:
        allowed_set = set(allowed)
        return [item for item in items if item in allowed_set]

    def _hard_repair_final_artifact(
            self,
            artifact: Dict[str, Any],
            trusted_bundle: Dict[str, Any],
            step_results: List[Dict],
            validation_issues: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        repaired = {
            "answer": str(artifact.get("answer", "")).strip(),
            "task_results": self._normalize_text_list(artifact.get("task_results", []), limit=12, item_limit=500),
            "confirmed_facts": self._filter_allowed_items(
                self._normalize_text_list(artifact.get("confirmed_facts", []), limit=8),
                trusted_bundle.get("confirmed_facts", []),
            ),
            "file_evidence": self._filter_allowed_items(
                self._normalize_text_list(artifact.get("file_evidence", []), limit=8),
                trusted_bundle.get("file_evidence", []),
            ),
            "failed_attempts": self._filter_allowed_items(
                self._normalize_text_list(artifact.get("failed_attempts", []), limit=8),
                trusted_bundle.get("failed_attempts", []),
            ),
            "unresolved_questions": self._normalize_text_list(artifact.get("unresolved_questions", []), limit=8),
            "citations": self._filter_allowed_items(
                self._normalize_text_list(artifact.get("citations", []), limit=8),
                trusted_bundle.get("citations", []),
            ),
        }
        if not repaired["task_results"]:
            repaired["task_results"] = self._build_task_results(step_results)

        if not repaired["confirmed_facts"]:
            repaired["confirmed_facts"] = trusted_bundle.get("confirmed_facts", [])[:5]
        if not repaired["file_evidence"]:
            repaired["file_evidence"] = trusted_bundle.get("file_evidence", [])[:5]
        if not repaired["failed_attempts"]:
            repaired["failed_attempts"] = trusted_bundle.get("failed_attempts", [])[:5]
        if not repaired["citations"]:
            repaired["citations"] = trusted_bundle.get("citations", [])[:5]
        if trusted_bundle.get("failed_attempts") and not repaired["unresolved_questions"]:
            repaired["unresolved_questions"] = trusted_bundle.get("unresolved_questions", [])[:5] or [
                "存在失败步骤，仍需进一步处理。"
            ]

        latest_read = trusted_bundle.get("latest_successful_read_file", "")
        if latest_read and not self._contains_any(
                "\n".join(repaired["file_evidence"] + repaired["citations"]),
                [latest_read]
        ):
            latest_read_fact = trusted_bundle.get("latest_successful_read_fact", "")
            if latest_read_fact and latest_read_fact in trusted_bundle.get("file_evidence", []):
                repaired["file_evidence"] = list(dict.fromkeys(repaired["file_evidence"] + [latest_read_fact]))[:8]
            if latest_read in trusted_bundle.get("citations", []):
                repaired["citations"] = list(dict.fromkeys(repaired["citations"] + [latest_read]))[:8]

        if not repaired["answer"]:
            successful = [step.get("response", "").strip() for step in step_results if step.get("success") and step.get("response")]
            if successful:
                repaired["answer"] = successful[-1][:300]
            else:
                actions = [step.get("action", "") for step in step_results if step.get("success") and step.get("action")]
                repaired["answer"] = "；".join(actions[:3]) or "执行完成。"

        if validation_issues and trusted_bundle.get("failed_attempts"):
            repaired["unresolved_questions"] = list(dict.fromkeys(
                repaired["unresolved_questions"] + trusted_bundle.get("unresolved_questions", [])[:5]
            ))[:8]

        return self._finalize_artifact(repaired)

    def _normalize_final_artifact(
            self,
            data: Dict[str, Any],
            step_results: List[Dict],
            trusted_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        artifact = {
            "answer": str(data.get("answer", "")).strip(),
            "task_results": self._normalize_text_list(data.get("task_results", []), limit=12, item_limit=500),
            "confirmed_facts": self._normalize_text_list(data.get("confirmed_facts", []), limit=8),
            "file_evidence": self._normalize_text_list(data.get("file_evidence", []), limit=8),
            "failed_attempts": self._normalize_text_list(data.get("failed_attempts", []), limit=8),
            "unresolved_questions": self._normalize_text_list(data.get("unresolved_questions", []), limit=8),
            "citations": self._normalize_text_list(data.get("citations", []), limit=8),
        }
        if not artifact["task_results"]:
            artifact["task_results"] = self._build_task_results(step_results)
        if not artifact["confirmed_facts"]:
            artifact["confirmed_facts"] = trusted_bundle.get("confirmed_facts", [])[:5]
        if not artifact["file_evidence"]:
            artifact["file_evidence"] = trusted_bundle.get("file_evidence", [])[:5]
        if not artifact["failed_attempts"]:
            artifact["failed_attempts"] = trusted_bundle.get("failed_attempts", [])[:5]
        if not artifact["unresolved_questions"]:
            artifact["unresolved_questions"] = trusted_bundle.get("unresolved_questions", [])[:5]
        if not artifact["citations"]:
            artifact["citations"] = trusted_bundle.get("citations", [])[:5]
        if not artifact["answer"]:
            successful = [step.get("action", "") for step in step_results if step.get("success") and step.get("action")]
            failed = [step.get("action", "") for step in step_results if not step.get("success") and step.get("action")]
            artifact["answer"] = "；".join((successful + failed)[:4]) or "执行完成，但模型未返回 answer。"
        return self._finalize_artifact(artifact)

    @staticmethod
    def _contains_any(text: str, candidates: List[str]) -> bool:
        return any(candidate and candidate in text for candidate in candidates)

    def _validate_final_answer_artifact(
            self,
            artifact: Dict[str, Any],
            trusted_bundle: Dict[str, Any],
            step_results: List[Dict],
    ) -> List[str]:
        issues = []
        allowed_confirmed = set(trusted_bundle.get("confirmed_facts", []))
        allowed_file_evidence = set(trusted_bundle.get("file_evidence", []))
        allowed_failed = set(trusted_bundle.get("failed_attempts", []))
        allowed_citations = set(trusted_bundle.get("citations", []))

        for fact in artifact.get("confirmed_facts", []):
            if fact not in allowed_confirmed:
                issues.append(f"confirmed_facts 包含未在 facts_ledger 中确认的事实: {fact}")

        for evidence in artifact.get("file_evidence", []):
            if evidence not in allowed_file_evidence:
                issues.append(f"file_evidence 包含未受信的文件结论: {evidence}")

        for item in artifact.get("failed_attempts", []):
            if item not in allowed_failed:
                issues.append(f"failed_attempts 包含未记录的失败步骤: {item}")

        for citation in artifact.get("citations", []):
            if citation not in allowed_citations:
                issues.append(f"citations 包含不在白名单中的引用: {citation}")

        latest_read = trusted_bundle.get("latest_successful_read_file", "")
        if latest_read and not self._contains_any(
                "\n".join(artifact.get("file_evidence", []) + artifact.get("citations", [])),
                [latest_read]
        ):
            issues.append(f"遗漏最近成功的 read_file 证据: {latest_read}")

        if trusted_bundle.get("failed_attempts") and not artifact.get("failed_attempts"):
            issues.append("存在失败步骤，但 failed_attempts 为空")

        if trusted_bundle.get("failed_attempts") and not artifact.get("unresolved_questions"):
            issues.append("存在失败步骤，但 unresolved_questions 为空")

        failed_actions = [step.get("action", "") for step in step_results if not step.get("success") and step.get("action")]
        if failed_actions and not artifact.get("unresolved_questions"):
            issues.append("步骤执行中存在失败动作，但 unresolved_questions 为空")

        confirmed_blob = "\n".join(artifact.get("confirmed_facts", []))
        for failed in trusted_bundle.get("failed_attempts", []):
            if failed and failed in confirmed_blob:
                issues.append(f"失败步骤被误写入 confirmed_facts: {failed}")

        return list(dict.fromkeys(issues))

    async def _generate_final_response_with_rescue(
            self,
            user_input: str,
            step_results: List[Dict],
            review_result: Dict,
            model_forward_fn,
            session: SessionContext,
            accumulated_context: Dict,
            temperature: float,
            top_p: float,
            max_tokens: int,
    ) -> Dict[str, Any]:
        trace_id = session.runtime_context.get("trace_id", "-") if getattr(session, "runtime_context", None) else "-"
        # 检查是否有「已读取文件但未解释」的情况
        read_files = []
        has_explanation = False
        for step in step_results:
            if step["success"]:
                for tc in step.get("tool_calls", []):
                    if tc.get("tool") == "read_file" and tc.get("success"):
                        read_files.append(tc.get("result", {}).get("path", ""))
            if "解释" in step.get("action", "") and step["success"]:
                has_explanation = True

        if read_files and not has_explanation:
            self.monitor.info(f"检测到文件已读取但未解释，触发自动补救: {read_files}")
            rescue_prompt = (
                f"用户需求中包含代码解释任务，你已成功读取了以下文件：{', '.join(read_files)}。"
                f"请基于文件内容，对代码进行解释。"
            )
            rescue_result = await self.react_framework.run(
                user_input=rescue_prompt,
                session=session,
                runtime_context={"run_mode": "tools", "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
                thread_id=f"{trace_id or 'multi'}_rescue"
            )
            rescue_response = rescue_result.get("response", "")
            if rescue_response:
                step_results.append({
                    "step_id": "rescue",
                    "action": "代码解释补救",
                    "success": True,
                    "response": rescue_response,
                })
                accumulated_context["step_outputs"]["rescue"] = rescue_response[:400]
                self.monitor.info("补救步骤成功，已生成代码解释")
        trusted_bundle = self._build_trusted_evidence_bundle(session, step_results, review_result)
        prompt_payload = {
            "user_input": user_input,
            "trusted_inputs": {
                "facts_ledger.confirmed_facts": trusted_bundle.get("confirmed_facts", []),
                "facts_ledger.file_facts": trusted_bundle.get("file_facts", []),
                "successful_tools": trusted_bundle.get("successful_tools", []),
                "final_artifact.evidence_used": trusted_bundle.get("citations", []),
                "failed_attempts": trusted_bundle.get("failed_attempts", []),
                "unresolved_questions": trusted_bundle.get("unresolved_questions", []),
            }
        }
        try:
            system_prompt = (
                "你是最终回答结构化器。只能基于提供的 trusted_inputs 产出 JSON。"
                "禁止引用未在 trusted_inputs 中出现的事实，禁止把失败写成成功。"
                "输出字段仅限：answer, confirmed_facts, file_evidence, failed_attempts, unresolved_questions, citations。"
                "answer 为简短结论；其余字段均为字符串数组。不要输出其他内容。"
            )
            raw_response = _safe_model_call(
                model_forward_fn,
                [{"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)}],
                system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            artifact = None
            if json_match:
                data = json.loads(json_match.group())
                artifact = self._normalize_final_artifact(data, step_results, trusted_bundle)
                validation_issues = self._validate_final_answer_artifact(artifact, trusted_bundle, step_results)
                if validation_issues:
                    correction_prompt = {
                        "instruction": "仅修正，不扩写。保留原 JSON 结构，只修复校验失败项。",
                        "validation_issues": validation_issues,
                        "trusted_inputs": prompt_payload["trusted_inputs"],
                        "previous_output": data,
                    }
                    correction_raw = _safe_model_call(
                        model_forward_fn,
                        [{"role": "user", "content": json.dumps(correction_prompt, ensure_ascii=False, indent=2)}],
                        system_prompt,
                        temperature=0,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    correction_match = re.search(r'\{[\s\S]*\}', correction_raw)
                    if correction_match:
                        corrected = json.loads(correction_match.group())
                        artifact = self._normalize_final_artifact(corrected, step_results, trusted_bundle)
                        validation_issues = self._validate_final_answer_artifact(artifact, trusted_bundle, step_results)
                    if validation_issues:
                        artifact = self._hard_repair_final_artifact(
                            artifact,
                            trusted_bundle,
                            step_results,
                            validation_issues=validation_issues,
                        )
                log_event(
                    "multi_agent_final_artifact",
                    "生成结构化最终产物",
                    trace_id=trace_id,
                    final_facts=len(artifact.get("final_facts", [])),
                    unresolved=len(artifact.get("unresolved_issues", [])),
                    evidence=len(artifact.get("evidence_used", [])),
                )
                return artifact
            artifact = self._build_fallback_structured_artifact(step_results, trusted_bundle)
            log_event(
                "multi_agent_final_artifact",
                "回退生成最终产物",
                trace_id=trace_id,
                final_facts=len(artifact.get("final_facts", [])),
                unresolved=len(artifact.get("unresolved_issues", [])),
                evidence=len(artifact.get("evidence_used", [])),
            )
            return artifact
        except Exception as e:
            self.monitor.error(f"生成最终回答失败: {e}")
            artifact = self._build_fallback_structured_artifact(step_results, trusted_bundle, error=str(e))
            log_event(
                "multi_agent_final_artifact_failed",
                "最终产物生成失败，已使用回退结果",
                trace_id=trace_id,
                error=str(e)[:200],
            )
            return artifact

    def _build_step_prompt(
            self,
            action: str,
            tool_hint: str,
            accumulated_context: Dict,
            step_id: int,
            total_steps: int,
            task_type: str = "tool",
            full_plan_steps: Optional[list] = None,
    ) -> str:
        parts = []
        if full_plan_steps and total_steps > 1:
            plan_overview = [f"📋 完整计划（共{total_steps}步）："]
            for s in full_plan_steps:
                sid = s.get("id", "?")
                sa = s.get("action", "")
                marker = "▶ 【当前】" if sid == step_id else ("  ✅ 已完成" if f"步骤{sid}" in " ".join(accumulated_context.get("completed_steps", [])) else f"  ⏳ 步骤{sid}")
                plan_overview.append(f"  {marker}: {sa}")
            parts.append("\n".join(plan_overview))
            parts.append("")

        parts.append(f"【计划执行 步骤 {step_id}/{total_steps}】{action}")

        if task_type == "knowledge":
            parts.append("📝 【知识问答步骤】：此步骤不需要调用任何工具，请直接根据已有知识回答，输出完整内容。")
            parts.append(
                "⚠️ 【防幻觉提醒】：对于引用、歌词、诗句、名人名言、具体数据等内容，"
                "请务必确认准确。如不确定，请在该内容后标注「（待核实）」，"
                "而非编造看似合理但实际错误的内容。"
            )
            if self._is_high_risk_knowledge_action(action):
                parts.append(
                    "🚨 【高风险知识题】：当前没有外部证据工具可用。"
                    "若任务要求歌词片段、原文引用、Top-N 排名、具体数量或细节，请不要自信生成。"
                    "可以给出高层概括，但所有不确定细节都必须标注「（待核实）」。"
                )
        elif tool_hint and tool_hint != "none":
            parts.append("")
            parts.append("⚠️ 工具调用强制格式要求：")
            parts.append(f"你必须输出如下标准格式（工具名独占一行，紧接着是 JSON 参数，参数名必须精确）：")
            parts.append(f"{tool_hint}")
            if tool_hint == "execute_python":
                parts.append('{"code": "你的Python代码"}')
            elif tool_hint == "read_file":
                parts.append('{"path": "文件路径"}')
            elif tool_hint == "write_file":
                parts.append('{"path": "文件路径", "content": "内容", "mode": "overwrite"}')
            elif tool_hint == "bash":
                parts.append('{"command": "shell命令"}')
            elif tool_hint == "list_dir":
                parts.append('{"path": "目录路径"}')
            elif tool_hint == "edit_file":
                parts.append('{"path": "文件路径", "old_content": "旧内容", "new_content": "新内容"}')
            else:
                parts.append('{"param": "value"}')
            parts.append("严禁使用其他参数名如 'param' 或命令行风格！")

        completed = accumulated_context.get("completed_steps", [])
        if completed:
            parts.append("✅ 已完成的前置步骤：" + "；".join(completed[-5:]))

        step_outputs: Dict = accumulated_context.get("step_outputs", {})
        if step_outputs:
            parts.append("📄 前置步骤执行结果摘要（可直接引用，无需重新执行）：")
            for sid in sorted(step_outputs.keys()):
                if sid < step_id:
                    out = step_outputs[sid]
                    if out:
                        parts.append(f"  步骤{sid}输出: {out[:300]}")

        if accumulated_context.get("failed_steps"):
            parts.append("⚠️ 以下步骤之前失败：" + "；".join(accumulated_context["failed_steps"]))

        parts.append("")
        if task_type == "knowledge":
            if self._is_high_risk_knowledge_action(action):
                parts.append("✅ 成功标准：给出谨慎回答；无证据的具体细节必须标注「（待核实）」，不得伪造原文或精确列表。")
            else:
                parts.append("✅ 成功标准：直接输出完整回答，无需调用工具。")
        else:
            parts.append("✅ 成功标准：执行完毕后输出【结果摘要】，说明做了什么、得到了什么结果。")

        return "\n".join(parts)
