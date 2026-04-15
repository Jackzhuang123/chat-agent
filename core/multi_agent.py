# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构"""

import inspect
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from core.monitor_logger import get_monitor_logger

_AVAILABLE_TOOLS = {"bash", "read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"}


def _safe_model_call(model_forward_fn: Callable, messages: list, system_prompt: str = "", **kwargs) -> str:
    """安全调用 model_forward_fn，兼容同步返回 str 和流式 generator 两种情况。

    GLMAgent.generate_stream_with_messages 是 generator，
    直接传入 re.search 会报 expected string or bytes-like object。
    此函数统一消费 generator，返回最终字符串。
    """
    try:
        result = model_forward_fn(messages, system_prompt, **kwargs)
    except Exception as e:
        return f"模型调用失败: {e}"

    # 处理 generator / iterator（流式返回）
    if inspect.isgenerator(result) or hasattr(result, "__next__"):
        chunks = []
        try:
            for chunk in result:
                if isinstance(chunk, str):
                    chunks.append(chunk)
        except Exception:
            pass
        # 流式返回通常是累积字符串，取最后一个（最完整）
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
    {{"id": 1, "action": "步骤描述（必须含具体文件名/目录名）", "tool": "工具名", "task_type": "tool|knowledge"}},
    {{"id": 2, "action": "步骤描述", "tool": "none", "task_type": "knowledge"}}
  ],
  "estimated_time": "预计耗时（秒）"
}}

核心规则（违反将导致执行失败）：
1. 步骤数量：严格限制 2-4 个，多个用户子任务合并为操作类别，不要每个子任务单独一步
2. action 描述必须具体：若涉及文件操作，必须写出具体文件名（如 "用 execute_python 扫描 core/ 目录整理类和方法写入 API.md" 而非 "读取源文件"）
3. task_type 字段：需要工具的步骤填 "tool"；纯知识问答（无文件操作）填 "knowledge"，tool 填 "none"
4. 工具选择策略（重要）：
   - 批量搜索/统计多文件 → bash（grep/find/wc）
   - 读单文件 → read_file
   - 写简单文件（几行内容）→ write_file
   - 需要「搜索+格式化+写入文件」→ 用 execute_python（代码中搜索、格式化、写入文件，最后必须 print 确认）
     ⚠️ 禁止用 write_file 写入从其他工具获取的大段内容！模型无法在参数中填入实际内容，只能写占位符
     ✅ 正确：execute_python 代码中完成搜索+格式化+写入+print确认
     ❌ 错误：bash搜索后用write_file写结果（模型只会写占位符如"<完整复制上方输出>"）
   - 【禁止】用 bash grep 直接重定向写文档：`grep ... > 文件.md` 只会输出 "文件名:行号:内容" 格式的原始数据，不是人类可读的文档
5. 禁止生成幻觉文件名（如 "源文件.py"、"文件名.py"），必须使用用户明确提到的真实文件名
6. 如果有已完成步骤，不要重复规划

示例（用户有3个子任务时如何合并为2-3步）：
用户: "扫描core目录找出所有类和方法整理成文档写入API.md，列出周杰伦10首歌，阅读main.py解释"
正确规划:
- 步骤1: "用execute_python扫描core/目录的所有.py文件，提取类和方法名，格式化为Markdown写入API.md，最后print确认" tool=execute_python task_type=tool
- 步骤2: "列出周杰伦最出名十首歌并解释" tool=none task_type=knowledge
- 步骤3: "用read_file读取main.py并解释代码" tool=read_file task_type=tool
（execute_python代码中必须包含print确认写入成功，避免stdout为空导致重试）

错误示例（不要这样做）：
- 步骤1: "用bash搜索后用write_file写入API.md" tool=bash  ← 错误！模型无法在write_file参数中填入bash的实际输出，只会写占位符
- 步骤1: "用execute_python扫描并写入API.md"（但代码没加print） ← 错误！stdout为空导致无限重试
- 步骤1: "用bash grep重定向写入API.md" tool=bash  ← 错误！bash grep输出不是人类可读的格式化文档"""
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
                    response = _safe_model_call(self.model_forward_fn, messages, "你是一个智能助手，请直接回答用户的问题，不要添加额外的解释或格式。", temperature=0.7, top_p=0.9, max_tokens=1024)
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
        if not isinstance(action, str):
            print(f"⚠️ 工具参数提取失败：action 类型错误 (期望 str，实际 {type(action).__name__})")
            return {"error": f"无效的步骤描述类型: {type(action).__name__}"}

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

    def __init__(self, react_framework, max_plan_steps: int = 4, max_retries: int = 1):
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
        # ⚠️ execute_python 必须加入白名单，否则 ReAct Executor 调用时会报"未知工具"
        self.planner = PlannerAgent(
            react_framework.model_forward_fn,
            available_tools=set(
                (["bash"] if react_framework.tool_executor.enable_bash else []) +
                ["read_file", "write_file", "edit_file", "list_dir", "execute_python", "none"]
            )
        )
        # Reviewer 复用 ReAct 的模型
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
        self.monitor.info(f"多 Agent 流程开始，输入长度: {len(user_input)}")
        # ── 1. 规划阶段 ──
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
            print(f"❌ 规划阶段异常: {e}\n{traceback.format_exc()}")
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

        # ── 2. ReAct 执行阶段 ──
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
            task_type = step.get("task_type", "tool")
            self.monitor.debug(f"执行步骤 {step_id}/{len(steps)}: {action[:50]}")

            # ── 新增：防御性检查 ─────────────────────────────────────────────
            if not action:
                print(f"⚠️ 步骤 {step_id} 缺少 action 描述，跳过")
                continue

            # 重置步骤级别状态
            session.task_context["completed_steps"] = []
            session.task_context["failed_attempts"] = []
            session.task_context["subtask_status"] = {}
            session.task_context["_plan_step_task_type"] = task_type
            session.task_context["_plan_step_id"] = step_id
            session.task_context["_plan_step_total"] = len(steps)
            session.task_context["_plan_step_action"] = action

            session.task_context["current_task"] = action  # 先临时设置为 action，稍后会被 fw.run 覆盖
            # 构建子任务描述
            try:
                sub_task = self._build_step_prompt(
                    action, tool_hint, accumulated_context, step_id, len(steps),
                    task_type=task_type, full_plan_steps=steps,
                )
            except Exception as e:
                print(f"❌ 构建步骤 {step_id} 提示词失败: {e}\n{traceback.format_exc()}")
                step_results.append({
                    "step_id": step_id, "action": action, "success": False,
                    "error": f"构建提示词异常: {e}"
                })
                continue

            session.task_context["current_task"] = sub_task
            step_runtime_ctx = dict(runtime_context or {})
            step_runtime_ctx.update({
                "run_mode": "tools",
                "task": user_input,
                "plan_step": step_id,
                "plan_total": len(steps),
                "max_timeout_seconds": min(len(steps) * 90, 300),
            })

            # ReAct 执行（最多 max_retries+1 次）
            react_result: Optional[Dict] = None
            for attempt in range(self.max_retries + 1):
                try:
                    react_result = await fw.run(
                        user_input=sub_task,
                        session=session,
                        history=None,
                        runtime_context=step_runtime_ctx,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    print(f"❌ 步骤 {step_id} 执行异常 (尝试 {attempt + 1}): {e}\n{traceback.format_exc()}")
                    react_result = {"response": f"执行异常: {e}", "error": str(e)}
                if not react_result.get("error") and not react_result.get("interrupted") and not react_result.get(
                        "loop_detected"):
                    break
                if attempt < self.max_retries:
                    step_runtime_ctx["_retry_reason"] = react_result.get("response", "")

            step_success = (
                    react_result is not None and
                    not react_result.get("error") and
                    not react_result.get("interrupted") and
                    not react_result.get("timed_out")
            )

            step_record = {
                "step_id": step_id,
                "action": action,
                "tool_hint": tool_hint,
                "task_type": task_type,
                "success": step_success,
                "response": react_result.get("response", "") if react_result else "",
                "tool_calls": react_result.get("tool_calls", []) if react_result else [],
                "iterations": react_result.get("iterations", 0) if react_result else 0,
                "reflection_summary": react_result.get("reflection_summary", "") if react_result else "",
            }
            step_results.append(step_record)

            if react_result and session.current_tool_chain_id:
                tool_chain_ids.append(session.current_tool_chain_id)

            # 更新累积上下文
            if step_success:
                self.monitor.debug(f"步骤 {step_id} 成功")
                accumulated_context["completed_steps"].append(f"步骤{step_id}: {action}")
                if "step_outputs" not in accumulated_context:
                    accumulated_context["step_outputs"] = {}
                accumulated_context["step_outputs"][step_id] = step_record["response"][:400]
            else:
                self.monitor.warning(f"步骤 {step_id} 失败")
                accumulated_context["failed_steps"].append(f"步骤{step_id}: {action}")

            if not step_success and step.get("critical", False):
                break

        # ── 3. 评审阶段 ──
        _review_steps = []
        for r in step_results:
            sid = r["step_id"]
            _step_out = accumulated_context.get("step_outputs", {}).get(sid, "")
            _result_text = _step_out or r["response"]
            if not r["success"]:
                _result_text = f"[执行失败] {r['response'][:200]}"
            _review_steps.append({
                "action": r["action"],
                "success": r["success"],
                "result": _result_text[:400],
                "tool_calls_count": len(r.get("tool_calls", [])),
            })
        review_result = self.reviewer.review(user_input, plan, _review_steps)

        # ── 4. 生成最终回答 ──
        final_summary = self._generate_final_response(
            user_input, step_results, review_result, fw.model_forward_fn,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        duration = time.time() - start_time

        self.monitor.info(
            f"多 Agent 流程完成，成功步骤: {sum(1 for r in step_results if r['success'])}/{len(step_results)}，"
            f"总耗时: {duration:.2f}s"
        )

        return {
            "success": True,
            "plan": plan,
            "step_results": step_results,
            "review": review_result.get("review", {}),
            "final_response": final_summary,
            "reflection_summary": fw.reflection.get_reflection_summary(
                session.reflection_history) if fw.reflection else "",
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
            task_type: str = "tool",
            full_plan_steps: Optional[list] = None,
    ) -> str:
        """将计划步骤转换为 ReAct 可理解的子任务 prompt。

        改进点：
        - 注入完整计划概览，让 ReAct 知道全局目标和当前所处位置
        - 知识类任务（task_type=knowledge）明确告知不需要工具
        - 前置完成步骤完整透传（最多5条），而非仅3条
        - 明确成功标准：必须完成什么才算完成本步骤
        """
        parts = []

        # ── 全局计划概览（首步或未完成步骤时注入）──────────────────────────────
        if full_plan_steps and total_steps > 1:
            plan_overview = [f"📋 完整计划（共{total_steps}步）："]
            for s in full_plan_steps:
                sid = s.get("id", "?")
                sa = s.get("action", "")
                marker = "▶ 【当前】" if sid == step_id else ("  ✅ 已完成" if f"步骤{sid}" in " ".join(accumulated_context.get("completed_steps", [])) else f"  ⏳ 步骤{sid}")
                plan_overview.append(f"  {marker}: {sa}")
            parts.append("\n".join(plan_overview))
            parts.append("")

        # ── 当前步骤指令 ──────────────────────────────────────────────────────
        parts.append(f"【计划执行 步骤 {step_id}/{total_steps}】{action}")

        # ── 任务类型处理 ──────────────────────────────────────────────────────
        if task_type == "knowledge":
            parts.append("📝 【知识问答步骤】：此步骤不需要调用任何工具，请直接根据已有知识回答，输出完整内容。")
            parts.append(
                "⚠️ 【防幻觉提醒】：对于引用、歌词、诗句、名人名言、具体数据等内容，"
                "请务必确认准确。如不确定，请在该内容后标注「（待核实）」，"
                "而非编造看似合理但实际错误的内容。模糊的记忆远好于自信的错误。"
            )
        elif tool_hint and tool_hint != "none":
            parts.append("")
            parts.append("⚠️ 工具调用强制格式要求：")
            parts.append(f"你必须输出如下标准格式（工具名独占一行，紧接着是 JSON 参数，参数名必须精确）：")
            parts.append(f"{tool_hint}")
            # 根据工具类型提供参数模板
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

        # ── 前置完成步骤（最多5条，完整透传）──────────────────────────────────
        completed = accumulated_context.get("completed_steps", [])
        if completed:
            parts.append("✅ 已完成的前置步骤：" + "；".join(completed[-5:]))

        # ── 前置步骤输出摘要（关键！让当前步骤知道前面发现了什么）────────────
        step_outputs: Dict = accumulated_context.get("step_outputs", {})
        if step_outputs:
            parts.append("📄 前置步骤执行结果摘要（可直接引用，无需重新执行）：")
            for sid in sorted(step_outputs.keys()):
                if sid < step_id:  # 只注入当前步骤之前的输出
                    out = step_outputs[sid]
                    if out:
                        parts.append(f"  步骤{sid}输出: {out[:300]}")

        if accumulated_context.get("failed_steps"):
            parts.append("⚠️ 以下步骤之前失败：" + "；".join(accumulated_context["failed_steps"]))

        # ── 成功标准 ────────────────────────────────────────────────────────
        parts.append("")
        if task_type == "knowledge":
            parts.append("✅ 成功标准：直接输出完整回答，无需调用工具。")
        else:
            parts.append("✅ 成功标准：执行完毕后输出【结果摘要】，说明做了什么、得到了什么结果。")

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
                # 知识类步骤（如歌曲列表、代码解释）响应可能很长，不能截断太短
                # 工具类步骤摘要一般较短；knowledge 步骤内容本身就是最终答案的一部分
                task_type = r.get("task_type", "tool")
                max_resp = 2000 if task_type == "knowledge" else 1500
                lines.append(f"   {r['response'][:max_resp]}")

        review = review_result.get("review", {})
        if review.get("issues"):
            lines.append("\n⚠️ 问题：" + "；".join(review["issues"][:3]))
        if review.get("suggestions"):
            lines.append("💡 建议：" + "；".join(review["suggestions"][:3]))

        prompt = "\n".join(lines)
        prompt += "\n\n请根据以上执行结果，给出完整详细的回答，不要省略重要信息。"

        try:
            return _safe_model_call(
                model_forward_fn,
                [{"role": "user", "content": prompt}],
                "你是智能助手，请整合执行结果回答用户。",
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
