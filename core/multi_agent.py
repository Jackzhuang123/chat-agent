# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构"""

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# 可用工具集合（与 ToolExecutor 保持一致）
_AVAILABLE_TOOLS = {"read_file", "write_file", "edit_file", "list_dir", "none"}
# bash 仅在 enable_bash=True 时追加，由 MultiAgentOrchestrator 控制


class PlannerAgent:
    """
    规划 Agent - 负责任务分解和规划

    职责：
    - 将复杂任务分解为具体步骤
    - 评估任务复杂度
    - 生成执行计划
    """

    def __init__(self, model_forward_fn: Callable, available_tools: Optional[set] = None):
        self.model_forward_fn = model_forward_fn
        # 可用工具白名单，由 MultiAgentOrchestrator 传入，默认使用全局集合
        self.available_tools: set = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)

    def plan(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """生成执行计划"""
        # 动态生成工具列表，只列出实际可用的工具
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
            # 注入上下文：已完成步骤 + 上一轮任务进度
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

            # 尝试解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan = json.loads(json_match.group())
                # 过滤掉不在可用工具白名单里的步骤工具，改为 none
                for step in plan.get("steps", []):
                    if step.get("tool", "none") not in self.available_tools:
                        step["tool"] = "none"
                return {
                    "success": True,
                    "plan": plan,
                    "raw_response": response
                }
            else:
                return {
                    "success": False,
                    "error": "无法解析计划",
                    "raw_response": response
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ExecutorAgent:
    """
    执行 Agent - 负责执行具体步骤

    职责：
    - 执行单个步骤
    - 调用工具
    - 返回执行结果
    """

    def __init__(self, tool_executor, available_tools: Optional[set] = None, model_forward_fn: Optional[Callable] = None):
        self.tool_executor = tool_executor
        self.available_tools: set = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)
        self.model_forward_fn = model_forward_fn  # 用于执行推理型任务

    def execute_step(self, step: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行单个步骤

        Args:
            step: 步骤信息
            context: 上下文信息，包含之前步骤的执行结果
        """
        tool = step.get("tool", "none")
        action = step.get("action", "")

        if tool == "none":
            # 对于推理型任务（如"列出周杰伦歌曲"、"解释代码"），调用LLM执行
            if self.model_forward_fn and self._is_reasoning_task(action):
                try:
                    # 构建上下文提示
                    context_str = ""
                    if context and context.get("previous_results"):
                        context_str = "\n\n已有信息：\n" + "\n".join([
                            f"- {r.get('action', '')}: {self._extract_content_summary(r.get('result', ''))}"
                            for r in context["previous_results"]
                            if r.get("success") and r.get("result")
                        ])

                    prompt = f"请完成以下任务：{action}{context_str}"
                    messages = [{"role": "user", "content": prompt}]

                    response = self.model_forward_fn(
                        messages,
                        system_prompt="你是一个智能助手，请直接回答用户的问题，不要添加额外的解释或格式。",
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=1024
                    )

                    return {
                        "success": True,
                        "step_id": step.get("id"),
                        "action": action,
                        "result": response,
                        "reasoning_task": True
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "step_id": step.get("id"),
                        "action": action,
                        "error": f"推理任务执行失败: {str(e)}"
                    }

            # 非推理型任务（如"等待"、"确认"等），直接标记完成
            return {
                "success": True,
                "step_id": step.get("id"),
                "action": action,
                "result": "步骤完成（无需工具）"
            }

        # 过滤不在可用工具白名单的工具，降级为 none（避免调用 bash 等不可用工具）
        if tool not in self.available_tools:
            return {
                "success": False,
                "step_id": step.get("id"),
                "action": action,
                "tool": tool,
                "error": f"工具 '{tool}' 不在可用列表中，跳过此步骤"
            }

        # 构造工具参数
        args = self._extract_tool_args(action, tool)

        # 如果参数提取返回错误，直接返回失败
        if isinstance(args, dict) and "error" in args:
            return {
                "success": False,
                "step_id": step.get("id"),
                "action": action,
                "tool": tool,
                "error": args["error"],
                "result": json.dumps({
                    "success": False,
                    "error": args["error"]
                }, ensure_ascii=False)
            }

        try:
            result = self.tool_executor.execute_tool(tool, args)
            # execute_tool 返回 JSON 字符串，先尝试解析判断是否错误
            try:
                result_obj = json.loads(result)
                success = not result_obj.get("error")
            except (json.JSONDecodeError, AttributeError):
                success = not result.startswith("Error:")

            return {
                "success": success,
                "step_id": step.get("id"),
                "action": action,
                "tool": tool,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "step_id": step.get("id"),
                "action": action,
                "tool": tool,
                "error": str(e)
            }

    def _extract_tool_args(self, action: str, tool: str) -> Dict[str, str]:
        """
        从动作描述中提取工具参数
        增强版：支持无扩展名文件、绝对路径、相对路径、中文描述等
        """
        if tool in ["read_file", "list_dir"]:
            # 策略1: 匹配带扩展名的文件名 (api.md, config.py) - 最精确，优先级最高
            # 匹配文件名.扩展名，但不包含中文字符
            ext_match = re.search(r'([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)', action)
            if ext_match:
                return {"path": ext_match.group(1)}

            # 策略2: 匹配引号内的内容（单引号或双引号）
            quoted = re.search(r'["\']([^"\']+)["\']', action)
            if quoted:
                return {"path": quoted.group(1).strip()}

            # 策略3: 匹配可能包含路径的关键词（文件/目录/路径）后的内容
            for kw in ["文件", "目录", "路径", "file", "dir", "path"]:
                # 匹配 "关键词: 内容" 或 "关键词 内容"
                match = re.search(rf'{kw}[：:\s]*([^\s"\'，。]+)', action, re.I)
                if match:
                    candidate = match.group(1).strip()
                    # 移除可能的中文后缀（如"内容"、"代码"等）
                    candidate = re.sub(r'(内容|代码|数据|信息)$', '', candidate)
                    if candidate and not candidate.isdigit() and len(candidate) > 1:
                        return {"path": candidate}

            # 策略4: 匹配相对路径（./xxx 或 ../xxx 或 core/xxx）
            rel_match = re.search(r'(?:\.{0,2}/)?[\w/-]+(?:/[\w/-]+)*', action)
            if rel_match:
                candidate = rel_match.group(0)
                # 过滤纯数字或过短的匹配
                if len(candidate) > 2 and not candidate.isdigit():
                    return {"path": candidate}

            # 策略5: 尝试匹配中文动词后的内容（读取 xxx，查看 xxx）
            cn_match = re.search(r'(?:读取|查看|打开|列出|扫描|阅读)\s*([^\s，。]+)', action)
            if cn_match:
                candidate = cn_match.group(1).strip()
                # 移除可能的中文后缀
                candidate = re.sub(r'(内容|代码|数据|信息|文件)$', '', candidate)
                if candidate and not candidate.isdigit() and len(candidate) > 1:
                    return {"path": candidate}

            # 如果都失败，返回错误标记，而不是默认值
            return {"error": f"无法从步骤描述中提取路径: {action}"}

        elif tool == "write_file":
            # 提取路径
            path_match = re.search(r'["\']?([^\s"\']+\.[a-zA-Z0-9]+)["\']?', action)
            if not path_match:
                # 尝试匹配无扩展名文件
                path_match = re.search(r'["\']?([\w/-]+)["\']?', action)
            path = path_match.group(1) if path_match else "output.txt"
            # 尝试提取内容（如果描述中包含引号包裹的内容）
            content_match = re.search(r'["\']([^"\']+)["\']', action)
            if content_match:
                content = content_match.group(1)
            else:
                # 如果无法提取，则提示需要额外输入
                content = f"[需要提供实际内容，当前步骤描述: {action}]"
            return {"path": path, "content": content}

        elif tool == "bash":
            # 提取命令
            cmd_match = re.search(r'命令[：:]?\s*(.+)', action)
            if cmd_match:
                return {"command": cmd_match.group(1).strip()}
            # 若没有命令关键词，则将整个 action 视为命令
            return {"command": action}

        elif tool == "edit_file":
            # 提取路径（简化版，实际可能需要更复杂）
            path_match = re.search(r'["\']?([^\s"\']+\.[a-zA-Z0-9]+)["\']?', action)
            if path_match:
                return {"path": path_match.group(1), "old_content": "", "new_content": ""}
            return {"error": f"无法从步骤描述中提取文件路径: {action}"}

        return {}

    def _is_reasoning_task(self, action: str) -> bool:
        """判断是否为需要LLM推理的任务"""
        reasoning_keywords = [
            "列出", "给出", "解释", "说明", "分析", "总结", "描述",
            "比较", "评价", "推荐", "建议", "预测", "判断",
            "截取", "片段", "含义", "阅读", "理解"
        ]
        return any(kw in action for kw in reasoning_keywords)

    def _extract_content_summary(self, result: str, max_len: int = 300) -> str:
        """从结果中提取内容摘要"""
        if not result:
            return ""

        try:
            # 尝试解析JSON
            result_obj = json.loads(result)
            if isinstance(result_obj, dict):
                content = result_obj.get("content", result_obj.get("stdout", str(result_obj)))
                if isinstance(content, str):
                    return content[:max_len] + ("..." if len(content) > max_len else "")
            return str(result_obj)[:max_len]
        except (json.JSONDecodeError, TypeError):
            # 非JSON，直接截取
            return result[:max_len] + ("..." if len(result) > max_len else "")


class ReviewerAgent:
    """
    审查 Agent - 负责结果审查和质量控制

    职责：
    - 检查执行结果
    - 评估完成度
    - 提出改进建议
    """

    def __init__(self, model_forward_fn: Callable):
        self.model_forward_fn = model_forward_fn

    def review(
        self,
        user_input: str,
        plan: Dict[str, Any],
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """审查执行结果"""
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

        # 构造审查上下文
        context = f"""用户需求：{user_input}

执行计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

执行结果：
{json.dumps(execution_results, ensure_ascii=False, indent=2)}"""

        messages = [{"role": "user", "content": context}]

        try:
            response = self.model_forward_fn(messages, system_prompt)

            # 尝试解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                review = json.loads(json_match.group())
                return {
                    "success": True,
                    "review": review,
                    "raw_response": response
                }
            else:
                # 简化审查
                all_success = all(r.get("success", False) for r in execution_results)
                return {
                    "success": True,
                    "review": {
                        "completed": all_success,
                        "quality": "good" if all_success else "fair",
                        "issues": [] if all_success else ["部分步骤执行失败"],
                        "suggestions": []
                    },
                    "raw_response": response
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class MultiAgentOrchestrator:
    """
    多 Agent 协调器 - 协调 Planner、Executor、Reviewer

    工作流程：
    1. Planner 生成执行计划（传入已完成步骤上下文，避免重复规划）
    2. Executor 逐步执行（过滤不可用工具）
    3. Reviewer 审查结果
    4. 根据审查结果决定是否重试
    """

    def __init__(
        self,
        model_forward_fn: Callable,
        tool_executor,
        max_retries: int = 1,
        enable_bash: bool = False,
    ):
        # 根据 enable_bash 确定可用工具集合
        available_tools = set(_AVAILABLE_TOOLS)
        if enable_bash:
            available_tools.add("bash")

        self.planner = PlannerAgent(model_forward_fn, available_tools=available_tools)
        self.executor = ExecutorAgent(tool_executor, available_tools=available_tools, model_forward_fn=model_forward_fn)
        self.reviewer = ReviewerAgent(model_forward_fn)
        self.max_retries = max_retries

    def run(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """运行多 Agent 协作

        Args:
            user_input: 用户输入
            context: 运行时上下文，支持以下字段以实现跨轮记忆：
                - completed_steps: 上一轮已完成的步骤列表
                - previous_task: 上一轮的任务描述
                - files_touched: 已操作的文件路径列表
                - current_task: 当前任务标识（SessionMemory 存储的值）
        """
        start_time = datetime.now()

        # 构建增强的规划上下文（加入跨轮任务进度）
        plan_context: Dict[str, Any] = {}
        if context:
            plan_context["completed_steps"] = context.get("completed_steps", [])
            plan_context["previous_task"] = context.get("previous_task") or context.get("current_task", "")
            plan_context["files_touched"] = context.get("files_touched", [])
            plan_context["current_task"] = context.get("current_task", "")

        # 1. 规划阶段
        plan_result = self.planner.plan(user_input, plan_context if any(plan_context.values()) else None)
        if not plan_result["success"]:
            return {
                "success": False,
                "stage": "planning",
                "error": plan_result.get("error", "规划失败")
            }

        plan = plan_result["plan"]

        # 2. 执行阶段
        execution_results = []
        for step in plan.get("steps", []):
            # 传递之前的执行结果作为上下文
            step_context = {
                "previous_results": execution_results
            }
            result = self.executor.execute_step(step, context=step_context)
            execution_results.append(result)

            # 如果关键步骤失败，提前终止
            if not result["success"] and step.get("critical", False):
                break

        # 3. 审查阶段
        review_result = self.reviewer.review(user_input, plan, execution_results)

        # 4. 决策：是否重试
        completed = False
        if review_result["success"]:
            review = review_result["review"]
            completed = review.get("completed", False)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "success": True,
            "completed": completed,
            "plan": plan,
            "execution_results": execution_results,
            "review": review_result.get("review", {}),
            "duration": duration,
            "timestamp": end_time.isoformat()
        }

    def _format_execution_summary(self, execution_results: List[Dict[str, Any]]) -> str:
        """
        将执行结果格式化为简洁摘要，供最终模型理解
        """
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
                    # 推理任务的结果需要完整展示，不截断
                    lines.append(f"  结果: {result}")
                elif not success and error:
                    lines.append(f"  错误: {error[:200]}")
            else:
                lines.append(f"步骤 {step_id}: 调用工具 {tool} ({action}) - {'成功' if success else '失败'}")
                if success:
                    # 提取结果摘要（前 300 字符）
                    try:
                        # 尝试解析 JSON 结果
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
                        # 非 JSON 结果直接截取
                        lines.append(f"  结果: {result[:200]}")
                else:
                    lines.append(f"  错误: {error[:200]}")
            lines.append("")  # 空行分隔
        return "\n".join(lines)

    def run_and_generate_response(
        self,
        user_input: str,
        model_forward_fn: Callable,
        context: Optional[Dict] = None,
        system_prompt: str = "你是一个智能助手。请根据执行结果，完整详细地回答用户的问题。对于列表、解释、分析等内容，请展示完整结果，不要省略。",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        运行多 Agent 协作并生成最终回答

        Args:
            user_input: 用户输入
            model_forward_fn: 模型调用函数，签名 fn(messages, system_prompt, **kwargs) -> str
            context: 可选的上下文（如已完成步骤等）
            system_prompt: 最终回答的系统提示
            temperature/top_p/max_tokens: 生成参数

        Returns:
            包含最终回答和完整执行结果的字典
        """
        # 1. 执行多 Agent 流程
        result = self.run(user_input, context)

        # 2. 格式化执行摘要
        execution_summary = self._format_execution_summary(result.get("execution_results", []))

        # 3. 构建最终回答的 prompt
        final_prompt = f"""用户问题：{user_input}

执行过程：
{execution_summary}

请根据以上执行结果回答用户。
- 如果执行中有错误，请明确指出失败的原因（例如文件不存在、工具不可用等）
- 如果执行成功，请整合各步骤的结果，给出完整详细的回答
- 对于推理任务的结果（如列表、解释、分析等），请直接展示完整内容，不要省略或概括"""

        messages = [{"role": "user", "content": final_prompt}]

        # 4. 调用模型生成最终回答
        try:
            final_response = model_forward_fn(
                messages,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            final_response = f"生成最终回答时出错: {e}"

        # 5. 返回包含最终回答的完整结果
        result["final_response"] = final_response
        return result