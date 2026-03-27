# -*- coding: utf-8 -*-
"""多 Agent 协作模块 - Planner + Executor + Reviewer 架构"""

import json
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
            import re
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

    def __init__(self, tool_executor, available_tools: Optional[set] = None):
        self.tool_executor = tool_executor
        self.available_tools: set = available_tools if available_tools is not None else set(_AVAILABLE_TOOLS)

    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个步骤"""
        tool = step.get("tool", "none")
        action = step.get("action", "")

        if tool == "none":
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

        # 构造工具参数（简化版，实际需要更智能的参数提取）
        args = self._extract_tool_args(action, tool)

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
        """从动作描述中提取工具参数（简化版）"""
        import re

        if tool in ["read_file", "list_dir"]:
            # 提取路径
            path_match = re.search(r'["\']?([^\s"\']+\.[a-z]+)["\']?', action)
            if path_match:
                return {"path": path_match.group(1)}
            # 提取目录
            dir_match = re.search(r'目录[：:]?\s*([^\s]+)', action)
            if dir_match:
                return {"path": dir_match.group(1)}
            return {"path": "."}

        elif tool == "write_file":
            # 提取路径和内容
            path_match = re.search(r'["\']?([^\s"\']+\.[a-z]+)["\']?', action)
            path = path_match.group(1) if path_match else "output.txt"
            return {"path": path, "content": action}

        elif tool == "bash":
            # 提取命令
            cmd_match = re.search(r'命令[：:]?\s*(.+)', action)
            if cmd_match:
                return {"command": cmd_match.group(1)}
            return {"command": action}

        return {}


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
            import re
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
        self.executor = ExecutorAgent(tool_executor, available_tools=available_tools)
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
            result = self.executor.execute_step(step)
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
