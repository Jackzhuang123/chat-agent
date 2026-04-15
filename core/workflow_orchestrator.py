# core/workflow_orchestrator.py
"""增强工作流协调器：断点续跑、交互挂起、智能阶段路由"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from core.langgraph_agent import LangGraphAgent
from .interaction import format_interaction_prompt as format_interaction, parse_interaction_response
from .mode_router import PreciseModeRouter, IntentType
from .phase_runner import PhaseRunner
from .plugin_registry import PluginRegistry
from .state_manager import WorkflowStateManager, SessionContext
from .workflow_dag import WorkflowDAG
from core.monitor_logger import get_monitor_logger

class WorkflowOrchestrator:
    def __init__(
        self,
        agent: LangGraphAgent,
        dag_config_path: str,
        project_root: Optional[str] = None,
        mode_router: Optional[PreciseModeRouter] = None,
    ):
        self.agent = agent
        self.dag = WorkflowDAG(dag_config_path)
        self.registry = PluginRegistry()
        if project_root:
            self.registry.load_project_config(project_root)
        self.phase_runner = PhaseRunner(self.registry)
        self.mode_router = mode_router
        self.state_manager: Optional[WorkflowStateManager] = None
        self.monitor = get_monitor_logger()

    def _init_state(self, workflow_id: str, user_input: str, runtime_context: Dict):
        self.state_manager = WorkflowStateManager(workflow_id)
        self.state_manager.data["user_input"] = user_input
        self.state_manager.data["runtime_context"] = runtime_context
        self.state_manager.save()

    async def run(
        self,
        user_input: str,
        session: "SessionContext",
        workflow_id: Optional[str] = None,
        runtime_context: Optional[Dict] = None,
        resume: bool = False,
        from_stage: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        import asyncio
        workflow_id = workflow_id or f"wf_{int(time.time())}"
        runtime_context = runtime_context or {}
        self.monitor.info(f"工作流开始，workflow_id={workflow_id}")

        if resume or from_stage:
            if not self.state_manager or self.state_manager.workflow_id != workflow_id:
                self.state_manager = WorkflowStateManager(workflow_id)
            start_stage = from_stage or self.state_manager.get_failed_stage() or self.state_manager.get_last_completed_stage()
            if not start_stage:
                start_stage = self._get_start_stage()
        else:
            self._init_state(workflow_id, user_input, runtime_context)
            start_stage = self._get_start_stage()

        # 使用待执行队列（支持并行分支注入）
        pending_stages: List[str] = [start_stage]

        while pending_stages:
            current_stage = pending_stages.pop(0)

            # 跳过已完成的阶段
            if self.state_manager.is_stage_completed(current_stage):
                next_stages = self.dag.get_next_stages(current_stage, self.state_manager.data)
                pending_stages = self._enqueue_stages(pending_stages, next_stages)
                continue

            # wait_for 门控：检查前置依赖是否全部完成
            stage_info = self.dag.stages.get(current_stage, {})
            wait_for = stage_info.get("wait_for", [])
            if isinstance(wait_for, str):
                wait_for = [wait_for]
            if wait_for:
                all_deps_done = all(
                    self.state_manager.is_stage_completed(dep) for dep in wait_for
                )
                if not all_deps_done:
                    # 依赖未完成，推迟到队列末尾等待
                    pending_stages.append(current_stage)
                    # 防止无限循环：若队列里全是等待状态的阶段，说明死锁
                    non_waiting = [s for s in pending_stages if s != current_stage]
                    if not non_waiting:
                        yield {"type": "error", "stage": current_stage,
                               "error": f"wait_for 死锁：{wait_for} 均未完成且没有其他可执行阶段"}
                        return
                    continue

            yield {"type": "progress", "stage": current_stage, "message": f"开始阶段 {current_stage}"}
            self.state_manager.mark_stage_start(current_stage)

            # 智能路由
            if self.mode_router and current_stage in ("analyze", "process"):
                intent = self.mode_router.route(user_input, context=self.state_manager.data.get("artifacts", {}))
                if intent.intent == IntentType.SKILLS and current_stage == "analyze":
                    self.state_manager.mark_stage_complete(current_stage, {"skip_reason": "技能模式跳过"})
                    next_stages = self.dag.get_next_stages(current_stage, self.state_manager.data)
                    pending_stages = self._enqueue_stages(pending_stages, next_stages)
                    continue

            # 前置钩子
            payload = {"input": self.state_manager.get_artifacts(), "user_input": user_input}
            modified_input, logs = self.phase_runner.run_hooks(
                current_stage, "before", payload, work_dir=str(Path.cwd())
            )
            self.state_manager.set_artifacts(modified_input)

            for log in logs:
                if log.get("status") == "pending_interactive":
                    interactive_cfg = log.get("interactive")
                    self.state_manager.data["pending_interaction_stage"] = current_stage
                    self.state_manager.data["pending_interaction_config"] = interactive_cfg
                    self.state_manager.save()
                    yield {
                        "type": "interaction_required",
                        "stage": current_stage,
                        "config": format_interaction(interactive_cfg),
                        "raw_config": interactive_cfg,
                    }
                    return

            # 主逻辑：判断是否并行执行
            parallel_next = stage_info.get("parallel", False)
            try:
                if parallel_next:
                    # 当前阶段并行触发多个子阶段（直接获取下一步并行执行）
                    next_stages_for_parallel = self.dag.get_next_stages(current_stage, self.state_manager.data)
                    self.monitor.info(f"阶段 {current_stage} 触发并行执行: {next_stages_for_parallel}")
                    if len(next_stages_for_parallel) > 1:
                        # 并行执行所有子阶段
                        parallel_tasks = [
                            self._execute_stage_with_agent(s, user_input, session, runtime_context)
                            for s in next_stages_for_parallel
                        ]
                        parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                        for stage_name, result in zip(next_stages_for_parallel, parallel_results):
                            if isinstance(result, Exception):
                                self.state_manager.mark_stage_failed(stage_name, str(result))
                                yield {"type": "error", "stage": stage_name, "error": str(result)}
                            else:
                                self.state_manager.mark_stage_complete(stage_name, result)
                                yield {"type": "progress", "stage": stage_name,
                                       "message": f"并行阶段 {stage_name} 完成"}
                        # 标记当前触发阶段完成（无实际输出，仅作调度）
                        stage_output = {"parallel_triggered": next_stages_for_parallel}
                        self.state_manager.mark_stage_complete(current_stage, stage_output)
                        # 并行阶段完成后，把后续汇聚阶段加入队列
                        for s in next_stages_for_parallel:
                            gather_next = self.dag.get_next_stages(s, self.state_manager.data)
                            pending_stages = self._enqueue_stages(pending_stages, gather_next)
                        continue
                    else:
                        # 只有一个后续，退化为串行
                        stage_output = await self._execute_stage_with_agent(current_stage, user_input, session, runtime_context)
                        self.state_manager.mark_stage_complete(current_stage, stage_output)
                else:
                    stage_output = await self._execute_stage_with_agent(current_stage, user_input, session, runtime_context)
                    self.state_manager.mark_stage_complete(current_stage, stage_output)
            except Exception as e:
                self.state_manager.mark_stage_failed(current_stage, str(e))
                yield {"type": "error", "stage": current_stage, "error": str(e)}
                return

            # 后置钩子
            payload = {"input": self.state_manager.get_artifacts(), "output": stage_output}
            modified_input, logs = self.phase_runner.run_hooks(
                current_stage, "after", payload, work_dir=str(Path.cwd())
            )
            self.state_manager.set_artifacts(modified_input)

            for log in logs:
                if log.get("status") == "pending_interactive":
                    interactive_cfg = log.get("interactive")
                    self.state_manager.data["pending_interaction_stage"] = current_stage
                    self.state_manager.data["pending_interaction_config"] = interactive_cfg
                    self.state_manager.save()
                    yield {
                        "type": "interaction_required",
                        "stage": current_stage,
                        "config": format_interaction(interactive_cfg),
                        "raw_config": interactive_cfg,
                    }
                    return

            if self.dag.is_checkpoint(current_stage) and not runtime_context.get("auto_confirm"):
                yield {
                    "type": "checkpoint",
                    "stage": current_stage,
                    "message": f"阶段 {current_stage} 已完成，是否继续？",
                }
                return

            # 非并行阶段：正常入队后续阶段
            if not parallel_next:
                self.monitor.info(f"工作流挂起等待交互，阶段 {current_stage}")
                next_stages = self.dag.get_next_stages(current_stage, self.state_manager.data)
                pending_stages = self._enqueue_stages(pending_stages, next_stages)

        yield {"type": "completed", "result": self.state_manager.data}
        return

    async def resume_with_response(self, response: Any, session: "SessionContext") -> AsyncGenerator[Dict, None]:
        if not self.state_manager:
            raise RuntimeError("状态管理器未初始化")
        pending_stage = self.state_manager.data.get("pending_interaction_stage")
        interactive_cfg = self.state_manager.data.get("pending_interaction_config")
        if not pending_stage or not interactive_cfg:
            raise RuntimeError("没有挂起的交互")
        try:
            injected = parse_interaction_response(interactive_cfg, response)
            self.state_manager.set_artifacts(injected)
        except RuntimeError as e:
            self.state_manager.mark_stage_failed(pending_stage, str(e))
            yield {"type": "error", "stage": pending_stage, "error": str(e)}
            return
        self.state_manager.data.pop("pending_interaction_stage", None)
        self.state_manager.data.pop("pending_interaction_config", None)
        self.state_manager.save()
        stage_output = await self._execute_stage_with_agent(pending_stage, self.state_manager.data["user_input"], session, self.state_manager.data.get("runtime_context", {}))
        self.state_manager.mark_stage_complete(pending_stage, stage_output)
        next_stages = self.dag.get_next_stages(pending_stage, self.state_manager.data)
        # 支持并行分支：取第一个作为 from_stage，其余由 run() 内队列机制自动补充
        next_stage = next_stages[0] if next_stages else None
        async for event in self.run(
            user_input=self.state_manager.data["user_input"],
            session=session,
            workflow_id=self.state_manager.workflow_id,
            runtime_context=self.state_manager.data.get("runtime_context", {}),
            from_stage=next_stage,
        ):
            yield event

    async def _execute_stage_with_agent(self, stage: str, user_input: str, session: "SessionContext", runtime_context: Dict) -> Dict:
        handler = self.dag.get_handler(stage)
        if not handler:
            return {"info": f"阶段 {stage} 无处理函数"}

        # ── 每个 workflow 阶段开始前重置迭代内部状态 ──────────────────────────
        # 问题：workflow 各阶段共享同一个 session，上一阶段积累的 tool_history /
        #       completed_steps / failed_attempts 会带入下一阶段，导致：
        #   1. 循环检测误触发（认为新阶段在"重复"旧阶段的操作）
        #   2. 反思引擎因旧失败记录提前中断
        #   3. 上下文消息跨阶段累积，压缩频率激增
        # 解决：清除迭代级别的状态，保留跨阶段的持久状态（如 files_read_cache）
        session.tool_history = []
        session.reflection_history = []
        session.task_context["completed_steps"] = []
        session.task_context["failed_attempts"] = []
        session.task_context["subtask_status"] = {}
        session.task_context["tool_result_log"] = []
        # 注意：不清除 files_read（跨阶段缓存避免重复读取）和 execute_python_outputs
        # ─────────────────────────────────────────────────────────────────────

        artifacts_ctx = self.state_manager.get_artifacts()
        # 不把全量 artifacts dump 进 prompt（可能很大），仅提取有意义的摘要字段
        artifacts_summary = {
            k: str(v)[:100] for k, v in artifacts_ctx.items()
            if k not in ("tool_calls",) and v
        }
        prompt = (
            f"执行工作流阶段: {stage}\n"
            f"用户需求: {user_input}\n"
            f"当前上下文: {json.dumps(artifacts_summary, ensure_ascii=False)}"
        )
        result = await self.agent.run(prompt, session=session, runtime_context=runtime_context)
        return {"response": result.get("response", ""), "tool_calls": result.get("tool_calls", [])}

    def _enqueue_stages(self, queue: List[str], new_stages: List[str]) -> List[str]:
        """将新阶段加入待执行队列（去重，避免重复入队）。"""
        for s in new_stages:
            if s not in queue:
                queue.append(s)
        return queue

    def _get_start_stage(self) -> str:
        """找到 DAG 中无前驱的起始节点（拓扑上的入口）。

        修复：不依赖 dict 遍历顺序（原 for..return 写法），
        而是显式计算哪些节点没有被任何其他节点的 next/feeds_into 引用。
        """
        all_stages = set(self.dag.stages.keys())
        referenced_as_next: set = set()
        for info in self.dag.stages.values():
            for nxt in info.get("next", []):
                referenced_as_next.add(nxt)
            feeds = info.get("feeds_into")
            if feeds:
                referenced_as_next.add(feeds)
        roots = all_stages - referenced_as_next
        if roots:
            # 若有多个入口，取字典中最先出现的（兼容声明顺序）
            for name in self.dag.stages:
                if name in roots:
                    return name
        # 降级：返回第一个阶段名
        return next(iter(self.dag.stages), "init")
