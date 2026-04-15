# core/workflow_dag.py
"""
声明式工作流 DAG 引擎 - 支持并行、等待、检查点
"""
import json
from typing import Dict, List, Optional


class WorkflowDAG:
    def __init__(self, config_path):
        """初始化 DAG。

        Args:
            config_path: 可以是文件路径字符串，也可以直接是 dict（内联配置）。
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        self.stages = self.config["stages"]
        self.handlers = self.config.get("handlers", {})

    def get_next_stages(self, current_stage: str, state: Dict) -> List[str]:
        """返回当前阶段执行完成后，下一步应该进入的阶段列表。

        修复：
          1. feeds_into 字段视为 next 的别名（原来忽略了此字段）
          2. wait_for 已移至 orchestrator 层处理，此处不重复校验
             （避免 orchestrator 的 wait_for 逻辑与 DAG 内部校验双重拦截）
          3. parallel=True 时返回全部 next 列表（供 orchestrator 并行执行）
        """
        stage_info = self.stages.get(current_stage, {})
        next_stages = list(stage_info.get("next", []))

        # feeds_into 视为 next 的别名（单值）
        feeds_into = stage_info.get("feeds_into")
        if feeds_into and feeds_into not in next_stages:
            next_stages.append(feeds_into)

        if not next_stages:
            return []

        parallel = stage_info.get("parallel", False)
        if parallel:
            return next_stages
        else:
            return [next_stages[0]]

    def is_checkpoint(self, stage: str) -> bool:
        return self.stages.get(stage, {}).get("checkpoint", False)

    def get_handler(self, stage: str) -> Optional[str]:
        return self.handlers.get(stage)


EXAMPLE_CONFIG = {
    "stages": {
        "init": {"next": ["analyze"], "checkpoint": False},
        # analyze 并行触发 process_a 和 process_b，orchestrator 会用 asyncio.gather 并行执行
        "analyze": {"next": ["process_a", "process_b"], "parallel": True, "checkpoint": True},
        # process_a / process_b 各自完成后，均流入 merge
        # merge 通过 wait_for 等待两个并行分支都完成（由 orchestrator 的队列机制保障）
        "process_a": {"next": ["merge"]},
        "process_b": {"next": ["merge"]},
        # merge 等待 process_a 和 process_b 都完成才开始
        "merge": {"next": ["report"], "wait_for": ["process_a", "process_b"], "checkpoint": False},
        "report": {"next": []}
    },
    "handlers": {
        "init": "handlers.init_handler",
        "analyze": "handlers.analyze_handler",
        "process_a": "handlers.process_handler",
        "process_b": "handlers.process_handler",
        "merge": "handlers.merge_handler",
        "report": "handlers.report_handler"
    }
}