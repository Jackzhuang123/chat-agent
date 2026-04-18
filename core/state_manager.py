# core/state_manager.py
"""统一工作流状态管理，支持断点续跑"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class SessionContext:
    """Agent 会话上下文，用于隔离并发请求的状态"""
    tool_history: List[Dict] = field(default_factory=list)
    reflection_history: List[Dict] = field(default_factory=list)
    read_files_cache: Dict[str, str] = field(default_factory=dict)
    task_context: Dict[str, Any] = field(default_factory=lambda: {
        "current_task": None,
        "completed_steps": [],
        "failed_attempts": [],
        # 结构化子任务状态：{subtask_index(int): {"desc": str, "status": "pending"|"done", "done_by": [tool_call_str]}}
        "subtask_status": {},
        "facts_ledger": {
            "confirmed_facts": [],
            "file_facts": [],
            "failed_actions": [],
            "open_questions": [],
        },
    })
    current_tool_chain_id: Optional[str] = None
    runtime_context: Dict[str, Any] = field(default_factory=dict)

class WorkflowStateManager:
    def __init__(self, workflow_id: str, state_dir: str = ".workflow_states"):
        self.workflow_id = workflow_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / f"{workflow_id}.json"
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "workflow_id": self.workflow_id,
            "status": "idle",
            "created_at": datetime.now().isoformat(),
            "updated_at": "",
            "current_stage": None,
            "stages": {},
            "artifacts": {},
            "user_input": "",
            "runtime_context": {},
            "pending_interaction_stage": None,
            "pending_interaction_config": None,
        }

    def save(self):
        self.data["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def mark_stage_start(self, stage: str):
        if stage not in self.data["stages"]:
            self.data["stages"][stage] = {"status": "pending", "started_at": None, "completed_at": None, "error": None}
        self.data["stages"][stage]["status"] = "running"
        self.data["stages"][stage]["started_at"] = datetime.now().isoformat()
        self.data["current_stage"] = stage
        self.save()

    def mark_stage_complete(self, stage: str, output: Dict = None):
        if stage in self.data["stages"]:
            self.data["stages"][stage]["status"] = "completed"
            self.data["stages"][stage]["completed_at"] = datetime.now().isoformat()
        if output:
            self.data["artifacts"].update(output)
        self.save()

    def mark_stage_failed(self, stage: str, error: str):
        if stage in self.data["stages"]:
            self.data["stages"][stage]["status"] = "failed"
            self.data["stages"][stage]["error"] = error
        self.data["status"] = "failed"
        self.save()

    def get_failed_stage(self) -> Optional[str]:
        for stage, info in self.data["stages"].items():
            if info.get("status") == "failed":
                return stage
        return None

    def get_last_completed_stage(self) -> Optional[str]:
        completed = [s for s, i in self.data["stages"].items() if i.get("status") == "completed"]
        return completed[-1] if completed else None

    def is_stage_completed(self, stage: str) -> bool:
        return self.data["stages"].get(stage, {}).get("status") == "completed"

    def set_artifacts(self, artifacts: Dict):
        self.data["artifacts"].update(artifacts)
        self.save()

    def get_artifacts(self) -> Dict:
        return self.data["artifacts"]
