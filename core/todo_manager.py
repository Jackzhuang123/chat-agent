#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO 管理模块 - Claude Code 风格的任务状态管理（四态机器）

提供 TodoItem（单任务状态机）和 TodoManager（任务集合管理器）。

状态流转：
  pending → in_progress → completed
                        → failed
                        → cancelled
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class TodoItem:
    """单个 TODO 任务项，包含完整状态机。"""

    VALID_STATES = {"pending", "in_progress", "completed", "failed", "cancelled"}

    def __init__(
        self,
        task_id: int,
        task: str,
        tool: str = "none",
        status: str = "pending",
        priority: str = "medium",
        notes: str = "",
    ):
        self.id = task_id
        self.task = task
        self.tool = tool  # 关联工具：none / read_file / write_file / bash / ...
        self.status = status
        self.priority = priority  # high / medium / low
        self.notes = notes
        self.result_preview: str = ""
        self.error: str = ""
        self.created_at: str = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.updated_at: str = self.created_at

    def transition(self, new_status: str, result_preview: str = "", error: str = "") -> None:
        """状态转换，附带结果/错误信息。"""
        if new_status not in self.VALID_STATES:
            raise ValueError(f"无效状态: {new_status}，合法值: {self.VALID_STATES}")
        self.status = new_status
        self.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if result_preview:
            self.result_preview = result_preview[:200]
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "tool": self.tool,
            "status": self.status,
            "priority": self.priority,
            "notes": self.notes,
            "result_preview": self.result_preview,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        item = cls(
            task_id=data.get("id", 0),
            task=data.get("task", ""),
            tool=data.get("tool", "none"),
            status=data.get("status", "pending"),
            priority=data.get("priority", "medium"),
            notes=data.get("notes", ""),
        )
        item.result_preview = data.get("result_preview", "")
        item.error = data.get("error", "")
        item.created_at = data.get("created_at", item.created_at)
        item.updated_at = data.get("updated_at", item.updated_at)
        return item


class TodoManager:
    """
    Claude Code 风格的 TODO 管理器。

    核心特性：
      1. 四态机器：pending → in_progress → completed / failed / cancelled
      2. 持久化：可序列化/反序列化为 JSON（供 session_log 保存）
      3. 状态查询：支持按状态批量查询
      4. 渲染接口：提供给中间件注入上下文的文本渲染

    与 Claude Code 的对比：
      Claude  → 模型主动调用 `todo_write` 工具更新
      本项目  → 双路径：① TaskExecutor 程序化更新 ② 模型可调用 todo_write 虚拟工具
    """

    STATUS_ICONS = {
        "pending":     "⏳",
        "in_progress": "▶️",
        "completed":   "✅",
        "failed":      "❌",
        "cancelled":   "⚪",
    }

    def __init__(self, title: str = "任务列表"):
        self.title = title
        self._items: List[TodoItem] = []
        self._id_counter: int = 0

    # ------------------------------------------------------------------ #
    #  CRUD                                                                #
    # ------------------------------------------------------------------ #

    def add(
        self,
        task: str,
        tool: str = "none",
        status: str = "pending",
        priority: str = "medium",
        notes: str = "",
        task_id: Optional[int] = None,
    ) -> TodoItem:
        """添加一个 TODO 项，返回该项实例。"""
        self._id_counter += 1
        item = TodoItem(
            task_id=task_id if task_id is not None else self._id_counter,
            task=task,
            tool=tool,
            status=status,
            priority=priority,
            notes=notes,
        )
        self._items.append(item)
        return item

    def update(self, task_id: int, new_status: str, result_preview: str = "", error: str = "") -> bool:
        """更新指定 ID 的 TODO 状态，返回是否成功。"""
        for item in self._items:
            if item.id == task_id:
                item.transition(new_status, result_preview=result_preview, error=error)
                return True
        return False

    def get(self, task_id: int) -> Optional[TodoItem]:
        """按 ID 获取单个任务。"""
        for item in self._items:
            if item.id == task_id:
                return item
        return None

    def get_by_status(self, status: str) -> List[TodoItem]:
        """按状态获取任务列表。"""
        return [item for item in self._items if item.status == status]

    def get_next_pending(self) -> Optional[TodoItem]:
        """获取下一个 pending 任务（按 id 顺序）。"""
        for item in self._items:
            if item.status == "pending":
                return item
        return None

    def get_in_progress(self) -> List[TodoItem]:
        """获取所有 in_progress 任务。"""
        return self.get_by_status("in_progress")

    def all_done(self) -> bool:
        """所有任务是否已完成（completed/failed/cancelled）。"""
        terminal = {"completed", "failed", "cancelled"}
        return all(item.status in terminal for item in self._items)

    def completion_rate(self) -> float:
        """完成率（completed / total）。"""
        if not self._items:
            return 0.0
        completed = sum(1 for i in self._items if i.status == "completed")
        return completed / len(self._items)

    def clear(self) -> None:
        """清空所有任务，重置计数器。"""
        self._items.clear()
        self._id_counter = 0

    # ------------------------------------------------------------------ #
    #  序列化 / 反序列化                                                    #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "items": [item.to_dict() for item in self._items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoManager":
        mgr = cls(title=data.get("title", "任务列表"))
        for item_data in data.get("items", []):
            item = TodoItem.from_dict(item_data)
            mgr._items.append(item)
            if item.id > mgr._id_counter:
                mgr._id_counter = item.id
        return mgr

    def load_from_todos_list(self, todos: List[Dict[str, Any]], title: str = "") -> None:
        """从 TaskPlanner 返回的 todos 列表初始化（兼容旧格式）。"""
        if title:
            self.title = title
        self.clear()
        for t in todos:
            self.add(
                task=t.get("task", ""),
                tool=t.get("tool", "none"),
                status=t.get("status", "pending"),
                priority=t.get("priority", "medium"),
                task_id=t.get("id"),
            )
            if (t.get("id") or 0) > self._id_counter:
                self._id_counter = t["id"]

    # ------------------------------------------------------------------ #
    #  渲染（供 TodoContextMiddleware 使用）                                #
    # ------------------------------------------------------------------ #

    def _render_item(self, item: TodoItem) -> str:
        """渲染单条 TODO 项为文本行（提取公共渲染逻辑）。"""
        icon = self.STATUS_ICONS.get(item.status, "❓")
        tool_tag = f" [{item.tool}]" if item.tool and item.tool != "none" else ""
        result_tag = (
            f" → {item.result_preview[:60]}"
            if item.result_preview and item.status == "completed"
            else ""
        )
        return f"  {icon} [{item.id}] {item.task}{tool_tag}{result_tag}"

    def render_for_context(self, show_completed: bool = True) -> str:
        """渲染为 LLM 上下文的紧凑文本（完整版，用于 Orchestrator）。"""
        if not self._items:
            return ""

        lines = [f"<todo_list title=\"{self.title}\">"]
        for item in self._items:
            if not show_completed and item.status == "completed":
                continue
            lines.append(self._render_item(item))
        lines.append("</todo_list>")
        return "\n".join(lines)

    def render_for_context_worker(self) -> str:
        """渲染为 Worker 专用格式：隐藏所有 pending 步骤，防止越权感知后续计划。"""
        if not self._items:
            return ""

        lines = [f"<todo_list title=\"{self.title}\">"]
        for item in self._items:
            if item.status == "pending":
                continue
            lines.append(self._render_item(item))
        lines.append("</todo_list>")
        return "\n".join(lines)

    def render_for_ui(self) -> str:
        """渲染为 UI 展示的 Markdown 格式。"""
        if not self._items:
            return ""
        lines = [f"📋 **{self.title}**\n"]
        for item in self._items:
            icon = self.STATUS_ICONS.get(item.status, "❓")
            tool_info = f" `{item.tool}`" if item.tool and item.tool != "none" else ""
            lines.append(f"{icon} **步骤{item.id}**: {item.task}{tool_info}")
            if item.result_preview and item.status in ("completed", "failed"):
                preview = item.result_preview[:80].replace("\n", " ")
                lines.append(f"   └ {preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  工具调用接口（供 TodoContextMiddleware 拦截 todo_write 使用）         #
    # ------------------------------------------------------------------ #

    def apply_tool_write(self, payload: Dict[str, Any]) -> str:
        """
        处理模型调用 todo_write 工具的请求。

        支持两种操作：
          - update: {"action": "update", "id": 1, "status": "completed"}
          - add:    {"action": "add", "task": "...", "tool": "none"}
        """
        action = payload.get("action", "update")

        if action == "update":
            task_id = payload.get("id")
            new_status = payload.get("status", "completed")
            result_preview = payload.get("result_preview", "")
            if task_id is None:
                return json.dumps({"success": False, "error": "缺少 id 参数"}, ensure_ascii=False)
            success = self.update(
                task_id=int(task_id),
                new_status=new_status,
                result_preview=result_preview,
                error=payload.get("error", ""),
            )
            return json.dumps({"success": success, "id": task_id, "status": new_status}, ensure_ascii=False)

        if action == "add":
            task = payload.get("task", "")
            if not task:
                return json.dumps({"success": False, "error": "缺少 task 参数"}, ensure_ascii=False)
            item = self.add(
                task=task,
                tool=payload.get("tool", "none"),
                priority=payload.get("priority", "medium"),
            )
            return json.dumps({"success": True, "id": item.id, "task": task}, ensure_ascii=False)

        return json.dumps({"success": False, "error": f"未知 action: {action}"}, ensure_ascii=False)

