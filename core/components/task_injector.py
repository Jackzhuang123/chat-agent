# -*- coding: utf-8 -*-
"""任务上下文注入（完整版）"""

import os
import re
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import SessionContext
    from core.vector_memory import VectorMemory

def inject_task_context(
    messages: List[Dict],
    session: "SessionContext",
    vector_memory: "VectorMemory" = None
) -> List[Dict]:
    current_task = session.task_context.get("current_task", "")
    if current_task is None:
        current_task = ""
    elif not isinstance(current_task, str):
        current_task = str(current_task)

    parts = []
    current_task = session.task_context.get("current_task", "")
    if current_task:
        parts.append(f"🎯 原始任务：{current_task}")

    completed = session.task_context.get("completed_steps", [])
    if completed:
        parts.append(f"📋 已完成步骤：{', '.join(completed[-5:])}")

    subtask_status = session.task_context.get("subtask_status", {})
    if subtask_status:
        done = [idx for idx, inf in subtask_status.items() if inf["status"] == "done"]
        pending = [idx for idx, inf in subtask_status.items() if inf["status"] == "pending"]
        board = [f"📊 子任务进度：{len(done)}/{len(subtask_status)} 完成"]
        for idx in sorted(subtask_status.keys()):
            inf = subtask_status[idx]
            icon = "✅" if inf["status"] == "done" else "⏳"
            board.append(f"  {icon} 步骤{idx}: {inf['desc'][:40]}")
        parts.append("\n".join(board))
        if pending:
            parts.append(f"▶ 当前待执行：步骤{pending[0]} - {subtask_status[pending[0]]['desc']}")

    # 未读文件
    read_files = set(os.path.basename(p) for p in session.read_files_cache.keys())
    for pat in [r'读取\s*([^\s，,。]+\.py)', r'查看\s*([^\s，,。]+\.py)']:
        for m in re.finditer(pat, current_task):
            fname = m.group(1).strip()
            if fname not in read_files:
                parts.append(f"⚠️ 尚未读取文件：{fname}")

    # 向量记忆
    if vector_memory and current_task:
        try:
            mems = vector_memory.search(query=current_task, top_k=2, filter_metadata={"type": "response"})
            if mems:
                parts.append("📚 相关历史记忆：" + "；".join(m["content"][:80] for m in mems))
        except Exception:
            pass

    if not parts:
        return messages

    context_msg = {"role": "system", "content": "\n".join(parts)}
    if len(messages) >= 2:
        return messages[:-1] + [context_msg] + messages[-1:]
    return messages + [context_msg]