# -*- coding: utf-8 -*-
"""任务上下文注入（修复版 - 避免向量记忆污染）"""

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

    facts_ledger = session.task_context.get("facts_ledger", {}) or {}
    confirmed_facts = facts_ledger.get("confirmed_facts", [])[:3]
    file_facts = facts_ledger.get("file_facts", [])[:2]
    failed_actions = facts_ledger.get("failed_actions", [])[:2]
    if confirmed_facts or file_facts or failed_actions:
        ledger_lines = ["🧾 事实账本："]
        for fact in confirmed_facts:
            ledger_lines.append(f"  - 已确认: {str(fact)[:120]}")
        for fact in file_facts:
            ledger_lines.append(f"  - 文件事实: {str(fact)[:120]}")
        for fact in failed_actions:
            ledger_lines.append(f"  - 失败记录: {str(fact)[:120]}")
        parts.append("\n".join(ledger_lines))

    # 未读文件提醒
    read_files = set(os.path.basename(p) for p in session.read_files_cache.keys())
    for pat in [r'读取\s*([^\s，,。]+\.py)', r'查看\s*([^\s，,。]+\.py)']:
        for m in re.finditer(pat, current_task):
            fname = m.group(1).strip()
            if fname not in read_files:
                parts.append(f"⚠️ 尚未读取文件：{fname}")

    # ---------- 修复：仅当尚无工具成功执行时才注入向量记忆 ----------
    has_successful_tool = len(session.task_context.get("completed_steps", [])) > 0

    if vector_memory and current_task and not has_successful_tool:
        try:
            mems = vector_memory.search_by_types(
                query=current_task,
                types=["assistant_response", "file_content"],
                top_k=2,
                min_score=0.35,
            )
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
