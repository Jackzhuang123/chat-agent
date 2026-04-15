# -*- coding: utf-8 -*-
"""上下文智能压缩"""

import json
import re
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import SessionContext

def compress_context_smart(
    messages: List[Dict],
    session: "SessionContext",
    model_forward_fn,
    vector_memory=None,
    limit: int = 12000,
    system_prompt_base: str = "",
) -> List[Dict]:
    total_chars = sum(len(json.dumps(m, default=str)) for m in messages)
    if total_chars / 1.5 < limit * 0.75 or len(messages) <= 6:
        return messages

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    recent_msgs = messages[-5:]
    start_idx = 1 if system_msg else 0
    middle_msgs = messages[start_idx:-5]
    if not middle_msgs:
        return messages

    summary_prompt = "请对以下对话进行摘要，保留关键信息（文件名、工具结果、已完成任务）。\n\n"
    for m in middle_msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        max_len = 1500 if any(k in content for k in ("execute_python", "read_file", "bash")) else 400
        summary_prompt += f"[{role}]: {content[:max_len]}\n---\n"

    try:
        summary = model_forward_fn(
            [{"role": "user", "content": summary_prompt}],
            system_prompt="你是上下文压缩助手，提取核心要点。",
            temperature=0.2, max_tokens=800,
        )
        known_tools = r'(?:execute_python|read_file|write_file|edit_file|list_dir|bash)'
        summary = re.sub(rf'(?:^|\n)\s*{known_tools}\s*\n\s*\{{', r'\n【历史已调用】', summary, flags=re.MULTILINE)
        compressed_msg = {"role": "system", "content": f"📦 [早期上下文摘要]\n{summary}"}
        new_messages = [system_msg] if system_msg else []
        new_messages.append(compressed_msg)
        new_messages.extend(recent_msgs)
        return new_messages
    except Exception:
        return messages[:2] + messages[-10:]