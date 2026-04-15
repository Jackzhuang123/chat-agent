# -*- coding: utf-8 -*-
"""循环检测逻辑"""

import hashlib
import json
import re
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_manager import SessionContext

def detect_loop(session: "SessionContext", max_same: int = 3) -> bool:
    def _normalize_python_code(code: str) -> str:
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        return '\n'.join(lines)

    def _make_key(h: Dict) -> str:
        tool = h.get("tool", "")
        if tool == "execute_python":
            try:
                args = json.loads(h.get("args", "{}"))
                code = args.get("code", h.get("args", ""))
            except Exception:
                code = h.get("args", "")
            norm_code = _normalize_python_code(code)
            code_hash = hashlib.md5(norm_code.encode("utf-8", errors="replace")).hexdigest()[:12]
            return f"execute_python|hash={code_hash}"
        return f"{tool}|{h.get('args', '')}"

    if len(session.tool_history) < max_same:
        return False

    recent_window = session.tool_history[-12:]
    recent = recent_window[-max_same:]
    first_key = _make_key(recent[0])
    if all(_make_key(h) == first_key for h in recent):
        return True

    tool_counts: Dict[str, int] = {}
    py_hashes: List[str] = []
    for h in recent_window:
        tool = h.get("tool", "")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool == "execute_python":
            key = _make_key(h)
            if key.startswith("execute_python|hash="):
                py_hashes.append(key.split("=")[1])
    for tool, cnt in tool_counts.items():
        if cnt >= 8:
            if tool == "execute_python":
                if len(set(py_hashes)) <= 3:
                    return True
            else:
                return True

    if hasattr(session, 'raw_response_cache') and len(session.raw_response_cache) >= 4:
        recent_responses = session.raw_response_cache[-8:]
        thought_patterns = []
        tool_name_re = re.compile(r'^(?:^|\n)\s*(execute_python|read_file|write_file|edit_file|list_dir|bash)\s*', re.MULTILINE)
        for resp in recent_responses:
            m = tool_name_re.search(resp)
            if m:
                tname = m.group(1)
                after = resp[m.end():m.end() + 100].strip()
                fingerprint = re.sub(r'\s+', ' ', after).strip()
                thought_patterns.append(f"{tname}|{fingerprint}")
        if len(thought_patterns) >= 4 and len(set(thought_patterns[-4:])) <= 2:
            return True
    return False