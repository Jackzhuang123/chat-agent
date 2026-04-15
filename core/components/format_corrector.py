# -*- coding: utf-8 -*-
"""格式纠错注入"""

from typing import Dict, List

def inject_format_correction(messages: List[Dict], response: str, work_dir: str) -> List[Dict]:
    correction = {
        "role": "user",
        "content": (
            "⚠️ 未检测到有效工具调用格式。\n"
            f"正确格式：\nread_file\n{{\"path\": \"{work_dir}/文件名.py\"}}\n"
            "请直接重新输出工具调用。"
        )
    }
    messages.append({"role": "assistant", "content": response})
    messages.append(correction)
    return messages