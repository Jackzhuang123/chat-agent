# -*- coding: utf-8 -*-
"""工具推荐注入"""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.components.deep_reflection import DeepReflectionEngine

def inject_tool_recommendations(messages: List[Dict], recommendations: List[Dict]) -> List[Dict]:
    if not recommendations:
        return messages
    rec_text = "\n".join(f"  - {r['tool']}: {r['confidence']:.0%} ({r['reason']})" for r in recommendations[:2])
    injection = {"role": "system", "content": f"💡 工具推荐：\n{rec_text}"}
    return messages[:-1] + [injection] + messages[-1:] if len(messages) >= 2 else messages + [injection]

def inject_efficient_sequences(messages: List[Dict], reflection: "DeepReflectionEngine") -> List[Dict]:
    if not reflection:
        return messages
    seqs = reflection.get_efficient_sequences(top_k=3)
    if not seqs:
        return messages
    seq_text = "；".join(f"{s['sequence']}（×{s['count']}）" for s in seqs)
    injection = {"role": "system", "content": f"📈 高效序列：{seq_text}"}
    return messages[:-1] + [injection] + messages[-1:] if len(messages) >= 2 else messages + [injection]