# -*- coding: utf-8 -*-
"""输出清理函数（增强版）"""

import re

def clean_react_tags(text: str) -> str:
    """
    增强版清理：移除所有 ReAct 推理标签（Thought/Action/Observation）以及可能的其他格式。
    """
    # 移除整行以 Thought/Action/Observation 开头的行（不区分大小写）
    text = re.sub(r'(?im)^\s*(thought|action|observation)\s*:.*$', '', text)
    # 移除可能存在的 XML 风格标签 <thought>...</thought> 等
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<action>.*?</action>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<observation>.*?</observation>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 合并多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def summarize_long_response(response: str, max_chars: int = 800) -> str:
    """
    如果响应超过指定长度，前面添加一句话摘要，并截断详细内容。
    """
    if len(response) <= max_chars:
        return response
    first_paragraph = re.split(r'\n\s*\n', response)[0]
    summary = first_paragraph[:200].strip()
    if not summary:
        summary = "详细回答如下"
    return f"📝 摘要：{summary}...\n\n（详细内容过长，已折叠，可在此处查看完整输出）\n\n{response}"

def strip_trailing_tool_call(text: str) -> str:
    # 保留原有函数不变
    code_block_pattern = re.compile(r'\n?\s*```[^\n]*\n[\s\S]*?```\s*$', re.DOTALL)
    result = text
    prev = None
    while prev != result:
        prev = result
        m = code_block_pattern.search(result)
        if m and any(h in m.group(0) for h in ('"path"', '"command"', '"content"')):
            result = result[:m.start()].rstrip()
    bare_pattern = r'\n?(read_file|write_file|edit_file|list_dir|bash)\s*\n\s*\{[^}]*\}\s*$'
    result = re.sub(bare_pattern, '', result, flags=re.DOTALL).rstrip()
    return result if result else text