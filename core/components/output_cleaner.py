# -*- coding: utf-8 -*-
"""输出清理函数"""

import re

def strip_trailing_tool_call(text: str) -> str:
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

def clean_react_tags(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    action_re = re.compile(r'^Action\s*:', re.I)
    thought_re = re.compile(r'^Thought\s*:\s*', re.I)
    for line in lines:
        if action_re.match(line.strip()):
            continue
        m = thought_re.match(line.strip())
        if m:
            rest = line.strip()[m.end():].strip()
            if rest:
                cleaned.append(rest)
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()