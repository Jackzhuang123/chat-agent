#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight Markdown helpers for log-oriented Gradio pages.

These pages only need a stable subset of Markdown features:
- headings
- ordered / unordered lists
- fenced code blocks
- inline code

Keeping the renderer local and deterministic avoids adding a heavy
dependency for simple session-log visualization.
"""

import html
import re


def render_markdown_html(text: str) -> str:
    """Render a small, safe Markdown subset to HTML for log pages."""
    if not text:
        return ""

    def render_inline(content: str) -> str:
        escaped = html.escape(content)
        return re.sub(r'`([^`]+)`', lambda m: f"<code>{m.group(1)}</code>", escaped)

    lines = text.splitlines()
    output = []
    in_code = False
    code_lang = ""
    code_lines = []
    list_mode = None

    def close_list():
        nonlocal list_mode
        if list_mode == "ul":
            output.append("</ul>")
        elif list_mode == "ol":
            output.append("</ol>")
        list_mode = None

    for line in lines:
        stripped = line.rstrip()
        fence = stripped.strip()
        if fence.startswith("```"):
            if in_code:
                code_html = html.escape("\n".join(code_lines))
                lang_attr = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                output.append(f"<pre><code{lang_attr}>{code_html}</code></pre>")
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                close_list()
                in_code = True
                code_lang = fence[3:].strip()
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped.strip():
            close_list()
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', stripped.strip())
        if heading:
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{render_inline(heading.group(2))}</h{level}>")
            continue
        ordered = re.match(r'^\s*\d+\.\s+(.+)$', stripped)
        if ordered:
            if list_mode != "ol":
                close_list()
                output.append("<ol>")
                list_mode = "ol"
            output.append(f"<li>{render_inline(ordered.group(1))}</li>")
            continue
        unordered = re.match(r'^\s*[-*]\s+(.+)$', stripped)
        if unordered:
            if list_mode != "ul":
                close_list()
                output.append("<ul>")
                list_mode = "ul"
            output.append(f"<li>{render_inline(unordered.group(1))}</li>")
            continue
        close_list()
        output.append(f"<p>{render_inline(stripped)}</p>")

    if in_code:
        code_html = html.escape("\n".join(code_lines))
        lang_attr = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
        output.append(f"<pre><code{lang_attr}>{code_html}</code></pre>")
    close_list()
    return "\n".join(output)


def build_markdown_preview(text: str, max_chars: int = 240) -> str:
    """Trim long Markdown safely so previews do not break fenced code blocks."""
    content = text or ""
    if len(content) <= max_chars:
        return content
    preview = content[:max_chars].rstrip()
    if preview.count("```") % 2 == 1:
        preview += "\n```"
    return preview + "\n\n..."
