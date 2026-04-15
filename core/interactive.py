# core/interactive.py
"""交互式确认处理（原始版本，与interaction.py配合）"""
from typing import Dict, Any


def format_interactive_prompt(interactive_cfg: Dict) -> str:
    mode = interactive_cfg.get("mode", "confirm")
    prompt = interactive_cfg.get("prompt", "请确认操作")
    options = interactive_cfg.get("options", [])
    if mode == "confirm":
        opts = "\n".join([f"  [{opt['id']}] {opt['label']}" for opt in options])
        return f"{prompt}\n{opts}\n请输入选项 ID:"
    elif mode == "choice":
        multi = " (可多选，逗号分隔)" if interactive_cfg.get("allow_multiple") else ""
        opts = "\n".join([f"  [{opt['id']}] {opt['label']}" for opt in options])
        return f"{prompt}{multi}\n{opts}\n请输入选项:"
    else:
        return prompt


def parse_interactive_response(interactive_cfg: Dict, response: str) -> Dict[str, Any]:
    mode = interactive_cfg.get("mode", "confirm")
    options = interactive_cfg.get("options", [])
    inject_as = interactive_cfg.get("inject_as", "user_feedback")

    if mode == "confirm":
        selected = response.strip()
        action = "continue"
        for opt in options:
            if opt["id"] == selected:
                action = opt.get("action", "continue")
                break
        if action == "block":
            raise RuntimeError("用户选择中止流程")
        return {inject_as: selected}
    elif mode == "choice":
        selected_ids = [s.strip() for s in response.split(",")]
        actions = []
        for sid in selected_ids:
            for opt in options:
                if opt["id"] == sid:
                    actions.append(opt.get("action", "continue"))
                    break
        if "block" in actions:
            raise RuntimeError("用户选择中止流程")
        return {inject_as: selected_ids}
    else:
        return {inject_as: response}