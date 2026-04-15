"""交互式确认处理，生成前端可用的配置"""
from typing import Dict, Any


def format_interaction_prompt(interactive_cfg: Dict) -> Dict[str, Any]:
        """将插件交互配置转换为 Gradio 可用的结构"""
        mode = interactive_cfg.get("mode", "confirm")
        prompt = interactive_cfg.get("prompt", "请确认操作")
        options = interactive_cfg.get("options", [])
        allow_multiple = interactive_cfg.get("allow_multiple", False)

        if mode == "confirm":
            choices = [(opt["id"], opt["label"]) for opt in options]
            return {
                "type": "confirm",
                "prompt": prompt,
                "choices": choices,
                "multiple": False,
            }
        elif mode == "choice":
            choices = [(opt["id"], opt["label"]) for opt in options]
            return {
                "type": "choice",
                "prompt": prompt,
                "choices": choices,
                "multiple": allow_multiple,
            }
        else:  # input
            return {
                "type": "input",
                "prompt": prompt,
            }

def parse_interaction_response(interactive_cfg: Dict, response: Any) -> Dict[str, Any]:
        """解析用户响应，返回注入到 artifacts 的数据"""
        mode = interactive_cfg.get("mode", "confirm")
        inject_as = interactive_cfg.get("inject_as", "user_feedback")
        options = interactive_cfg.get("options", [])

        if mode == "confirm":
            selected_id = response
            action = "continue"
            for opt in options:
                if opt["id"] == selected_id:
                    action = opt.get("action", "continue")
                    break
            if action == "block":
                raise RuntimeError("用户选择中止流程")
            return {inject_as: selected_id}
        elif mode == "choice":
            selected_ids = response if isinstance(response, list) else [response]
            actions = []
            for sid in selected_ids:
                for opt in options:
                    if opt["id"] == sid:
                        actions.append(opt.get("action", "continue"))
                        break
            if "block" in actions:
                raise RuntimeError("用户选择中止流程")
            return {inject_as: selected_ids}
        else:  # input
            return {inject_as: response}