# core/plugin_executor.py
"""
插件执行器 - 支持 skill/agent/webhook/script/interactive
"""
import json
import subprocess
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class PluginResult:
    def __init__(self, status: str, message: str = "", gate_value: Optional[float] = None,
                 modified_input: Optional[Dict] = None, interactive: Optional[Dict] = None,
                 duration_ms: int = 0):
        self.status = status
        self.message = message
        self.gate_value = gate_value
        self.modified_input = modified_input
        self.interactive = interactive
        self.duration_ms = duration_ms

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "gate_value": self.gate_value,
            "modified_input": self.modified_input,
            "interactive": self.interactive,
            "duration_ms": self.duration_ms
        }


def execute_plugin(plugin_config: Dict, payload: Dict, work_dir: Optional[str] = None) -> PluginResult:
    ptype = plugin_config.get("type", "skill")
    ref = plugin_config.get("ref", "")
    timeout = plugin_config.get("timeout", 60)

    if not ref:
        return PluginResult("error", "插件引用(ref)为空")

    if ptype == "script":
        return _run_script(ref, payload, timeout, work_dir)
    elif ptype == "webhook":
        return _call_webhook(ref, payload, timeout, plugin_config.get("token"))
    elif ptype == "skill":
        return PluginResult("pending_skill", f"请调用 Skill: {ref}\n参数: {json.dumps(payload, ensure_ascii=False)}")
    elif ptype == "agent":
        return PluginResult("pending_agent", f"请派遣 Agent: {ref}\n参数: {json.dumps(payload, ensure_ascii=False)}")
    elif ptype == "interactive":
        interactive_cfg = plugin_config.get("interactive", {})
        return PluginResult("pending_interactive", "等待用户交互", interactive=interactive_cfg)
    else:
        return PluginResult("error", f"未知插件类型: {ptype}")


def _run_script(ref: str, payload: Dict, timeout: int, cwd: Optional[str]) -> PluginResult:
    import os
    start = time.time()
    env = os.environ.copy()
    env["PLUGIN_PAYLOAD"] = json.dumps(payload, ensure_ascii=False)
    try:
        if ref.endswith(".py"):
            cmd = ["python3", ref]
        else:
            cmd = ["bash", ref]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd
        )
        duration = int((time.time() - start) * 1000)
        if result.returncode != 0:
            return PluginResult("error", f"脚本失败: {result.stderr[:200]}", duration_ms=duration)
        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout) if stdout.startswith("{") else {}
        except json.JSONDecodeError:
            data = {"message": stdout}
        gate = data.get("gate_value")
        return PluginResult("ok", data.get("message", stdout), gate_value=gate,
                           modified_input=data.get("modified_input"), duration_ms=duration)
    except subprocess.TimeoutExpired:
        return PluginResult("error", f"脚本超时 ({timeout}s)")
    except Exception as e:
        return PluginResult("error", str(e))


def _call_webhook(url: str, payload: Dict, timeout: int, token: Optional[str]) -> PluginResult:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    start = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            duration = int((time.time() - start) * 1000)
            try:
                resp_data = json.loads(body) if body.strip().startswith("{") else {}
            except json.JSONDecodeError:
                resp_data = {"message": body}
            gate = resp_data.get("gate_value")
            return PluginResult("ok", resp_data.get("message", body), gate_value=gate, duration_ms=duration)
    except urllib.error.HTTPError as e:
        return PluginResult("error", f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        return PluginResult("error", str(e))