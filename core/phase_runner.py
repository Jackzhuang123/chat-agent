# core/phase_runner.py
"""
阶段执行引擎 - 负责一个阶段内插件的拓扑排序、执行、门控判定
"""
import json
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple
from .plugin_registry import PluginRegistry
from .plugin_executor import execute_plugin, PluginResult


def topological_sort(hooks: List[Dict]) -> List[List[Dict]]:
    """Kahn 算法分层排序，支持 depends_on"""
    id_to_hook = {h.get("id", h["ref"]): h for h in hooks if h.get("enabled", True)}
    in_degree = defaultdict(int)
    adj = defaultdict(list)

    for hid, hook in id_to_hook.items():
        deps = hook.get("depends_on", [])
        if isinstance(deps, str):
            deps = [deps]
        for dep in deps:
            if dep in id_to_hook:
                adj[dep].append(hid)
                in_degree[hid] += 1

    queue = deque([hid for hid in id_to_hook if in_degree[hid] == 0])
    layers = []
    while queue:
        layer = [id_to_hook[hid] for hid in queue]
        layers.append(layer)
        next_queue = deque()
        for hid in queue:
            for neighbor in adj[hid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        queue = next_queue

    remaining = [h for hid, h in id_to_hook.items() if in_degree[hid] > 0]
    if remaining:
        layers.append(remaining)
    return layers


def check_gate(hook: Dict, result: PluginResult) -> Optional[str]:
    gate = hook.get("gate")
    if not gate or result.gate_value is None:
        return None
    threshold = gate.get("pass_threshold")
    if threshold is not None and result.gate_value < threshold:
        return f"门控失败: gate_value={result.gate_value:.2%} < {threshold:.2%}"
    return None


class PhaseRunner:
    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    def run_hooks(self, phase_id: str, hook_point: str, payload: Dict, work_dir: Optional[str] = None) -> Tuple[Dict, List[Dict]]:
        hooks = self.registry.get_phase_hooks(phase_id, hook_point)
        if not hooks:
            return payload.get("input", {}), []

        layers = topological_sort(hooks)
        modified_input = payload.get("input", {})
        log_entries = []

        for layer in layers:
            for hook in layer:
                result = execute_plugin(hook, {"input": modified_input, "phase": phase_id, "trigger": hook_point}, work_dir)
                log_entries.append({
                    "hook": hook.get("id", hook["ref"]),
                    "status": result.status,
                    "message": result.message,
                    "gate_value": result.gate_value
                })

                if result.status == "error":
                    on_fail = hook.get("on_failure", "warn")
                    if on_fail == "block":
                        raise RuntimeError(f"插件 {hook['ref']} 失败且策略为 block: {result.message}")
                elif result.status == "pending_interactive":
                    log_entries[-1]["interactive"] = result.interactive
                    return modified_input, log_entries
                elif result.status.startswith("pending_"):
                    pass
                else:
                    gate_fail = check_gate(hook, result)
                    if gate_fail:
                        on_fail = hook.get("gate", {}).get("on_fail", "block")
                        if on_fail == "block":
                            raise RuntimeError(gate_fail)
                    if result.modified_input:
                        modified_input = result.modified_input

        return modified_input, log_entries