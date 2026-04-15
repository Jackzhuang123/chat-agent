# core/plugin_registry.py
"""
插件注册表 - 加载、合并多层插件配置
支持：系统默认 -> 用户配置 -> 项目配置 -> CLI 参数
"""
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Optional

DEFAULT_CONFIG_PATH = Path.home() / ".your_app" / "plugins.json"
PROJECT_CONFIG_NAME = ".your_app_plugins.json"


class PluginRegistry:
    def __init__(self, cli_hooks: Optional[List[str]] = None):
        self.system_config = self._load_json(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
        self.user_config = self._load_user_config()
        self.project_config = {}
        self.cli_config = self._parse_cli_hooks(cli_hooks or [])
        self._merged = None

    def _load_json(self, path: Path) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_user_config(self) -> Dict:
        user_path = Path.home() / ".your_app_user_plugins.json"
        return self._load_json(user_path)

    def _parse_cli_hooks(self, hooks: List[str]) -> Dict:
        """解析 CLI 参数，如 'phase:before=my_skill' """
        config = {"phases": {}}
        for spec in hooks:
            if ":" not in spec:
                continue
            phase_part, hook_part = spec.split(":", 1)
            phase = phase_part.strip()
            hook_name = "before" if "before" in hook_part else "after"
            ref = hook_part.split("=")[-1].strip()
            hook = {"type": "skill", "ref": ref, "enabled": True}
            config.setdefault("phases", {}).setdefault(phase, {}).setdefault("hooks", {}).setdefault(hook_name, []).append(hook)
        return config

    def load_project_config(self, project_root: str):
        path = Path(project_root) / PROJECT_CONFIG_NAME
        if path.exists():
            self.project_config = self._load_json(path)
        self._merged = None

    @property
    def merged_config(self) -> Dict:
        if self._merged is None:
            self._merged = self._deep_merge(
                self.system_config,
                self.user_config,
                self.project_config,
                self.cli_config
            )
        return self._merged

    def _deep_merge(self, *configs: Dict) -> Dict:
        """递归合并，后者覆盖前者"""
        result = {}
        for cfg in configs:
            for key, value in cfg.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value)
        return result

    def get_phase_hooks(self, phase_id: str, hook_point: str) -> List[Dict]:
        phase = self.merged_config.get("phases", {}).get(phase_id, {})
        hooks = phase.get("hooks", {}).get(hook_point, [])
        return [h for h in hooks if h.get("enabled", True)]