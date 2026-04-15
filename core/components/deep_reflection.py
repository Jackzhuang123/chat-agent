# -*- coding: utf-8 -*-
"""深度反思引擎"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class DeepReflectionEngine:
    """失败分析 + 成功模式学习 + 双向同步 AdaptiveToolLearner"""

    FAILURE_PATTERNS = {
        "file_not_found": {
            "patterns": [r"not found", r"不存在", r"找不到", r"No such file"],
            "category": "parameter_error",
            "fixes": ["使用绝对路径", "先list_dir确认", "检查文件名拼写"]
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"权限不足"],
            "category": "tool_error",
            "fixes": ["检查文件权限", "更换工作目录"]
        },
        "syntax_error": {
            "patterns": [r"syntax error", r"invalid syntax", r"语法错误"],
            "category": "parameter_error",
            "fixes": ["检查JSON格式", "验证参数类型"]
        },
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"超时"],
            "category": "execution_error",
            "fixes": ["增加超时时间", "拆分任务", "优化命令"]
        }
    }

    def __init__(self):
        self.failure_memory: Dict[str, List] = {}
        self.success_patterns: Dict[str, Dict] = {}
        self._tool_learner = None

    def attach_tool_learner(self, learner):
        self._tool_learner = learner

    def reflect_on_result(self, tool_name: str, result: Dict, context: Dict = None, history: List[Dict] = None) -> Dict:
        context = context or {}
        history = history if history is not None else []
        error = result.get("error")
        if not error:
            self._record_success(tool_name, context, history)
            return {"success": True, "level": "surface", "analysis": "执行成功", "suggestions": [], "action": "continue"}

        error_analysis = self._analyze_error(error)
        is_repeated = self._is_repeated_failure(error_analysis["category"], tool_name)
        recent_tools = context.get("recent_tools", [])
        is_loop = self._detect_loop(recent_tools)

        if is_loop:
            level, action, suggestions = "meta", "break_loop", ["强制切换策略", "更换工具类型", "请求用户澄清"]
        elif is_repeated:
            level, action, suggestions = "strategic", "escalate", [
                f"重复错误！停止使用{tool_name}",
                f"尝试: {error_analysis['fixes'][0] if error_analysis['fixes'] else '替代方案'}",
                "如仍失败则上报BLOCKED"
            ]
        else:
            level, action, suggestions = "operational", "retry_with_fix", error_analysis["fixes"][:2]

        reflection = {
            "success": False, "level": level, "category": error_analysis["category"],
            "root_cause": error_analysis["root_cause"], "analysis": error_analysis["description"],
            "suggestions": suggestions, "action": action, "is_repeated": is_repeated,
            "is_loop": is_loop, "confidence": 0.9 if is_repeated else 0.7
        }
        history.append({
            "timestamp": datetime.now().isoformat(), "tool": tool_name, "success": False,
            "error": error[:100], "category": reflection.get("category", "unknown"),
            "args_sig": context.get("tool_args_sig", ""), "reflection": reflection
        })
        cat = error_analysis["category"]
        self.failure_memory.setdefault(cat, []).append({"tool": tool_name, "error": error, "timestamp": datetime.now().isoformat()})
        if self._tool_learner:
            self._tool_learner.record_usage(
                task_type=context.get("task", "general"), tool_name=tool_name, success=False,
                execution_time=result.get("_execution_time", 0), context=context,
                previous_tool=recent_tools[-1] if recent_tools else None, error_message=error[:200]
            )
        return reflection

    def _record_success(self, tool_name: str, context: Dict, history: List[Dict]) -> None:
        recent_tools = context.get("recent_tools", [])
        exec_time = context.get("_execution_time", 0)
        task_type = context.get("task", "general")
        history.append({
            "timestamp": datetime.now().isoformat(), "tool": tool_name, "success": True,
            "reflection": {"level": "surface", "analysis": "执行成功", "action": "continue"}
        })
        if recent_tools:
            prev = recent_tools[-1]
            seq_key = f"{prev}->{tool_name}"
            if seq_key not in self.success_patterns:
                self.success_patterns[seq_key] = {"count": 0, "avg_time": 0.0, "task_types": []}
            pat = self.success_patterns[seq_key]
            pat["count"] += 1
            pat["avg_time"] = (pat["avg_time"] * (pat["count"] - 1) + exec_time) / pat["count"]
            if task_type not in pat["task_types"]:
                pat["task_types"].append(task_type)
        if self._tool_learner:
            self._tool_learner.record_usage(
                task_type=task_type, tool_name=tool_name, success=True, execution_time=exec_time,
                context=context, previous_tool=recent_tools[-1] if recent_tools else None
            )

    def get_efficient_sequences(self, top_k: int = 5) -> List[Dict]:
        sorted_patterns = sorted(self.success_patterns.items(), key=lambda x: x[1]["count"], reverse=True)
        return [{"sequence": k, "count": v["count"], "avg_time": v["avg_time"]} for k, v in sorted_patterns[:top_k]]

    def _analyze_error(self, error: str) -> Dict:
        error_lower = error.lower()
        for error_type, info in self.FAILURE_PATTERNS.items():
            for pattern in info["patterns"]:
                if re.search(pattern, error_lower, re.I):
                    return {
                        "category": info["category"], "type": error_type,
                        "description": f"{error_type}: {error[:100]}",
                        "root_cause": info["category"], "fixes": info["fixes"], "confidence": 0.85
                    }
        return {
            "category": "unknown_error", "type": "unknown", "description": f"未知错误: {error[:100]}",
            "root_cause": "需要进一步分析", "fixes": ["记录错误详情", "尝试替代工具", "简化任务"], "confidence": 0.3
        }

    def _is_repeated_failure(self, category: str, tool_name: str) -> bool:
        failures = self.failure_memory.get(category, [])
        recent = [f for f in failures if f["tool"] == tool_name and (datetime.now() - datetime.fromisoformat(f["timestamp"])).seconds < 300]
        return len(recent) > 0

    def _detect_loop(self, recent_tools: List[str]) -> bool:
        if len(recent_tools) < 4:
            return False
        last_4 = recent_tools[-4:]
        return len(set(last_4)) <= 2 and last_4[0] == last_4[2]

    def should_continue(self, history: List[Dict], max_failed: int = 3) -> Tuple[bool, str]:
        failed_history = [h for h in history if isinstance(h, dict) and not h.get("success", True)]
        if len(failed_history) < max_failed:
            return True, ""
        recent_failed = failed_history[-max_failed:]
        strategic_count = sum(1 for h in recent_failed if isinstance(h, dict) and (h.get("level") == "strategic" or (isinstance(h.get("reflection"), dict) and h["reflection"].get("level") == "strategic")))
        if strategic_count >= max_failed:
            return False, "连续战略级失败，建议重新规划或人工介入"
        meta_count = sum(1 for h in recent_failed if isinstance(h, dict) and (h.get("level") == "meta" or (isinstance(h.get("reflection"), dict) and h["reflection"].get("level") == "meta")))
        if meta_count >= 2:
            return False, "检测到工具调用循环，建议更换策略"
        signatures = []
        for h in recent_failed:
            cat = h.get("category") or (h.get("reflection") or {}).get("category", "unknown")
            tool = h.get("tool", "")
            args_sig = h.get("args_sig", "")
            signatures.append((cat, tool, args_sig))
        if signatures and len(set(signatures)) == 1 and signatures[0][0] != "unknown":
            cat, tool, _ = signatures[0]
            return False, f"连续 {len(recent_failed)} 次相同错误({cat})且重复工具/参数({tool})，建议更换策略"
        return True, ""

    def get_reflection_summary(self, history: List[Dict]) -> str:
        if not history:
            return "暂无反思记录"
        recent = history[-5:]
        lines = ["## 近期反思记录"]
        for r in recent:
            refl = r.get("reflection", {})
            icon = "✅" if r.get("success") else "❌"
            lines.append(f"{icon} {r['tool']}: {refl.get('analysis', '')[:50]}...")
        return "\n".join(lines)