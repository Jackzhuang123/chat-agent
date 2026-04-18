# -*- coding: utf-8 -*-
"""增强反思引擎：错误分类、智能重试、策略调整"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.monitor_logger import get_monitor_logger


class EnhancedReflectionEngine:
    """失败分析 + 成功模式学习，支持分级处理与策略切换"""

    FAILURE_PATTERNS = {
        "file_not_found": {
            "patterns": [r"not found", r"不存在", r"找不到", r"No such file"],
            "category": "parameter_error",
            "fixes": ["使用绝对路径", "先list_dir确认", "检查文件名拼写"],
            "retry_strategy": "correct_path"
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"权限不足"],
            "category": "tool_error",
            "fixes": ["检查文件权限", "更换工作目录"],
            "retry_strategy": "abort"
        },
        "syntax_error": {
            "patterns": [r"syntax error", r"invalid syntax", r"语法错误"],
            "category": "parameter_error",
            "fixes": ["检查JSON格式", "验证参数类型"],
            "retry_strategy": "reformat"
        },
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"超时"],
            "category": "execution_error",
            "fixes": ["增加超时时间", "拆分任务", "优化命令"],
            "retry_strategy": "increase_timeout"
        },
        "empty_output": {
            "patterns": [r"stdout.*空", r"no output", r"未产生任何输出"],
            "category": "execution_error",
            "fixes": ["改用bash命令", "检查代码是否有print"],
            "retry_strategy": "switch_tool"
        }
    }

    def __init__(self):
        self.failure_memory: Dict[str, List] = {}
        self.success_patterns: Dict[str, Dict] = {}
        self._tool_learner = None
        self.monitor = get_monitor_logger()

    def attach_tool_learner(self, learner):
        self._tool_learner = learner

    def record_result(
        self,
        tool_name: str,
        result: Dict,
        context: Dict = None,
        history: List[Dict] = None
    ):
        """记录工具执行结果，更新反思历史"""
        context = context or {}
        history = history if history is not None else []
        error = result.get("error")

        if not error:
            self._record_success(tool_name, context, history)
            return

        analysis = self._analyze_error(error)
        is_repeated = self._is_repeated_failure(analysis["category"], tool_name)
        recent_tools = context.get("recent_tools", [])
        is_loop = self._detect_loop(recent_tools)

        if is_loop:
            level, action, suggestions = "meta", "break_loop", ["强制切换策略", "更换工具类型", "请求用户澄清"]
        elif is_repeated:
            level, action, suggestions = "strategic", "escalate", [
                f"重复错误！停止使用{tool_name}",
                f"尝试: {analysis['fixes'][0] if analysis['fixes'] else '替代方案'}"
            ]
        else:
            level, action, suggestions = "operational", "retry_with_fix", analysis["fixes"][:2]

        if level in ("strategic", "meta"):
            self.monitor.warning(f"反思触发 {level} 级别: {tool_name} - {analysis['category']} - {action}")

        reflection = {
            "success": False,
            "level": level,
            "category": analysis["category"],
            "analysis": analysis["description"],
            "suggestions": suggestions,
            "action": action,
            "retry_strategy": analysis.get("retry_strategy", "default"),
            "is_repeated": is_repeated,
            "is_loop": is_loop,
        }

        history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": False,
            "error": error[:100],
            "reflection": reflection
        })

        cat = analysis["category"]
        self.failure_memory.setdefault(cat, []).append({
            "tool": tool_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

        if self._tool_learner:
            self._tool_learner.record_usage(
                task_type=context.get("task", "general"),
                tool_name=tool_name,
                success=False,
                execution_time=result.get("_execution_time", 0),
                context=context,
                previous_tool=recent_tools[-1] if recent_tools else None,
                error_message=error[:200]
            )

    def _record_success(self, tool_name: str, context: Dict, history: List[Dict]):
        recent_tools = context.get("recent_tools", [])
        exec_time = context.get("_execution_time", 0)
        task_type = context.get("task", "general")

        history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": True,
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
                task_type=task_type,
                tool_name=tool_name,
                success=True,
                execution_time=exec_time,
                context=context,
                previous_tool=recent_tools[-1] if recent_tools else None
            )

    def _analyze_error(self, error: str) -> Dict:
        error_lower = error.lower()
        for error_type, info in self.FAILURE_PATTERNS.items():
            for pattern in info["patterns"]:
                if re.search(pattern, error_lower, re.I):
                    return {
                        "category": info["category"],
                        "type": error_type,
                        "description": f"{error_type}: {error[:100]}",
                        "fixes": info["fixes"],
                        "retry_strategy": info.get("retry_strategy", "default"),
                        "confidence": 0.85
                    }
        return {
            "category": "unknown_error",
            "type": "unknown",
            "description": f"未知错误: {error[:100]}",
            "fixes": ["记录错误详情", "尝试替代工具", "简化任务"],
            "retry_strategy": "default",
            "confidence": 0.3
        }

    def _is_repeated_failure(self, category: str, tool_name: str) -> bool:
        failures = self.failure_memory.get(category, [])
        recent = [
            f for f in failures
            if f["tool"] == tool_name and
            (datetime.now() - datetime.fromisoformat(f["timestamp"])).seconds < 300
        ]
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
        strategic_count = sum(1 for h in recent_failed if h.get("reflection", {}).get("level") == "strategic")
        if strategic_count >= max_failed:
            return False, "连续战略级失败，建议重新规划或人工介入"

        meta_count = sum(1 for h in recent_failed if h.get("reflection", {}).get("level") == "meta")
        if meta_count >= 2:
            return False, "检测到工具调用循环，建议更换策略"

        signatures = []
        for h in recent_failed:
            cat = h.get("reflection", {}).get("category", "unknown")
            tool = h.get("tool", "")
            signatures.append((cat, tool))
        if len(set(signatures)) == 1 and signatures[0][0] != "unknown":
            return False, f"连续 {len(recent_failed)} 次相同错误 ({signatures[0][0]})，建议更换策略"

        return True, ""

    def get_reflection_summary(self, history: List[Dict]) -> str:
        if not history:
            return "暂无反思记录"
        recent = history[-3:]
        lines = ["## 近期反思记录"]
        for r in recent:
            refl = r.get("reflection", {})
            icon = "✅" if r.get("success") else "❌"
            lines.append(f"{icon} {r['tool']}: {refl.get('analysis', '')[:50]}...")
        return "\n".join(lines)

    def get_efficient_sequences(self, top_k: int = 5) -> List[Dict]:
        sorted_patterns = sorted(
            self.success_patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        return [
            {"sequence": k, "count": v["count"], "avg_time": v["avg_time"]}
            for k, v in sorted_patterns[:top_k]
        ]

    def get_failure_statistics(self, history: Optional[List[Dict]] = None, top_k: int = 5) -> Dict[str, object]:
        """统计失败模式、工具失败分布和循环/重复错误情况。"""
        history = history or []
        failed_items = [h for h in history if isinstance(h, dict) and not h.get("success", True)]

        by_category: Dict[str, int] = {}
        by_tool: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        repeated_failures = 0
        loop_failures = 0

        for item in failed_items:
            reflection = item.get("reflection", {}) or {}
            category = reflection.get("category", "unknown")
            level = reflection.get("level", "unknown")
            tool = item.get("tool", "unknown")

            by_category[category] = by_category.get(category, 0) + 1
            by_tool[tool] = by_tool.get(tool, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1

            if reflection.get("is_repeated"):
                repeated_failures += 1
            if reflection.get("is_loop"):
                loop_failures += 1

        def _top_items(counter: Dict[str, int]) -> List[Dict[str, object]]:
            ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            return [{"name": name, "count": count} for name, count in ranked[:top_k]]

        return {
            "total_failures": len(failed_items),
            "repeated_failures": repeated_failures,
            "loop_failures": loop_failures,
            "top_failure_categories": _top_items(by_category),
            "top_failed_tools": _top_items(by_tool),
            "failure_levels": _top_items(by_level),
            "top_success_sequences": self.get_efficient_sequences(top_k=top_k),
        }

    def get_reflection_report(self, history: Optional[List[Dict]] = None, top_k: int = 5) -> str:
        stats = self.get_failure_statistics(history=history, top_k=top_k)
        lines = ["## 反思统计报告"]
        lines.append(f"- 总失败次数: {stats['total_failures']}")
        lines.append(f"- 重复失败次数: {stats['repeated_failures']}")
        lines.append(f"- 循环失败次数: {stats['loop_failures']}")

        if stats["top_failure_categories"]:
            lines.append("- 高频失败类别: " + "；".join(f"{x['name']}({x['count']})" for x in stats["top_failure_categories"]))
        if stats["top_failed_tools"]:
            lines.append("- 高频失败工具: " + "；".join(f"{x['name']}({x['count']})" for x in stats["top_failed_tools"]))
        if stats["top_success_sequences"]:
            lines.append("- 高效工具序列: " + "；".join(f"{x['sequence']}({x['count']})" for x in stats["top_success_sequences"]))

        return "\n".join(lines)
