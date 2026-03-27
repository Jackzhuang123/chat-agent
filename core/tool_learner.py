# -*- coding: utf-8 -*-
"""工具学习模块 - 基于历史自动选择工具"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


class ToolLearner:
    """
    工具学习系统 - 基于任务类型和历史自动推荐工具

    特性：
    - 任务分类（文件操作、代码分析、系统命令等）
    - 工具成功率统计
    - 基于上下文的工具推荐
    - 持久化学习结果
    """

    def __init__(self, memory_dir: str = ".agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # 任务类型 -> 工具映射
        self.task_patterns = {
            "文件读取": {
                "keywords": ["读取", "查看", "打开", "read", "view", "cat"],
                "tools": ["read_file"],
                "priority": 1.0
            },
            "文件写入": {
                "keywords": ["写入", "创建", "保存", "write", "create", "save"],
                "tools": ["write_file"],
                "priority": 1.0
            },
            "文件编辑": {
                "keywords": ["修改", "编辑", "替换", "edit", "modify", "replace"],
                "tools": ["edit_file"],
                "priority": 1.0
            },
            "目录浏览": {
                "keywords": ["列出", "浏览", "扫描", "list", "browse", "scan"],
                "tools": ["list_dir"],
                "priority": 0.9
            },
            "代码分析": {
                "keywords": ["分析", "解析", "查找类", "查找方法", "analyze", "parse", "class", "method"],
                "tools": ["bash", "read_file"],
                "priority": 0.8
            },
            "命令执行": {
                "keywords": ["执行", "运行", "命令", "execute", "run", "command", "grep", "find"],
                "tools": ["bash"],
                "priority": 0.7
            }
        }

        # 工具使用历史：{task_type: {tool: {"success": int, "failed": int}}}
        self.tool_history: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0, "failed": 0})
        )

        # 持久化文件
        self.learner_file = self.memory_dir / "tool_learner.json"
        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载学习结果"""
        if self.learner_file.exists():
            try:
                with open(self.learner_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换回 defaultdict
                    for task_type, tools in data.get("tool_history", {}).items():
                        for tool, stats in tools.items():
                            self.tool_history[task_type][tool] = stats
            except Exception:
                pass

    def save_to_disk(self):
        """保存学习结果到磁盘"""
        try:
            # 转换为普通 dict 以便序列化
            data = {
                "tool_history": {
                    task_type: dict(tools)
                    for task_type, tools in self.tool_history.items()
                },
                "last_update": str(Path(__file__).stat().st_mtime)
            }
            with open(self.learner_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def classify_task(self, user_input: str) -> List[str]:
        """任务分类"""
        user_input_lower = user_input.lower()
        matched_types = []

        for task_type, pattern in self.task_patterns.items():
            keywords = pattern["keywords"]
            priority = pattern["priority"]

            # 检查关键词匹配
            match_count = sum(1 for kw in keywords if kw in user_input_lower)
            if match_count > 0:
                matched_types.append((task_type, priority, match_count))

        # 按优先级和匹配数排序
        matched_types.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return [task_type for task_type, _, _ in matched_types]

    def recommend_tools(
        self,
        user_input: str,
        top_k: int = 3,
        context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """推荐工具"""
        # 1. 任务分类
        task_types = self.classify_task(user_input)

        if not task_types:
            # 无法分类，返回默认推荐
            return [
                {"tool": "read_file", "confidence": 0.5, "reason": "默认推荐"},
                {"tool": "list_dir", "confidence": 0.4, "reason": "默认推荐"},
                {"tool": "bash", "confidence": 0.3, "reason": "默认推荐"}
            ]

        # 2. 收集候选工具
        candidates = []

        for task_type in task_types:
            pattern = self.task_patterns[task_type]
            base_tools = pattern["tools"]
            priority = pattern["priority"]

            for tool in base_tools:
                # 计算成功率
                history = self.tool_history.get(task_type, {}).get(tool, {"success": 0, "failed": 0})
                total = history["success"] + history["failed"]
                success_rate = history["success"] / total if total > 0 else 0.5

                # 综合评分：优先级 * 成功率
                confidence = priority * success_rate

                candidates.append({
                    "tool": tool,
                    "confidence": confidence,
                    "task_type": task_type,
                    "success_rate": success_rate,
                    "total_uses": total,
                    "reason": f"任务类型: {task_type}"
                })

        # 3. 去重并排序
        tool_scores = {}
        for cand in candidates:
            tool = cand["tool"]
            if tool not in tool_scores or cand["confidence"] > tool_scores[tool]["confidence"]:
                tool_scores[tool] = cand

        # 4. 返回 top-k
        recommendations = sorted(tool_scores.values(), key=lambda x: x["confidence"], reverse=True)
        return recommendations[:top_k]

    def record_usage(self, task_type: str, tool: str, success: bool):
        """记录工具使用"""
        if success:
            self.tool_history[task_type][tool]["success"] += 1
        else:
            self.tool_history[task_type][tool]["failed"] += 1

    def get_tool_stats(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """获取工具统计"""
        if task_type:
            return dict(self.tool_history.get(task_type, {}))

        # 全局统计
        stats = {}
        for task_type, tools in self.tool_history.items():
            for tool, history in tools.items():
                if tool not in stats:
                    stats[tool] = {"success": 0, "failed": 0}
                stats[tool]["success"] += history["success"]
                stats[tool]["failed"] += history["failed"]

        return stats

    def suggest_next_tool(
        self,
        user_input: str,
        executed_tools: List[str],
        context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """根据已执行工具推荐下一个工具"""
        # 获取推荐
        recommendations = self.recommend_tools(user_input, top_k=5, context=context)

        # 过滤已执行的工具
        for rec in recommendations:
            if rec["tool"] not in executed_tools:
                return rec

        return None
