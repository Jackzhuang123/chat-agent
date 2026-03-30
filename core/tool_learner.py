# tool_learner.py - 智能增强版

import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class ToolUsagePattern:
    """工具使用模式"""
    task_type: str
    tool_sequence: List[str]  # 工具调用序列
    success_rate: float
    avg_execution_time: float
    context_features: Dict[str, Any]  # 上下文特征
    last_used: datetime

    def to_dict(self):
        return {
            **asdict(self),
            "last_used": self.last_used.isoformat()
        }


class AdaptiveToolLearner:
    """
    自适应工具学习器

    特性：
    - 序列模式学习（不只是单工具）
    - 上下文感知推荐
    - 负样本学习（失败模式）
    - 跨任务迁移
    """

    def __init__(self, memory_dir: str = ".agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # 工具统计
        self.tool_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "success": 0, "failed": 0,
                "avg_time": 0, "last_success": None
            }
        )

        # 序列模式：工具A -> 工具B 的转移概率
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # 任务模式库
        self.patterns: List[ToolUsagePattern] = []

        # 失败模式库（负样本学习）
        self.failure_patterns: List[Dict] = []

        # 上下文特征提取器
        self.context_extractor = ContextFeatureExtractor()

        self._load_from_disk()

    def record_usage(
            self,
            task_type: str,
            tool_name: str,
            success: bool,
            execution_time: float = 0,
            context: Dict[str, Any] = None,
            previous_tool: Optional[str] = None,
            error_message: str = None
    ):
        """记录工具使用（增强版）

        Args:
            previous_tool: 前一个执行的工具名，用于更新转移矩阵。
                          由 DeepReflectionEngine 调用时自动传入。
        """
        # 基础统计
        stats = self.tool_stats[tool_name]
        if success:
            stats["success"] += 1
            stats["last_success"] = datetime.now().isoformat()
        else:
            stats["failed"] += 1
            # 记录失败模式
            if error_message:
                self.failure_patterns.append({
                    "tool": tool_name,
                    "task_type": task_type,
                    "error": error_message,
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                })

        # 更新平均时间
        total = stats["success"] + stats["failed"]
        stats["avg_time"] = (stats["avg_time"] * (total - 1) + execution_time) / total

        # 更新转移矩阵：previous_tool -> tool_name（仅成功时才记录为有效转移）
        if previous_tool and success:
            self.transition_matrix[previous_tool][tool_name] += 1

        # 提取上下文特征
        context_features = self.context_extractor.extract(context or {})

        # 查找或创建模式，并更新序列
        pattern = self._find_or_create_pattern(
            task_type, tool_name, context_features
        )
        # 如果有前驱工具，更新序列（将 previous_tool -> tool_name 添加到序列中）
        if previous_tool and previous_tool not in pattern.tool_sequence:
            idx = pattern.tool_sequence.index(previous_tool) if previous_tool in pattern.tool_sequence else -1
            if idx >= 0 and tool_name not in pattern.tool_sequence:
                pattern.tool_sequence.insert(idx + 1, tool_name)
            elif tool_name not in pattern.tool_sequence:
                pattern.tool_sequence.append(tool_name)

        # 成功率只统计 success/failed（忽略 avg_time 等非计数字段）
        s = self.tool_stats[tool_name]["success"]
        f = self.tool_stats[tool_name]["failed"]
        pattern.success_rate = s / max(1, s + f)
        pattern.avg_execution_time = stats["avg_time"]
        pattern.last_used = datetime.now()

        self._save_to_disk()

    def recommend_next_tools(
            self,
            current_task: str,
            executed_tools: List[str],
            current_context: Dict[str, Any] = None,
            top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        推荐下一个工具（序列感知）
        """
        # 1. 基于转移概率
        next_tool_scores = defaultdict(float)

        if executed_tools:
            last_tool = executed_tools[-1]
            transitions = self.transition_matrix.get(last_tool, {})
            total_trans = sum(transitions.values())

            if total_trans > 0:
                for next_tool, count in transitions.items():
                    if next_tool not in executed_tools:  # 避免重复
                        next_tool_scores[next_tool] += count / total_trans * 0.4

        # 2. 基于任务类型匹配
        task_features = self.context_extractor.extract_text(current_task)

        for pattern in self.patterns:
            # 任务类型相似度
            type_sim = self._text_similarity(
                task_features,
                self.context_extractor.extract_text(pattern.task_type)
            )

            # 上下文相似度
            context_sim = 0
            if current_context and pattern.context_features:
                context_sim = self._context_similarity(
                    current_context,
                    pattern.context_features
                )

            # 综合评分
            for tool in pattern.tool_sequence:
                if tool not in executed_tools:
                    score = (
                            type_sim * 0.3 +
                            context_sim * 0.2 +
                            pattern.success_rate * 0.3 +
                            (1 / (1 + len(executed_tools))) * 0.2  # 序列位置偏好
                    )
                    next_tool_scores[tool] = max(next_tool_scores[tool], score)

        # 3. 基于失败模式避免
        for failure in self.failure_patterns[-10:]:  # 最近10次失败
            if failure["tool"] in next_tool_scores:
                # 如果当前上下文与失败上下文相似，降低推荐度
                if self._context_similarity(
                        current_context or {},
                        failure.get("context", {})
                ) > 0.7:
                    next_tool_scores[failure["tool"]] *= 0.5

        # 排序返回
        sorted_tools = sorted(
            next_tool_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "tool": tool,
                "confidence": score,
                "reason": self._generate_recommend_reason(tool, executed_tools)
            }
            for tool, score in sorted_tools[:top_k]
        ]

    def predict_success_probability(
            self,
            tool_name: str,
            task_type: str,
            context: Dict[str, Any]
    ) -> float:
        """预测工具执行成功率"""
        _ts = self.tool_stats[tool_name]
        base_rate = (
                _ts["success"] /
                max(1, _ts["success"] + _ts["failed"])
        )

        # 上下文调整
        context_boost = 0
        for pattern in self.patterns:
            if tool_name in pattern.tool_sequence:
                sim = self._context_similarity(
                    context,
                    pattern.context_features
                )
                if sim > 0.8:
                    context_boost = max(context_boost, pattern.success_rate - base_rate)

        # 时间衰减（长时间未使用的工具成功率降低）
        last_success = self.tool_stats[tool_name].get("last_success")
        if last_success:
            days_since = (datetime.now() - datetime.fromisoformat(last_success)).days
            time_decay = max(0, 1 - days_since / 30)  # 30天后完全衰减
        else:
            time_decay = 0.5

        return min(1.0, (base_rate + context_boost * 0.3) * (0.7 + 0.3 * time_decay))

    def _find_or_create_pattern(
            self,
            task_type: str,
            tool: str,
            context_features: Dict
    ) -> ToolUsagePattern:
        """查找或创建模式"""
        for pattern in self.patterns:
            if (pattern.task_type == task_type and
                    tool in pattern.tool_sequence):
                return pattern

        # 创建新模式
        new_pattern = ToolUsagePattern(
            task_type=task_type,
            tool_sequence=[tool],
            success_rate=0.5,
            avg_execution_time=0,
            context_features=context_features,
            last_used=datetime.now()
        )
        self.patterns.append(new_pattern)
        return new_pattern

    def _text_similarity(self, text1: str, text2: str) -> float:
        """文本相似度（简化版Jaccard）"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """上下文相似度"""
        if not ctx1 or not ctx2:
            return 0.0

        # 比较共同的键
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            v1, v2 = ctx1[key], ctx2[key]
            if isinstance(v1, str) and isinstance(v2, str):
                similarities.append(self._text_similarity(v1, v2))
            elif v1 == v2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

        return sum(similarities) / len(similarities)

    def _generate_recommend_reason(
            self,
            tool: str,
            executed: List[str]
    ) -> str:
        """生成推荐理由"""
        if not executed:
            return f"基于任务类型推荐 {tool}"

        last = executed[-1]
        trans_count = self.transition_matrix.get(last, {}).get(tool, 0)
        if trans_count > 0:
            return f"{last} 后常接 {tool}（{trans_count}次）"

        stats = self.tool_stats[tool]
        total = max(1, stats["success"] + stats["failed"])
        return f"历史成功率 {stats['success'] / total:.0%}"

    def _save_to_disk(self):
        """持久化"""
        data = {
            "tool_stats": dict(self.tool_stats),
            "transitions": dict(self.transition_matrix),
            "patterns": [p.to_dict() for p in self.patterns],
            "failures": self.failure_patterns[-100:]  # 保留最近100条
        }
        with open(self.memory_dir / "tool_learner_v2.pkl", "wb") as f:
            pickle.dump(data, f)

    def _load_from_disk(self):
        """加载"""
        file_path = self.memory_dir / "tool_learner_v2.pkl"
        if not file_path.exists():
            return

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            self.tool_stats = defaultdict(
                lambda: {"success": 0, "failed": 0, "avg_time": 0, "last_success": None},
                data.get("tool_stats", {})
            )
            self.transition_matrix = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in data.get("transitions", {}).items()}
            )
            # 恢复 patterns
            raw_patterns = data.get("patterns", [])
            self.patterns = []
            for p in raw_patterns:
                try:
                    last_used = datetime.fromisoformat(p.get("last_used", datetime.now().isoformat()))
                    self.patterns.append(ToolUsagePattern(
                        task_type=p.get("task_type", "general"),
                        tool_sequence=p.get("tool_sequence", []),
                        success_rate=p.get("success_rate", 0.5),
                        avg_execution_time=p.get("avg_execution_time", 0),
                        context_features=p.get("context_features", {}),
                        last_used=last_used,
                    ))
                except Exception:
                    pass
            # 恢复失败模式
            self.failure_patterns = data.get("failures", [])

        except Exception as e:
            print(f"加载工具学习器失败: {e}")


class ContextFeatureExtractor:
    """上下文特征提取器"""

    def extract(self, context: Dict) -> Dict[str, Any]:
        """提取结构化特征"""
        features = {}

        # 文件类型特征
        if "path" in context:
            path = context["path"]
            features["file_ext"] = Path(path).suffix if "." in path else "none"
            features["path_depth"] = len(Path(path).parts)

        # 文本长度特征
        if "content" in context:
            content = context["content"]
            features["content_length"] = len(content)
            features["has_code"] = any(kw in content for kw in ["def ", "class ", "import "])

        # 时间特征
        features["hour"] = datetime.now().hour

        return features

    def extract_text(self, text: str) -> str:
        """提取文本特征"""
        return text.lower()