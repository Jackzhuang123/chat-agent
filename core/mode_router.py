# mode_router.py - 精准意图识别

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple
from core.monitor_logger import get_monitor_logger

class IntentType(Enum):
    CHAT = "chat"
    TOOLS = "tools"
    SKILLS = "skills"
    HYBRID = "hybrid"
    PLAN = "plan"
    MULTI_AGENT = "multi_agent"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    raw_confidence: float
    calibrated_confidence: float
    reasoning: str
    router_type: str
    alternatives: List[Tuple[IntentType, float]]
    risk_level: str
    suggested_params: Dict[str, Any]


class ConfidenceCalibrator:
    def __init__(self):
        self.calibration_history: List[Tuple[float, bool]] = []
        self.temperature = 1.0
        self.monitor = get_monitor_logger()

    def calibrate(self, raw_confidence: float, router_type: str) -> float:
        router_bias = {"rule": 0.05, "llm": -0.05, "ensemble": 0.0}.get(router_type, 0.0)
        adjusted = raw_confidence / self.temperature
        calibrated = adjusted + router_bias
        return max(0.1, min(0.99, calibrated))

    def update(self, predicted_conf: float, was_correct: bool):
        self.calibration_history.append((predicted_conf, was_correct))
        if len(self.calibration_history) >= 50:
            self._update_temperature()

    def _update_temperature(self):
        recent = self.calibration_history[-50:]
        conf_buckets = {}
        for conf, correct in recent:
            bucket = round(conf * 10) / 10
            if bucket not in conf_buckets:
                conf_buckets[bucket] = []
            conf_buckets[bucket].append(correct)
        total_error = 0
        for bucket, outcomes in conf_buckets.items():
            avg_conf = bucket
            avg_acc = sum(outcomes) / len(outcomes)
            total_error += abs(avg_conf - avg_acc)
        if total_error > 0.2:
            self.temperature *= 1.1
        elif total_error < 0.1:
            self.temperature *= 0.95
        self.calibration_history = []


class ReliabilityAssessor:
    def __init__(self):
        self.uncertainty_patterns = [r"可能|也许|大概|应该", r"不确定|不清楚|不知道", r"could|might|maybe|possibly", r"unclear|ambiguous|confusing"]
        self.contradiction_patterns = [r"但是|然而|不过|although|but|however"]

    def assess(self, llm_response: str, intent_result: Dict) -> Dict[str, Any]:
        risk_factors = []
        score = 1.0
        uncertainty_count = sum(1 for p in self.uncertainty_patterns if re.search(p, llm_response, re.I))
        if uncertainty_count > 0:
            score -= 0.1 * uncertainty_count
            risk_factors.append(f"检测到{uncertainty_count}处不确定性表达")
        contradiction_count = sum(1 for p in self.contradiction_patterns if re.search(p, llm_response, re.I))
        if contradiction_count > 0:
            score -= 0.15 * contradiction_count
            risk_factors.append("检测到潜在矛盾")
        conf = intent_result.get("confidence", 0.5)
        if conf > 0.95 and uncertainty_count > 0:
            score -= 0.2
            risk_factors.append("高置信度与不确定性表达冲突")
        try:
            json_str = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group())
                required_keys = ["mode", "confidence", "reason"]
                missing = [k for k in required_keys if k not in data]
                if missing:
                    score -= 0.1 * len(missing)
                    risk_factors.append(f"缺少字段: {missing}")
        except:
            score -= 0.3
            risk_factors.append("JSON解析失败")
        recommendation = "proceed" if score > 0.8 else "verify" if score > 0.6 else "fallback"
        return {"reliability_score": max(0, score), "risk_factors": risk_factors, "recommendation": recommendation}


class PreciseModeRouter:
    def __init__(
            self,
            llm_forward_fn: Optional[Callable] = None,
            confidence_threshold: float = 0.7,
            enable_calibration: bool = True
    ):
        self.llm_forward_fn = llm_forward_fn
        self.threshold = confidence_threshold
        self.calibrator = ConfidenceCalibrator() if enable_calibration else None
        self.reliability_assessor = ReliabilityAssessor()
        self.patterns = {
            IntentType.CHAT: [
                {"patterns": [r"什么是", r"为什么", r"怎么样", r"如何理解", r"解释", r"介绍", r"聊天", r"你好", r"谢谢", r"再见"], "weight": 1.0},
            ],
            IntentType.TOOLS: [
                {"patterns": [r"读取", r"写入", r"修改", r"删除", r"列出", r"扫描", r"查找", r"调用工具", r"用工具", r"使用工具", r"执行", r"运行", r"命令", r"grep", r"find", r"ls", r"cat", r"bash", r"read", r"write", r"edit", r"list", r"search", r"查看文件", r"打开文件", r"看看", r"帮我看", r"分析文件", r"读文件", r"写文件", r"代码", r"目录", r"文件夹", r"\.py", r"\.js", r"\.java", r"\.md"], "weight": 1.0, "require_path": True},
            ],
            IntentType.PLAN: [
                {"patterns": [r"分析", r"重构", r"优化", r"设计", r"实现", r"开发", r"生成", r"创建项目", r"搭建", r"部署", r"复杂", r"多步骤", r"整体", r"全面"], "weight": 1.0, "complexity_indicators": [r"整体", r"全面", r"所有"]},
            ],
            IntentType.SKILLS: [
                {"patterns": [r"技能", r"知识库", r"参考文档", r"查阅", r"学习", r"pdf", r"文档处理", r"代码审查", r"web测试"], "weight": 1.0},
            ],
            IntentType.HYBRID: [
                {"patterns": [r"结合", r"同时", r"一边", r"既要", r"也要"], "weight": 1.0},
            ],
            IntentType.MULTI_AGENT: [
                {"patterns": [r"规划并执行", r"分析并生成", r"审查", r"评估质量", r"端到端", r"完整流程", r"全链路"], "weight": 1.0},
            ],
        }
        self.monitor = get_monitor_logger()

    def route(self, user_input: str, context: Dict = None) -> IntentResult:
        context = context or {}
        rule_result = self._rule_route(user_input, context)
        skill_result = self._skill_route(user_input, context)
        path_signals = self._detect_path_signals(user_input)

        max_rule_conf = max(rule_result["confidence"], skill_result["confidence"])

        if path_signals["has_path"]:
            result = self._build_result(IntentType.TOOLS, 0.95, "rule", path_signals["reason"], alternatives=[])
            self.monitor.info(
                f"意图路由: {result.intent.value} (置信度: {result.confidence:.2f}) "
                f"路由方式: {result.router_type}, 风险: {result.risk_level}"
            )
            return result

        if max_rule_conf >= self.threshold:
            best = rule_result if rule_result["confidence"] > skill_result["confidence"] else skill_result
            calibrated = self._calibrate(best["confidence"], "rule") if self.calibrator else best["confidence"]
            result = self._build_result(best["intent"], calibrated, "rule", best["reason"],
                               alternatives=self._get_alternatives([rule_result, skill_result]))
            self.monitor.info(
                f"意图路由: {result.intent.value} (置信度: {result.confidence:.2f}) "
                f"路由方式: {result.router_type}, 风险: {result.risk_level}"
            )
            return result

        if self.llm_forward_fn:
            llm_result = self._llm_route(user_input, context)
            reliability = self.reliability_assessor.assess(llm_result.get("raw_response", ""), llm_result)
            if reliability["recommendation"] == "fallback":
                result = self._build_result(rule_result["intent"],
                                   self._calibrate(rule_result["confidence"] * 0.9, "ensemble") if self.calibrator else
                                   rule_result["confidence"], "ensemble",
                                   f"LLM不可靠({reliability['reliability_score']:.2f})，回退规则: {rule_result['reason']}",
                                   alternatives=[(llm_result["intent"], llm_result["confidence"])], risk_level="high")
                self.monitor.info(
                    f"意图路由: {result.intent.value} (置信度: {result.confidence:.2f}) "
                    f"路由方式: {result.router_type}, 风险: {result.risk_level}"
                )
                return result
            ensemble_conf = self._fuse_confidence(rule_result["confidence"], llm_result["confidence"], reliability["reliability_score"])
            calibrated = self._calibrate(ensemble_conf, "ensemble") if self.calibrator else ensemble_conf
            result = self._build_result(llm_result["intent"], calibrated, "ensemble",
                               f"LLM: {llm_result['reason']} | 规则: {rule_result['reason']}",
                               alternatives=self._get_alternatives([rule_result, skill_result, llm_result]),
                               risk_level="low" if reliability["reliability_score"] > 0.8 else "medium")
            self.monitor.info(
                f"意图路由: {result.intent.value} (置信度: {result.confidence:.2f}) "
                f"路由方式: {result.router_type}, 风险: {result.risk_level}"
            )
            return result
        result = self._build_result(rule_result["intent"],
                           self._calibrate(rule_result["confidence"], "rule") if self.calibrator else rule_result[
                               "confidence"], "rule", rule_result["reason"],
                           alternatives=[(skill_result["intent"], skill_result["confidence"])])
        self.monitor.info(
            f"意图路由: {result.intent.value} (置信度: {result.confidence:.2f}) "
            f"路由方式: {result.router_type}, 风险: {result.risk_level}"
        )
        return result

    def _rule_route(self, user_input: str, context: Dict) -> Dict:
        scores = {intent: 0.0 for intent in IntentType}
        user_lower = user_input.lower()
        for intent, patterns in self.patterns.items():
            for pattern_def in patterns:
                pattern_matches = sum(1 for p in pattern_def["patterns"] if re.search(p, user_lower, re.I))
                if "negative_patterns" in pattern_def:
                    neg_matches = sum(1 for p in pattern_def["negative_patterns"] if re.search(p, user_lower, re.I))
                    if neg_matches > 0:
                        continue
                if pattern_def.get("require_path"):
                    if not self._has_file_path(user_input):
                        pattern_matches *= 0.5
                if "complexity_indicators" in pattern_def:
                    complex_matches = sum(1 for c in pattern_def["complexity_indicators"] if re.search(c, user_lower, re.I))
                    pattern_matches += complex_matches * 0.5
                scores[intent] += pattern_matches * pattern_def.get("weight", 1.0)
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        total_score = sum(scores.values()) or 1
        confidence = min(best_score / 3, 0.95)
        return {"intent": best_intent, "confidence": confidence, "reason": f"匹配{self._count_matches(best_intent, user_lower)}个模式", "all_scores": scores}

    def _skill_route(self, user_input: str, context: Dict) -> Dict:
        available_skills = context.get("available_skills", [])
        if not available_skills:
            return {"intent": IntentType.CHAT, "confidence": 0.3, "reason": "无可用技能"}
        user_lower = user_input.lower()
        matched_skills = []
        for skill in available_skills:
            skill_tags = skill.get("tags", [])
            skill_name = skill.get("name", "").lower()
            match_score = 0
            for tag in skill_tags:
                if tag.lower() in user_lower:
                    match_score += 2
            if skill_name in user_lower:
                match_score += 3
            if match_score > 0:
                matched_skills.append((skill, match_score))
        if matched_skills:
            matched_skills.sort(key=lambda x: x[1], reverse=True)
            best_match = matched_skills[0]
            if best_match[1] >= 3:
                return {"intent": IntentType.SKILLS, "confidence": min(best_match[1] / 5, 0.9), "reason": f"匹配技能: {best_match[0]['name']}", "matched_skills": [s[0]["id"] for s in matched_skills[:3]]}
        return {"intent": IntentType.CHAT, "confidence": 0.4, "reason": "无技能匹配"}

    def _llm_route(self, user_input: str, context: Dict) -> Dict:
        system_prompt = """你是意图识别专家。分析用户输入，判断意图类型。

可选意图:
- chat: 闲聊、知识问答、情感交流
- tools: 文件操作、命令执行、代码分析
- skills: 使用特定技能知识库
- plan: 复杂任务规划、多步骤执行
- hybrid: 结合多种模式

输出JSON格式:
{
    "mode": "tools",
    "confidence": 0.85,
    "reason": "用户要求读取文件",
    "key_indicators": ["读取", "文件"],
    "uncertainty": []
}"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"分析意图: {user_input}"}]
        try:
            response = self.llm_forward_fn(messages, system_prompt="")
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if not json_match:
                raise ValueError("无JSON输出")
            result = json.loads(json_match.group())
            intent_map = {"chat": IntentType.CHAT, "tools": IntentType.TOOLS, "skills": IntentType.SKILLS, "plan": IntentType.PLAN, "hybrid": IntentType.HYBRID}
            return {
                "intent": intent_map.get(result.get("mode", "chat"), IntentType.CHAT),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", "LLM分析"),
                "raw_response": response,
                "indicators": result.get("key_indicators", []),
                "uncertainty": result.get("uncertainty", [])
            }
        except Exception as e:
            return {"intent": IntentType.CHAT, "confidence": 0.3, "reason": f"LLM路由失败: {e}", "raw_response": "", "error": str(e)}

    # 文件操作动词（中英文），路径信号必须配合这些动词才能强制 TOOLS 模式
    _FILE_ACTION_VERBS = (
        "读取", "写入", "修改", "删除", "扫描", "查找", "列出", "创建", "编辑",
        "阅读", "查看", "打开", "分析", "生成", "写", "改",
        "read", "write", "edit", "list", "scan", "find", "create", "delete", "open",
        "grep", "bash", "run", "execute",
    )

    def _detect_path_signals(self, user_input: str) -> Dict:
        """检测用户输入中的文件路径信号。

        改进：仅当输入中**同时存在文件操作动词**时才判定为 TOOLS 模式，
        避免纯知识问答中出现路径关键词（如文件名作为举例）时被误判。
        """
        path_patterns = [
            r'[\'"]?([a-zA-Z]:\\[^\'"]+)[\'"]?',
            r'[\'"]?(/[^\'"]+)[\'"]?',
            r'[\'"]?(\./[^\'"]+)[\'"]?',
            r'[\'"]?(\w+\.(py|js|md|txt|json|yaml|yml|sh|toml|cfg|ini))[\'"]?',
        ]
        matches = []
        for pattern in path_patterns:
            for match in re.finditer(pattern, user_input):
                matches.append(match.group(1))

        if not matches:
            return {"has_path": False}

        # 必须同时检测到文件操作动词，才能确定是工具任务
        user_lower = user_input.lower()
        has_action_verb = any(verb in user_lower for verb in self._FILE_ACTION_VERBS)
        if has_action_verb:
            return {"has_path": True, "paths": matches, "reason": f"检测到路径: {', '.join(matches[:2])}"}

        # 有路径但没有操作动词——可能只是作为背景提及，降级为普通规则判断
        return {"has_path": False, "paths": matches, "reason": "路径存在但无操作动词，不强制 TOOLS 模式"}

    def _has_file_path(self, text: str) -> bool:
        return bool(re.search(r'[./\\]\w+\.\w+', text))

    def _count_matches(self, intent: IntentType, text: str) -> int:
        patterns = self.patterns.get(intent, [])
        count = 0
        for p in patterns:
            for pattern in p["patterns"]:
                if re.search(pattern, text, re.I):
                    count += 1
        return count

    def _fuse_confidence(self, rule_conf: float, llm_conf: float, reliability: float, weights: List[float] = None) -> float:
        weights = weights or [0.4, 0.6]
        llm_weight = weights[1] * reliability
        rule_weight = weights[0]
        total = rule_weight + llm_weight
        normalized_weights = [rule_weight / total, llm_weight / total]
        fused = normalized_weights[0] * rule_conf + normalized_weights[1] * llm_conf
        uncertainty = abs(rule_conf - llm_conf)
        penalty = uncertainty * 0.1
        return min(fused - penalty, 0.99)

    def _calibrate(self, raw_conf: float, router_type: str) -> float:
        if self.calibrator:
            return self.calibrator.calibrate(raw_conf, router_type)
        return raw_conf

    def _get_alternatives(self, results: List[Dict]) -> List[Tuple[IntentType, float]]:
        alternatives = []
        seen = set()
        for r in results:
            intent = r["intent"]
            if intent not in seen:
                alternatives.append((intent, r["confidence"]))
                seen.add(intent)
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[1:3] if len(alternatives) > 1 else []

    def _build_result(self, intent: IntentType, confidence: float, router_type: str, reasoning: str, alternatives: List[Tuple[IntentType, float]], risk_level: str = "low") -> IntentResult:
        suggested_params = self._suggest_params(intent)
        return IntentResult(
            intent=intent,
            confidence=confidence,
            raw_confidence=confidence,
            calibrated_confidence=confidence,
            reasoning=reasoning,
            router_type=router_type,
            alternatives=alternatives,
            risk_level=risk_level,
            suggested_params=suggested_params
        )

    def _suggest_params(self, intent: IntentType) -> Dict[str, Any]:
        params = {
            IntentType.CHAT: {"temperature": 0.8, "max_tokens": 1024, "tools_enabled": False},
            IntentType.TOOLS: {"temperature": 0.3, "max_tokens": 2048, "tools_enabled": True, "plan_mode": False},
            IntentType.PLAN: {"temperature": 0.4, "max_tokens": 4096, "tools_enabled": True, "plan_mode": True, "max_iterations": 20},
            IntentType.SKILLS: {"temperature": 0.5, "max_tokens": 2048, "tools_enabled": True, "skill_mode": True},
        }
        return params.get(intent, params[IntentType.CHAT])