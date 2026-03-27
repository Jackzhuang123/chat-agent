# -*- coding: utf-8 -*-
"""模式路由器 - 自动识别用户意图并切换模式

支持两级路由：
1. 规则路由（快速）：正则 + 关键词匹配，适合明确意图
2. LLM 路由（精确）：调用语言模型进行语义分析，适合模糊意图
   - 规则路由置信度 < 阈值时，自动升级为 LLM 路由
   - LLM 路由结果不会写入 session_log（由调用方控制）
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple


class ModeRouter:
    """
    智能模式路由器 - 自动识别用户意图并推荐模式

    支持的模式：
    - chat: 纯对话模式（闲聊、问答）
    - tools: 工具模式（文件操作、代码分析、命令执行）
    - skills: 技能模式（使用外部知识库）
    - hybrid: 混合模式（技能 + 工具）
    - plan: 计划模式（复杂任务分解）
    - multi_agent: 多Agent模式（需要规划-执行-审查）
    - streaming: 流式模式（实时展示进度）
    """

    def __init__(self, llm_forward_fn: Optional[Callable] = None, llm_confidence_threshold: float = 0.70):
        """
        Args:
            llm_forward_fn: 可选的 LLM 调用函数，签名 fn(messages, system_prompt="") -> str。
                            若提供，当规则路由置信度低于阈值时自动调用 LLM 进行语义分析。
            llm_confidence_threshold: 规则路由置信度低于此值时触发 LLM 路由（默认 0.70）。
        """
        self.llm_forward_fn = llm_forward_fn
        self.llm_confidence_threshold = llm_confidence_threshold

        # 模式识别规则
        self.mode_patterns = {
            "chat": {
                "keywords": ["什么是", "为什么", "怎么样", "如何理解", "解释", "介绍", "聊天", "你好", "谢谢", "再见"],
                "priority": 0.3,
                "description": "纯对话模式 - 直接回答问题，不调用工具"
            },
            "tools": {
                "keywords": [
                    "读取", "写入", "修改", "删除", "列出", "扫描", "查找", "调用工具", "用工具", "使用工具",
                    "执行", "运行", "命令", "grep", "find", "ls", "cat", "bash",
                    "read", "write", "edit", "list", "search",
                    "查看文件", "打开文件", "看看", "帮我看", "分析文件", "读文件", "写文件",
                    "代码", "目录", "文件夹", ".py", ".js", ".java", ".md",
                ],
                "priority": 0.9,
                "description": "工具模式 - 通过工具收集事实"
            },
            "skills": {
                "keywords": [
                    "技能", "知识库", "参考文档", "查阅", "学习",
                    "pdf", "文档处理", "代码审查", "web测试"
                ],
                "priority": 0.75,
                "description": "技能模式 - 使用外部知识库"
            },
            "hybrid": {
                "keywords": [
                    "结合", "同时", "一边", "既要", "也要"
                ],
                "priority": 0.8,
                "description": "混合模式 - 技能 + 工具"
            },
            "plan": {
                "keywords": [
                    "分析", "重构", "优化", "设计", "实现", "开发",
                    "生成", "创建项目", "搭建", "部署",
                    "复杂", "多步骤", "整体", "全面"
                ],
                "priority": 0.7,
                "description": "计划模式 - 分解复杂任务"
            },
            "multi_agent": {
                "keywords": [
                    "规划并执行", "分析并生成", "审查", "评估质量",
                    "端到端", "完整流程", "全链路"
                ],
                "priority": 0.85,
                "description": "多Agent模式 - 规划-执行-审查"
            },
            "streaming": {
                "keywords": [
                    "实时", "流式", "进度", "展示过程",
                    "看到执行", "监控"
                ],
                "priority": 0.6,
                "description": "流式模式 - 实时展示进度"
            }
        }

        # 任务复杂度识别
        self.complexity_patterns = {
            "simple": {
                "keywords": ["读取", "查看", "列出", "打开", "显示"],
                "max_steps": 2
            },
            "medium": {
                "keywords": ["修改", "编辑", "查找", "分析"],
                "max_steps": 4
            },
            "complex": {
                "keywords": ["重构", "优化", "设计", "实现", "生成"],
                "max_steps": 10
            }
        }

    # 文件/目录路径强信号正则（绝对路径、相对路径 ./xxx、Windows 盘符）
    _PATH_PATTERN = re.compile(
        r'(?:'
        r'/[A-Za-z0-9_./-]+[A-Za-z0-9_.-]'   # Unix 绝对路径 /xxx/yyy
        r'|(?:\./|\.\./)[\w./-]+'              # 相对路径 ./xxx 或 ../xxx
        r'|[A-Z]:\\[\w\\.-]+'                  # Windows 盘符 C:\xxx
        r')'
    )

    # LLM 意图路由的系统提示
    _LLM_ROUTER_SYSTEM = """
    你是一个意图路由专家。你的任务是分析用户的自然语言输入，判断其背后的真实意图。
    请仔细分析，用户可能使用口语、省略或模糊的表达。

    **模式定义：**
    - chat: 闲聊、知识问答、情感交流。不需要操作文件或执行代码。
    - tools: 需要读取/写入文件、列出目录、执行命令、分析现有代码、查看日志。
    - plan: 需要设计架构、生成新项目、重构代码、解决复杂Bug。这通常涉及多步骤思考。
    - skills: 需要查阅特定的内部知识库或文档。

    **分析指南：**
    1. 如果用户提到“看看”、“读读”、“文件”、“日志”、“代码”或包含路径（如 .py, /src），请路由到 `tools`。
    2. 如果用户说“帮我写”、“设计一个”、“怎么实现”、“重构”，请路由到 `plan`。
    3. 如果用户问“什么是”、“为什么”、“解释一下”，请路由到 `chat`。

    **输出要求：**
    严格返回 JSON，不要输出任何其他内容。
    {
        "mode": "tools",
        "confidence": 0.9,
        "reason": "用户要求查看文件内容"
    }
    """

    def detect_mode(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        检测用户意图并推荐模式（两级路由）

        返回格式：
        {
            "recommended_mode": "tools",
            "confidence": 0.85,
            "alternatives": [{"mode": "plan", "confidence": 0.65}],
            "complexity": "medium",
            "reasoning": "检测到文件操作关键词，推荐工具模式",
            "router": "rule" | "llm"
        }
        """
        # 先用规则路由
        rule_result = self._rule_based_detect(user_input, context)

        # 若规则路由置信度足够高，直接返回
        if rule_result["confidence"] >= self.llm_confidence_threshold:
            rule_result["router"] = "rule"
            return rule_result

        # 规则路由置信度低 → 尝试 LLM 路由
        if self.llm_forward_fn is not None:
            llm_result = self._llm_based_detect(user_input, context)
            if llm_result is not None:
                # LLM 路由成功，合并字段
                llm_result["router"] = "llm"
                llm_result.setdefault("alternatives", rule_result.get("alternatives", []))
                llm_result.setdefault("complexity", rule_result.get("complexity", "simple"))
                return llm_result

        # 兜底：返回规则路由结果
        rule_result["router"] = "rule"
        return rule_result

    def _rule_based_detect(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """纯规则路由（原有逻辑）"""
        from .tool_enforcement_middleware import DirectCommandDetector

        user_input_lower = user_input.lower()

        # 0. 直接命令检测（最高优先级）
        direct_cmd = DirectCommandDetector.detect(user_input)
        if direct_cmd["is_direct_command"]:
            return {
                "recommended_mode": "tools",
                "confidence": direct_cmd["confidence"],
                "alternatives": [],
                "complexity": "simple",
                "reasoning": direct_cmd["reason"],
                "suggested_tool": direct_cmd.get("tool_name")
            }

        # 1. 路径强信号先行判断：输入包含文件/目录路径时直接路由为 tools 模式
        if self._PATH_PATTERN.search(user_input):
            complexity = self._detect_complexity(user_input_lower)
            return {
                "recommended_mode": "tools",
                "confidence": 0.92,
                "alternatives": [],
                "complexity": complexity,
                "reasoning": "检测到文件/目录路径，直接路由为工具模式"
            }

        # 1. 计算各模式的匹配分数
        mode_scores = []
        for mode, pattern in self.mode_patterns.items():
            keywords = pattern["keywords"]
            priority = pattern["priority"]

            match_count = sum(1 for kw in keywords if kw in user_input_lower)

            if match_count > 0:
                score = match_count * priority
                mode_scores.append({
                    "mode": mode,
                    "confidence": min(score / 3.0, 1.0),
                    "match_count": match_count,
                    "description": pattern["description"]
                })

        # 2. 排序并选择最佳模式
        mode_scores.sort(key=lambda x: x["confidence"], reverse=True)

        if not mode_scores:
            return {
                "recommended_mode": "chat",
                "confidence": 0.3,  # 降低默认置信度，更容易触发 LLM 路由
                "alternatives": [],
                "complexity": "simple",
                "reasoning": "未检测到特定意图，默认对话模式"
            }

        best = mode_scores[0]
        alternatives = mode_scores[1:3] if len(mode_scores) > 1 else []

        # 3. 检测任务复杂度
        complexity = self._detect_complexity(user_input_lower)

        # 4. 自动升级模式
        recommended_mode = best["mode"]
        reasoning = f"检测到 {best['match_count']} 个关键词，推荐{best['description']}"

        # 复杂任务自动启用计划模式
        if complexity == "complex" and recommended_mode == "tools":
            recommended_mode = "plan"
            reasoning += "；任务复杂度高，自动启用计划模式"

        # 多步骤任务自动启用多Agent
        if "并" in user_input and best["confidence"] > 0.7:
            recommended_mode = "multi_agent"
            reasoning += "；检测到多步骤任务，推荐多Agent模式"

        return {
            "recommended_mode": recommended_mode,
            "confidence": best["confidence"],
            "alternatives": alternatives,
            "complexity": complexity,
            "reasoning": reasoning
        }

    def _llm_based_detect(self, user_input: str, context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 进行语义意图分析。

        调用轻量级 prompt，让模型输出 JSON 格式的路由结果。
        若 LLM 调用失败或解析失败，返回 None（调用方降级到规则路由）。
        """
        # 构建对话历史（可选：注入最近的对话上下文）
        messages = []
        if context and context.get("history"):
            history = context["history"]
            # 只取最近 2 轮，避免上下文过长
            for pair in history[-2:]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    u, a = pair
                    if u:
                        messages.append({"role": "user", "content": str(u)})
                    if a:
                        messages.append({"role": "assistant", "content": str(a)[:100]})

        messages.append({"role": "user", "content": f"用户输入：{user_input}\n\n请判断应使用哪种模式，输出 JSON："})

        try:
            raw = self.llm_forward_fn(messages, system_prompt=self._LLM_ROUTER_SYSTEM)
            if not raw or not raw.strip():
                return None

            # 提取 JSON（模型可能在 JSON 前后输出额外文字）
            # 这个正则能匹配包含换行、引号的复杂 JSON
            json_match = re.search(r'\{[\s\S]*"mode"[^\}]*"reason"[^\}]*\}', raw, re.DOTALL)
            if not json_match:
                # 如果精确匹配失败，尝试找第一个 { ... } 包裹的内容
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            llm_mode = data.get("mode", "").lower()
            llm_conf = float(data.get("confidence", 0.7))
            llm_reason = data.get("reason", "LLM 语义分析")


            # 校验 mode 合法性
            valid_modes = {"chat", "tools", "skills", "plan", "hybrid", "multi_agent"}
            if llm_mode not in valid_modes:
                return None

            complexity = self._detect_complexity(user_input.lower())
            return {
                "recommended_mode": llm_mode,
                "confidence": min(llm_conf, 1.0),
                "complexity": complexity,
                "reasoning": f"[LLM] {llm_reason}",
            }

        except Exception:
            return None

    def set_llm_forward(self, llm_forward_fn: Callable) -> None:
        """运行时注入 LLM 调用函数（延迟绑定，避免循环依赖）。"""
        self.llm_forward_fn = llm_forward_fn

    def _detect_complexity(self, user_input_lower: str) -> str:
        """检测任务复杂度"""
        for complexity, pattern in self.complexity_patterns.items():
            keywords = pattern["keywords"]
            match_count = sum(1 for kw in keywords if kw in user_input_lower)
            if match_count > 0:
                return complexity

        if len(user_input_lower) > 100:
            return "complex"
        elif len(user_input_lower) > 30:
            return "medium"
        else:
            return "simple"

    def auto_switch_mode(
        self,
        user_input: str,
        current_mode: str = "chat",
        context: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        自动切换模式

        返回：(new_mode, reason)
        """
        detection = self.detect_mode(user_input, context)
        recommended = detection["recommended_mode"]
        confidence = detection["confidence"]

        if confidence > 0.6 and recommended != current_mode:
            return recommended, detection["reasoning"]

        return current_mode, f"保持当前模式（推荐置信度: {confidence:.2f}）"

    def suggest_parameters(self, mode: str, user_input: str) -> Dict[str, Any]:
        """根据模式推荐参数"""
        params = {
            "run_mode": mode,
            "plan_mode": False,
            "enable_streaming": False,
            "max_iterations": 50
        }

        if mode == "plan":
            params["plan_mode"] = True
            params["max_iterations"] = 55

        if mode == "multi_agent":
            params["use_multi_agent"] = True
            params["max_iterations"] = 50

        if mode == "streaming":
            params["enable_streaming"] = True

        if mode == "tools":
            if "读取" in user_input or "查看" in user_input:
                params["max_iterations"] = 50
            else:
                params["max_iterations"] = 50

        return params


class AutoModeMiddleware:
    """
    自动模式中间件 - 在执行前自动识别并设置模式
    """

    def __init__(self, router: Optional[ModeRouter] = None):
        self.router = router or ModeRouter()

    def process_before_run(
        self,
        user_input: str,
        runtime_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """在运行前处理"""
        if runtime_context.get("run_mode"):
            return runtime_context

        detection = self.router.detect_mode(user_input, runtime_context)

        runtime_context["run_mode"] = detection["recommended_mode"]
        runtime_context["mode_detection"] = detection

        suggested_params = self.router.suggest_parameters(
            detection["recommended_mode"],
            user_input
        )
        runtime_context.update(suggested_params)

        return runtime_context


def create_auto_mode_framework(framework_class, model_forward_fn, **kwargs):
    """
    创建自动模式框架

    使用示例：
    framework = create_auto_mode_framework(
        QwenAgentFramework,
        model_forward_fn=model_forward,
        enable_parallel=True
    )

    result = framework.run("读取core/agent_framework.py")
    # 自动检测为 tools 模式
    """
    from functools import wraps

    router = ModeRouter()
    auto_middleware = AutoModeMiddleware(router)

    framework = framework_class(model_forward_fn=model_forward_fn, **kwargs)

    original_run = framework.run

    @wraps(original_run)
    def auto_run(user_input: str, history=None, runtime_context=None):
        if runtime_context is None:
            runtime_context = {}

        runtime_context = auto_middleware.process_before_run(user_input, runtime_context)

        if runtime_context.get("mode_detection"):
            detection = runtime_context["mode_detection"]
            print(f"🔍 自动检测模式: {detection['recommended_mode']} "
                  f"(置信度: {detection['confidence']:.2f}) [{detection.get('router', 'rule')}]")
            print(f"💡 原因: {detection['reasoning']}")

        return original_run(user_input, history, runtime_context)

    framework.run = auto_run
    framework.router = router

    return framework
