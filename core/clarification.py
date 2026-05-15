#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式澄清模块
当模型输出包含 NEEDS_CONTEXT / BLOCKED 或检测到提问意图时，
主动向用户追问缺失信息，积累澄清结果后生成最终完整回答。
"""

import re
from typing import Dict, List, Optional, Any

from core.monitor_logger import get_monitor_logger


class ClarificationManager:
    """跨轮次澄清状态管理器，挂载在 SessionContext 或 runtime_context 上使用"""

    MAX_CLARIFY_ROUNDS = 2          # 最多连续澄清轮数
    CLARIFY_FLAG = "_in_clarification"

    def __init__(self):
        self.monitor = get_monitor_logger()

    # ------------------------------------------------------------------
    # 状态存取（统一存放在 runtime_context 中，避免污染 SessionContext）
    # ------------------------------------------------------------------
    @staticmethod
    def is_clarification_active(context: Dict[str, Any]) -> bool:
        return bool(context.get(ClarificationManager.CLARIFY_FLAG))

    @staticmethod
    def set_clarification_active(context: Dict[str, Any], active: bool):
        context[ClarificationManager.CLARIFY_FLAG] = active
        if not active:
            context.pop("clarify_round", None)
            context.pop("pending_clarify_question", None)
        else:
            context.setdefault("clarify_round", 0)

    @staticmethod
    def get_clarify_round(context: Dict[str, Any]) -> int:
        return context.get("clarify_round", 0)

    @staticmethod
    def inc_clarify_round(context: Dict[str, Any]):
        context["clarify_round"] = context.get("clarify_round", 0) + 1

    @staticmethod
    def add_clarified_fact(context: Dict[str, Any], fact: str):
        facts = context.setdefault("clarified_facts", [])
        facts.append(fact)

    @staticmethod
    def get_clarified_facts(context: Dict[str, Any]) -> List[str]:
        return context.get("clarified_facts", [])

    # ------------------------------------------------------------------
    # 判断是否需要触发澄清
    # ------------------------------------------------------------------
    def should_trigger_clarification(
        self,
        response: str,
        context: Dict[str, Any],
        session_context: Optional[Any] = None,
    ) -> bool:
        """
        根据模型回答判断是否应该暂停流程，向用户发起澄清提问。
        触发条件（满足任一即可）：
          1. 回答中包含 NEEDS_CONTEXT / BLOCKED 状态标签
          2. 回答明确向用户提问（问句 + 选项）
          3. 在工具模式下，工具调用全部失败且回答过于简短
        """
        if self.is_clarification_active(context):
            # 已经处于澄清流程中，不再重复触发
            return False

        # 条件1：检测完成状态协议
        status_protocol = re.search(
            r"STATUS:\s*(NEEDS_CONTEXT|BLOCKED)", response, re.IGNORECASE
        )
        if status_protocol:
            self.monitor.info(f"澄清触发: 检测到 {status_protocol.group(1)}")
            return True

        # 条件2：明确的提问意图（利用已有 AskUserQuestionMiddleware 的逻辑）
        question_signals = [
            "请问", "是否", "需要确认", "请确认", "您是否", "是否需要",
            "需要我", "要不要", "是否要", "请选择", "可以告诉我",
            "请提供", "需要更多信息", "需要澄清", "还不清楚",
        ]
        if any(signal in response for signal in question_signals):
            self.monitor.info("澄清触发: 模型输出包含提问信号")
            return True

        # 条件3：回答过短且似乎放弃（tools模式）
        run_mode = context.get("run_mode", "")
        if run_mode in ("tools", "hybrid"):
            # 回答很短，且没有工具调用结果的引用
            if len(response.strip()) < 80 and not any(
                kw in response for kw in ("已执行", "结果", "成功", "失败", "✅", "❌")
            ):
                self.monitor.info("澄清触发: tools 模式下回答过短且无实质内容")
                return True

        return False

    # ===== 新增：统一澄清入口 =====
    def trigger_clarification(
        self,
        runtime_context: Dict[str, Any],
        missing_info: List[str],
        source: str = "system"
    ) -> str:
        """
        统一澄清卡片生成入口，自动管理轮次、状态，并返回格式化卡片。
        所有需要澄清的路径都应调用此方法，而不是手动拼接 STATUS 文本。
        """
        if self.get_clarify_round(runtime_context) >= self.MAX_CLARIFY_ROUNDS:
            self.monitor.warning("已达最大澄清轮次，不再追问")
            facts = self.get_clarified_facts(runtime_context)
            facts_text = "；".join(facts) if facts else "无"
            return (
                "⚠️ 已多次向您确认信息，但仍不足以完成任务。\n"
                f"已收集的事实：{facts_text}\n\n"
                "系统将基于现有信息尝试执行，如有偏差请重新描述需求。"
            )

        self.set_clarification_active(runtime_context, True)
        self.inc_clarify_round(runtime_context)
        current_round = self.get_clarify_round(runtime_context)

        runtime_context.setdefault("_pending_clarify_items", [])
        runtime_context["_pending_clarify_items"] = missing_info[:]

        return self.ask_clarification(runtime_context, missing_info)
        # ==========================================

    # ------------------------------------------------------------------
    # 生成澄清状态时的系统指令（注入到下一轮对话中）
    # ------------------------------------------------------------------
    def build_clarification_context_message(
        self, context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        当用户回答了之前的澄清问题后，构建一条系统消息，
        包含已澄清的事实，并要求模型基于这些事实给出最终完整答案。
        """
        facts = self.get_clarified_facts(context)
        if not facts:
            return {
                "role": "system",
                "content": (
                    "用户针对你的问题提供了额外信息，请结合对话历史生成完整回答。"
                ),
            }

        facts_text = "\n".join(f"  - {f}" for f in facts)
        return {
            "role": "system",
            "content": (
                "【澄清状态 - 用户已补充以下信息】\n"
                f"{facts_text}\n\n"
                "请基于以上事实以及对话历史，直接给出**完整、准确**的最终回答。\n"
                "**不要再次向用户提问！** 如果信息仍然不足，请根据已有信息给出最佳推断，"
                "并明确指出哪些部分是推测的。"
            ),
        }

    def ask_clarification(self, context: Dict[str, Any], missing_info: List[str]) -> str:
        """
        生成结构化的澄清卡片，提示用户提供缺失信息。
        Args:
            context: runtime_context
            missing_info: 描述缺失信息的字符串列表，如 ["文件路径", "操作类型"]
        Returns:
            格式化的 Markdown 字符串
        """
        lines = [
            "🔍 **需要您确认以下信息**",
            "",
            "为了准确完成任务，请补充以下内容："
        ]
        for i, item in enumerate(missing_info, 1):
            lines.append(f"{i}. {item}")

        lines.append("")
        lines.append("💡 您可以直接回复，或提供更多上下文。")
        return "\n".join(lines)


    def process_user_clarification(
        self,
        user_input: str,
        context: Dict[str, Any],
        vector_memory: Any = None
    ) -> bool:
        """
        处理用户针对澄清问题的回复。
        将用户补充的信息存入事实账本，并写入向量记忆（高重要性）。
        Returns:
            bool: 是否成功处理
        """
        if not user_input or not isinstance(user_input, str):
            return False

        # 1. 记录到澄清事实列表
        facts = context.setdefault("clarified_facts", [])
        facts.append(user_input.strip())

        # 2. 同时追加到 confirmed_facts（以便后续步骤自动引用）
        confirmed = context.setdefault("confirmed_facts", [])
        confirmed.append({"kind": "user_clarification", "content": user_input.strip()})

        # 3. 存入向量记忆，标记为高重要性（如果有向量记忆对象）
        if vector_memory and hasattr(vector_memory, "add"):
            try:
                vector_memory.add(
                    content=f"用户澄清补充: {user_input}",
                    metadata={
                        "type": "user_clarification",
                        "importance": 0.95,
                        "source": "clarification_manager"
                    },
                    importance=0.95,
                    auto_score=False,
                    skip_duplicate=True
                )
                self.monitor.info("澄清信息已写入向量记忆")
            except Exception as e:
                self.monitor.warning(f"写入向量记忆失败: {e}")

        # 4. 更新澄清状态
        self.set_clarification_active(context, False)
        return True

    # ------------------------------------------------------------------
    # 主入口：处理一轮对话的澄清逻辑
    # ------------------------------------------------------------------
    def handle_clarification_flow(
        self,
        user_input: str,
        context: Dict[str, Any],
        previous_response: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        返回值:
          - 如果需要澄清且不应继续执行工具/Agent，返回 dict:
                {"action": "clarify", "message": <澄清问题>}
          - 如果澄清已结束，返回 dict:
                {"action": "finalize", "clarified_facts": [...]}
          - 如果不需要特殊处理，返回 None，继续正常流程
        """
        # 1. 如果当前轮次是用户正在回答一个澄清问题
        if self.is_clarification_active(context):
            # 记录用户的回答
            self.add_clarified_fact(context, user_input)
            current_round = self.get_clarify_round(context)
            self.monitor.debug(f"澄清轮次 {current_round}，已累积事实: {self.get_clarified_facts(context)}")
            # 结束澄清状态，允许模型生成最终答案
            self.set_clarification_active(context, False)
            return {
                "action": "finalize",
                "clarified_facts": self.get_clarified_facts(context),
            }

        # 2. 如果不在澄清中，但模型上一轮的回答触发了澄清
        if previous_response and self.should_trigger_clarification(
            previous_response, context
        ):
            # 检查是否超过最大轮次
            if self.get_clarify_round(context) >= self.MAX_CLARIFY_ROUNDS:
                self.monitor.warning("已达最大澄清轮次，强制结束")
                self.set_clarification_active(context, False)
                return None

            self.set_clarification_active(context, True)
            self.inc_clarify_round(context)
            # 把模型的澄清问题直接作为一条用户消息返回（让前端展示）
            return {
                "action": "clarify",
                "message": previous_response.strip(),
            }

        return None

