#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
意图路由模块 - 自动分析用户意图，决定 run_mode 和技能加载

包含两个路由器：
  - IntentRouter    : 基于规则/关键词的快速意图路由（零 LLM 开销）
  - AIIntentRouter  : AI 驱动的语义意图分析 + 合并任务规划（单次 LLM 调用）

设计原则：
  规则路由作为"零延迟快速路径"，AI 路由作为"语义理解增强层"。
  有明确文件路径/操作关键词时直接走规则路由，否则调用 AI 补充判断。
"""

import json
import re
from typing import Any, Dict, List, Optional


class IntentRouter:
    """
    意图路由器：自动分析用户输入，决定运行模式和需要加载的技能。

    参考 mini Claude Code v4 的 Skills 机制：
    - 无需用户手动勾选工具模式/技能模式
    - 基于规则+关键词快速判断意图（无 LLM 开销）
    - 技能以"追加消息"方式注入上下文（缓存友好）
    """

    # ---- 工具模式触发关键词（需具体，避免过宽匹配） ----
    FILE_OP_KEYWORDS = {
        "读取", "查看文件", "打开文件", "阅读文件", "看看文件",
        "写入文件", "写入", "创建文件", "新建文件", "保存到文件",
        "修改文件", "编辑文件", "更新文件", "追加到文件",
        "列目录", "列出目录", "列文件", "目录结构", "文件列表",
        "read_file", "write_file", "list_dir",
        "分析文件", "文件内容", "查看目录",
    }
    FILE_PATH_PATTERN = re.compile(
        r"(?:/[^\s，。；：\u4e00-\u9fff]{2,}"
        r"|(?:\./|\.\./)[\w./\-]+"
        r"|[\w\-]+\.(?:py|md|txt|json|yaml|yml|sh|js|ts|html|css|csv|log|conf|ini|toml|pdf))"
    )
    BASH_KEYWORDS = {
        "执行", "运行", "命令", "bash", "shell", "终端", "execute", "run command",
    }

    # ---- Skills 触发规则 ----
    SKILL_TRIGGERS: List[Dict[str, Any]] = [
        {
            "skill_id": "pdf",
            "keywords": {"pdf", ".pdf", "pdf文件", "pdf内容", "pdf处理", "提取pdf"},
        },
        {
            # python-dev 和 python_dev 是同一技能的两种目录命名，合并关键词统一匹配
            "skill_id": "python-dev",
            "keywords": {"python", "python脚本", "python代码", "python开发", "pip", "虚拟环境", "venv"},
        },
        {
            "skill_id": "code-review",
            "keywords": {"代码审查", "code review", "review", "代码质量", "安全漏洞", "审计", "检查代码", "代码检查"},
        },
        {
            "skill_id": "web-qa",
            "keywords": {
                "web测试", "qa测试", "页面测试", "浏览器测试", "自动化测试",
                "playwright", "selenium", "端到端测试", "e2e", "用户流程测试",
                "部署检查", "生产验证", "响应式测试", "表单测试", "截图",
                "web qa", "网页测试", "前端测试",
            },
        },
    ]

    # ---- 追问/继续意图关键词（对话接续信号） ----
    FOLLOWUP_KEYWORDS = {
        "输出", "展示", "显示", "打印", "继续", "接着", "然后", "再",
        "详细", "具体", "完整", "全部", "更多", "说明", "解释",
        "show", "print", "continue", "more", "detail", "explain",
        "给我看", "给我", "告诉我", "怎么", "如何",
    }

    def __init__(self, available_skill_ids: Optional[List[str]] = None):
        """
        Args:
            available_skill_ids: 当前系统中实际存在的 skill id 列表（用于过滤无效 skill）
        """
        self.available_skill_ids: set = set(available_skill_ids or [])

    def analyze(
        self,
        user_message: str,
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析用户消息，返回路由决策。支持历史上下文感知。

        Args:
            user_message: 当前用户消息
            uploaded_files: 已上传文件列表
            chat_history: 对话历史 [(user_msg, bot_msg), ...] 用于上下文感知

        Returns:
            {
                "run_mode": "chat" | "tools" | "skills" | "hybrid",
                "skill_ids": [...],   # 需要加载的技能列表
                "reasons": [...],     # 决策原因（调试日志用）
                "needs_tools": bool,
                "needs_skills": bool,
                "inherited_context": str,  # 追问继承时携带的上一轮摘要
            }
        """
        msg = user_message or ""
        msg_lower = msg.lower()
        reasons: List[str] = []

        # ---- 0. 历史上下文感知：检测是否是对近期工具操作的追问 ----
        inherited_tools = False
        _inherited_assistant_summary = ""
        if chat_history:
            recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history

            recent_file_op_user_msg = ""
            any_recent_file_op = False
            last_user_msg = ""
            last_assistant_msg = ""

            for h in reversed(recent_history):
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    u_msg = str(h[0] or "")
                    a_msg = str(h[1] or "")
                    if not last_user_msg:
                        last_user_msg = u_msg
                        last_assistant_msg = a_msg
                    _had_op = (
                        any(kw in u_msg for kw in self.FILE_OP_KEYWORDS)
                        or bool(self.FILE_PATH_PATTERN.search(u_msg))
                        or any(kw in u_msg for kw in {"read_file", "write_file", "list_dir"})
                    )
                    if _had_op and not any_recent_file_op:
                        any_recent_file_op = True
                        recent_file_op_user_msg = u_msg
                elif isinstance(h, dict):
                    role = h.get("role", "")
                    content = str(h.get("content", ""))
                    if role == "assistant" and not last_assistant_msg:
                        last_assistant_msg = content
                    elif role == "user":
                        if not last_user_msg:
                            last_user_msg = content
                        _had_op = (
                            any(kw in content for kw in self.FILE_OP_KEYWORDS)
                            or bool(self.FILE_PATH_PATTERN.search(content))
                        )
                        if _had_op and not any_recent_file_op:
                            any_recent_file_op = True
                            recent_file_op_user_msg = content

            is_followup = (
                len(msg.strip()) < 30
                or any(kw in msg for kw in self.FOLLOWUP_KEYWORDS)
            )
            reset_kws = {"你好", "hello", "你是谁", "介绍", "新话题", "换个话题"}
            if any_recent_file_op and is_followup and not any(kw in msg_lower for kw in reset_kws):
                inherited_tools = True
                reasons.append(f"继承上下文工具模式（近期操作: {recent_file_op_user_msg[:30]}...）")
                _inherited_assistant_summary = last_assistant_msg[:500] if last_assistant_msg else ""

        # ---- 1. 判断是否需要工具 ----
        has_path = bool(self.FILE_PATH_PATTERN.search(msg))
        has_file_op = any(kw in msg for kw in self.FILE_OP_KEYWORDS)
        has_bash_op = any(kw in msg_lower for kw in self.BASH_KEYWORDS)
        has_uploaded = bool(uploaded_files)

        needs_tools = inherited_tools
        if has_path and has_file_op:
            needs_tools = True
            reasons.append("检测到文件路径+操作意图")
        elif has_path:
            needs_tools = True
            reasons.append("检测到文件路径")
        elif has_file_op:
            needs_tools = True
            reasons.append("检测到文件操作关键词")
        elif has_bash_op:
            needs_tools = True
            reasons.append("检测到命令执行意图")
        elif has_uploaded:
            needs_tools = True
            reasons.append("存在已上传文件，启用工具模式")

        # ---- 2. 判断需要加载哪些 Skills ----
        skill_ids: List[str] = []
        for trigger in self.SKILL_TRIGGERS:
            skill_id = trigger["skill_id"]
            if self.available_skill_ids and skill_id not in self.available_skill_ids:
                continue
            keywords = trigger.get("keywords", set())
            matched = [kw for kw in keywords if kw.lower() in msg_lower or kw in msg]
            if matched:
                if skill_id not in skill_ids:
                    skill_ids.append(skill_id)
                reasons.append(f"技能 [{skill_id}] 命中关键词: {matched[:2]}")

        needs_skills = bool(skill_ids)

        # ---- 3. 决定 run_mode ----
        if needs_tools and needs_skills:
            run_mode = "hybrid"
        elif needs_tools:
            run_mode = "tools"
        elif needs_skills:
            run_mode = "skills"
        else:
            run_mode = "chat"

        return {
            "run_mode": run_mode,
            "skill_ids": skill_ids,
            "reasons": reasons,
            "needs_tools": needs_tools,
            "needs_skills": needs_skills,
            "inherited_context": _inherited_assistant_summary if inherited_tools else "",
        }


class AIIntentRouter:
    """
    AI 驱动的意图路由器（优化版）。

    架构：
      规则路由（IntentRouter）作为"零延迟快速路径"
      AI 路由（AIIntentRouter）作为"语义理解增强层"

    【优化重点】合并意图分析 + 任务规划为单次 LLM 调用：
      旧版：AIIntentRouter 调 1 次 LLM → needs_breakdown=true → TaskPlanner 再调 1 次 LLM
      新版：单次 LLM 调用同时返回 run_mode + needs_breakdown + todos（可选）
            若 needs_breakdown=true，直接附带 todos，TaskPlanner 不再单独调用

    Prompt 设计原则（轻量、快速）：
      - 系统 prompt 控制在 400 tokens 内
      - temperature=0.1，保证稳定 JSON 输出
      - 仅在规则路由低置信度时才调用
    """

    ROUTER_SYSTEM_PROMPT = """\
你是一个意图分析器和任务规划助手。分析用户消息并返回 JSON，不要输出任何其他内容。

可用的 run_mode：
- "chat": 普通对话，无需工具
- "tools": 需要读写文件、执行命令等工具操作
- "skills": 需要加载领域知识（pdf/python/code-review）
- "hybrid": 同时需要工具和技能

返回格式（严格 JSON，不加 markdown）：
{
  "run_mode": "chat|tools|skills|hybrid",
  "is_followup": true|false,
  "needs_breakdown": true|false,
  "skill_ids": [],
  "reason": "一句话说明原因",
  "title": "任务标题（needs_breakdown=true 时填写，一句话概括）",
  "todos": []
}

判断规则：
- is_followup=true: 当前消息是对上一轮对话的追问/继续（如"输出代码""详细说明""继续"）
- needs_breakdown=true: 任务包含 3 个以上明显步骤，适合拆解为 TODO 列表
- skill_ids: 从 [pdf, python-dev, code-review] 中选择相关技能
- title: 当 needs_breakdown=true 时，填写一句话任务标题（如"分析并优化代码"）；否则为空字符串
- todos: 当 needs_breakdown=true 时，填入 3-6 个步骤；否则为空数组 []
  todos 格式: [{"id":1,"task":"步骤描述","tool":"read_file|write_file|edit_file|list_dir|bash|none","status":"pending"}]
- tool 字段必须是以上合法值之一，禁止使用 python-dev、code-review 等技能名作为 tool 值；需要执行命令/代码时用 bash
- todos 步骤必须相互独立、不重复，禁止两个步骤描述同一件事
- 步骤描述必须具体明确操作对象（如"用grep扫描core目录的.py文件"而非"扫描目录"）
- 后续步骤必须基于前一步结果做进一步处理，禁止重复前一步已执行的命令
- 本项目是 Python 项目：涉及代码文件操作时，目标文件为 .py 而非 .java/.class
- 【合并规则】"整理成文档"和"写入文件"是同一个操作，必须合并为一个步骤（tool=write_file），禁止拆成两步
- 【合并规则】"生成内容"和"保存到文件"是同一个操作，不能分开
- 运行命令时使用 python3 而非 python（macOS 环境）"""

    def __init__(
        self,
        model_forward_fn,
        rule_router: Optional["IntentRouter"] = None,
        available_skill_ids: Optional[List[str]] = None,
        enable_ai_routing: bool = True,
    ):
        """
        Args:
            model_forward_fn: 轻量模型调用函数
            rule_router: 规则路由器实例（作为快速路径）
            available_skill_ids: 可用技能 ID 列表
            enable_ai_routing: 是否启用 AI 路由（可运行时关闭）
        """
        self.model_forward_fn = model_forward_fn
        self.rule_router = rule_router or IntentRouter(available_skill_ids)
        self.available_skill_ids = set(available_skill_ids or [])
        self.enable_ai_routing = enable_ai_routing

    def _call_llm_for_intent_and_plan(
        self,
        user_message: str,
        chat_history: Optional[List[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        单次 LLM 调用：同时完成意图分析 + 任务拆解规划。

        返回解析后的 dict（含 todos），或 None（失败时降级）。
        """
        # 构建历史摘要（只取最近 2 轮，控制 token）
        history_summary = ""
        if chat_history:
            recent = chat_history[-2:] if len(chat_history) > 2 else chat_history
            parts = []
            for h in recent:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    u = str(h[0] or "")[:60]
                    a = str(h[1] or "")[:60]
                    if u:
                        parts.append(f"用户: {u}")
                    if a:
                        parts.append(f"助手: {a}")
            if parts:
                history_summary = "【近期对话】\n" + "\n".join(parts) + "\n\n"

        prompt_content = (
            f"{history_summary}"
            f"【当前消息】\n{user_message[:400]}\n\n"
            "请分析意图、判断是否需要任务拆解，并返回 JSON（若需要拆解则填写 todos 字段）。"
        )

        messages = [{"role": "user", "content": prompt_content}]

        try:
            raw = self.model_forward_fn(
                messages,
                system_prompt=self.ROUTER_SYSTEM_PROMPT,
                temperature=0.1,
                top_p=0.9,
                max_tokens=400,
            )
            raw = raw.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # 校验并补全 todos 字段
                todos = data.get("todos", [])
                if isinstance(todos, list) and len(todos) >= 2:
                    for i, t in enumerate(todos):
                        t.setdefault("id", i + 1)
                        t.setdefault("tool", "none")
                        t.setdefault("status", "pending")
                    data["todos"] = todos
                    data["needs_breakdown"] = True
                else:
                    data["todos"] = []
                    # 如果模型声称 needs_breakdown 但 todos 不够，降级
                    if len(todos) < 2:
                        data["needs_breakdown"] = False
                return data
        except Exception:
            pass
        return None

    def analyze(
        self,
        user_message: str,
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析意图，融合规则路由 + AI 路由结果。

        策略：
          - 规则路由有高置信度结果（明确关键词/路径）→ 直接使用，跳过 AI
          - 规则路由结果不确定 → 调用 AI 补充判断（含任务拆解规划）

        返回字段：
          - run_mode: chat/tools/skills/hybrid
          - skill_ids: 技能列表
          - reasons: 路由决策原因
          - needs_breakdown: 是否需要任务拆解
          - todos: 任务拆解结果（若 needs_breakdown=True，直接可用，无需二次调用 TaskPlanner）
          - is_followup: 是否为追问
          - router: "rule" / "ai" / "rule_fallback"
        """
        # 第一步：规则路由快速判断
        rule_result = self.rule_router.analyze(
            user_message=user_message,
            uploaded_files=uploaded_files,
            chat_history=chat_history,
        )

        # 高置信度场景：有明确的文件路径/操作关键词
        has_explicit_signal = any(
            r in ("检测到文件路径+操作意图", "检测到文件路径", "检测到文件操作关键词", "检测到命令执行意图")
            for r in rule_result.get("reasons", [])
        )
        # 通过 inherited_context 继承了工具模式（追问场景）
        has_inherited_tools = any(
            r.startswith("继承上下文工具模式")
            for r in rule_result.get("reasons", [])
        )

        # 追问继承：短路返回，不做任务拆解
        if has_inherited_tools:
            rule_result["router"] = "rule"
            rule_result.setdefault("needs_breakdown", False)
            rule_result.setdefault("todos", [])
            rule_result.setdefault("is_followup", True)
            return rule_result

        # 调用 AI 补充判断（has_explicit_signal 时 run_mode 锁定规则结果，但仍判断 needs_breakdown）
        if self.enable_ai_routing and self.model_forward_fn is not None:
            ai_result = self._call_llm_for_intent_and_plan(user_message, chat_history)
            if ai_result and isinstance(ai_result, dict):
                # run_mode：has_explicit_signal 时锁定规则结果，防止 AI 把 tools 降级为 chat
                if has_explicit_signal:
                    run_mode = rule_result["run_mode"]
                else:
                    run_mode = ai_result.get("run_mode", rule_result["run_mode"])
                    if run_mode not in ("chat", "tools", "skills", "hybrid"):
                        run_mode = rule_result["run_mode"]

                # 合并技能（规则检测到的 + AI 推荐的）
                ai_skill_ids = [
                    s for s in (ai_result.get("skill_ids") or [])
                    if s in self.available_skill_ids
                ]
                merged_skills = list(
                    dict.fromkeys(rule_result.get("skill_ids", []) + ai_skill_ids)
                )

                reasons = list(rule_result.get("reasons", []))
                if ai_result.get("reason"):
                    reasons.append(f"AI分析: {ai_result['reason']}")
                if ai_result.get("is_followup"):
                    reasons.append("AI判断: 追问/继续上一轮对话")
                if ai_result.get("needs_breakdown"):
                    todos = ai_result.get("todos", [])
                    reasons.append(f"AI规划: 任务已拆解为 {len(todos)} 步")

                return {
                    "run_mode": run_mode,
                    "skill_ids": merged_skills,
                    "reasons": reasons,
                    "needs_tools": run_mode in ("tools", "hybrid"),
                    "needs_skills": bool(merged_skills),
                    "needs_breakdown": bool(ai_result.get("needs_breakdown", False)),
                    "todos": ai_result.get("todos", []),
                    "title": ai_result.get("title", ""),
                    "is_followup": bool(ai_result.get("is_followup", False)),
                    "router": "ai",
                    "inherited_context": rule_result.get("inherited_context", ""),
                }

        # 降级：使用规则路由结果（AI 路由未启用或调用失败）
        rule_result["router"] = "rule" if has_explicit_signal else "rule_fallback"
        rule_result.setdefault("needs_breakdown", False)
        rule_result.setdefault("todos", [])
        rule_result.setdefault("is_followup", False)
        return rule_result

