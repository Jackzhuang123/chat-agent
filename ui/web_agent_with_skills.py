#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B Agent Web UI - 豆包风格完美版（恢复原始气泡样式，优化按钮布局）
支持本地 Qwen 模型 和 智谱 GLM-4-Flash API（免费）双引擎切换。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 在任何导入前，优先加载项目根目录下的 .env 文件
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _ef:
        for _el in _ef:
            _el = _el.strip()
            if _el and not _el.startswith("#") and "=" in _el:
                _ek, _ev = _el.split("=", 1)
                _ek, _ev = _ek.strip(), _ev.strip()
                if _ek and _ek not in os.environ:  # 系统环境变量优先
                    os.environ[_ek] = _ev

# GLMAgent 支持
try:
    from glm_agent import GLMAgent, validate_api_key
    HAS_GLM = True
except ImportError:
    HAS_GLM = False

# 从环境变量读取 GLM API Key
_GLM_API_KEY = os.environ.get("GLM_API_KEY", "").strip()
_GLM_AUTO_ENABLED = HAS_GLM and bool(_GLM_API_KEY)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import (
    PlanModeMiddleware,
    QwenAgentFramework,
    RuntimeModeMiddleware,
    SkillInjector,
    SkillManager,
    SkillsContextMiddleware,
    ToolResultGuardMiddleware,
    UploadedFilesMiddleware,
    create_example_skills,
    create_qwen_model_forward,
    MultiAgentOrchestrator,
    create_streaming_wrapper,
    # 已移除：TodoManager, TaskPlanner, IntentRouter, AIIntentRouter
    # 如需使用，请改用 ModeRouter
)
from core.prompts import get_system_prompt, inject_few_shot_examples
from core.tool_enforcement_middleware import ToolEnforcementMiddleware
from session_logger import get_logger

try:
    import PyPDF2

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class QwenAgent:
    def __init__(self, model_path="./model/qwen2.5-0.5b", logger=None):
        print("正在初始化模型 (CPU模式)...")
        # 确保模型路径相对于项目根目录
        if model_path.startswith("./"):
            # 获取项目根目录（ui/web_agent_with_skills.py 的父目录的父目录）
            project_root = Path(__file__).parent.parent
            self.model_path = str((project_root / model_path[2:]).resolve())
        else:
            self.model_path = model_path

        print(f"使用模型路径: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.default_system_prompt = "你是一个智能个人助手,名字叫小Q。请用简洁、幽默的风格回答。"
        self.logger = logger
        print("✅ 模型加载完毕!")

    def generate_stream(self, message, history, system_prompt=None, temperature=0.7, top_p=0.9, max_tokens=8192):
        """标准的流式生成方法 - 接收 history (二维列表)，并记录模型调用"""
        import time
        sys_prompt = system_prompt if system_prompt and system_prompt.strip() else self.default_system_prompt
        messages = [{"role": "system", "content": sys_prompt}]

        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        # 构建完整的提示词用于日志记录
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            model_inputs, streamer=streamer, max_new_tokens=int(max_tokens),
            temperature=float(temperature), top_p=float(top_p), do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        start_time = time.time()
        thread.start()

        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

        # 流式生成完成后，记录完整调用
        execution_time = time.time() - start_time
        if self.logger:
            try:
                # 计算tokens
                input_tokens = len(model_inputs['input_ids'][0])
                output_tokens = len(self.tokenizer.encode(partial_message))

                self.logger.log_model_call(
                    prompt=text[:500] + "..." if len(text) > 500 else text,  # 限制长度
                    response=partial_message,
                    execution_time=execution_time,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    model_name="Qwen2.5-0.5B"
                )
            except Exception as e:
                print(f"日志记录错误: {e}")

    def generate_stream_with_messages(self, messages, temperature=0.7, top_p=0.9, max_tokens=8192):
        """新方法 - 直接接收 messages (字典列表)，用于 Skills 系统，并记录模型调用"""
        import time
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            model_inputs, streamer=streamer, max_new_tokens=int(max_tokens),
            temperature=float(temperature), top_p=float(top_p), do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        start_time = time.time()
        thread.start()

        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

        # 流式生成完成后，记录完整调用
        execution_time = time.time() - start_time
        if self.logger:
            try:
                # 计算tokens
                input_tokens = len(model_inputs['input_ids'][0])
                output_tokens = len(self.tokenizer.encode(partial_message))

                self.logger.log_model_call(
                    prompt=text[:500] + "..." if len(text) > 500 else text,  # 限制长度
                    response=partial_message,
                    execution_time=execution_time,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    model_name="Qwen2.5-0.5B"
                )
            except Exception as e:
                print(f"日志记录错误: {e}")


def create_ui_with_skills():
    logger = get_logger()

    # 当前激活的 Agent（可在运行时切换）
    _active_agent = {"instance": None, "type": "local"}  # type: 'local' | 'glm'

    def get_or_init_local_agent():
        """懒加载本地 Qwen 模型（首次调用时才加载）。"""
        if _active_agent["instance"] is None or _active_agent["type"] != "local":
            print("正在加载本地 Qwen2.5-0.5B 模型...")
            _active_agent["instance"] = QwenAgent(logger=logger)
            _active_agent["type"] = "local"
        return _active_agent["instance"]

    # 启动时自动从环境变量初始化 GLM Agent
    if _GLM_AUTO_ENABLED:
        try:
            _active_agent["instance"] = GLMAgent(
                api_key=_GLM_API_KEY,
                model="glm-4-flash",
                logger=logger,
            )
            _active_agent["type"] = "glm"
            print(f"✅ 已自动启用 GLM-4-Flash（环境变量 GLM_API_KEY）")
        except Exception as _e:
            print(f"⚠️  GLM 初始化失败，将使用本地模型: {_e}")
    else:
        print("ℹ️  未检测到 GLM_API_KEY 环境变量，将使用本地 Qwen 模型")

    print("🧪 初始化 Skills 系统...")
    create_example_skills()
    skill_manager = SkillManager()
    skill_injector = SkillInjector(skill_manager)
    print(f"✅ 发现 {len(skill_manager.skills_metadata)} 个技能")

    # 初始化意图路由器（AI 增强版，自动感知可用 skill 列表）
    _available_skill_ids = list(skill_manager.skills_metadata.keys())

    # 使用新的 ModeRouter 替代旧的 IntentRouter
    # LLM forward 函数延迟绑定（在 dynamic_routing_forward 定义后调用 set_llm_forward）
    from core import ModeRouter
    mode_router = ModeRouter(llm_forward_fn=None, llm_confidence_threshold=0.70)

    # 兼容层：模拟旧的 intent_router 接口
    class IntentRouterCompat:
        """兼容层：将 ModeRouter 适配为旧的 IntentRouter 接口。

        两级路由策略：
        1. ModeRouter 规则路由（快速）：正则 + 关键词，置信度 >= 0.55 直接返回
        2. ModeRouter LLM 路由（精确）：调用语言模型语义分析，置信度 < 0.55 时触发
        """
        def __init__(self, mode_router, available_skill_ids):
            self.mode_router = mode_router
            self.available_skill_ids = available_skill_ids

        def route(self, user_input, context=None):
            """模拟 route 方法"""
            detection = self.mode_router.detect_mode(user_input, context)
            return {
                "matched_skills": [],
                "mode": detection["recommended_mode"],
                "confidence": detection["confidence"]
            }

        def analyze(self, user_message, uploaded_files=None, chat_history=None):
            """适配 analyze 接口：将 ModeRouter.detect_mode 结果转换为标准意图格式。

            两级路由：
            - 规则路由置信度 >= 0.55：直接返回，router="rule"
            - 规则路由置信度 < 0.55 且 LLM 可用：调用 LLM，router="llm"
            - LLM 路由失败：兜底规则路由，router="rule"

            追问继承（inherited_context）：
            - 若上一轮是工具模式且有有效响应，且当前轮检测为 chat/tools，
              则提取上一轮摘要注入 inherited_context，避免模型幻觉。
            - 若用户在追问"之前/上次/刚才"等，强制继承上下文。

            返回格式：
            {
                "run_mode": "chat|tools|skills|hybrid",
                "skill_ids": [],
                "reasons": ["..."],
                "needs_breakdown": False,
                "router": "rule|llm",
                "inherited_context": "上一轮工具结果摘要（或空字符串）",
            }
            """
            import re as _re

            context = {}
            if uploaded_files:
                context["uploaded_files"] = uploaded_files
            if chat_history:
                context["history"] = chat_history

            detection = self.mode_router.detect_mode(user_message, context)
            recommended = detection["recommended_mode"]
            confidence = detection["confidence"]
            reasoning = detection.get("reasoning", "")
            router_type = detection.get("router", "rule")

            # 将 ModeRouter 的模式映射到旧接口的 run_mode
            mode_map = {
                "chat": "chat",
                "tools": "tools",
                "skills": "skills",
                "hybrid": "hybrid",
                "plan": "tools",       # plan 模式映射为 tools（由 plan_mode 控制）
                "multi_agent": "multi_agent",  # 多 Agent 模式：Planner-Executor-Reviewer
                "streaming": "tools",           # 流式模式：走工具循环 + StreamingFramework
            }
            run_mode = mode_map.get(recommended, "chat")

            # 技能匹配：如果是 skills/hybrid 模式，尝试匹配技能
            skill_ids = []
            if run_mode in ("skills", "hybrid") and self.available_skill_ids:
                msg_lower = user_message.lower()
                for sid in self.available_skill_ids:
                    if sid.lower() in msg_lower or any(
                        kw in msg_lower for kw in sid.lower().replace("-", " ").split()
                    ):
                        skill_ids.append(sid)
                # 没有精确匹配时，默认取第一个技能（保持可用）
                if not skill_ids and run_mode == "skills":
                    skill_ids = self.available_skill_ids[:1]

            # 是否需要任务拆解
            needs_breakdown = (
                recommended in ("plan", "multi_agent")
                or detection.get("complexity") == "complex"
            )

            # ---- 追问上下文继承 (inherited_context) ----
            # 解决问题：用户说"查询聊天记录"/"我之前问什么了"时，
            # 模型应基于当前会话上下文回答，而不是说"我无法访问历史"
            inherited_context = ""
            _followup_patterns = _re.compile(
                r'(之前|上次|刚才|刚刚|刚刚|上一轮|上一条|查看|回顾|总结|再说一遍|'
                r'什么问题|聊过什么|聊了什么|问过什么|说了什么|记录|历史|'
                r'前面|那个|之前说|你刚|you said|previous|last|above)',
                _re.IGNORECASE
            )
            _is_followup = bool(_followup_patterns.search(user_message))

            if chat_history and (_is_followup or run_mode == "chat"):
                # 从最近 3 轮历史中提取有效的 assistant 响应摘要
                _ctx_parts = []
                for _pair in chat_history[-3:]:
                    if not (isinstance(_pair, (list, tuple)) and len(_pair) == 2):
                        continue
                    _u, _a = _pair
                    if not _u or not _a:
                        continue
                    # 过滤掉错误消息
                    _skip_prefixes = ("[⚠️", "[GLM", "[在进行中", "[未设置")
                    if any(_a.startswith(p) for p in _skip_prefixes):
                        continue
                    # 截取摘要（最多 200 字符/轮）
                    _a_summary = _a[:200] + ("…" if len(_a) > 200 else "")
                    _ctx_parts.append(f"用户: {str(_u)[:80]}\n助手: {_a_summary}")

                if _ctx_parts:
                    inherited_context = "\n\n".join(_ctx_parts)
                    # 追问时更新 reasoning
                    if _is_followup:
                        reasoning = f"检测到追问意图，注入上一轮上下文（{len(_ctx_parts)}轮）"
                        router_type = router_type  # 保持原有路由类型

            reason_text = reasoning if reasoning else f"模式检测置信度: {confidence:.2f}"
            if router_type == "llm":
                reason_text = f"[LLM路由] {reason_text}"

            return {
                "run_mode": run_mode,
                "recommended_mode_raw": recommended,  # 未经 mode_map 映射的原始推荐模式
                "skill_ids": skill_ids,
                "reasons": [reason_text],
                "needs_breakdown": needs_breakdown,
                "router": router_type,
                "inherited_context": inherited_context,
            }

    intent_router = IntentRouterCompat(mode_router, _available_skill_ids)

    def _get_or_init_ai_router():
        """兼容函数：返回 intent_router（已集成 LLM 路由）"""
        return intent_router

    print(f"🧭 智能模式路由器已就绪（ModeRouter + LLM语义路由），可感知技能: {_available_skill_ids}")
    print("🔄 自动模式检测已激活（规则路由 + LLM兜底）")

    middlewares = [
        RuntimeModeMiddleware(),
        PlanModeMiddleware(),
        SkillsContextMiddleware(),
        UploadedFilesMiddleware(),
        ToolResultGuardMiddleware(),
    ]

    def dynamic_model_forward(messages, system_prompt="", **kwargs):
        """动态前向函数：每次调用时从 _active_agent 读取当前 Agent 实例。"""
        current_agent = _active_agent.get("instance")
        if current_agent is None:
            # 如果还没有初始化任何 Agent，则加载本地模型
            current_agent = get_or_init_local_agent()
        forward_fn = create_qwen_model_forward(current_agent)
        return forward_fn(messages, system_prompt=system_prompt, **kwargs)

    def dynamic_routing_forward(messages, system_prompt="", **kwargs):
        """意图路由专用 forward 函数：不记录任何日志，避免路由阶段的 LLM 调用污染 session 日志。

        AI 意图路由（AIIntentRouter）需要调用 LLM 进行意图分析，
        但这次调用属于"基础设施调用"，不是用户可见的 Agent 操作，不应写入 session_logger。
        通过临时置空 agent.logger，确保 log_model_call 不会在路由阶段被触发。
        """
        current_agent = _active_agent.get("instance")
        if current_agent is None:
            current_agent = get_or_init_local_agent()

        # 临时禁用日志记录（意图路由属于基础设施调用，不应污染用户会话日志）
        _orig_logger = getattr(current_agent, "logger", None)
        try:
            if _orig_logger is not None:
                current_agent.logger = None
            forward_fn = create_qwen_model_forward(current_agent)
            return forward_fn(messages, system_prompt=system_prompt, **kwargs)
        finally:
            # 恢复原始 logger（无论是否异常）
            if _orig_logger is not None:
                current_agent.logger = _orig_logger

    def dynamic_stream_forward(messages, system_prompt="", **kwargs):
        """真流式前向函数：直接调用 Agent 的流式生成接口，逐 token yield 累积文本。
        供 process_message_direct_stream 在 chat/skills 模式下实现逐 token 输出。
        """
        current_agent = _active_agent.get("instance")
        if current_agent is None:
            current_agent = get_or_init_local_agent()

        # 构建含 system 的完整消息（避免重复注入）
        full_messages = list(messages)
        if system_prompt and system_prompt.strip():
            _already = (
                full_messages
                and full_messages[0].get("role") == "system"
                and full_messages[0].get("content", "").strip() == system_prompt.strip()
            )
            if not _already:
                full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        if hasattr(current_agent, "generate_stream_with_messages"):
            yield from current_agent.generate_stream_with_messages(
                full_messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 512),
            )
        else:
            # 降级：直接用非流式调用返回整体字符串
            yield dynamic_model_forward(messages, system_prompt=system_prompt, **kwargs)

    # 延迟绑定 LLM forward 到 mode_router，激活语义路由能力
    # dynamic_routing_forward 临时关闭 session logger，避免路由调用污染用户日志
    mode_router.set_llm_forward(dynamic_routing_forward)
    print("🤖 LLM 语义路由已绑定（路由调用不写入 session_log）")

    agent_framework = QwenAgentFramework(
        model_forward_fn=dynamic_model_forward,
        work_dir=str(Path(__file__).parent.parent),
        enable_bash=True,  # 启用bash工具用于批量扫描
        max_iterations=100,
        tools_in_system_prompt=True,
        middlewares=middlewares + [ToolEnforcementMiddleware(max_retries=2)],
        default_runtime_context={"run_mode": "chat", "plan_mode": False},
    )

    # 修复后的 CSS + JavaScript - 恢复原始气泡样式
    custom_head = """
    <style>
    /* ========== 全局样式 ========== */
    * {
        box-sizing: border-box;
    }

    body, html {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
    }

    .gradio-container {
        padding: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        height: 100vh !important;
    }

    /* ========== 主容器 ========== */
    #main-container {
        display: flex;
        width: 100%;
        height: 100vh;
        overflow: hidden;
    }

    /* ========== 侧边栏 ========== */
    #app-sidebar {
        width: 300px;
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid #e5e7eb;
        overflow-y: auto;
        overflow-x: hidden;
        transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), 
                    margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        flex-shrink: 0;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    #app-sidebar.collapsed {
        width: 0;
        margin-left: -300px;
        overflow: hidden;
    }

    #app-sidebar::-webkit-scrollbar {
        width: 8px;
    }

    #app-sidebar::-webkit-scrollbar-track {
        background: transparent;
    }

    #app-sidebar::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }

    #app-sidebar::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }

    /* ========== 主内容区 ========== */
    #main-content-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #f5f7fa;
        height: 100vh;
        overflow: hidden;
    }

    /* ========== 顶部栏 ========== */
    #top-bar-area {
        background: #fff;
        border-bottom: 1px solid #e5e7eb;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        flex-shrink: 0;
        height: 60px;
    }

    /* 切换按钮 */
    #toggle-btn {
        padding: 8px 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.2);
    }

    #toggle-btn:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    /* ========== 对话区域 ========== */
    #chat-wrapper {
        flex: 1;
        display: flex;
        flex-direction: column;
        padding: 0;
        overflow: hidden;
        min-height: 0;
        background: #f5f7fa;
    }

    /* Chatbot 容器 - 保持原始框架样式 */
    .chatbot-box {
        flex: 1 1 auto !important;
        background: #fff !important;
        border-radius: 16px !important;
        border: 1px solid #e5e7eb !important;
        margin: 20px 24px 16px 24px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 0 !important;
    }

    /* Gradio Chatbot 内部结构 */
    .chatbot-box > * {
        flex: 1 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    .chatbot-box .wrap {
        flex: 1 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    .chatbot-box .wrap > div {
        flex: 1 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 20px !important;
    }

    /* ========== 执行日志 ========== */
    #chat-wrapper .accordion {
        margin: 0 24px 12px 24px !important;
        background: #fff;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        flex-shrink: 0;
        max-height: 100px;
        overflow: hidden;
    }

    .log-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        line-height: 1.6;
        max-height: 80px;
        overflow-y: auto;
        color: #374151;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ========== 输入区域 - 横向布局 ========== */
    #input-area-fixed {
        flex-shrink: 0;
        background: #f5f7fa;
        padding: 0 24px 24px 24px;
    }

    /* 输入框和按钮在同一行 */
    .input-row-container {
        background: #fff;
        border-radius: 24px;
        border: 1.5px solid #e5e7eb;
        padding: 16px 20px;
        display: flex;
        gap: 14px;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .input-row-container:focus-within {
        border-color: #6366f1;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15);
    }

    .input-text-area {
        flex: 1;
        min-width: 0;
    }

    .input-text-area textarea {
        border: none !important;
        background: transparent !important;
        min-height: 28px !important;
        max-height: 120px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 4px 0 !important;
        resize: none !important;
        color: #1f2937;
        width: 100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    .input-text-area textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    .input-text-area textarea::placeholder {
        color: #b4b8c1;
        font-size: 15px;
    }

    /* ========== 按钮组 - 横向排列 ========== */
    .button-row {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-shrink: 0;
    }

    .upload-btn-inline {
        background: transparent !important;
        color: #6b7280 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 8px !important;
        font-size: 18px !important;
        cursor: pointer;
        transition: all 0.2s;
        height: 32px;
        min-width: 32px;
        width: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        opacity: 0.6;
    }

    .upload-btn-inline:hover {
        background: #f3f4f6 !important;
        color: #4b5563 !important;
        opacity: 1;
    }

    .send-btn-inline {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-size: 0 !important;
        font-weight: 600 !important;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 32px;
        min-width: 32px;
        width: 32px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        white-space: nowrap;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .send-btn-inline::before {
        content: "➤";
        font-size: 16px;
    }

    .send-btn-inline:hover {
        opacity: 0.95;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
    }

    .send-btn-inline:active {
        transform: translateY(0);
    }

    /* ========== 隐藏元素 ========== */
    footer {
        display: none !important;
    }

    /* ========== 滚动条美化 ========== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }

    /* ========== Gradio 组件覆盖 ========== */
    .gradio-container .prose {
        max-width: none !important;
    }

    .contain {
        max-width: 100% !important;
    }

    /* ========== 标题样式 ========== */
    #top-bar-area h2 {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
        margin: 0;
        flex: 1;
    }

    /* ========== 侧边栏内边距优化 ========== */
    #app-sidebar > div:first-child {
        padding: 24px 18px !important;
        display: flex;
        flex-direction: column;
        overflow: visible !important;
    }

    /* ========== 侧边栏滚动容器 ========== */
    #sidebar-scroll-container {
        display: flex;
        flex-direction: column;
        gap: 4px;
        overflow: visible !important;
    }

    /* 禁用Gradio内部的所有滚动容器 */
    #sidebar-scroll-container * {
        overflow: visible !important;
    }

    #sidebar-scroll-container .wrap {
        overflow: visible !important;
        min-height: unset !important;
    }

    #sidebar-scroll-container .gradio-column {
        overflow: visible !important;
        min-height: unset !important;
    }

    /* 确保所有div都不创建自己的滚动上下文 */
    #sidebar-scroll-container div {
        overflow: visible !important;
    }

    /* ========== 侧边栏分组样式 - 无框设计 ========== */
    #sidebar-scroll-container > h2 {
        margin: 0 0 20px 0 !important;
        padding: 0 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #111827 !important;
        letter-spacing: -0.3px !important;
    }

    /* Markdown 标题样式 */
    #sidebar-scroll-container > div:has(> h4) {
        margin-top: 20px !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }

    #sidebar-scroll-container h4 {
        margin: 0 0 14px 0 !important;
        padding: 0 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        color: #6b7280 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* Checkbox 和其他输入组件样式 */
    #sidebar-scroll-container .gradio-checkbox {
        margin-bottom: 10px !important;
        padding: 8px 0 !important;
    }

    #sidebar-scroll-container .gradio-checkbox label {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    #sidebar-scroll-container .gradio-slider {
        margin-bottom: 16px !important;
    }

    #sidebar-scroll-container .gradio-slider label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #4f46e5 !important;
    }

    #sidebar-scroll-container .gradio-textbox,
    #sidebar-scroll-container .gradio-dropdown {
        margin-bottom: 14px !important;
    }

    #sidebar-scroll-container .gradio-textbox label,
    #sidebar-scroll-container .gradio-dropdown label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #4f46e5 !important;
    }

    /* 隐藏所有的Group边框 */
    #app-sidebar .gradio-group {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        overflow: visible !important;
    }

    /* 侧边栏Markdown段落样式 */
    #sidebar-scroll-container p {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 侧边栏输入框美化 */
    #sidebar-scroll-container input,
    #sidebar-scroll-container textarea {
        border-color: #e5e7eb !important;
        border-radius: 6px !important;
        font-size: 13px !important;
    }

    #sidebar-scroll-container input:focus,
    #sidebar-scroll-container textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* 侧边栏下拉菜单美化 */
    #sidebar-scroll-container select {
        border-color: #e5e7eb !important;
        border-radius: 6px !important;
        background-color: #fff !important;
        color: #374151 !important;
        padding: 8px 12px !important;
    }

    #sidebar-scroll-container select:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* 侧边栏Slider样式 */
    #sidebar-scroll-container input[type="range"] {
        accent-color: #6366f1 !important;
    }

    /* ========== 文件上传按钮美化 ========== */
    .upload-btn-inline input[type="file"] {
        display: none !important;
    }

    .upload-btn-inline button {
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }

    .upload-btn-inline button:hover {
        background: #f3f4f6 !important;
    }

    .upload-btn-inline label {
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        margin: 0 !important;
        font-size: 18px;
    }

    /* ========== 发送按钮优化 ========== */
    .send-btn-inline {
        letter-spacing: 0 !important;
    }
    </style>

    <script>
    // 侧边栏切换脚本
    (function() {
        let isCollapsed = false;

        function setupToggle() {
            const sidebar = document.getElementById('app-sidebar');
            const toggleBtn = document.getElementById('toggle-btn');

            if (!sidebar || !toggleBtn) {
                setTimeout(setupToggle, 100);
                return;
            }

            toggleBtn.onclick = function() {
                isCollapsed = !isCollapsed;

                if (isCollapsed) {
                    sidebar.classList.add('collapsed');
                    toggleBtn.innerHTML = '▶ 展开侧边栏';
                } else {
                    sidebar.classList.remove('collapsed');
                    toggleBtn.innerHTML = '◀ 收起侧边栏';
                }
            };
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupToggle);
        } else {
            setupToggle();
        }

        setTimeout(setupToggle, 300);
        setTimeout(setupToggle, 600);
        setTimeout(setupToggle, 1000);
        setTimeout(setupToggle, 2000);
    })();
    </script>
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Qwen2.5 Assistant", head=custom_head) as demo:

        with gr.Row(elem_id="main-container"):
            # 侧边栏
            with gr.Column(elem_id="app-sidebar", scale=0, min_width=300):
                with gr.Column(elem_id="sidebar-scroll-container"):
                    gr.Markdown("### 🤖 智能助手")

                    # ===== 模型引擎选择 =====
                    gr.Markdown("#### 🚀 模型引擎")

                    # 初始引擎状态
                    _init_engine_value = "⚡ GLM-4-Flash" if _GLM_AUTO_ENABLED else "🏠 本地 Qwen2.5-0.5B"
                    model_engine = gr.Radio(
                        choices=["🏠 本地 Qwen2.5-0.5B", "⚡ GLM-4-Flash"],
                        value=_init_engine_value,
                        label=None,
                        show_label=False,
                        interactive=_GLM_AUTO_ENABLED,  # 有 Key 才允许切换
                    )

                    # GLM 状态面板（只读，无需输入 Key）
                    _init_glm_status = (
                        f"✅ 已启用（模型: glm-4-flash）"
                        if _GLM_AUTO_ENABLED
                        else "⚪ 未配置（请设置环境变量 GLM_API_KEY）"
                    )
                    with gr.Column(visible=True) as glm_config_panel:
                        gr.Markdown("#### ⚙️ GLM 配置")
                        glm_status = gr.Textbox(
                            label="状态",
                            value=_init_glm_status,
                            interactive=False,
                            lines=1,
                            max_lines=2,
                        )
                        glm_model_choice = gr.Dropdown(
                            choices=[
                                "glm-4-flash",
                                "glm-4-flash-250414",
                                "glm-4-air",
                                "glm-4",
                            ],
                            value="glm-4-flash",
                            label="GLM 模型",
                            visible=_GLM_AUTO_ENABLED,
                        )

                    # 数据分析链接 - 高级分析
                    gr.Markdown("#### 📊 数据分析")
                    gr.HTML("""
                    <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; margin-top: 8px;">
                        <a href="http://127.0.0.1:7862" target="_blank" style="
                            display: block;
                            padding: 14px 16px;
                            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                            color: white;
                            border-radius: 10px;
                            text-align: center;
                            text-decoration: none;
                            font-weight: 600;
                            font-size: 15px;
                            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
                            letter-spacing: 0.3px;
                        " onmouseover="this.style.opacity='0.95'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 16px rgba(99, 102, 241, 0.35)'" onmouseout="this.style.opacity='1'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(99, 102, 241, 0.25)'">
                            🔬 高级分析 →
                        </a>
                    </div>
                    """)

                    gr.Markdown("#### 🧭 AI 自动模式")
                    gr.HTML("""
                    <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 12px;font-size:13px;color:#166534;margin-bottom:4px;">
                        <b>🤖 双层意图路由已启用</b><br>
                        <span style="color:#15803d;">📏 规则路由</span>：零延迟，识别明确文件/命令意图<br>
                        <span style="color:#7c3aed;">🧠 AI路由</span>：语义理解，处理模糊追问/复杂需求<br>
                        <span style="color:#b45309;">📋 任务拆解</span>：复杂需求自动生成TODO逐步执行<br>
                        &nbsp;• 无需手动切换模式
                    </div>
                    """)
                    # 当前模式状态（由 bot_response 更新，仅展示）
                    current_mode_display = gr.Textbox(
                        value="💬 等待输入...",
                        label="当前模式",
                        interactive=False,
                        lines=2,
                    )
                    # TODO 进度展示（任务拆解时显示）
                    todo_progress_display = gr.HTML(
                        value="",
                        label="任务进度",
                        visible=True,
                    )
                    plan_mode = gr.Checkbox(label="🗂️ 计划模式（AI 先规划再执行）", value=False)

                    gr.Markdown("#### 模型参数")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                    max_tokens = gr.Slider(64, 16384, value=8192, step=64, label="Max Tokens")

                    gr.Markdown("#### 系统提示")
                    system_prompt = gr.Textbox(
                        value="你是一个智能助手。",
                        lines=3,
                        placeholder="自定义系统提示...",
                        show_label=False
                    )

                    gr.Markdown("#### 可用技能")
                    _skill_list_text = "\n".join(
                        f"• {s.get('name', s['id'])}：{s.get('description', '')}"
                        for s in skill_manager.get_skills_list()
                    ) or "（暂无技能）"
                    gr.Textbox(
                        value=_skill_list_text,
                        label="已发现技能（自动加载）",
                        interactive=False,
                        lines=max(2, len(skill_manager.get_skills_list())),
                    )

            # 主内容区
            with gr.Column(elem_id="main-content-area", scale=1):
                # 顶部栏
                with gr.Row(elem_id="top-bar-area"):
                    gr.HTML("<button id='toggle-btn'>◀ 收起侧边栏</button>")
                    gr.Markdown("## 💬 Chatbot")

                # 对话区域
                with gr.Column(elem_id="chat-wrapper"):
                    # 1. Chatbot - 恢复原始框架样式
                    chatbot = gr.Chatbot(
                        label=None,
                        show_copy_button=True,
                        show_label=False,
                        container=False,
                        elem_classes="chatbot-box"
                    )

                    # 2. 输入区域 - 横向布局（输入框 + 上传 + 发送）
                    with gr.Column(elem_id="input-area-fixed"):
                        with gr.Row(elem_classes="input-row-container"):
                            # 输入框
                            msg = gr.Textbox(
                                label=None,
                                placeholder="请输入问题或需求",
                                lines=1,
                                max_lines=6,
                                show_label=False,
                                container=False,
                                elem_classes="input-text-area"
                            )

                            # 按钮组（横向）
                            with gr.Row(elem_classes="button-row"):
                                pdf_file = gr.File(
                                    label="📎",
                                    file_count="multiple",
                                    file_types=[".pdf"],
                                    elem_classes="upload-btn-inline",
                                    container=False,
                                    visible=True,
                                    scale=0
                                )
                                send_btn = gr.Button("", elem_classes="send-btn-inline", scale=0)

        # ========== 事件处理（保持不变）==========

        def user_input(user_message, history):
            return "", history + [[user_message, None]]

        def extract_pdf_text(pdf_files):
            if not pdf_files or not HAS_PYPDF:
                return ""
            pdf_content = ""
            files_to_process = pdf_files if isinstance(pdf_files, list) else [pdf_files]

            for pdf_item in files_to_process:
                try:
                    pdf_path = pdf_item.get('name') if isinstance(pdf_item, dict) else pdf_item
                    with open(pdf_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        filename = Path(pdf_path).name if pdf_path else "unknown.pdf"
                        pdf_content += f"\n【文件: {filename}】\n"
                        for page_num, page in enumerate(reader.pages, 1):
                            text = page.extract_text()
                            if text.strip():
                                pdf_content += f"【第 {page_num} 页】\n{text}\n"
                except Exception as e:
                    pdf_content += f"\n❌ 提取失败: {str(e)}\n"
            return pdf_content.strip()

        def extract_uploaded_file_meta(pdf_files):
            if not pdf_files:
                return []

            files_to_process = pdf_files if isinstance(pdf_files, list) else [pdf_files]
            uploaded_files = []
            for pdf_item in files_to_process:
                pdf_path = pdf_item.get('name') if isinstance(pdf_item, dict) else pdf_item
                if not pdf_path:
                    continue

                path_obj = Path(pdf_path)
                try:
                    file_size = path_obj.stat().st_size if path_obj.exists() else 0
                except Exception:
                    file_size = 0

                uploaded_files.append({
                    "filename": path_obj.name,
                    "path": str(path_obj),
                    "size": file_size,
                })
            return uploaded_files

        def get_active_agent():
            """获取当前激活的 Agent 实例（GLM 或本地 Qwen）。"""
            return _active_agent.get("instance") or get_or_init_local_agent()

        def bot_response(history, sys_prompt, temp, top_p_val, max_tok, plan_mode_enabled, pdf_files):
            """
            统一响应入口。

            由 IntentRouter 自动分析用户意图，决定：
            - run_mode: chat / tools / skills / hybrid
            - 需要加载的 skills（内容追加至对话末尾，缓存友好）
            - 是否进入 agent_framework 工具循环

            不再依赖手动 checkbox（工具模式 / Skills 模式）。
            """
            if not history or history[-1][1] is not None:
                return history

            user_message = history[-1][0]
            history_without_last = history[:-1]
            logger = get_logger()

            # 创建新会话
            logger.create_session()

            import time
            start_time = time.time()

            pdf_info = ""
            uploaded_files_meta = extract_uploaded_file_meta(pdf_files)
            if uploaded_files_meta and HAS_PYPDF:
                pdf_text = extract_pdf_text(pdf_files)
                if pdf_text:
                    pdf_info = f"\n\n【PDF内容】:\n{pdf_text[:1000]}..."

            # 过滤掉包含错误响应/占位响应的历史轮次，避免污染后续对话上下文
            # 问题根因（日志 170319）：
            #   工具模式下 iteration=0 的中间计划文本（"好的，以下是计划：..."）
            #   会被当作 assistant 回复写入历史，下一轮 chat 模式下模型看到这条幻觉
            #   历史，导致凭空编造不存在的方法和类继承关系。
            _filter_prefixes = (
                # 错误/超限
                "已达到最大迭代次数",
                "[⚠️ 工具模式错误]",
                "[GLM API 错误]",
                # 占位消息（session_logger 写入的临时状态）
                "[在进行中...]",
                "[未设置]",
            )
            # 过滤掉以计划文本起头的幻觉响应：
            # 工具模式 iteration=0 的计划消息通常以"好的，以下是"/"以下是完成任务"开头，
            # 后面还会出现明确的工具调用，最终 bot_response 应替换为工具结果摘要。
            # 如果这类文本意外留在历史中，需要过滤。
            # 判断标准：响应内包含工具调用标记但又不是最终结论（代表未被工具结果替换）
            def _is_stale_tool_plan(a_text: str) -> bool:
                """检测是否是未被替换的工具模式计划文本（应被过滤出历史）。"""
                if not a_text:
                    return False
                # 包含 XML 工具标签（说明是中间态，未被最终结果替换）
                if "<tool>" in a_text and "</tool>" in a_text:
                    return True
                if "<input>" in a_text and "</input>" in a_text:
                    return True
                return False

            # 构建干净的对话历史：
            #  - 过滤掉 assistant 侧是错误/占位/中间态计划的轮次
            #  - 注意：过滤时不能因为 a 无效就同时丢弃 u（用户消息）
            #    因为下一轮意图路由需要扫描历史中的用户消息来判断是否是文件操作追问
            #    若整轮丢弃，会导致"输出文件内容"等追问无法继承工具模式（日志 175616）
            _clean_pairs: list = []
            for u, a in history_without_last:
                if not u:
                    continue  # 用户消息为空直接跳过
                # 判断 assistant 响应是否有效
                _a_invalid = (
                    not a
                    or any(a.startswith(p) for p in _filter_prefixes)
                    or _is_stale_tool_plan(a)
                )
                if _a_invalid:
                    # assistant 无效时，保留用户消息但用空字符串占位，
                    # 这样后续的 base_messages 构建中会被 if u and a 跳过（不加入模型上下文），
                    # 但 _routing_history 构建时用 u 做文件操作检测仍然有效
                    _clean_pairs.append([u, None])
                else:
                    _clean_pairs.append([u, a])

            # 给模型用的历史（u 和 a 都有效）
            chat_history = [[u, a] for u, a in _clean_pairs if u and a]

            # ---- 1. 懒初始化 AI 路由器 + 分析意图 ----
            active_router = _get_or_init_ai_router()
            # 为意图路由构建"轻量历史"：基于 _clean_pairs（包含 a=None 的轮次）。
            # 使用 _clean_pairs 而非 chat_history，目的是保留用户消息（u有效但a无效的情况），
            # 让规则路由的文件操作检测可以扫描到历史中的文件路径，正确触发工具模式继承。
            # assistant 内容截断到 120 字符，避免过长文本干扰 AI 路由的 JSON 输出。
            _routing_history = [
                [u, (a[:120] + "…" if a and len(a) > 120 else (a or ""))]
                for u, a in _clean_pairs[-3:]
                if u  # 只要用户消息有效即可（a可以为None/空）
            ]
            intent = active_router.analyze(
                user_message=user_message + pdf_info,
                uploaded_files=uploaded_files_meta,
                chat_history=_routing_history,
            )
            run_mode = intent["run_mode"]
            skill_ids = intent["skill_ids"]
            intent_reasons = intent["reasons"]
            needs_breakdown = intent.get("needs_breakdown", False)
            router_type = intent.get("router", "rule")
            # 追问工具操作时，上一轮 assistant 的响应摘要（工具执行结果）
            # 用于在 chat 模式下追问时，让模型"记住"上次读取的内容，避免幻觉
            inherited_context = intent.get("inherited_context", "")

            # 构建模式展示文字
            _mode_icons = {"chat": "💬", "tools": "🔧", "skills": "🎓", "hybrid": "⚡"}
            _mode_labels = {"chat": "对话模式", "tools": "工具模式", "skills": "技能模式", "hybrid": "混合模式"}
            _router_badge = "🧠AI" if router_type == "ai" else "📏规则"
            mode_display_text = f"{_mode_icons.get(run_mode, '🤖')} {_mode_labels.get(run_mode, run_mode)} [{_router_badge}]"
            if skill_ids:
                mode_display_text += f" | 技能: {', '.join(skill_ids)}"
            if intent_reasons:
                mode_display_text += f"\n原因: {'; '.join(intent_reasons[:2])}"
            if needs_breakdown:
                mode_display_text += " | 📋任务拆解中..."

            runtime_context = {
                "run_mode": run_mode,
                "plan_mode": bool(plan_mode_enabled),
                "uploaded_files": uploaded_files_meta,
                "selected_skills": skill_ids,
                "intent_reasons": intent_reasons,
            }
            execution_log = [{
                "iteration": -1,
                "type": "intent_analysis",
                "run_mode": run_mode,
                "skill_ids": skill_ids,
                "reasons": intent_reasons,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }]

            # ---- 2. 构建基础消息列表 ----
            # 使用优化的 system prompt
            if not sys_prompt or sys_prompt.strip() == "":
                sys_prompt = get_system_prompt(
                    mode=run_mode,
                    work_dir=str(Path.cwd()),
                    skills_context=skills_text if selected_skills else ""
                )

            base_messages = [{"role": "system", "content": sys_prompt}]

            # tools 模式注入 Few-Shot 示例
            if run_mode == "tools":
                base_messages = inject_few_shot_examples(base_messages, max_examples=2)

            for u, a in chat_history:
                base_messages.append({"role": "user", "content": u})
                base_messages.append({"role": "assistant", "content": a})

            # 若是对上一轮工具操作的追问（chat 模式继承工具上下文），
            # 将上一轮的工具执行结果摘要作为额外上下文注入，
            # 避免模型因缺少工具结果而幻觉（如日志 170319 中凭空编造类结构）。
            _user_msg_with_ctx = user_message + pdf_info
            if inherited_context and run_mode == "chat":
                _user_msg_with_ctx = (
                    f"【上一轮工具执行结果摘要（请基于此内容回答）】\n"
                    f"{inherited_context}\n\n"
                    f"【当前问题】\n{_user_msg_with_ctx}"
                )

            base_messages.append({"role": "user", "content": _user_msg_with_ctx})

            # ---- 3. 缓存友好方式注入 Skills 内容（追加到末尾） ----
            if skill_ids:
                skill_content_parts = []
                skill_infos = []
                for sid in skill_ids:
                    content = skill_manager.get_skill_detail(sid)
                    if not content:
                        continue
                    meta = skill_manager.skills_metadata.get(sid, {})
                    skill_name = meta.get("name", sid)
                    skill_content_parts.append(
                        f'<skill-loaded name="{sid}">\n# 技能: {skill_name}\n\n{content}\n</skill-loaded>'
                    )
                    skill_infos.append({
                        "id": sid,
                        "name": skill_name,
                        "description": meta.get("description", ""),
                        "tags": meta.get("tags", []),
                    })

                if skill_content_parts:
                    runtime_context["skill_contexts"] = skill_infos
                    skill_inject_msg = (
                        "\n\n".join(skill_content_parts)
                        + "\n\n请参考上方技能知识完成任务。"
                    )
                    # 追加到末尾（不修改历史，保持 prompt cache 有效）
                    base_messages.append({"role": "user", "content": skill_inject_msg})
                    execution_log.append({
                        "iteration": -1,
                        "type": "skill_inject",
                        "skill_ids": skill_ids,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    })

            response_started = False

            # ---- 4. 路由分发 ----
            # 分支 A：复杂任务 → MultiAgentOrchestrator（Planner-Executor-Reviewer）
            if needs_breakdown or run_mode == "multi_agent":
                print("🤝 多 Agent 模式：Planner → Executor → Reviewer")
                try:
                    _ma_forward = create_qwen_model_forward(get_active_agent())
                    _orchestrator = MultiAgentOrchestrator(
                        model_forward_fn=_ma_forward,
                        tool_executor=agent_framework.tool_executor,
                        max_retries=1,
                    )
                    # 使用 run_and_generate_response 获取最终自然语言回答
                    _ma_result = _orchestrator.run_and_generate_response(
                        user_input=user_message + pdf_info,
                        model_forward_fn=_ma_forward,  # 用于最终回答生成
                        context=runtime_context,
                        system_prompt="你是一个智能助手。请根据以下执行结果，用简洁的语言回答用户的问题。如果执行失败，请说明失败原因。",
                        temperature=temp,
                        top_p=top_p_val,
                        max_tokens=max_tok,
                    )
                    # 提取最终回答
                    final_response = _ma_result.get("final_response", "")
                    if not final_response:
                        final_response = "多 Agent 执行完成，但未能生成最终回答。"

                    # 记录执行日志（可选，保留原始执行信息）
                    execution_log.append({
                        "iteration": 0,
                        "type": "multi_agent_run",
                        "completed": _ma_result.get("completed", False),
                        "duration": _ma_result.get("duration", 0),
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    })

                    # 将最终回答写入历史
                    history[-1][1] = final_response
                    response_started = True
                    yield history, mode_display_text
                except Exception as e:
                    print(f"⚠️ 多Agent模式异常，降级为工具模式: {e}")
                    needs_breakdown = False  # 降级为工具模式
                    execution_log.append({
                        "type": "multi_agent_error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    })

            # 分支 B：工具模式（无任务拆解）
            if not needs_breakdown and run_mode in ("tools", "hybrid"):
                # 分支 B1：流式模式 → StreamingFramework 实时展示执行进度
                _use_streaming = (intent.get("recommended_mode_raw") == "streaming")
                if _use_streaming:
                    print("🌊 流式模式：StreamingFramework 实时展示进度")
                    try:
                        _sf = create_streaming_wrapper(agent_framework)
                        _chat_history_dicts = [
                            {"role": "user", "content": u} if i % 2 == 0
                            else {"role": "assistant", "content": a}
                            for u, a in chat_history
                            for i, _ in [(0, u), (1, a)]
                        ]
                        _sf_progress_lines = []
                        for _evt in _sf.run_stream(
                            user_input=user_message + pdf_info,
                            history=_chat_history_dicts or None,
                            runtime_context=runtime_context,
                        ):
                            _evt_type = _evt.event_type
                            _evt_data = _evt.data
                            if _evt_type == "start":
                                _sf_progress_lines = ["## 🌊 流式执行进度\n"]
                            elif _evt_type == "thought":
                                if _evt_data.get("content"):
                                    _sf_progress_lines.append(
                                        f"💭 **思考**: {_evt_data['content'][:100]}"
                                    )
                            elif _evt_type == "tool_call":
                                _tool_nm = _evt_data.get("tool", "")
                                _mode_nm = _evt_data.get("mode", "sequential")
                                if _tool_nm:
                                    _sf_progress_lines.append(
                                        f"🔧 **工具调用** [{_mode_nm}]: `{_tool_nm}`"
                                    )
                            elif _evt_type == "tool_result":
                                _ok_icon = "✅" if _evt_data.get("success") else "❌"
                                _sf_progress_lines.append(
                                    f"{_ok_icon} **工具结果**: {str(_evt_data.get('result',''))[:80]}"
                                )
                            elif _evt_type == "reflection":
                                _sf_progress_lines.append(
                                    f"🔍 **反思**: {_evt_data.get('analysis','')}"
                                )
                            elif _evt_type == "progress":
                                _sf_progress_lines.append(
                                    f"⏳ {_evt_data.get('message','')}"
                                )
                            elif _evt_type == "error":
                                _sf_progress_lines.append(
                                    f"⚠️ **错误**: {_evt_data.get('message', _evt_data.get('reason',''))}"
                                )
                            elif _evt_type == "complete":
                                _final_resp = _evt_data.get("response", "")
                                _iters = _evt_data.get("iterations", 0)
                                _dur = _evt_data.get("duration", 0)
                                _sf_progress_lines.append(
                                    f"\n---\n✅ **完成** | 迭代: {_iters} 轮 | 耗时: {_dur:.1f}s"
                                )
                                if _final_resp:
                                    _sf_progress_lines.append(f"\n### 最终结果\n{_final_resp}")
                                execution_log.append({
                                    "iteration": _iters,
                                    "type": "streaming_complete",
                                    "duration": _dur,
                                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                })
                            # 每次事件后实时刷新 UI
                            history[-1][1] = "\n".join(_sf_progress_lines)
                            response_started = True
                            yield history, mode_display_text
                    except Exception as e:
                        print(f"⚠️ 流式模式异常，降级为工具模式: {e}")
                        _use_streaming = False  # 降级
                        execution_log.append({
                            "type": "streaming_error",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        })

                # 分支 B2：普通工具模式 → agent_framework 完整工具循环
                if not _use_streaming:
                    try:
                        response, execution_log_fw, runtime_context = agent_framework.process_message(
                            user_message + pdf_info,
                            chat_history,
                            system_prompt_override=sys_prompt,
                            runtime_context=runtime_context,
                            return_runtime_context=True,
                            temperature=temp,
                            top_p=top_p_val,
                            max_tokens=max_tok,
                        )
                        execution_log.extend(execution_log_fw)
                        history[-1][1] = response
                        response_started = True
                        yield history, mode_display_text
                    except Exception as e:
                        error_msg = f"[⚠️ 工具模式错误] {str(e)}"
                        execution_log.append({
                            "iteration": 0,
                            "type": "runtime_error",
                            "content": error_msg,
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        })
                        # 降级：直接调用模型（携带已注入的 skill 内容）
                        for text_chunk in get_active_agent().generate_stream_with_messages(
                                base_messages, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                        ):
                            history[-1][1] = error_msg + "\n\n" + text_chunk
                            response_started = True
                            yield history, mode_display_text

            # 分支 C：纯对话 / Skills 知识模式
            elif not needs_breakdown:
                # 纯对话 / Skills 知识模式：经过完整中间件链（PlanMode / RuntimeMode 等）后调用模型
                # 使用 process_message_direct_stream 替代直通 generate_stream_with_messages，
                # 确保 plan_mode、runtime_mode 等中间件提示统一注入，不再需要手动打补丁
                # stream_forward_fn 提供真流式输出（逐 token），提升用户体验
                for response_chunk, _ in agent_framework.process_message_direct_stream(
                    base_messages,
                    system_prompt_override=sys_prompt,
                    runtime_context=runtime_context,
                    stream_forward_fn=dynamic_stream_forward,
                    temperature=temp,
                    top_p=top_p_val,
                    max_tokens=max_tok,
                ):
                    history[-1][1] = response_chunk
                    response_started = True
                    yield history, mode_display_text
                execution_log.append({
                    "iteration": 0,
                    "type": "model_response",
                    "content": history[-1][1] if history[-1][1] else "",
                    "run_mode": run_mode,
                    "plan_mode": runtime_context.get("plan_mode", False),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })

            # ---- 5. 记录对话到日志 ----
            if response_started and history[-1][1]:
                execution_time = time.time() - start_time
                _cur_agent = _active_agent.get("instance")
                _agent_type = _active_agent.get("type", "local")
                if _cur_agent is not None and hasattr(_cur_agent, "model"):
                    # GLMAgent / QwenAgent 均有 model 属性，直接取
                    _cur_model_name = _cur_agent.model
                elif _agent_type == "glm":
                    # agent 实例不存在但类型是 glm，记录为 glm-4-flash
                    _cur_model_name = "glm-4-flash"
                else:
                    _cur_model_name = "Qwen2.5-0.5B"
                logger.log_message(
                    user_message=user_message,
                    bot_response=history[-1][1],
                    execution_time=execution_time,
                    tokens_used=0,
                    model=_cur_model_name,
                    runtime_context=runtime_context,
                    execution_log=execution_log,
                )

        # ===== GLM 引擎切换事件 =====
        def on_engine_change(engine_choice):
            """切换引擎（本地 / GLM）。"""
            is_glm = "GLM" in engine_choice
            if is_glm:
                if not _GLM_AUTO_ENABLED:
                    return "⚪ 未配置（请设置环境变量 GLM_API_KEY）"
                # 切回 GLM（用当前已选模型）
                current_model = _active_agent["instance"].model if (
                    _active_agent.get("instance") and _active_agent["type"] == "glm"
                ) else "glm-4-flash"
                if _active_agent["type"] != "glm":
                    try:
                        _active_agent["instance"] = GLMAgent(
                            api_key=_GLM_API_KEY,
                            model=current_model,
                            logger=logger,
                        )
                        _active_agent["type"] = "glm"
                    except Exception as e:
                        return f"❌ 切换失败: {e}"
                return f"✅ 已启用（模型: {current_model}）"
            else:
                # 切换到本地模式（懒加载，首次对话时才真正加载模型）
                _active_agent["instance"] = None
                _active_agent["type"] = "local"
                return "⚪ 本地 Qwen 模式（将在首次对话时加载模型）"

        def on_model_change(model_name):
            """切换 GLM 模型版本（使用已有 Key 重建 Agent）。"""
            if not _GLM_AUTO_ENABLED:
                return "⚪ 未配置 GLM_API_KEY，无法切换"
            try:
                _active_agent["instance"] = GLMAgent(
                    api_key=_GLM_API_KEY,
                    model=model_name,
                    logger=logger,
                )
                _active_agent["type"] = "glm"
                return f"✅ 已切换模型: {model_name}"
            except Exception as e:
                return f"❌ 切换失败: {e}"

        # 绑定引擎切换
        model_engine.change(
            on_engine_change,
            inputs=[model_engine],
            outputs=[glm_status],
        )

        # 绑定 GLM 模型版本切换
        glm_model_choice.change(
            on_model_change,
            inputs=[glm_model_choice],
            outputs=[glm_status],
        )

        # 绑定对话事件（IntentRouter 已内置自动路由，无需传入模式 checkbox）
        msg.submit(user_input, [msg, chatbot], [msg, chatbot]).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, plan_mode, pdf_file],
            [chatbot, current_mode_display]
        )

        send_btn.click(user_input, [msg, chatbot], [msg, chatbot]).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, plan_mode, pdf_file],
            [chatbot, current_mode_display]
        )

    return demo


if __name__ == "__main__":
    if _env_file.exists():
        print(f"✅ 已加载环境变量文件: {_env_file}")

    print("🚀 正在启动 Qwen2.5 Agent...")

    try:
        import gradio_client.utils as gcu
        original_internal = gcu._json_schema_to_python_type

        def safe_json_schema_to_python_type(schema, defs=None):
            try:
                if isinstance(schema, bool):
                    return "bool"
                if not isinstance(schema, dict):
                    return "unknown"
                return original_internal(schema, defs)
            except TypeError as e:
                if "argument of type 'bool' is not iterable" in str(e):
                    return "any"
                raise

        gcu._json_schema_to_python_type = safe_json_schema_to_python_type
        print("✅ 已应用 Gradio JSON Schema 安全修补")
    except Exception as e:
        print(f"⚠️  JSON Schema 修补过程中出错: {e}")

    demo = create_ui_with_skills()

    import socket

    def find_free_port(start=7860, attempts=10):
        for port in range(start, start + attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result != 0:
                    return port
            except:
                pass
        return 7860

    port = find_free_port()
    print(f"✅ 使用端口: {port}")

    os.environ['GRADIO_DISABLE_API'] = '1'

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        share=False,
        show_error=True,
        show_api=False
    )