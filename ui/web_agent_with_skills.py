#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B Agent Web UI - 豆包风格完美版（恢复原始气泡样式，优化按钮布局）
"""

import sys
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import QwenAgentFramework, create_qwen_model_forward, SkillManager, SkillInjector, create_example_skills, \
    ToolExecutor
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

    def generate_stream(self, message, history, system_prompt=None, temperature=0.7, top_p=0.9, max_tokens=512):
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

    def generate_stream_with_messages(self, messages, temperature=0.7, top_p=0.9, max_tokens=512):
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
    qwen_agent = QwenAgent(logger=logger)
    tool_executor = ToolExecutor(enable_bash=False)

    print("🧪 初始化 Skills 系统...")
    create_example_skills()
    skill_manager = SkillManager()
    skill_injector = SkillInjector(skill_manager)
    print(f"✅ 发现 {len(skill_manager.skills_metadata)} 个技能")

    agent_framework = QwenAgentFramework(
        model_forward_fn=create_qwen_model_forward(qwen_agent),
        enable_bash=False, max_iterations=5, tools_in_system_prompt=True
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
                    gr.Markdown("### 🤖 Qwen2.5 Assistant")

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

                    gr.Markdown("#### 模式设置")
                    use_tools = gr.Checkbox(label="🔧 工具模式", value=False)
                    use_skills = gr.Checkbox(label="🎓 Skills 系统", value=True)

                    gr.Markdown("#### 模型参数")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                    max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")

                    gr.Markdown("#### 系统提示")
                    system_prompt = gr.Textbox(
                        value="你是一个智能助手。",
                        lines=3,
                        placeholder="自定义系统提示...",
                        show_label=False
                    )

                    gr.Markdown("#### 技能配置")
                    skills_selector = gr.Dropdown(
                        choices=[s["id"] for s in skill_manager.get_skills_list()],
                        multiselect=True,
                        label="选择技能"
                    )
                    auto_match_skills = gr.Checkbox(label="自动匹配", value=True)

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

        def bot_response(history, sys_prompt, temp, top_p_val, max_tok, use_tools_mode, use_skills_mode,
                         selected_skills, auto_match, pdf_files):
            if not history or history[-1][1] is not None:
                return history

            user_message = history[-1][0]
            history_without_last = history[:-1]
            logger = get_logger()

            # 创建新会话
            logger.create_session()

            pdf_info = ""
            if pdf_files and HAS_PYPDF:
                pdf_text = extract_pdf_text(pdf_files)
                if pdf_text:
                    pdf_info = f"\n\n【PDF内容】:\n{pdf_text[:1000]}..."

            chat_history = [[u, a] for u, a in history_without_last if u and a]

            use_custom_messages = False
            enhanced_messages = None
            skills_to_inject = []

            if use_skills_mode:
                if auto_match and not selected_skills:
                    matched_skills = skill_manager.find_skills_for_task(user_message + pdf_info)
                    skills_to_inject = [s["id"] for s in matched_skills[:3]]
                else:
                    skills_to_inject = selected_skills if selected_skills else []

                if skills_to_inject:
                    enhanced_messages = [{"role": "system", "content": sys_prompt}]
                    for u, a in chat_history:
                        enhanced_messages.append({"role": "user", "content": u})
                        enhanced_messages.append({"role": "assistant", "content": a})
                    enhanced_messages.append({"role": "user", "content": user_message + pdf_info})

                    enhanced_messages = skill_injector.inject_skills_to_context(
                        enhanced_messages, skills_to_inject, include_full_content=False
                    )
                    use_custom_messages = True

            response_started = False
            import time
            start_time = time.time()

            if use_tools_mode:
                try:
                    response, exec_log = agent_framework.process_message(
                        user_message + pdf_info, chat_history,
                        system_prompt_override=sys_prompt, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                    )
                    history[-1][1] = response
                    response_started = True
                    yield history
                except Exception as e:
                    error_msg = f"[⚠️ 工具模式错误] {str(e)}"

                    if use_custom_messages:
                        for text_chunk in qwen_agent.generate_stream_with_messages(
                                enhanced_messages, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                        ):
                            history[-1][1] = error_msg + "\n\n" + text_chunk
                            response_started = True
                            yield history
                    else:
                        for text_chunk in qwen_agent.generate_stream(
                                user_message + pdf_info, chat_history,
                                system_prompt=sys_prompt, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                        ):
                            history[-1][1] = error_msg + "\n\n" + text_chunk
                            response_started = True
                            yield history
            else:
                if use_custom_messages:
                    for response in qwen_agent.generate_stream_with_messages(
                            enhanced_messages, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                    ):
                        history[-1][1] = response
                        response_started = True
                        yield history
                else:
                    for response in qwen_agent.generate_stream(
                            user_message + pdf_info, chat_history,
                            system_prompt=sys_prompt, temperature=temp, top_p=top_p_val, max_tokens=max_tok
                    ):
                        history[-1][1] = response
                        response_started = True
                        yield history

            # 记录对话到日志
            if response_started and history[-1][1]:
                execution_time = time.time() - start_time
                logger.log_message(
                    user_message=user_message,
                    bot_response=history[-1][1],
                    execution_time=execution_time,
                    tokens_used=0
                )

        # 绑定事件
        msg.submit(user_input, [msg, chatbot], [msg, chatbot]).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, use_tools, use_skills,
             skills_selector, auto_match_skills, pdf_file],
            [chatbot]
        )

        send_btn.click(user_input, [msg, chatbot], [msg, chatbot]).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, use_tools, use_skills,
             skills_selector, auto_match_skills, pdf_file],
            [chatbot]
        )

    return demo


if __name__ == "__main__":
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

    import os
    os.environ['GRADIO_DISABLE_API'] = '1'

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        share=False,
        show_error=True,
        show_api=False
    )