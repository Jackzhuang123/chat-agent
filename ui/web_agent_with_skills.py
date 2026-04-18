#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B Agent Web UI - 豆包风格完美版（恢复原始气泡样式，优化按钮布局）
支持本地 Qwen 模型 和 智谱 GLM-4-Flash API（免费）双引擎切换。
"""
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，使 core 模块可被正确导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.monitor_logger import get_monitor_logger, log_startup, log_shutdown,set_log_level

import gradio as gr

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

from ui.chat_controller import ChatController

def create_ui_with_skills():
    controller = ChatController()

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
                    _init_engine_value = "⚡ GLM-4-Flash" if controller._GLM_AUTO_ENABLED else "🏠 本地 Qwen2.5-0.5B"
                    model_engine = gr.Radio(
                        choices=["🏠 本地 Qwen2.5-0.5B", "⚡ GLM-4-Flash"],
                        value=_init_engine_value,
                        label=None,
                        show_label=False,
                        interactive=controller._GLM_AUTO_ENABLED,
                    )

                    # GLM 状态面板
                    _init_glm_status = (
                        f"✅ 已启用（模型: glm-4-flash）"
                        if controller._GLM_AUTO_ENABLED
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
                            visible=controller._GLM_AUTO_ENABLED,
                        )

                    # 数据分析链接
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
                    current_mode_display = gr.Textbox(
                        value="💬 等待输入...",
                        label="当前模式",
                        interactive=False,
                        lines=2,
                    )
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
                        for s in controller.skill_manager.get_skills_list()
                    ) or "（暂无技能）"
                    gr.Textbox(
                        value=_skill_list_text,
                        label="已发现技能（自动加载）",
                        interactive=False,
                        lines=max(2, len(controller.skill_manager.get_skills_list())),
                    )

            # 主内容区
            with gr.Column(elem_id="main-content-area", scale=1):
                # 顶部栏
                with gr.Row(elem_id="top-bar-area"):
                    gr.HTML("<button id='toggle-btn'>◀ 收起侧边栏</button>")
                    gr.Markdown("## 💬 Chatbot")

                # 对话区域
                with gr.Column(elem_id="chat-wrapper"):
                    chatbot = gr.Chatbot(
                        label=None,
                        show_copy_button=True,
                        show_label=False,
                        container=False,
                        elem_classes="chatbot-box"
                    )

                    with gr.Column(elem_id="input-area-fixed"):
                        with gr.Row(elem_classes="input-row-container"):
                            msg = gr.Textbox(
                                label=None,
                                placeholder="请输入问题或需求",
                                lines=1,
                                max_lines=6,
                                show_label=False,
                                container=False,
                                elem_classes="input-text-area"
                            )
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

        # ========== 事件处理 ==========
        def user_input(user_message, history):
            return "", history + [[user_message, None]]

        model_engine.change(
            controller.on_engine_change,
            inputs=[model_engine],
            outputs=[glm_status],
        )
        glm_model_choice.change(
            controller.on_model_change,
            inputs=[glm_model_choice],
            outputs=[glm_status],
        )

        msg.submit(user_input, [msg, chatbot], [msg, chatbot]).then(
            controller.handle_message_with_workflow_resume,
            [chatbot, system_prompt, temperature, top_p, max_tokens, plan_mode, pdf_file],
            [chatbot, current_mode_display]
        )
        send_btn.click(user_input, [msg, chatbot], [msg, chatbot]).then(
            controller.handle_message_with_workflow_resume,
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

    monitor = get_monitor_logger()
    port = find_free_port()
    log_startup("Web Agent with Skills", port=port)
    monitor.info(f"启动 Gradio 服务，端口 {port}")

    try:
        set_log_level("DEBUG")  # 开启 DEBUG 级别日志
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            inbrowser=True,
            share=False,
            show_error=True,
            show_api=False
        )
    except Exception as e:
        monitor.exception("Gradio 服务启动失败")
        raise
    finally:
        log_shutdown("Web Agent with Skills")