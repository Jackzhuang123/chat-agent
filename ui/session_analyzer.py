#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级会话分析页面 - 提供可视化、JSON编辑和深度分析功能
"""

import json

import gradio as gr

from session_logger import get_logger


def create_session_analyzer():
    """
    创建高级会话分析界面

    优化改进：
    1. ✅ 移除手动刷新按钮 - 会话总览页面自动加载
    2. ✅ 移除会话详情下拉菜单 - 直接显示最新会话，每5秒自动刷新
    3. ✅ 会话列表项可点击 - 点击任意会话可查看详情
    4. ✅ 移除数据导出tab - 简化界面
    5. ✅ 自动显示最新会话 - 无需手动选择

    Returns:
        demo: Gradio Blocks 应用
    """
    logger = get_logger()

    # 自定义CSS - 增强可视化
    custom_css = """
    <script>
    // 全局函数用于选择会话并切换到详情Tab
    window.selectSession = function(sessionId) {
        try {
            console.log('Selecting session:', sessionId);

            // 1. 找到隐藏的textbox元素并设置值
            const selector = document.getElementById('session_id_selector');
            if (selector) {
                console.log('Found selector element');

                // 查找textarea或input元素
                const input = selector.querySelector('textarea') || selector.querySelector('input');
                if (input) {
                    console.log('Found input element, setting value:', sessionId);
                    input.value = sessionId;

                    // 触发input事件让Gradio检测到值的变化
                    const inputEvent = new Event('input', { bubbles: true });
                    input.dispatchEvent(inputEvent);

                    // 延迟后触发change事件
                    setTimeout(() => {
                        const changeEvent = new Event('change', { bubbles: true });
                        input.dispatchEvent(changeEvent);
                        console.log('Dispatched change event');
                    }, 100);
                } else {
                    console.log('No input element found');
                }
            } else {
                console.log('No selector element found');
            }

            // 2. 点击刷新按钮以加载会话详情
            setTimeout(() => {
                const refreshBtn = document.getElementById('refresh_session_btn');
                if (refreshBtn) {
                    console.log('Found refresh button, clicking it');
                    const btn = refreshBtn.querySelector('button');
                    if (btn) {
                        btn.click();
                        console.log('Clicked refresh button');
                    }
                } else {
                    console.log('No refresh button found');
                }
            }, 150);

            // 3. 切换到"会话详情"Tab
            setTimeout(() => {
                // 查找所有的tab按钮
                const tabButtons = document.querySelectorAll('[role="tab"]');
                console.log('Found', tabButtons.length, 'tab buttons');

                // 遍历找到包含"会话详情"的tab
                for (let btn of tabButtons) {
                    const text = btn.textContent || btn.innerText;
                    console.log('Tab text:', text);
                    if (text.includes('会话详情')) {
                        console.log('Found session detail tab, clicking it');
                        btn.click();
                        break;
                    }
                }
            }, 300);

        } catch(e) {
            console.error('Error selecting session:', e);
        }
    };
    </script>
    <style>
    /* ========== 全局样式 ========== */
    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    /* ========== 会话统计卡片 ========== */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transition: all 0.3s;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }

    .stat-card.alt1 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .stat-card.alt2 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .stat-card.alt3 {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .stat-card.alt4 {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .stat-value {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }

    .stat-label {
        font-size: 13px;
        opacity: 0.9;
        letter-spacing: 0.5px;
    }

    /* ========== 会话表格 ========== */
    .session-item {
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 10px 0;
        background: #fff;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .session-item:hover {
        background: #f9fafb;
        border-color: #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .session-info {
        flex: 1;
    }

    .session-id {
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 13px;
        color: #667eea;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .session-meta {
        font-size: 12px;
        color: #999;
        margin: 2px 0;
    }

    .session-actions {
        display: flex;
        gap: 8px;
    }

    /* ========== JSON查看器 ========== */
    .json-container {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 16px;
        border-radius: 8px;
        font-family: 'Fira Code', 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-size: 12px;
        line-height: 1.6;
        overflow-x: auto;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .json-key {
        color: #9cdcfe;
    }

    .json-string {
        color: #ce9178;
    }

    .json-number {
        color: #b5cea8;
    }

    .json-boolean {
        color: #569cd6;
    }

    .json-null {
        color: #569cd6;
    }

    /* ========== 对话气泡 ========== */
    .message-bubble {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid;
    }

    .user-bubble {
        background: #eff6ff;
        border-left-color: #3b82f6;
    }

    .bot-bubble {
        background: #f0fdf4;
        border-left-color: #10b981;
    }

    .bubble-header {
        font-weight: 600;
        margin-bottom: 6px;
        font-size: 13px;
    }

    .bubble-content {
        font-size: 13px;
        line-height: 1.5;
        color: #374151;
    }

    .bubble-meta {
        font-size: 11px;
        color: #999;
        margin-top: 8px;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        padding-top: 6px;
    }

    /* ========== 模型调用详情 ========== */
    .model-call-item {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }

    .model-call-header {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .model-call-params {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 8px;
        font-size: 12px;
        margin: 8px 0;
        background: #fff;
        padding: 8px;
        border-radius: 4px;
    }

    .param-item {
        padding: 4px;
        background: #f3f4f6;
        border-radius: 3px;
    }

    .param-label {
        color: #666;
        font-size: 11px;
    }

    .param-value {
        color: #667eea;
        font-weight: 600;
        margin-top: 2px;
    }

    .code-block {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 8px;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 12px;
        overflow-x: auto;
        margin: 4px 0;
    }

    /* ========== 标签页标题 ========== */
    .tab-label {
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ========== 饼图和图表 ========== */
    .chart-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    /* ========== 按钮 ========== */
    .btn-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    /* ========== 搜索框 ========== */
    .search-box {
        display: flex;
        gap: 8px;
        margin: 16px 0;
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

    /* ========== 响应式 ========== */
    @media (max-width: 768px) {
        .stat-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .session-item {
            flex-direction: column;
            align-items: flex-start;
        }

        .session-actions {
            width: 100%;
            margin-top: 8px;
        }
    }
    </style>
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="🔬 高级会话分析", head=f"<head>{custom_css}</head>") as demo:
        gr.Markdown("# 🔬 高级会话分析")
        gr.Markdown("深度分析、可视化和JSON编辑会话数据")

        # 全局状态 - 存储选中的session_id，用于在tabs之间共享
        selected_session_id = gr.State(value="")

        # 隐藏的状态变量用于传递session_id
        selected_session_state = gr.State(value=None)

        with gr.Tabs() as tabs:
            # ===== Tab 1: 会话总览 =====
            with gr.TabItem("📊 会话总览"):
                # 统计卡片容器
                with gr.Row():
                    total_sessions_html = gr.HTML()
                    total_messages_html = gr.HTML()
                    avg_duration_html = gr.HTML()
                    total_tokens_html = gr.HTML()

                gr.Markdown("### 📋 最近的会话")
                sessions_html = gr.HTML()

                def get_all_sessions_info():
                    """获取所有会话信息"""
                    sessions = logger.get_all_sessions()

                    total_sessions = len(sessions)
                    total_messages = sum(s["message_count"] for s in sessions)
                    avg_duration = sum(s["total_duration"] for s in sessions) / max(1, total_sessions)
                    total_tokens = sum(s["total_tokens"] for s in sessions)

                    # 生成统计卡片
                    total_sessions_card = f"""
                    <div class="stat-card">
                        <div class="stat-label">📁 总会话数</div>
                        <div class="stat-value">{total_sessions}</div>
                    </div>
                    """

                    total_messages_card = f"""
                    <div class="stat-card alt1">
                        <div class="stat-label">💬 总对话数</div>
                        <div class="stat-value">{total_messages}</div>
                    </div>
                    """

                    avg_duration_card = f"""
                    <div class="stat-card alt2">
                        <div class="stat-label">⏱️ 平均耗时(s)</div>
                        <div class="stat-value">{avg_duration:.2f}</div>
                    </div>
                    """

                    total_tokens_card = f"""
                    <div class="stat-card alt3">
                        <div class="stat-label">🔤 总Token数</div>
                        <div class="stat-value">{total_tokens}</div>
                    </div>
                    """

                    # 生成会话列表 - 添加点击交互
                    sessions_html_str = '<div style="display: flex; flex-direction: column; gap: 12px;">'

                    if not sessions:
                        sessions_html_str += '<div style="padding: 20px; text-align: center; color: #999; background: #f5f5f5; border-radius: 8px;">暂无会话记录</div>'
                    else:
                        for idx, session in enumerate(sessions[:10], 1):  # 显示最近10个
                            session_id = session["session_id"]
                            created_at = session["created_at"][:19]  # 去掉毫秒
                            msg_count = session["message_count"]
                            call_count = session["call_count"]
                            duration = session["total_duration"]
                            tokens = session["total_tokens"]

                            # 根据性能指标选择背景颜色
                            bg_color = "#f0f4ff" if idx % 2 == 0 else "#ffffff"

                            sessions_html_str += f"""
                            <div class="session-item" onclick="window.selectSession('{session_id}')" style="cursor: pointer; background: {bg_color}; padding: 12px; border-radius: 8px; border-left: 4px solid #667eea; transition: all 0.2s;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <div style="font-weight: 600; color: #667eea; font-size: 14px;">🆔 {session_id}</div>
                                    <div style="color: #999; font-size: 12px;">📅 {created_at}</div>
                                </div>
                                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 13px;">
                                    <div style="background: #e5e7ff; padding: 6px; border-radius: 4px; text-align: center;"><span style="color: #667eea; font-weight: 600;">{msg_count}</span><div style="color: #999; font-size: 11px;">对话</div></div>
                                    <div style="background: #e5f0ff; padding: 6px; border-radius: 4px; text-align: center;"><span style="color: #4facfe; font-weight: 600;">{call_count}</span><div style="color: #999; font-size: 11px;">调用</div></div>
                                    <div style="background: #fff0e5; padding: 6px; border-radius: 4px; text-align: center;"><span style="color: #fa709a; font-weight: 600;">{duration:.1f}s</span><div style="color: #999; font-size: 11px;">耗时</div></div>
                                    <div style="background: #e5ffe5; padding: 6px; border-radius: 4px; text-align: center;"><span style="color: #43e97b; font-weight: 600;">{tokens}</span><div style="color: #999; font-size: 11px;">Tokens</div></div>
                                </div>
                            </div>
                            """

                    sessions_html_str += '</div>'

                    return total_sessions_card, total_messages_card, avg_duration_card, total_tokens_card, sessions_html_str

                # 页面加载时自动获取会话信息
                demo.load(
                    get_all_sessions_info,
                    outputs=[total_sessions_html, total_messages_html, avg_duration_html, total_tokens_html, sessions_html]
                )

            # ===== Tab 2: 会话详情查看 =====
            with gr.TabItem("🔍 会话详情"):
                gr.Markdown("### 会话详情自动刷新")

                # 隐藏的session_id输入框，用于存储选中的会话ID
                current_session_id = gr.Textbox(
                    value="",
                    visible=False,
                    elem_id="session_id_selector"
                )

                # 隐藏的刷新按钮，用于手动触发会话详情刷新
                refresh_btn = gr.Button(visible=False, elem_id="refresh_session_btn")

                # 自动刷新定时器状态
                refresh_timer = gr.Textbox(visible=False, value="0")

                with gr.Tabs():
                    # ===== 会话摘要 =====
                    with gr.TabItem("📈 会话摘要"):
                        gr.Markdown("### 📊 统计信息")
                        with gr.Row():
                            summary_stats = gr.HTML()

                        gr.Markdown("### 💭 对话流")
                        messages_summary = gr.HTML()

                    # ===== 模型调用分析 =====
                    with gr.TabItem("🧠 模型调用分析"):
                        gr.Markdown("### 模型调用详情")
                        model_calls_detail = gr.HTML()

                    # ===== JSON原始数据 =====
                    with gr.TabItem("📋 JSON数据"):
                        gr.Markdown("### 原始JSON数据（可复制编辑）")
                        json_editor = gr.Code(
                            language="json",
                            label="JSON编辑器",
                            interactive=True,
                            lines=30
                        )

                        with gr.Row():
                            # 注意：gr.DownloadButton 在 gradio 4.44.1 + gradio_client 1.3.0
                            # 中会触发 json_schema_to_python_type 的版本兼容 bug，改用 gr.File
                            download_json_btn = gr.File(label="⬇️ 下载JSON", scale=1, visible=False)
                            copy_status = gr.Textbox(interactive=False, label="操作提示", scale=2)

                    # ===== 消息详情 =====
                    with gr.TabItem("💬 消息详情"):
                        gr.Markdown("### 所有对话消息")
                        all_messages_detail = gr.HTML()

                def load_session_details(session_id_input=None):
                    """加载会话详情 - 自动从最新会话获取或使用指定的session_id"""
                    # 如果没有指定session_id，获取最新的会话
                    if not session_id_input:
                        sessions = logger.get_all_sessions()
                        if not sessions:
                            return "", "", "", "", ""
                        session_id = sessions[0]["session_id"]  # 获取最新的会话
                    else:
                        session_id = session_id_input

                    session_data = logger.get_session_details(session_id)

                    if not session_data:
                        return "会话不存在", "", "", "", ""

                    stats = session_data["statistics"]

                    # 1. 摘要统计
                    summary_html = f"""
                    <div class="stat-grid">
                        <div class="stat-card">
                            <div class="stat-label">💬 对话数</div>
                            <div class="stat-value">{stats['total_messages']}</div>
                        </div>
                        <div class="stat-card alt1">
                            <div class="stat-label">🔧 调用数</div>
                            <div class="stat-value">{stats['total_calls']}</div>
                        </div>
                        <div class="stat-card alt2">
                            <div class="stat-label">⏱️ 总耗时(s)</div>
                            <div class="stat-value">{stats['total_duration']:.2f}</div>
                        </div>
                        <div class="stat-card alt3">
                            <div class="stat-label">🔤 总Token</div>
                            <div class="stat-value">{stats['total_tokens_used']}</div>
                        </div>
                    </div>
                    """

                    # 2. 对话流摘要
                    messages_html = ""
                    for idx, msg in enumerate(session_data["messages"], 1):
                        user_msg = msg["user_message"][:100] + "..." if len(msg["user_message"]) > 100 else msg["user_message"]
                        bot_response = msg["bot_response"][:100] + "..." if len(msg["bot_response"]) > 100 else msg["bot_response"]
                        exec_time = msg["execution_time"]
                        tokens = msg["tokens_used"]
                        model_calls_count = len(msg.get("model_calls", []))

                        messages_html += f"""
                        <div style="margin: 12px 0; padding: 12px; border-radius: 8px; background: #f9fafb; border-left: 4px solid #667eea;">
                            <div style="font-weight: 600; color: #667eea; margin-bottom: 6px;">💬 对话 #{idx}</div>
                            <div class="message-bubble user-bubble">
                                <div class="bubble-header">👤 用户:</div>
                                <div class="bubble-content">{user_msg}</div>
                            </div>
                            <div class="message-bubble bot-bubble">
                                <div class="bubble-header">🤖 助手:</div>
                                <div class="bubble-content">{bot_response}</div>
                            </div>
                            <div class="bubble-meta">
                                ⏱️ {exec_time:.2f}s | 🔤 {tokens} tokens | 🧠 {model_calls_count} 次模型调用
                            </div>
                        </div>
                        """

                    # 3. 模型调用详情
                    model_calls_html = ""
                    total_model_calls = sum(len(msg.get("model_calls", [])) for msg in session_data["messages"])

                    if total_model_calls > 0:
                        model_calls_html += f"""
                        <div style="margin-bottom: 16px; padding: 12px; background: #f0f4ff; border-radius: 8px; border-left: 4px solid #667eea;">
                            <div style="font-weight: 600; color: #667eea; margin-bottom: 8px;">🧠 总计 {total_model_calls} 次模型调用</div>
                        """

                        for msg_idx, msg in enumerate(session_data["messages"], 1):
                            model_calls = msg.get("model_calls", [])
                            if model_calls:
                                for call_idx, call in enumerate(model_calls, 1):
                                    prompt = call.get("prompt", "")[:150]
                                    response = call.get("response", "")[:150]
                                    exec_time = call.get("execution_time", 0)
                                    input_tokens = call.get("tokens_input", 0)
                                    output_tokens = call.get("tokens_output", 0)
                                    temp = call.get("parameters", {}).get("temperature", 0)
                                    top_p = call.get("parameters", {}).get("top_p", 0)

                                    model_calls_html += f"""
                                    <div class="model-call-item">
                                        <div class="model-call-header">
                                            <span>🔗 对话 {msg_idx} - 调用 {call_idx}</span>
                                            <span style="color: #999; font-size: 12px; font-weight: normal;">⏱️ {exec_time:.3f}s</span>
                                        </div>
                                        <div class="model-call-params">
                                            <div class="param-item">
                                                <div class="param-label">📥 输入Token</div>
                                                <div class="param-value">{input_tokens}</div>
                                            </div>
                                            <div class="param-item">
                                                <div class="param-label">📤 输出Token</div>
                                                <div class="param-value">{output_tokens}</div>
                                            </div>
                                            <div class="param-item">
                                                <div class="param-label">🌡️ Temperature</div>
                                                <div class="param-value">{temp}</div>
                                            </div>
                                            <div class="param-item">
                                                <div class="param-label">🎯 Top-P</div>
                                                <div class="param-value">{top_p}</div>
                                            </div>
                                        </div>
                                        <div style="margin-top: 8px;">
                                            <div style="font-size: 12px; font-weight: 600; color: #667eea; margin-bottom: 4px;">📥 提示词:</div>
                                            <div class="code-block">{prompt}</div>
                                        </div>
                                        <div style="margin-top: 8px;">
                                            <div style="font-size: 12px; font-weight: 600; color: #10b981; margin-bottom: 4px;">📤 输出:</div>
                                            <div class="code-block">{response}</div>
                                        </div>
                                    </div>
                                    """

                        model_calls_html += "</div>"

                    # 4. JSON原始数据
                    json_str = json.dumps(session_data, ensure_ascii=False, indent=2)

                    # 5. 详细消息信息
                    all_messages_html = ""
                    for idx, msg in enumerate(session_data["messages"], 1):
                        timestamp = msg["timestamp"]
                        user_msg = msg["user_message"]
                        bot_response = msg["bot_response"]
                        exec_time = msg["execution_time"]
                        tokens = msg["tokens_used"]
                        model_calls = msg.get("model_calls", [])

                        all_messages_html += f"""
                        <div style="margin: 16px 0; padding: 16px; border-radius: 8px; background: #fff; border: 1px solid #e5e7eb;">
                            <div style="font-weight: 600; color: #667eea; margin-bottom: 12px;">📌 消息 #{idx}</div>
                            <div style="color: #999; font-size: 12px; margin-bottom: 8px;">⏰ {timestamp}</div>

                            <div class="message-bubble user-bubble" style="margin: 8px 0;">
                                <div class="bubble-header">👤 用户消息:</div>
                                <div class="bubble-content" style="white-space: pre-wrap; word-break: break-word;">{user_msg}</div>
                            </div>

                            <div class="message-bubble bot-bubble" style="margin: 8px 0;">
                                <div class="bubble-header">🤖 助手回复:</div>
                                <div class="bubble-content" style="white-space: pre-wrap; word-break: break-word;">{bot_response}</div>
                            </div>

                            <div class="bubble-meta">
                                ⏱️ 执行时间: {exec_time:.3f}s | 🔤 Token数: {tokens} | 🧠 模型调用: {len(model_calls)} 次
                            </div>
                        """

                        # 显示该消息的所有模型调用
                        if model_calls:
                            all_messages_html += f"""
                            <div style="margin-top: 12px; padding: 12px; background: #f0f4ff; border-radius: 8px; border-left: 4px solid #667eea;">
                                <div style="font-weight: 600; color: #667eea; margin-bottom: 8px;">🔍 模型调用详情 ({len(model_calls)} 次)</div>
                            """
                            for call_idx, call in enumerate(model_calls, 1):
                                prompt = call.get("prompt", "")
                                response = call.get("response", "")
                                exec_time_call = call.get("execution_time", 0)
                                input_tokens = call.get("tokens_input", 0)
                                output_tokens = call.get("tokens_output", 0)
                                temp = call.get("parameters", {}).get("temperature", 0)
                                top_p = call.get("parameters", {}).get("top_p", 0)

                                all_messages_html += f"""
                                <div style="margin-top: 8px; padding: 8px; background: #fff; border-radius: 4px; border-left: 3px solid #667eea;">
                                    <div style="font-weight: 600; color: #667eea; margin-bottom: 6px; font-size: 12px;">调用 #{call_idx}</div>
                                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 12px; margin-bottom: 8px;">
                                        <div style="background: #f3f4f6; padding: 6px; border-radius: 3px;">
                                            <div style="color: #999; font-size: 11px;">📥 输入</div>
                                            <div style="color: #667eea; font-weight: 600;">{input_tokens} tokens</div>
                                        </div>
                                        <div style="background: #f3f4f6; padding: 6px; border-radius: 3px;">
                                            <div style="color: #999; font-size: 11px;">📤 输出</div>
                                            <div style="color: #667eea; font-weight: 600;">{output_tokens} tokens</div>
                                        </div>
                                        <div style="background: #f3f4f6; padding: 6px; border-radius: 3px;">
                                            <div style="color: #999; font-size: 11px;">🌡️ 温度</div>
                                            <div style="color: #667eea; font-weight: 600;">{temp}</div>
                                        </div>
                                        <div style="background: #f3f4f6; padding: 6px; border-radius: 3px;">
                                            <div style="color: #999; font-size: 11px;">⏱️ 耗时</div>
                                            <div style="color: #667eea; font-weight: 600;">{exec_time_call:.3f}s</div>
                                        </div>
                                    </div>
                                    <div style="background: #fafafa; padding: 6px; border-radius: 3px; margin-bottom: 6px;">
                                        <div style="color: #999; font-size: 11px; margin-bottom: 2px;">📝 提示词:</div>
                                        <div style="font-family: 'Monaco', 'Consolas', monospace; font-size: 11px; color: #666; white-space: pre-wrap; word-break: break-all;">{prompt}</div>
                                    </div>
                                    <div style="background: #fafafa; padding: 6px; border-radius: 3px;">
                                        <div style="color: #999; font-size: 11px; margin-bottom: 2px;">💬 输出:</div>
                                        <div style="font-family: 'Monaco', 'Consolas', monospace; font-size: 11px; color: #666; white-space: pre-wrap; word-break: break-all;">{response}</div>
                                    </div>
                                </div>
                                """

                            all_messages_html += "</div>"

                        all_messages_html += "</div>"

                    return summary_html, messages_html, model_calls_html, json_str, all_messages_html

                def prepare_json_download(json_content: str):
                    """将 JSON 内容写入临时文件，供 gr.File 下载组件使用。"""
                    import tempfile
                    if not json_content or json_content.strip() == "":
                        return gr.update(visible=False, value=None), "❌ 没有可下载的数据，请先选择一个会话"
                    try:
                        tmp = tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", encoding="utf-8",
                            delete=False, prefix="session_"
                        )
                        tmp.write(json_content)
                        tmp.flush()
                        tmp.close()
                        return gr.update(value=tmp.name, visible=True), "✅ 文件已准备好，点击下方文件名即可下载"
                    except Exception as e:
                        return gr.update(visible=False, value=None), f"❌ 准备下载失败: {e}"

                # 页面加载时和定时刷新
                demo.load(
                    load_session_details,
                    inputs=current_session_id,
                    outputs=[summary_stats, messages_summary, model_calls_detail, json_editor, all_messages_detail],
                    every=5  # 每5秒自动刷新一次
                )

                # 当用户从列表中选择会话时立即加载
                current_session_id.change(
                    load_session_details,
                    inputs=current_session_id,
                    outputs=[summary_stats, messages_summary, model_calls_detail, json_editor, all_messages_detail]
                )

                # 当点击刷新按钮时加载会话详情
                refresh_btn.click(
                    load_session_details,
                    inputs=current_session_id,
                    outputs=[summary_stats, messages_summary, model_calls_detail, json_editor, all_messages_detail]
                )

                # 下载 JSON 按钮（json_editor 内容变化时自动准备下载）
                json_editor.change(
                    prepare_json_download,
                    inputs=json_editor,
                    outputs=[download_json_btn, copy_status]
                )

    return demo


if __name__ == "__main__":
    import socket

    def find_free_port(start=7862, attempts=10):
        for port in range(start, start + attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result != 0:
                    return port
            except:
                pass
        return 7862

    port = find_free_port()
    print(f"✅ 高级分析页面运行在端口 {port}")

    demo = create_session_analyzer()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        share=False,
        show_error=True,
    )

