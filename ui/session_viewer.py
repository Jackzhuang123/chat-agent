#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会话日志查看器 - 提供一个专门的Gradio界面用于查看和分析对话日志
"""

import json

import gradio as gr

from session_logger import get_logger


def create_session_viewer():
    """
    创建会话日志查看器界面

    Returns:
        demo: Gradio Blocks 应用
    """
    logger = get_logger()

    # 自定义CSS
    custom_css = """
    <style>
    .session-list-item {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 8px 0;
        background: #f9fafb;
        cursor: pointer;
        transition: all 0.2s;
    }

    .session-list-item:hover {
        background: #f0f1f3;
        border-color: #d1d5db;
    }

    .session-stat {
        display: inline-block;
        margin-right: 16px;
        font-size: 13px;
    }

    .call-trace {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Monaco', 'Consolas', 'SF Mono', monospace;
        font-size: 12px;
        line-height: 1.6;
        overflow-x: auto;
    }

    .message-item {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }

    .user-msg {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
    }

    .bot-msg {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        margin: 8px;
    }

    .stat-value {
        font-size: 24px;
        font-weight: bold;
        margin: 8px 0;
    }

    .stat-label {
        font-size: 12px;
        opacity: 0.9;
    }
    </style>
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Qwen2.5 会话日志", head=f"<head>{custom_css}</head>") as demo:
        gr.Markdown("# 📋 会话日志查看器")
        gr.Markdown("查看、分析和导出所有对话会话记录")

        with gr.Tabs():
            # ===== Tab 1: 会话列表 =====
            with gr.TabItem("📁 会话列表"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新列表", variant="primary")
                    delete_session_btn = gr.Button("🗑️ 删除选中会话", variant="stop")

                sessions_table = gr.Dataframe(
                    headers=["会话ID", "创建时间", "消息数", "调用数", "总耗时(s)", "Token数"],
                    interactive=False,
                    label="所有会话"
                )

                selected_session_id = gr.Textbox(visible=False)

                def refresh_sessions():
                    """刷新会话列表"""
                    sessions = logger.get_all_sessions()
                    data = []
                    for session in sessions:
                        data.append([
                            session["session_id"],
                            session["created_at"],
                            session["message_count"],
                            session["call_count"],
                            f"{session['total_duration']:.2f}",
                            session["total_tokens"]
                        ])
                    return data

                def on_session_select(evt):
                    """选中会话时的回调"""
                    if evt and len(evt.index) > 0:
                        row_idx = evt.index[0]
                        sessions = logger.get_all_sessions()
                        if row_idx < len(sessions):
                            return sessions[row_idx]["session_id"]
                    return ""

                # 会话表格选择事件
                sessions_table.select(
                    on_session_select,
                    outputs=selected_session_id
                )

                refresh_btn.click(refresh_sessions, outputs=sessions_table)

                # 初始化表格
                refresh_sessions()

            # ===== Tab 2: 会话详情 =====
            with gr.TabItem("🔍 会话详情"):
                gr.Markdown("### 会话统计信息")

                with gr.Row():
                    stat_msgs = gr.HTML()
                    stat_calls = gr.HTML()
                    stat_duration = gr.HTML()
                    stat_tokens = gr.HTML()

                gr.Markdown("### 对话记录")
                messages_html = gr.HTML(label="消息记录")

                gr.Markdown("### 调用追踪")
                calls_html = gr.HTML(label="技能调用记录")

                # 详情页面初始化文本
                view_session_btn = gr.Button("查看选中会话详情", variant="primary")

                def show_session_details(session_id: str):
                    """显示会话详情"""
                    if not session_id:
                        return "请先从会话列表中选择一个会话", "", "", "", "", ""

                    session_data = logger.get_session_details(session_id)
                    if not session_data:
                        return "会话不存在", "", "", "", "", ""

                    # 统计信息
                    stats = session_data["statistics"]

                    msg_stat = f"""
                    <div class="stat-card">
                        <div class="stat-label">对话条数</div>
                        <div class="stat-value">{stats['total_messages']}</div>
                    </div>
                    """

                    call_stat = f"""
                    <div class="stat-card">
                        <div class="stat-label">技能调用</div>
                        <div class="stat-value">{stats['total_calls']}</div>
                    </div>
                    """

                    duration_stat = f"""
                    <div class="stat-card">
                        <div class="stat-label">总耗时(秒)</div>
                        <div class="stat-value">{stats['total_duration']:.2f}</div>
                    </div>
                    """

                    tokens_stat = f"""
                    <div class="stat-card">
                        <div class="stat-label">总Token数</div>
                        <div class="stat-value">{stats['total_tokens_used']}</div>
                    </div>
                    """

                    # 对话消息HTML
                    messages_html_str = "<div>"
                    for msg in session_data["messages"]:
                        timestamp = msg["timestamp"]
                        user_msg = msg["user_message"]
                        bot_response = msg["bot_response"][:200] + "..." if len(msg["bot_response"]) > 200 else msg["bot_response"]
                        exec_time = msg["execution_time"]
                        tokens = msg["tokens_used"]
                        model_calls = msg.get("model_calls", [])

                        messages_html_str += f"""
                        <div class="message-item">
                            <div style="color: #666; font-size: 12px;">{timestamp}</div>
                            <div class="user-msg" style="margin-top: 8px; padding: 8px;">
                                <strong>👤 用户:</strong> {user_msg}
                            </div>
                            <div class="bot-msg" style="margin-top: 8px; padding: 8px;">
                                <strong>🤖 助手:</strong> {bot_response}
                            </div>
                            <div style="color: #999; font-size: 12px; margin-top: 8px;">
                                ⏱️ {exec_time:.2f}s | 🔤 {tokens} tokens
                            </div>
                        """

                        # 显示模型调用详情
                        if model_calls:
                            messages_html_str += f"""
                            <div style="margin-top: 12px; border-top: 1px solid #e5e7eb; padding-top: 8px;">
                                <strong style="color: #667eea;">🔍 模型调用详情 ({len(model_calls)} 次)</strong>
                            """
                            for idx, call in enumerate(model_calls, 1):
                                call_prompt = call.get("prompt", "")[:150] + "..." if len(call.get("prompt", "")) > 150 else call.get("prompt", "")
                                call_response = call.get("response", "")[:150] + "..." if len(call.get("response", "")) > 150 else call.get("response", "")
                                call_exec_time = call.get("execution_time", 0)
                                call_tokens_input = call.get("tokens_input", 0)
                                call_tokens_output = call.get("tokens_output", 0)
                                call_temp = call.get("parameters", {}).get("temperature", 0)
                                call_top_p = call.get("parameters", {}).get("top_p", 0)

                                messages_html_str += f"""
                                <div style="margin-top: 8px; padding: 8px; background: #f0f4ff; border-radius: 4px;">
                                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">
                                        <strong>调用 #{idx}</strong> | {call.get("model", "Unknown")} | ⏱️ {call_exec_time:.2f}s
                                    </div>
                                    <div style="font-size: 11px; color: #999;">
                                        📥 输入: {call_tokens_input} tokens | 📤 输出: {call_tokens_output} tokens |
                                        🌡️ T:{call_temp} | TopP:{call_top_p}
                                    </div>
                                    <div style="font-size: 11px; background: #fff; padding: 4px; border-radius: 2px; margin-top: 4px; color: #666;">
                                        <strong>提示词:</strong> {call_prompt}
                                    </div>
                                    <div style="font-size: 11px; background: #fff; padding: 4px; border-radius: 2px; margin-top: 4px; color: #10b981;">
                                        <strong>输出:</strong> {call_response}
                                    </div>
                                </div>
                                """
                            messages_html_str += "</div>"

                        messages_html_str += "</div>"
                    messages_html_str += "</div>"

                    # 调用追踪HTML
                    calls_html_str = "<div>"
                    for call in session_data["calls"]:
                        timestamp = call["timestamp"]
                        skill_name = call["skill_name"]
                        status = "✅ 成功" if call["status"] == "success" else "❌ 失败"
                        exec_time = call["execution_time"]

                        calls_html_str += f"""
                        <div class="call-trace">
                            <div style="font-weight: bold; color: #667eea;">{skill_name} {status}</div>
                            <div style="color: #666; margin: 4px 0;">时间: {timestamp}</div>
                            <div style="color: #666; margin: 4px 0;">耗时: {exec_time:.2f}s</div>
                            <div style="margin-top: 8px; background: #fff; padding: 8px; border-radius: 4px;">
                                <pre style="margin: 0;">{json.dumps(call.get("output", {}), ensure_ascii=False, indent=2)}</pre>
                            </div>
                        </div>
                        """
                    calls_html_str += "</div>"

                    return msg_stat, call_stat, duration_stat, tokens_stat, messages_html_str, calls_html_str

                view_session_btn.click(
                    show_session_details,
                    inputs=selected_session_id,
                    outputs=[stat_msgs, stat_calls, stat_duration, stat_tokens, messages_html, calls_html]
                )

            # ===== Tab 3: 数据导出 =====
            with gr.TabItem("💾 数据导出"):
                gr.Markdown("### 导出会话数据")

                export_format = gr.Radio(
                    choices=["JSON"],
                    value="JSON",
                    label="导出格式"
                )

                with gr.Row():
                    export_btn = gr.Button("📥 导出会话", variant="primary")
                    export_msg = gr.Textbox(
                        label="导出结果",
                        interactive=False
                    )

                gr.Markdown("### 批量操作")

                with gr.Row():
                    clear_logs_btn = gr.Button("🗑️ 清空所有日志", variant="stop")
                    clear_msg = gr.Textbox(
                        label="操作结果",
                        interactive=False
                    )

                def export_session(session_id: str):
                    """导出会话"""
                    if not session_id:
                        return "❌ 请先从会话列表中选择一个会话"

                    export_path = f"./exported_session_{session_id}.json"
                    if logger.export_session(session_id, export_path):
                        return f"✅ 会话已导出到: {export_path}"
                    else:
                        return "❌ 导出失败"

                def clear_all_logs():
                    """清空所有日志"""
                    sessions = logger.get_all_sessions()
                    deleted_count = 0
                    for session in sessions:
                        if logger.delete_session(session["session_id"]):
                            deleted_count += 1
                    return f"✅ 已删除 {deleted_count} 个会话"

                export_btn.click(
                    export_session,
                    inputs=selected_session_id,
                    outputs=export_msg
                )

                clear_logs_btn.click(
                    clear_all_logs,
                    outputs=clear_msg
                )

    return demo


if __name__ == "__main__":
    import socket

    def find_free_port(start=7861, attempts=10):
        for port in range(start, start + attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result != 0:
                    return port
            except:
                pass
        return 7861

    port = find_free_port()
    print(f"✅ 会话查看器运行在端口 {port}")

    demo = create_session_viewer()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        share=False,
        show_error=True
    )

