#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B Agent Web UI - 终极版 (含 Skills)
基于 Gradio 构建的 AI 助手界面,支持工具调用、技能系统和参数自定义

新增特性:
  - 🎓 Skills 系统 (知识外置化)
  - 📚 技能库管理
  - 🔍 智能技能匹配
  - 💉 上下文注入 (保留缓存)
"""

import json
import sys
from pathlib import Path
from threading import Thread
from typing import List

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import QwenAgentFramework, create_qwen_model_forward, SkillManager, SkillInjector, create_example_skills, ToolExecutor


# ============================================================================
# 核心类: QwenAgent - 负责模型加载和响应生成
# ============================================================================

class QwenAgent:
    """
    Qwen 模型代理类
    负责模型加载、初始化和流式生成响应
    """

    def __init__(self, model_path="./model/qwen2.5-0.5b"):
        """初始化模型"""
        print("正在初始化模型 (CPU模式)...")
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.default_system_prompt = "你是一个智能个人助手,名字叫小Q。请用简洁、幽默的风格回答。"
        print("✅ 模型加载完毕!")

    def generate_stream(self, message, history, system_prompt=None, temperature=0.7, top_p=0.9, max_tokens=512):
        """流式生成响应"""
        sys_prompt = system_prompt if system_prompt and system_prompt.strip() else self.default_system_prompt
        messages = [{"role": "system", "content": sys_prompt}]

        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

    def generate_stream_text(self, messages: List[dict], temperature=0.7, top_p=0.9, max_tokens=512):
        """直接从消息列表生成流式文本"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message


# ============================================================================
# UI 创建函数 - 使用 Gradio Blocks 构建自定义界面
# ============================================================================

def create_ui_with_skills():
    """创建支持 Skills 的 Web UI"""

    # 初始化组件
    qwen_agent = QwenAgent()
    tool_executor = ToolExecutor(enable_bash=False)

    # 初始化 Skills 系统
    print("🧪 初始化 Skills 系统...")
    create_example_skills()  # 创建示例技能
    skill_manager = SkillManager()
    skill_injector = SkillInjector(skill_manager)

    print(f"✅ 发现 {len(skill_manager.skills_metadata)} 个技能")

    # 初始化 Agent 框架
    agent_framework = QwenAgentFramework(
        model_forward_fn=create_qwen_model_forward(qwen_agent),
        enable_bash=False,
        max_iterations=5,
        tools_in_system_prompt=True
    )

    # 自定义 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .skill-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 12px;
    }
    .skill-info {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    footer {
        visibility: hidden;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Qwen2.5 Agent (终极版)") as demo:

        # 标题
        gr.Markdown("""
            # 🤖 Qwen2.5-0.5B Agent 终极版 (含 Skills)

            这是一个功能最完整的 AI 助手:
            - 🔧 **工具调用**: 文件操作、目录探索
            - 🎓 **Skills 系统**: 按需加载领域知识
            - 💬 **多轮对话**: 完整的对话历史
            - 📚 **技能库**: 自动技能匹配和注入
        """)

        # 主布局
        with gr.Row():
            # 左侧: 对话区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话窗口",
                    height=500,
                    show_copy_button=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="输入消息",
                        placeholder="在这里输入你的问题...",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("发送 📤", variant="primary", scale=1)

                with gr.Row():
                    retry_btn = gr.Button("🔄 重试")
                    undo_btn = gr.Button("↩️ 撤销")
                    clear_btn = gr.Button("🗑️ 清空")

                with gr.Row():
                    use_tools = gr.Checkbox(label="启用工具模式", value=False)
                    use_skills = gr.Checkbox(label="启用 Skills", value=True)
                    show_execution_log = gr.Checkbox(label="显示执行日志", value=False)

                # 技能建议
                with gr.Accordion("📚 可用技能", open=False):
                    skills_display = gr.Textbox(
                        label="技能列表",
                        value=_get_skills_info(skill_manager),
                        lines=10,
                        interactive=False
                    )

                # 执行日志
                with gr.Accordion("📋 执行日志", open=False):
                    execution_log = gr.Textbox(
                        label="执行记录",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )

            # 右侧: 设置区
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 高级设置")

                system_prompt = gr.Textbox(
                    label="系统提示词",
                    value="你是一个智能个人助手,名字叫小Q。请用简洁、幽默的风格回答。",
                    lines=4,
                    placeholder="自定义助手的行为和性格..."
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="越高越随机"
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="控制多样性"
                )

                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="最大 Token 数",
                    info="控制长度"
                )

                # Skills 设置
                gr.Markdown("### 🎓 Skills 设置")

                skills_selector = gr.Dropdown(
                    choices=[s["id"] for s in skill_manager.get_skills_list()],
                    multiselect=True,
                    label="选择要使用的技能",
                    info="留空则自动匹配"
                )

                auto_match_skills = gr.Checkbox(
                    label="自动匹配技能",
                    value=True,
                    info="根据任务自动选择相关技能"
                )

        # 事件处理
        def user_input(user_message, history):
            return "", history + [[user_message, None]]

        def bot_response(history, sys_prompt, temp, top_p_val, max_tok, use_tools_mode, use_skills_mode,
                        show_log, selected_skills, auto_match):
            """生成响应,并可选使用工具和 Skills"""
            if not history or history[-1][1] is not None:
                return history, ""

            user_message = history[-1][0]
            history_without_last = history[:-1]
            execution_info = ""

            # Skills 处理
            messages_with_skills = [[u, a] for u, a in history_without_last if u and a]

            if use_skills_mode:
                # 确定要使用的技能
                if auto_match and not selected_skills:
                    # 自动匹配
                    matched_skills = skill_manager.find_skills_for_task(user_message)
                    skills_to_inject = [s["id"] for s in matched_skills[:3]]  # 最多 3 个
                else:
                    # 使用手选的技能
                    skills_to_inject = selected_skills if selected_skills else []

                if skills_to_inject and show_log:
                    execution_info = f"✅ 已注入技能: {', '.join(skills_to_inject)}\n\n"

                # 构建消息列表
                messages = [{"role": "system", "content": sys_prompt}]
                for u, a in messages_with_skills:
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": user_message})

                # 注入 Skills (关键: 保留缓存!)
                if skills_to_inject:
                    messages = skill_injector.inject_skills_to_context(
                        messages,
                        skills_to_inject,
                        include_full_content=False
                    )
            else:
                messages = None

            # 处理响应
            if use_tools_mode:
                try:
                    response, exec_log = agent_framework.process_message(
                        user_message,
                        messages_with_skills,
                        system_prompt_override=sys_prompt,
                        temperature=temp,
                        top_p=top_p_val,
                        max_tokens=max_tok
                    )

                    if show_log:
                        execution_info += json.dumps(exec_log, ensure_ascii=False, indent=2)

                    history[-1][1] = response
                    yield history, execution_info
                except Exception as e:
                    error_msg = f"[工具模式错误] {str(e)}\n\n使用普通模式回答..."
                    for text_chunk in qwen_agent.generate_stream(
                        user_message,
                        messages_with_skills,
                        system_prompt=sys_prompt,
                        temperature=temp,
                        top_p=top_p_val,
                        max_tokens=max_tok
                    ):
                        history[-1][1] = error_msg + "\n" + text_chunk
                        yield history, execution_info
            else:
                # 普通模式
                for response in qwen_agent.generate_stream(
                    user_message,
                    messages_with_skills,
                    system_prompt=sys_prompt,
                    temperature=temp,
                    top_p=top_p_val,
                    max_tokens=max_tok
                ):
                    history[-1][1] = response
                    yield history, execution_info

        def retry_last(history, sys_prompt, temp, top_p_val, max_tok, use_tools_mode, use_skills_mode,
                      show_log, selected_skills, auto_match):
            if not history:
                return history, ""
            if history[-1][1] is not None:
                history[-1][1] = None
            yield from bot_response(history, sys_prompt, temp, top_p_val, max_tok, use_tools_mode,
                                   use_skills_mode, show_log, selected_skills, auto_match)

        def undo_last(history):
            if history:
                return history[:-1], ""
            return history, ""

        # 绑定事件
        msg.submit(
            user_input,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, use_tools, use_skills,
             show_execution_log, skills_selector, auto_match_skills],
            [chatbot, execution_log]
        )

        send_btn.click(
            user_input,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens, use_tools, use_skills,
             show_execution_log, skills_selector, auto_match_skills],
            [chatbot, execution_log]
        )

        retry_btn.click(
            retry_last,
            [chatbot, system_prompt, temperature, top_p, max_tokens, use_tools, use_skills,
             show_execution_log, skills_selector, auto_match_skills],
            [chatbot, execution_log]
        )

        undo_btn.click(undo_last, chatbot, [chatbot, execution_log])
        clear_btn.click(lambda: ([], ""), None, [chatbot, execution_log])

    return demo


def _get_skills_info(skill_manager: SkillManager) -> str:
    """获取技能信息用于显示"""
    skills = skill_manager.get_skills_list()
    if not skills:
        return "暂无可用技能"

    lines = []
    for skill in skills:
        lines.append(f"**{skill['name']}**")
        lines.append(f"{skill['description']}")
        if skill.get("tags"):
            lines.append(f"标签: {', '.join(skill['tags'])}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("🚀 正在启动 Qwen2.5 Agent 终极版...")

    demo = create_ui_with_skills()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )

