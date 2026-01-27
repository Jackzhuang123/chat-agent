#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B 个人助手 Web UI
基于 Gradio 构建的本地 AI 助手界面,支持流式输出和参数自定义
"""

from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# ============================================================================
# 核心类: QwenAgent - 负责模型加载和响应生成
# ============================================================================

class QwenAgent:
    """
    Qwen 模型代理类
    负责模型加载、初始化和流式生成响应
    """

    def __init__(self, model_path="./model/qwen2.5-0.5b"):
        """
        初始化模型

        Args:
            model_path (str): 模型文件路径,默认为 ./model/qwen2.5-0.5b
        """
        print("正在初始化模型 (CPU模式)...")
        self.model_path = model_path

        # 加载 tokenizer (分词器)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # 加载模型
        # torch_dtype=torch.float32: CPU 推荐使用 float32 (GPU 可用 float16)
        # device_map="cpu": 明确指定在 CPU 上运行
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        # 默认系统提示词 (定义助手的性格和行为)
        self.default_system_prompt = "你是一个智能个人助手,名字叫小Q。请用简洁、幽默的风格回答。"

        print("✅ 模型加载完毕!")

    def generate_stream(self, message, history, system_prompt=None, temperature=0.7, top_p=0.9, max_tokens=512):
        """
        流式生成响应 (打字机效果)

        Args:
            message (str): 用户当前输入的消息
            history (list): 历史对话列表,格式: [[user_msg, bot_msg], ...]
            system_prompt (str, optional): 自定义系统提示词,为 None 则使用默认值
            temperature (float): 温度参数 (0.1-2.0)
                - 低值: 更保守、确定
                - 高值: 更随机、创造性
            top_p (float): 核采样参数 (0.1-1.0)
                - 只考虑概率累积前 P% 的词汇
            max_tokens (int): 最大生成 token 数,控制回复长度

        Yields:
            str: 逐步生成的响应文本 (部分文本)
        """
        # 1. 确定使用的系统提示词 (自定义 or 默认)
        sys_prompt = system_prompt if system_prompt and system_prompt.strip() else self.default_system_prompt

        # 2. 构建 Qwen 格式的对话消息列表
        # Qwen 格式: [{"role": "system/user/assistant", "content": "..."}]
        messages = [{"role": "system", "content": sys_prompt}]

        # 3. 转换历史对话格式
        # Gradio 格式: [['用户消息', '助手回复'], ...]
        # 转换为 Qwen 格式
        for user_msg, bot_msg in history:
            if user_msg:  # 防止空消息
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        # 4. 添加当前用户问题
        messages.append({"role": "user", "content": message})

        # 5. 应用 chat template (Qwen 特定的对话格式)
        # tokenize=False: 返回字符串而非 token IDs
        # add_generation_prompt=True: 添加生成提示符 (如 "assistant:")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 6. Tokenize 输入文本
        # return_tensors="pt": 返回 PyTorch tensor
        # .to("cpu"): 确保在 CPU 上运行
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")

        # 7. 创建流式输出器 (实现打字机效果的关键)
        # skip_prompt=True: 不返回输入部分,只返回生成的新内容
        # skip_special_tokens=True: 不返回特殊 token (如 <|im_start|>, <|im_end|>)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 8. 准备生成参数
        generation_kwargs = dict(
            model_inputs,                      # 输入的 token IDs
            streamer=streamer,                 # 流式输出器
            max_new_tokens=int(max_tokens),    # 最大生成 token 数
            temperature=float(temperature),     # 温度参数
            top_p=float(top_p),                # 核采样参数
            do_sample=True,                    # 启用采样 (必须为 True 才能使用 temperature 和 top_p)
        )

        # 9. 在子线程中运行模型生成
        # 为什么用子线程?
        # - 主线程负责从 streamer 读取 token 并 yield 给 Gradio
        # - 子线程负责模型推理,生成的 token 会自动放入 streamer
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 10. 从 streamer 逐步读取生成的 token 并 yield (实现打字机效果)
        partial_message = ""
        for new_token in streamer:  # streamer 是一个迭代器
            partial_message += new_token
            yield partial_message  # Gradio 会自动更新界面显示


# ============================================================================
# UI 创建函数 - 使用 Gradio Blocks 构建自定义界面
# ============================================================================

def create_ui():
    """
    创建 Gradio Web UI 界面

    Returns:
        gr.Blocks: Gradio 应用实例
    """
    # 初始化模型代理 (只在启动时加载一次)
    agent = QwenAgent()

    # 自定义 CSS 样式
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .chat-message {
        font-size: 16px;
    }
    footer {
        visibility: hidden;  /* 隐藏 Gradio 默认的页脚 */
    }
    """

    # 使用 Blocks 创建自定义布局
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Qwen2.5 个人助手") as demo:

        # ===== 标题和说明 =====
        gr.Markdown(
            """
            # 🤖 Qwen2.5-0.5B 智能个人助手

            这是一个运行在本地 CPU 上的轻量级 AI 助手,基于 Qwen2.5-0.5B-Instruct 模型构建。

            ✨ **特性**:
            - 💬 支持多轮对话,保留历史记忆
            - ⚡ 流式输出,打字机效果
            - 🎨 可自定义系统提示词和生成参数
            - 🔒 完全本地运行,保护隐私
            """
        )

        # ===== 主要布局: 左右分栏 =====
        with gr.Row():
            # ----- 左侧: 对话区 (占 3/4 宽度) -----
            with gr.Column(scale=3):
                # 聊天窗口
                chatbot = gr.Chatbot(
                    label="对话窗口",
                    height=500,                          # 窗口高度
                    show_copy_button=True               # 显示复制按钮
                    # avatar_images=(None, "🤖")          # 头像 (用户, 助手) - 已注释,避免文件路径错误
                )

                # 输入区域
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入消息",
                        placeholder="在这里输入你的问题...",
                        scale=4,          # 占 4/5 宽度
                        lines=2           # 2行高度
                    )
                    send_btn = gr.Button("发送 📤", variant="primary", scale=1)

                # 功能按钮行
                with gr.Row():
                    retry_btn = gr.Button("🔄 重试")    # 重新生成最后一个回答
                    undo_btn = gr.Button("↩️ 撤销")     # 删除最后一轮对话
                    clear_btn = gr.Button("🗑️ 清空")   # 清空所有对话

                # 示例问题 (快速开始)
                gr.Examples(
                    examples=[
                        "你好,请介绍一下你自己",
                        "写一首关于春天的短诗",
                        "用Python写一个冒泡排序算法",
                        "解释一下什么是机器学习",
                        "给我讲个笑话",
                        "如何学好编程?"
                    ],
                    inputs=msg,
                    label="💡 示例问题"
                )

            # ----- 右侧: 参数设置区 (占 1/4 宽度) -----
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 高级设置")

                # 系统提示词输入框
                system_prompt = gr.Textbox(
                    label="系统提示词",
                    value="你是一个智能个人助手,名字叫小Q。请用简洁、幽默的风格回答。",
                    lines=4,
                    placeholder="自定义助手的行为和性格..."
                )

                # Temperature 滑块
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (温度)",
                    info="越高越随机,越低越确定"
                )

                # Top P 滑块
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P (核采样)",
                    info="控制生成多样性"
                )

                # Max Tokens 滑块
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="最大Token数",
                    info="控制回复长度"
                )

                # 参数说明
                gr.Markdown(
                    """
                    ---
                    ### 📊 参数说明

                    **Temperature**: 控制输出的随机性
                    - 低值(0.1-0.5): 更保守、确定
                    - 中值(0.6-0.9): 平衡创造力
                    - 高值(1.0-2.0): 更随机、创新

                    **Top P**: 核采样阈值
                    - 只考虑概率累积前 P% 的词
                    - 建议值: 0.8-0.95
                    """
                )

        # ===== 事件处理函数 =====

        def user_input(user_message, history):
            """
            处理用户输入

            Args:
                user_message (str): 用户输入的消息
                history (list): 当前对话历史

            Returns:
                tuple: (清空的输入框, 更新后的历史)
            """
            # 清空输入框,并在历史中添加新消息 (助手回复先设为 None)
            return "", history + [[user_message, None]]

        def bot_response(history, sys_prompt, temp, top_p_val, max_tok):
            """
            生成助手的响应

            Args:
                history (list): 对话历史
                sys_prompt (str): 系统提示词
                temp (float): Temperature 参数
                top_p_val (float): Top P 参数
                max_tok (int): Max Tokens 参数

            Yields:
                list: 更新后的对话历史 (逐步更新)
            """
            # 安全检查: 如果历史为空或最后一条已有回复,直接返回
            if not history or history[-1][1] is not None:
                return history

            # 获取用户最后的问题
            user_message = history[-1][0]
            # 历史记录 (不包括当前问题)
            history_without_last = history[:-1]

            # 调用模型生成响应 (流式)
            for response in agent.generate_stream(
                user_message,
                history_without_last,
                system_prompt=sys_prompt,
                temperature=temp,
                top_p=top_p_val,
                max_tokens=max_tok
            ):
                # 更新历史中的助手回复
                history[-1][1] = response
                yield history  # yield 给 Gradio,界面会自动更新

        def retry_last(history, sys_prompt, temp, top_p_val, max_tok):
            """
            重试最后一个回答 (重新生成)

            Args:
                history (list): 对话历史
                其他参数同 bot_response

            Yields:
                list: 更新后的对话历史
            """
            if not history:
                return history

            # 移除最后的回答,保留问题
            if history[-1][1] is not None:
                history[-1][1] = None

            # 重新调用 bot_response 生成
            yield from bot_response(history, sys_prompt, temp, top_p_val, max_tok)

        def undo_last(history):
            """
            撤销最后一轮对话

            Args:
                history (list): 对话历史

            Returns:
                list: 删除最后一轮后的历史
            """
            if history:
                return history[:-1]
            return history

        # ===== 绑定事件 =====

        # 输入框按 Enter 或点击发送按钮
        # .submit(): 输入框提交事件
        # .then(): 链式调用,先执行 user_input,再执行 bot_response
        msg.submit(
            user_input,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens],
            chatbot
        )

        # 点击发送按钮 (同上)
        send_btn.click(
            user_input,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            bot_response,
            [chatbot, system_prompt, temperature, top_p, max_tokens],
            chatbot
        )

        # 重试按钮
        retry_btn.click(
            retry_last,
            [chatbot, system_prompt, temperature, top_p, max_tokens],
            chatbot
        )

        # 撤销按钮
        undo_btn.click(undo_last, chatbot, chatbot)

        # 清空按钮 (使用 lambda 返回空列表)
        clear_btn.click(lambda: [], None, chatbot)

    return demo


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("🚀 正在启动 Qwen2.5 个人助手...")

    # 创建 UI
    demo = create_ui()

    # 启动服务
    demo.launch(
        server_name="127.0.0.1",  # 本地访问 (改为 "0.0.0.0" 可局域网访问)
        server_port=7860,         # 端口号 (Gradio 4.16.0 使用 server_port 而不是 port)
        inbrowser=True,           # 自动打开浏览器
        share=False               # 设为 True 可生成临时公网链接 (需要 Gradio 账号)
    )

