#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class QwenAgent:
    def __init__(self, model_path="./model/qwen2.5-0.5b", logger=None):
        print("正在初始化模型 (CPU模式)...")
        # 确保模型路径相对于项目根目录
        if model_path.startswith("./"):
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

        execution_time = time.time() - start_time
        if self.logger:
            try:
                input_tokens = len(model_inputs['input_ids'][0])
                output_tokens = len(self.tokenizer.encode(partial_message))
                self.logger.log_model_call(
                    prompt=text[:500] + "..." if len(text) > 500 else text,
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

        execution_time = time.time() - start_time
        if self.logger:
            try:
                input_tokens = len(model_inputs['input_ids'][0])
                output_tokens = len(self.tokenizer.encode(partial_message))
                self.logger.log_model_call(
                    prompt=text[:500] + "..." if len(text) > 500 else text,
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

