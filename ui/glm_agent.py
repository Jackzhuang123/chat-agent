#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLMAgent - 智谱 GLM-4-Flash API 封装
兼容 QwenAgent 接口，可直接替换本地模型。

使用方式:
    from glm_agent import GLMAgent
    agent = GLMAgent(api_key="your_api_key")

免费模型: glm-4-flash（永久免费，需在 https://open.bigmodel.cn/ 注册获取 API Key）
"""

import time
from typing import Generator, List, Dict, Optional, Tuple


class GLMAgent:
    """
    智谱 GLM API 封装类，接口与 QwenAgent 完全兼容。
    支持流式输出，可无缝替换本地模型。
    """

    AVAILABLE_MODELS = {
        "glm-4-flash": "GLM-4-Flash",
        "glm-4-flash-250414": "GLM-4-Flash 最新版",
        "glm-4-air": "GLM-4-Air",
        "glm-4": "GLM-4",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        logger=None,
    ):
        """
        初始化 GLMAgent。

        Args:
            api_key: 智谱 AI API Key（在 https://open.bigmodel.cn/ 获取）
            model: 模型名称，默认 glm-4-flash（永久免费）
            logger: 可选的日志记录器（兼容 SessionLogger）
        """
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.default_system_prompt = "你是一个智能个人助手，名字叫小Q。请用简洁、幽默的风格回答。"

        # 延迟导入，避免在未安装 zhipuai 时报错
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化智谱 AI 客户端。"""
        try:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=self.api_key)
            print(f"✅ GLM 客户端初始化成功，使用模型: {self.model}")
        except ImportError:
            raise ImportError(
                "缺少 zhipuai 依赖，请运行: pip install zhipuai"
            )

    def _log_call(self, prompt_preview: str, response: str, execution_time: float,
                  tokens_input: int, tokens_output: int, temperature: float, top_p: float):
        """记录模型调用（与 QwenAgent 日志格式保持一致）。"""
        if not self.logger:
            return
        try:
            self.logger.log_model_call(
                prompt=prompt_preview,
                response=response,
                execution_time=execution_time,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                temperature=temperature,
                top_p=top_p,
                model_name=self.model,
            )
        except Exception as e:
            print(f"日志记录错误: {e}")

    def generate_stream(
        self,
        message: str,
        history: List,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """
        流式生成 - 兼容 QwenAgent.generate_stream 接口。
        接收 message + history（二维列表格式）。
        """
        sys_prompt = system_prompt if system_prompt and system_prompt.strip() else self.default_system_prompt
        messages = [{"role": "system", "content": sys_prompt}]

        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        yield from self._stream_from_messages(messages, temperature, top_p, max_tokens)

    def generate_stream_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """
        流式生成 - 兼容 QwenAgent.generate_stream_with_messages 接口。
        直接接收 messages 列表（字典格式）。
        """
        yield from self._stream_from_messages(messages, temperature, top_p, max_tokens)

    def _stream_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """内部流式生成核心方法。"""
        start_time = time.time()
        partial_message = ""
        tokens_input = 0
        tokens_output = 0

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    partial_message += delta
                    yield partial_message

                # 提取 token 用量（在最后一个 chunk 中）
                if hasattr(chunk, "usage") and chunk.usage:
                    tokens_input = getattr(chunk.usage, "prompt_tokens", 0)
                    tokens_output = getattr(chunk.usage, "completion_tokens", 0)

        except Exception as e:
            err_str = str(e)
            # 针对 429 速率限制给出友好提示
            if "429" in err_str or "1302" in err_str or "rate limit" in err_str.lower():
                error_msg = (
                    "⚠️ **API 请求频率超限（429）**\n\n"
                    "您的账户已达到调用速率限制，请稍等 10～30 秒后重试。\n\n"
                    "**建议：**\n"
                    "- 等待片刻后重新发送消息\n"
                    "- 如频繁遇到此问题，可考虑升级到付费套餐（glm-4-air 等）\n"
                    "- 复杂任务可拆分为较小步骤分次提交"
                )
            elif "401" in err_str or "authentication" in err_str.lower() or "unauthorized" in err_str.lower():
                error_msg = (
                    "❌ **API Key 认证失败（401）**\n\n"
                    "请检查 API Key 是否正确，或前往 https://open.bigmodel.cn/ 重新获取。"
                )
            else:
                error_msg = f"[GLM API 错误] {err_str}"
            print(error_msg)
            yield error_msg
            return

        # 记录日志
        execution_time = time.time() - start_time
        prompt_preview = str(messages)[:500] + "..." if len(str(messages)) > 500 else str(messages)
        self._log_call(
            prompt_preview=prompt_preview,
            response=partial_message,
            execution_time=execution_time,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            temperature=temperature,
            top_p=top_p,
        )


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    验证 API Key 是否有效。

    Returns:
        (is_valid, message)
    """
    if not api_key or not api_key.strip():
        return False, "API Key 不能为空"

    api_key = api_key.strip()

    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key)
        # 用最小请求验证 key
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            stream=False,
        )
        if resp and resp.choices:
            return True, "✅ API Key 验证成功！"
        return False, "❌ API Key 验证失败：响应异常"
    except ImportError:
        return False, "❌ 缺少 zhipuai 包，请运行: pip install zhipuai"
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "authentication" in err.lower() or "unauthorized" in err.lower():
            return False, f"❌ API Key 无效：{err}"
        return False, f"❌ 验证出错：{err}"

