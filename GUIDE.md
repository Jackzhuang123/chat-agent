# 开发指南

本文档说明代码架构和实现细节,适合开发者阅读。

---

## 📐 代码架构

### 核心类: QwenAgent

负责模型加载和响应生成。

```python
class QwenAgent:
    def __init__(self, model_path="./model/qwen2.5-0.5b")
        # 加载模型和 tokenizer

    def generate_stream(self, message, history, system_prompt, temperature, top_p, max_tokens)
        # 流式生成响应
        # 返回 generator,逐步 yield 输出
```

### UI 创建函数: create_ui()

使用 Gradio Blocks 创建自定义界面。

```python
def create_ui():
    agent = QwenAgent()  # 初始化模型

    with gr.Blocks() as demo:
        # 左侧: 对话区 (Chatbot, Textbox, Buttons)
        # 右侧: 参数设置区 (Sliders, Textbox)

    return demo
```

---

## 🔑 关键技术实现

### 1. 流式输出 (打字机效果)

使用 `TextIteratorStreamer` 在子线程中生成,主线程逐步 yield:

```python
# 创建 streamer
streamer = TextIteratorStreamer(
    self.tokenizer,
    skip_prompt=True,           # 不返回输入部分
    skip_special_tokens=True    # 不返回特殊 token
)

# 在子线程生成
generation_kwargs = dict(
    model_inputs,
    streamer=streamer,
    max_new_tokens=int(max_tokens),
    temperature=float(temperature),
    top_p=float(top_p),
    do_sample=True
)
thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
thread.start()

# 主线程逐步 yield
partial_message = ""
for new_token in streamer:
    partial_message += new_token
    yield partial_message  # Gradio 会自动更新界面
```

---

### 2. 对话历史管理

Gradio 和 Qwen 的对话格式不同,需要转换:

```python
# Gradio 格式: [['用户消息1', '助手回复1'], ['用户消息2', '助手回复2']]
# Qwen 格式: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

# 转换代码
messages = [{"role": "system", "content": sys_prompt}]

for user_msg, bot_msg in history:
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
    if bot_msg:
        messages.append({"role": "assistant", "content": bot_msg})

messages.append({"role": "user", "content": message})  # 当前问题
```

---

### 3. Chat Template 应用

Qwen 模型使用特定的对话格式:

```python
# 使用 tokenizer 的 chat_template
text = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,              # 返回字符串而非 token IDs
    add_generation_prompt=True   # 添加生成提示符
)

# 然后再 tokenize
model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
```

---

### 4. CPU 优化配置

```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype=torch.float32,  # CPU 推荐 float32 (GPU 用 float16)
    device_map="cpu"            # 明确指定 CPU
)
```

---

### 5. Gradio 事件绑定

实现用户交互逻辑:

```python
# 用户输入 → 添加到历史 → 生成响应
msg.submit(
    user_input,              # 处理用户输入
    [msg, chatbot],          # 输入组件
    [msg, chatbot]           # 输出组件
).then(
    bot_response,            # 生成响应
    [chatbot, system_prompt, temperature, top_p, max_tokens],
    chatbot
)

# 重试功能
retry_btn.click(
    retry_last,
    [chatbot, system_prompt, temperature, top_p, max_tokens],
    chatbot
)

# 撤销和清空
undo_btn.click(undo_last, chatbot, chatbot)
clear_btn.click(lambda: [], None, chatbot)
```

---

## 🎨 自定义主题和样式

### CSS 样式

```python
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.chat-message {
    font-size: 16px;
}
footer {
    visibility: hidden;  /* 隐藏 Gradio 底部信息 */
}
"""
```

### Gradio 主题

```python
gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(),  # 柔和主题
    title="Qwen2.5 个人助手"
)
```

---

## 🔄 函数调用流程

### 用户发送消息时

```
1. user_input(user_message, history)
   └─ 返回 ("", history + [[user_message, None]])

2. bot_response(history, sys_prompt, temp, top_p_val, max_tok)
   └─ 调用 agent.generate_stream(...)
      └─ yield 部分响应 → Gradio 自动更新界面
```

### 重试功能

```
retry_last(history, sys_prompt, temp, top_p_val, max_tok)
├─ 移除最后的回答 (保留问题)
└─ 调用 bot_response 重新生成
```

### 撤销功能

```
undo_last(history)
└─ 返回 history[:-1] (移除最后一轮对话)
```

---

## 📦 依赖说明

### 核心依赖

```
torch>=2.0.0          # PyTorch 深度学习框架
transformers>=4.35.0  # Hugging Face Transformers
gradio==4.16.0        # Web UI 框架 (指定版本避免兼容问题)
accelerate>=0.20.0    # 模型加速库
```

### 版本兼容

- `gradio==4.16.0` + `huggingface_hub<0.20` 组合避免导入错误
- Python 3.8+ (推荐 3.9 或 3.10)
- CPU 推理: `torch_dtype=torch.float32`
- GPU 推理: `torch_dtype=torch.float16`, `device_map="cuda"`

---

## 🛠️ 扩展开发

### 添加新功能

#### 1. 保存对话历史

```python
import json

def save_history(history, filepath="chat_history.json"):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(filepath="chat_history.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 在 UI 中添加保存/加载按钮
save_btn = gr.Button("💾 保存对话")
load_btn = gr.Button("📂 加载对话")

save_btn.click(save_history, chatbot, None)
load_btn.click(load_history, None, chatbot)
```

#### 2. 预设角色快速切换

```python
PRESETS = {
    "编程导师": "你是资深 Python 开发者,擅长教学和代码审查。",
    "创意作家": "你是浪漫主义诗人,文笔优美,富有想象力。",
    "学习助手": "你是耐心的老师,用简单语言解释复杂概念。"
}

preset_dropdown = gr.Dropdown(
    choices=list(PRESETS.keys()),
    label="角色预设"
)

def load_preset(preset_name):
    return PRESETS.get(preset_name, "")

preset_dropdown.change(load_preset, preset_dropdown, system_prompt)
```

#### 3. GPU 加速支持

```python
import torch

class QwenAgent:
    def __init__(self, model_path="./model/qwen2.5-0.5b", use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if use_gpu else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=self.device
        )
```

---

## 🧪 调试技巧

### 1. 打印生成参数

```python
print(f"Temperature: {temperature}, Top P: {top_p}, Max Tokens: {max_tokens}")
print(f"Messages: {messages}")
```

### 2. 查看 Tokenizer 输出

```python
text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Template output:\n{text}")
```

### 3. 测试模型加载

```python
if __name__ == "__main__":
    print("测试模型加载...")
    agent = QwenAgent()
    print("✅ 模型加载成功")
```

---

## 🚀 性能优化

### 1. 模型量化 (减小内存占用)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. 缓存优化

```python
# 使用 KV cache 加速推理
output = model.generate(
    **model_inputs,
    use_cache=True,  # 启用 KV cache
    ...
)
```

### 3. 批量推理 (多用户)

```python
# 处理多个请求
model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**model_inputs)
```

---

## 📝 代码风格

### 命名规范

- 类名: `PascalCase` (如 `QwenAgent`)
- 函数名: `snake_case` (如 `generate_stream`)
- 常量: `UPPER_CASE` (如 `DEFAULT_SYSTEM_PROMPT`)

### 注释规范

```python
def generate_stream(self, message, history, system_prompt=None, temperature=0.7, top_p=0.9, max_tokens=512):
    """
    流式生成响应

    Args:
        message (str): 用户当前输入
        history (list): 历史对话列表 [[user, bot], ...]
        system_prompt (str): 自定义系统提示词
        temperature (float): 温度参数 (0.1-2.0)
        top_p (float): 核采样参数 (0.1-1.0)
        max_tokens (int): 最大生成 token 数

    Yields:
        str: 部分生成的响应文本
    """
```

---

## 🎯 最佳实践

1. **模型加载**: 在程序启动时一次性加载,避免重复加载
2. **错误处理**: 添加 try-except 捕获生成错误
3. **参数验证**: 验证用户输入的参数范围
4. **日志记录**: 使用 logging 模块记录关键信息
5. **资源清理**: 程序退出时释放模型资源

---

## 📚 参考资源

- [Gradio 官方文档](https://www.gradio.app/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [Qwen2.5 模型文档](https://huggingface.co/Qwen)
- [PyTorch 文档](https://pytorch.org/docs/)

---

🎊 祝开发愉快!

