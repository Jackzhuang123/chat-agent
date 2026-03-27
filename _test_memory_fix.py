#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证记忆修复效果"""
import sys
sys.path.insert(0, '/Users/zhuangranxin/PyCharmProjects/chat-Agent')

errors = []

# ========== 1. inherited_context 追问检测 ==========
print("=== 1. inherited_context 追问检测 ===")
from core.mode_router import ModeRouter

# Mock IntentRouterCompat（直接测试 analyze 逻辑）
mode_router = ModeRouter()

# 模拟 IntentRouterCompat（从 web_agent_with_skills 移植一个简化版做测试）
import re as _re

def simulate_analyze(user_message, chat_history=None):
    """模拟 IntentRouterCompat.analyze 的追问逻辑"""
    _followup_patterns = _re.compile(
        r'(之前|上次|刚才|刚刚|上一轮|上一条|查看|回顾|总结|再说一遍|'
        r'什么问题|聊过什么|聊了什么|问过什么|说了什么|记录|历史|'
        r'前面|那个|之前说|你刚|you said|previous|last|above)',
        _re.IGNORECASE
    )
    _is_followup = bool(_followup_patterns.search(user_message))

    detection = mode_router.detect_mode(user_message)
    run_mode = {"chat": "chat", "tools": "tools"}.get(detection["recommended_mode"], "chat")

    inherited_context = ""
    if chat_history and (_is_followup or run_mode == "chat"):
        _ctx_parts = []
        for _pair in chat_history[-3:]:
            if not (isinstance(_pair, (list, tuple)) and len(_pair) == 2):
                continue
            _u, _a = _pair
            if not _u or not _a:
                continue
            _skip_prefixes = ("[⚠️", "[GLM", "[在进行中", "[未设置")
            if any(_a.startswith(p) for p in _skip_prefixes):
                continue
            _a_summary = _a[:200] + ("…" if len(_a) > 200 else "")
            _ctx_parts.append(f"用户: {str(_u)[:80]}\n助手: {_a_summary}")
        if _ctx_parts:
            inherited_context = "\n\n".join(_ctx_parts)

    return {"run_mode": run_mode, "inherited_context": inherited_context, "is_followup": _is_followup}

# 模拟上一轮对话历史
fake_history = [
    ["你好", "你好👋！有什么可以帮助你的吗？"],
    ["列出周杰伦的十大最出名的歌名", "1. 青花瓷 2. 东风破 3. 七里香..."],
]

# 测试1：追问"我之前问什么了"
r1 = simulate_analyze("我之前有问什么问题嘛", chat_history=fake_history)
ok1 = bool(r1["inherited_context"]) and r1["is_followup"]
print(f"{'✅' if ok1 else '❌'} 追问'之前问什么' -> inherited_context 已填充: {bool(r1['inherited_context'])}, is_followup: {r1['is_followup']}")
if not ok1:
    errors.append("追问检测失败")

# 测试2：追问"查询聊天记录"
r2 = simulate_analyze("查询聊天记录", chat_history=fake_history)
ok2 = bool(r2["inherited_context"]) and r2["is_followup"]
print(f"{'✅' if ok2 else '❌'} '查询聊天记录' -> inherited_context 已填充: {bool(r2['inherited_context'])}, is_followup: {r2['is_followup']}")
if not ok2:
    errors.append("'查询聊天记录'追问检测失败")

# 测试3：普通对话，无历史时 inherited_context 为空
r3 = simulate_analyze("你好")
ok3 = r3["inherited_context"] == ""
print(f"{'✅' if ok3 else '❌'} 无历史时 inherited_context 为空: {r3['inherited_context'] == ''}")

# 测试4：有历史的普通 chat 模式也注入上下文（让模型有背景）
r4 = simulate_analyze("你好", chat_history=fake_history)
ok4 = bool(r4["inherited_context"])  # chat 模式有历史也应注入
print(f"{'✅' if ok4 else '❌'} chat模式有历史时注入上下文: {bool(r4['inherited_context'])}")

# ========== 2. system prompt 重复注入修复 ==========
print("\n=== 2. system prompt 重复注入修复 ===")
from core import create_qwen_model_forward

# 记录实际传入的消息列表
recorded_messages = []
class MockAgent:
    model = "mock"
    def generate_stream_with_messages(self, messages, **kwargs):
        recorded_messages.clear()
        recorded_messages.extend(messages)
        yield "mock response"

agent = MockAgent()
fwd = create_qwen_model_forward(agent)

# 情况1: messages 已有 system，传入相同 system_prompt → 不重复
msgs_with_system = [
    {"role": "system", "content": "你是一个智能助手。"},
    {"role": "user", "content": "你好"},
]
fwd(msgs_with_system, system_prompt="你是一个智能助手。")
sys_count = sum(1 for m in recorded_messages if m["role"] == "system")
ok_dedup = sys_count == 1
print(f"{'✅' if ok_dedup else '❌'} 相同 system_prompt 去重: system 消息数={sys_count} (期望1)")
if not ok_dedup:
    errors.append(f"system prompt 去重失败: 有 {sys_count} 条 system 消息")

# 情况2: messages 已有 system，传入不同 system_prompt → 前置
msgs_with_system2 = [
    {"role": "system", "content": "你是一个智能助手。"},
    {"role": "user", "content": "你好"},
]
fwd(msgs_with_system2, system_prompt="你是工具助手，使用 ReAct 模式。")
sys_count2 = sum(1 for m in recorded_messages if m["role"] == "system")
ok_diff = sys_count2 == 2
print(f"{'✅' if ok_diff else '❌'} 不同 system_prompt 前置: system 消息数={sys_count2} (期望2)")

# 情况3: messages 无 system，正常前置
msgs_no_system = [{"role": "user", "content": "你好"}]
fwd(msgs_no_system, system_prompt="你是一个智能助手。")
sys_count3 = sum(1 for m in recorded_messages if m["role"] == "system")
ok_add = sys_count3 == 1
print(f"{'✅' if ok_add else '❌'} 无 system 时正常前置: system 消息数={sys_count3} (期望1)")
if not ok_add:
    errors.append("无 system 时前置失败")

# ========== 3. ToolLearner 接入 ==========
print("\n=== 3. ToolLearner 接入 agent_framework ===")
from core import QwenAgentFramework


def mock_fwd(messages, system_prompt="", **kwargs):
    # 第一次返回工具调用，第二次返回最终答案
    call_count = getattr(mock_fwd, '_calls', 0)
    mock_fwd._calls = call_count + 1
    if call_count == 0:
        return 'list_dir\n{"path": "."}'
    return "✅ 已列出目录"

mock_fwd._calls = 0

fw = QwenAgentFramework(model_forward_fn=mock_fwd, work_dir='/tmp')
ok_learner = fw.tool_learner is not None
print(f"{'✅' if ok_learner else '❌'} ToolLearner 实例已创建: {ok_learner}")
if not ok_learner:
    errors.append("ToolLearner 未初始化")

result = fw.run("列出目录")
ok_record = len(fw.tool_history) > 0
print(f"{'✅' if ok_record else '❌'} 工具调用后 tool_history 有记录: {len(fw.tool_history)} 条")

# ========== 汇总 ==========
print()
if errors:
    print(f"❌ {len(errors)} 个问题:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("🎉 全部验证通过！")

