#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复效果 - 验证意图识别、模式匹配和工具解析

运行方式：
    python test_fixes.py
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.mode_router import ModeRouter
from core.tool_enforcement_middleware import DirectCommandDetector
from core.agent_tools import ToolParser
from core.prompts import get_system_prompt, inject_few_shot_examples


def print_section(title: str):
    """打印分隔符"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_direct_command_detection():
    """测试直接命令检测"""
    print_section("测试 1: 直接命令检测")

    test_cases = [
        ("读取 core/agent_tools.py", True, "read_file"),
        ("调用工具进行读取", True, None),
        ("列出 core 目录", True, "list_dir"),
        ("执行 pytest 测试", True, "bash"),
        ("写入 test.py 文件", True, "write_file"),
        ("用 read_file 工具读取配置", True, "read_file"),
        ("你好，今天天气怎么样？", False, None),
        ("什么是 Python？", False, None),
    ]

    passed = 0
    failed = 0

    for user_input, expected_is_direct, expected_tool in test_cases:
        result = DirectCommandDetector.detect(user_input)
        is_correct = (result["is_direct_command"] == expected_is_direct)

        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{status} | 输入: {user_input}")
        print(f"      | 预期: is_direct={expected_is_direct}, tool={expected_tool}")
        print(f"      | 实际: is_direct={result['is_direct_command']}, tool={result['tool_name']}, confidence={result['confidence']:.2f}")
        print(f"      | 原因: {result['reason']}")
        print()

        if is_correct:
            passed += 1
        else:
            failed += 1

    print(f"📊 结果: {passed} 通过, {failed} 失败, 总计 {len(test_cases)} 个测试")


def test_tool_parsing():
    """测试工具解析"""
    print_section("测试 2: 工具调用解析")

    test_cases = [
        # 格式 1: 标准裸格式
        ('read_file\n{"path": "core/agent_tools.py"}', [("read_file", {"path": "core/agent_tools.py"})]),

        # 格式 2: JSON 对象格式
        ('{"tool": "read_file", "input": {"path": "test.py"}}', [("read_file", {"path": "test.py"})]),

        # 格式 3: OpenAI function_call 格式
        ('{"name": "list_dir", "arguments": {"path": "core"}}', [("list_dir", {"path": "core"})]),

        # 格式 4: Markdown JSON 代码块
        ('```json\n{"tool": "list_dir", "input": {"path": "core"}}\n```', [("list_dir", {"path": "core"})]),

        # 格式 5: 工具名 + JSON（宽松）
        ('read_file {"path": "test.py"}', [("read_file", {"path": "test.py"})]),

        # 格式 6: GLM 常见错误格式
        ('{"api": "read_file", "path": "test.py"}', [("read_file", {"path": "test.py"})]),

        # 格式 7: 带噪声的裸格式（最宽松）
        ('我要调用 read_file 工具 {"path": "core/agent_tools.py"}', [("read_file", {"path": "core/agent_tools.py"})]),

        # 非工具调用
        ('已读取并输出文件内容...', []),
    ]

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result = ToolParser.parse_tool_calls(text)
        is_correct = (result == expected)

        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{status} | 输入: {text[:60]}...")
        print(f"      | 预期: {expected}")
        print(f"      | 实际: {result}")
        print()

        if is_correct:
            passed += 1
        else:
            failed += 1

    print(f"📊 结果: {passed} 通过, {failed} 失败, 总计 {len(test_cases)} 个测试")


def test_mode_routing():
    """测试模式路由"""
    print_section("测试 3: 模式路由")

    # 创建路由器（不使用 LLM）
    router = ModeRouter(llm_forward_fn=None, llm_confidence_threshold=0.70)

    test_cases = [
        ("读取 core/agent_tools.py", "tools", 0.90),
        ("列出 core 目录", "tools", 0.80),
        ("执行 pytest 测试", "tools", 0.80),
        ("分析并重构 agent_tools.py", "plan", 0.60),
        ("你好，今天天气怎么样？", "chat", 0.30),
        ("/Users/test/file.py 的内容是什么", "tools", 0.90),  # 路径强信号
    ]

    passed = 0
    failed = 0

    for user_input, expected_mode, min_confidence in test_cases:
        result = router.detect_mode(user_input)
        is_correct = (
            result["recommended_mode"] == expected_mode and
            result["confidence"] >= min_confidence
        )

        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{status} | 输入: {user_input}")
        print(f"      | 预期: mode={expected_mode}, confidence>={min_confidence}")
        print(f"      | 实际: mode={result['recommended_mode']}, confidence={result['confidence']:.2f}")
        print(f"      | 原因: {result['reasoning']}")
        print()

        if is_correct:
            passed += 1
        else:
            failed += 1

    print(f"📊 结果: {passed} 通过, {failed} 失败, 总计 {len(test_cases)} 个测试")


def test_system_prompts():
    """测试 system prompt 生成"""
    print_section("测试 4: System Prompt 生成")

    modes = ["chat", "tools", "plan", "hybrid"]

    for mode in modes:
        prompt = get_system_prompt(mode, work_dir="/test/dir")

        print(f"模式: {mode}")
        print(f"长度: {len(prompt)} 字符")

        # 验证关键内容
        if mode == "tools":
            has_format = "tool_name" in prompt and '{"param"' in prompt
            has_examples = "【示例" in prompt
            has_rules = "【核心规则】" in prompt or "【禁止" in prompt

            print(f"✓ 包含工具格式说明: {has_format}")
            print(f"✓ 包含示例: {has_examples}")
            print(f"✓ 包含规则约束: {has_rules}")

            if not (has_format and has_examples and has_rules):
                print("❌ tools 模式 prompt 缺少关键内容")
        elif mode == "chat":
            has_persona = "智能助手" in prompt or "小Q" in prompt
            print(f"✓ 包含角色设定: {has_persona}")

        print(f"前 150 字符预览:\n{prompt[:150]}...")
        print()


def test_few_shot_injection():
    """测试 Few-Shot 示例注入"""
    print_section("测试 5: Few-Shot 示例注入")

    messages = [
        {"role": "system", "content": "你是智能助手"},
        {"role": "user", "content": "读取文件"}
    ]

    injected = inject_few_shot_examples(messages, tool_name="read_file", max_examples=2)

    print(f"原始消息数: {len(messages)}")
    print(f"注入后消息数: {len(injected)}")
    print(f"预期消息数: {len(messages) + 4}  (2个示例 * 2条消息)")

    if len(injected) == len(messages) + 4:
        print("✅ PASS: Few-Shot 示例注入成功")
    else:
        print("❌ FAIL: Few-Shot 示例数量不正确")

    print("\n注入的消息列表:")
    for i, msg in enumerate(injected):
        role = msg["role"]
        content = msg["content"][:50]
        print(f"  {i+1}. [{role}] {content}...")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("  Chat-Agent 修复效果验证测试")
    print("🚀" * 30)

    try:
        test_direct_command_detection()
        test_tool_parsing()
        test_mode_routing()
        test_system_prompts()
        test_few_shot_injection()

        print("\n" + "=" * 60)
        print("  ✅ 所有测试完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
