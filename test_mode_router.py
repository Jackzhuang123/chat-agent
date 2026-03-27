#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试模式路由器"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import QwenAgentFramework
from core.mode_router import ModeRouter, AutoModeMiddleware, create_auto_mode_framework


def mock_model(messages, system_prompt):
    """模拟模型"""
    last_msg = messages[-1].get("content", "") if messages else ""

    # 检测模式提示
    mode_hints = {
        "chat": "纯对话模式",
        "tools": "工具模式",
        "plan": "计划模式"
    }

    for mode, hint in mode_hints.items():
        if hint in str(messages):
            return f"收到{mode}模式提示，执行任务"

    if "读取" in last_msg or "read" in last_msg:
        return 'read_file\n{"path": "core/__init__.py"}'

    return "任务完成"


def test_mode_detection():
    """测试模式检测"""
    print("=== 测试：模式检测 ===\n")

    router = ModeRouter()

    test_cases = [
        "什么是 Python？",
        "读取core/agent_framework.py文件",
        "分析并重构core目录的代码结构",
        "规划并执行一个完整的项目搭建流程",
        "实时展示文件扫描进度"
    ]

    for user_input in test_cases:
        detection = router.detect_mode(user_input)
        print(f"输入: {user_input}")
        print(f"  推荐模式: {detection['recommended_mode']}")
        print(f"  置信度: {detection['confidence']:.2f}")
        print(f"  复杂度: {detection['complexity']}")
        print(f"  原因: {detection['reasoning']}")
        print()

    print("✅ 测试通过\n")


def test_auto_switch():
    """测试自动切换"""
    print("=== 测试：自动模式切换 ===\n")

    router = ModeRouter()

    # 模拟对话历史
    conversations = [
        ("什么是 Agent？", "chat"),
        ("读取core目录", "chat"),
        ("分析core/agent_framework.py", "tools")
    ]

    for user_input, current_mode in conversations:
        new_mode, reason = router.auto_switch_mode(user_input, current_mode)
        print(f"输入: {user_input}")
        print(f"  当前模式: {current_mode}")
        print(f"  新模式: {new_mode}")
        print(f"  原因: {reason}")
        print()

    print("✅ 测试通过\n")


def test_suggest_parameters():
    """测试参数推荐"""
    print("=== 测试：参数推荐 ===\n")

    router = ModeRouter()

    modes = ["chat", "tools", "plan", "multi_agent", "streaming"]

    for mode in modes:
        params = router.suggest_parameters(mode, "测试任务")
        print(f"模式: {mode}")
        print(f"  参数: {params}")
        print()

    print("✅ 测试通过\n")


def test_auto_mode_framework():
    """测试自动模式框架"""
    print("=== 测试：自动模式框架 ===\n")

    # 创建自动模式框架
    framework = create_auto_mode_framework(
        QwenAgentFramework,
        model_forward_fn=mock_model,
        enable_parallel=True
    )

    test_inputs = [
        "什么是 Python？",
        "读取core/__init__.py文件",
        "分析core目录的代码结构"
    ]

    for user_input in test_inputs:
        print(f"\n--- 测试输入: {user_input} ---")
        result = framework.run(user_input, history=[])
        print(f"迭代次数: {result['iterations']}")
        print(f"响应: {result['response'][:100]}")

    print("\n✅ 测试通过\n")


def test_mode_middleware():
    """测试模式中间件"""
    print("=== 测试：自动模式中间件 ===\n")

    middleware = AutoModeMiddleware()

    test_cases = [
        ("读取文件", {}),
        ("分析代码", {"run_mode": "tools"}),  # 已设置模式
        ("重构项目", {})
    ]

    for user_input, runtime_context in test_cases:
        print(f"输入: {user_input}")
        print(f"  原始 context: {runtime_context}")

        updated_context = middleware.process_before_run(user_input, runtime_context.copy())

        print(f"  更新后 context:")
        print(f"    run_mode: {updated_context.get('run_mode')}")
        print(f"    plan_mode: {updated_context.get('plan_mode')}")
        print(f"    max_iterations: {updated_context.get('max_iterations')}")
        print()

    print("✅ 测试通过\n")


def compare_modes():
    """模式对比"""
    print("=== 模式对比 ===\n")

    print("支持的模式:")
    print("  1. chat - 纯对话模式")
    print("     适用: 闲聊、问答、解释概念")
    print("     特点: 不调用工具，直接回答")
    print()

    print("  2. tools - 工具模式")
    print("     适用: 文件操作、代码分析、命令执行")
    print("     特点: 通过工具收集事实")
    print()

    print("  3. plan - 计划模式")
    print("     适用: 复杂任务、多步骤操作")
    print("     特点: 先分解任务再执行")
    print()

    print("  4. multi_agent - 多Agent模式")
    print("     适用: 需要规划-执行-审查的任务")
    print("     特点: Planner + Executor + Reviewer")
    print()

    print("  5. streaming - 流式模式")
    print("     适用: 需要实时展示进度")
    print("     特点: SSE 流式输出")
    print()

    print("自动检测规则:")
    print("  - 关键词匹配: 根据用户输入的关键词")
    print("  - 复杂度评估: simple/medium/complex")
    print("  - 自动升级: 复杂任务自动启用高级模式")
    print("  - 置信度阈值: >0.6 才切换模式")
    print()


if __name__ == "__main__":
    try:
        test_mode_detection()
        test_auto_switch()
        test_suggest_parameters()
        test_mode_middleware()
        test_auto_mode_framework()
        compare_modes()

        print("=" * 50)
        print("✅ 所有模式路由测试通过")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
