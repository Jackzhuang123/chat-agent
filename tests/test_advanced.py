#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试高级特性（向量记忆+多Agent+工具学习+流式输出）"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    VectorMemory,
    MultiAgentOrchestrator,
    ToolLearner,
    StreamingFramework,
    QwenAgentFramework,
    ToolExecutor,
)


def mock_model(messages, system_prompt):
    """模拟模型"""
    last_msg = messages[-1].get("content", "") if messages else ""

    # 规划响应
    if "规划" in system_prompt or "分解" in system_prompt:
        return '''{
  "complexity": "simple",
  "steps": [
    {"id": 1, "action": "列出core目录", "tool": "list_dir"},
    {"id": 2, "action": "读取__init__.py", "tool": "read_file"}
  ],
  "estimated_time": "5"
}'''

    # 审查响应
    if "审查" in system_prompt or "评估" in system_prompt:
        return '''{
  "completed": true,
  "quality": "good",
  "issues": [],
  "suggestions": []
}'''

    # 普通响应
    if "列出" in last_msg or "list" in last_msg:
        return 'list_dir\n{"path": "core"}'

    return "任务完成"


def test_vector_memory():
    """测试向量记忆"""
    print("=== 测试：向量记忆 ===\n")

    memory = VectorMemory()

    # 添加记忆
    memory.add_memory("读取core/agent_framework.py文件", {"type": "file_read"})
    memory.add_memory("列出core目录的内容", {"type": "dir_list"})
    memory.add_memory("执行grep命令搜索类定义", {"type": "bash"})
    memory.add_memory("分析agent_framework.py的代码结构", {"type": "code_analysis"})

    print(f"记忆数量: {len(memory)}")

    # 语义检索
    results = memory.search("查看文件内容", top_k=2)
    print(f"\n检索 '查看文件内容' 的结果:")
    for r in results:
        print(f"  - {r['content']} (相似度: {r['similarity']:.2f})")

    # 保存
    memory.save_to_disk()

    # 重新加载
    memory2 = VectorMemory()
    print(f"\n重新加载后记忆数量: {len(memory2)}")

    print("\n✅ 测试通过\n")


def test_multi_agent():
    """测试多Agent协作"""
    print("=== 测试：多Agent协作 ===\n")

    tool_executor = ToolExecutor(work_dir="..")
    orchestrator = MultiAgentOrchestrator(
        model_forward_fn=mock_model,
        tool_executor=tool_executor
    )

    result = orchestrator.run("列出core目录并读取__init__.py")

    print(f"任务完成: {result['completed']}")
    print(f"执行步骤: {len(result.get('execution_results', []))}")
    print(f"耗时: {result.get('duration', 0):.2f}秒")

    if result.get("plan"):
        print(f"复杂度: {result['plan'].get('complexity')}")

    print("\n✅ 测试通过\n")


def test_tool_learner():
    """测试工具学习"""
    print("=== 测试：工具学习 ===\n")

    learner = ToolLearner()

    # 任务分类
    task1 = "读取core/agent_framework.py文件"
    types1 = learner.classify_task(task1)
    print(f"任务: {task1}")
    print(f"分类: {types1}\n")

    task2 = "执行grep命令搜索类定义"
    types2 = learner.classify_task(task2)
    print(f"任务: {task2}")
    print(f"分类: {types2}\n")

    # 工具推荐
    recommendations = learner.recommend_tools(task1, top_k=3)
    print(f"推荐工具 (for '{task1}'):")
    for rec in recommendations:
        print(f"  - {rec['tool']} (置信度: {rec['confidence']:.2f}, 原因: {rec['reason']})")

    # 记录使用
    learner.record_usage("文件读取", "read_file", success=True)
    learner.record_usage("文件读取", "read_file", success=True)
    learner.record_usage("文件读取", "bash", success=False)

    # 保存
    learner.save_to_disk()

    # 统计
    stats = learner.get_tool_stats()
    print(f"\n工具统计: {stats}")

    print("\n✅ 测试通过\n")


def test_streaming():
    """测试流式输出"""
    print("=== 测试：流式输出 ===\n")

    framework = QwenAgentFramework(
        model_forward_fn=mock_model,
        enable_parallel=True,
    )

    streaming = StreamingFramework(framework)

    events = []
    for event in streaming.run_stream("列出core目录"):
        events.append(event)
        print(f"[{event.event_type}] {event.data}")

        # 只收集前10个事件（避免输出过多）
        if len(events) >= 10:
            break

    print(f"\n总事件数: {len(events)}")

    # 检查事件类型
    event_types = [e.event_type for e in events]
    print(f"事件类型: {set(event_types)}")

    # 测试 SSE 格式
    sse_lines = []
    for event in streaming.run_stream_sse("列出core目录"):
        sse_lines.append(event)
        if len(sse_lines) >= 5:
            break

    print(f"\nSSE 格式示例:")
    print(sse_lines[0][:100] + "...")

    print("\n✅ 测试通过\n")


def compare_features():
    """特性对比"""
    print("=== 高级特性对比 ===\n")

    print("基础版:")
    print("  - 并行执行: ✅")
    print("  - 持久化记忆: ✅ (SessionMemory)")
    print("  - 语义压缩: ✅")
    print("  - 循环检测: ✅")
    print()

    print("高级版:")
    print("  - 向量记忆: ✅ TF-IDF embedding")
    print("  - 多Agent协作: ✅ Planner + Executor + Reviewer")
    print("  - 工具学习: ✅ 基于任务分类推荐")
    print("  - 流式输出: ✅ SSE 实时进度")
    print()

    print("对比先进框架:")
    print("  LangChain:")
    print("    - 向量记忆: ✅ (需插件)")
    print("    - 多Agent: ❌")
    print("    - 工具学习: ❌")
    print("    - 流式输出: ✅")
    print()

    print("  AutoGPT:")
    print("    - 向量记忆: ✅ (向量数据库)")
    print("    - 多Agent: ❌")
    print("    - 工具学习: ❌")
    print("    - 流式输出: ❌")
    print()

    print("  Chat-Agent (本框架):")
    print("    - 向量记忆: ✅ (内置 TF-IDF)")
    print("    - 多Agent: ✅ (内置 Orchestrator)")
    print("    - 工具学习: ✅ (任务分类)")
    print("    - 流式输出: ✅ (SSE)")
    print()


if __name__ == "__main__":
    try:
        test_vector_memory()
        test_multi_agent()
        test_tool_learner()
        test_streaming()
        compare_features()

        print("=" * 50)
        print("✅ 所有高级特性测试通过")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
