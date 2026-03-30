#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 DeepReflectionEngine ↔ AdaptiveToolLearner 双向同步"""
import os
import shutil
import sys

sys.path.insert(0, '.')

# 清理之前的测试数据，确保测试环境干净
_mem_dir = '/tmp/test_agent_mem_fresh'
shutil.rmtree(_mem_dir, ignore_errors=True)
os.makedirs(_mem_dir, exist_ok=True)

from core.tool_learner import AdaptiveToolLearner
from core.agent_framework import DeepReflectionEngine

# 创建组件（使用全新目录，无历史数据）
learner = AdaptiveToolLearner(memory_dir=_mem_dir)
engine = DeepReflectionEngine()
engine.attach_tool_learner(learner)

# 模拟成功工具序列: read_file -> bash -> write_file
ctx1 = {'recent_tools': [], 'task': 'code_review', '_execution_time': 0.5}
engine.reflect_on_result('read_file', {'output': 'ok'}, context=ctx1)

ctx2 = {'recent_tools': ['read_file'], 'task': 'code_review', '_execution_time': 1.2}
engine.reflect_on_result('bash', {'output': 'ok'}, context=ctx2)

ctx3 = {'recent_tools': ['read_file', 'bash'], 'task': 'code_review', '_execution_time': 0.3}
engine.reflect_on_result('write_file', {'output': 'ok'}, context=ctx3)

# 模拟失败
ctx4 = {'recent_tools': ['write_file'], 'task': 'code_review', '_execution_time': 0.1}
engine.reflect_on_result('read_file', {'error': 'not found'}, context=ctx4)

print('[1] 成功模式记录（engine.success_patterns）:')
for k, v in engine.success_patterns.items():
    print(f'  {k}: count={v["count"]}, avg_time={v["avg_time"]:.2f}s')

print()
print('[2] ToolLearner 转移矩阵（成功路径）:')
for src, targets in learner.transition_matrix.items():
    for dst, cnt in targets.items():
        print(f'  {src} -> {dst}: {cnt}')

print()
print('[3] ToolLearner 工具统计:')
for tool, stats in learner.tool_stats.items():
    print(f'  {tool}: success={stats["success"]}, failed={stats["failed"]}')

print()
print('[4] 推荐（after read_file）:')
recs = learner.recommend_next_tools('code_review', ['read_file'], top_k=3)
for r in recs:
    print(f'  {r["tool"]}: confidence={r["confidence"]:.3f} | {r["reason"]}')

print()
print('[5] 高效序列 Top-3:')
for s in engine.get_efficient_sequences(top_k=3):
    print(f'  {s["sequence"]}: x{s["count"]} avg {s["avg_time"]:.2f}s')

# 断言验证
assert learner.transition_matrix['read_file']['bash'] == 1, \
    f"read_file->bash 转移应为1，实际: {learner.transition_matrix['read_file']['bash']}"
assert learner.transition_matrix['bash']['write_file'] == 1, \
    f"bash->write_file 转移应为1，实际: {learner.transition_matrix['bash']['write_file']}"
assert learner.tool_stats['read_file']['success'] == 1, \
    f"read_file 应有1次成功，实际: {learner.tool_stats['read_file']['success']}"
assert learner.tool_stats['read_file']['failed'] == 1, \
    f"read_file 应有1次失败，实际: {learner.tool_stats['read_file']['failed']}"
assert 'read_file->bash' in engine.success_patterns, \
    "应记录 read_file->bash 序列"
assert 'bash->write_file' in engine.success_patterns, \
    "应记录 bash->write_file 序列"

# 验证推荐结果中包含 bash（最高置信度）
assert len(recs) > 0, "应有推荐结果"
assert recs[0]['tool'] == 'bash', f"推荐第一位应为 bash，实际: {recs[0]['tool']}"

# 验证反思历史
assert len(engine.reflection_history) == 4, \
    f"应有4条反思记录，实际: {len(engine.reflection_history)}"
success_count = sum(1 for r in engine.reflection_history if r.get('success'))
assert success_count == 3, \
    f"应有3次成功反思，实际: {success_count}"

print()
print('✅ 所有断言通过！双向同步逻辑验证成功！')
print()
print('[反思摘要]:')
print(engine.get_reflection_summary())

