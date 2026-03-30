#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 VectorMemory 兼容接口修复"""
import os
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.vector_memory import VectorMemory

TMP_DIR = '/tmp/test_vm_compat'

def test_compat_methods():
    vm = VectorMemory(memory_dir=TMP_DIR)

    # 1. add_context
    vm.add_context('current_task', '扫描 core 目录')
    print('✅ add_context OK')

    # 2. update_tool_stats
    vm.update_tool_stats('read_file', True, 0.5)
    vm.update_tool_stats('bash', False, 1.2)
    vm.update_tool_stats('read_file', True, 0.3)
    print('✅ update_tool_stats OK')

    # 3. build_context_summary
    msgs = [
        {'role': 'user', 'content': '你好'},
        {'role': 'assistant', 'content': '有什么需要帮助的吗？'},
    ]
    summary = vm.build_context_summary(msgs)
    assert '用户' in summary or '助手' in summary, f"summary 格式异常: {summary}"
    print(f'✅ build_context_summary: {summary}')

    # 4. get_tool_recommendation
    recs = vm.get_tool_recommendation('general')
    assert isinstance(recs, list), f"应为列表: {recs}"
    assert 'read_file' in recs, f"read_file 应在推荐中（成功率最高）: {recs}"
    print(f'✅ get_tool_recommendation: {recs}')

    # 5. 空上下文/空列表的边界情况
    vm.add_context('empty_val', '')  # 不应出错
    empty_summary = vm.build_context_summary([])
    assert '无历史上下文' in empty_summary or '压缩' in empty_summary
    empty_recs = VectorMemory(memory_dir=TMP_DIR + '_empty').get_tool_recommendation('x')
    assert empty_recs == []
    print('✅ 边界情况 OK')

    print('\n🎉 VectorMemory 所有兼容方法验证通过！')

if __name__ == '__main__':
    try:
        test_compat_methods()
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        shutil.rmtree(TMP_DIR + '_empty', ignore_errors=True)

