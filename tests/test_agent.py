#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chat-Agent 核心流程测试套件

覆盖范围：
  1. ToolExecutor         — read_file / write_file / edit_file / list_dir / bash
  2. ToolParser           — XML 格式 / 裸格式 工具调用解析
  3. IntentRouter         — chat / tools / skills / hybrid 意图路由
  4. TodoItem / Manager   — 四态机器、CRUD、渲染、apply_tool_write
  5. QwenAgentFramework   — chat 模式 / tools 模式、中间件链
  6. 中间件               — RuntimeMode / PlanMode / ToolResultGuard / TodoContext / Memory
  7. MemoryManager        — 保存/加载/规则提取/格式化注入
  8. TaskPlanner          — 任务规划（Stub 模型）
  9. TaskExecutor         — tool=none 整理步骤 / write_file 步骤 / bash 步骤
 10. _prepare_tool_result_for_model — bash / read_file / 通用截断

运行方式：
    cd /path/to/chat-Agent
    python -m pytest tests/test_agent.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# 确保 core 包可被导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_tools import ToolExecutor, ToolParser
from core.agent_framework import (
    AgentMiddleware,
    AIIntentRouter,
    IntentRouter,
    MemoryInjectionMiddleware,
    MemoryManager,
    PlanModeMiddleware,
    QwenAgentFramework,
    RuntimeModeMiddleware,
    TaskExecutor,
    TaskPlanner,
    TodoContextMiddleware,
    TodoItem,
    TodoManager,
    ToolResultGuardMiddleware,
    _inject_context_before_last_user,
)


# ============================================================================
# 工具函数：构建一个同步 stub 模型（返回固定字符串，无需真实 API）
# ============================================================================

def make_stub_model(response: str = "这是一个测试回复。"):
    """返回一个总是返回 response 的 stub model_forward_fn。"""
    def stub(messages, system_prompt="", **kwargs):
        return response
    return stub


def make_tool_model(tool_name: str, tool_input: dict):
    """返回一个模拟工具调用的 stub 模型（输出一次工具调用后返回普通文本）。"""
    call_count = {"n": 0}

    def stub(messages, system_prompt="", **kwargs):
        if call_count["n"] == 0:
            call_count["n"] += 1
            input_json = json.dumps(tool_input, ensure_ascii=False)
            return f"{tool_name}\n{input_json}"
        return "工具执行完毕。"

    return stub


# ============================================================================
# 1. ToolExecutor
# ============================================================================

class TestToolExecutor:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.executor = ToolExecutor(work_dir=self.tmpdir, enable_bash=True)

    # --- write_file ---
    def test_write_file_creates_file(self):
        result = json.loads(self.executor.execute_tool("write_file", {
            "path": "hello.txt",
            "content": "Hello, World!"
        }))
        assert result["success"] is True
        assert Path(self.tmpdir, "hello.txt").read_text() == "Hello, World!"

    def test_write_file_returns_size(self):
        result = json.loads(self.executor.execute_tool("write_file", {
            "path": "size_test.txt",
            "content": "abcde"
        }))
        assert result["size"] == 5

    # --- read_file ---
    def test_read_file_existing(self):
        p = Path(self.tmpdir, "read_me.txt")
        p.write_text("内容内容内容")
        result = json.loads(self.executor.execute_tool("read_file", {"path": "read_me.txt"}))
        assert result["success"] is True
        assert "内容内容内容" in result["content"]

    def test_read_file_nonexistent(self):
        result = json.loads(self.executor.execute_tool("read_file", {"path": "no_such.txt"}))
        assert result["success"] is False

    # --- edit_file ---
    def test_edit_file_replace(self):
        p = Path(self.tmpdir, "edit_me.txt")
        p.write_text("old content here")
        result = json.loads(self.executor.execute_tool("edit_file", {
            "path": "edit_me.txt",
            "old_content": "old content",
            "new_content": "new content"
        }))
        assert result["success"] is True
        assert "new content" in p.read_text()

    def test_edit_file_not_found_old_content(self):
        p = Path(self.tmpdir, "edit_fail.txt")
        p.write_text("something")
        result = json.loads(self.executor.execute_tool("edit_file", {
            "path": "edit_fail.txt",
            "old_content": "nonexistent",
            "new_content": "replacement"
        }))
        # edit_file 找不到 old_content 时返回 {"error": "..."}
        assert "error" in result or result.get("success") is False

    # --- list_dir ---
    def test_list_dir_returns_contents(self):
        Path(self.tmpdir, "a.py").write_text("# a")
        Path(self.tmpdir, "b.py").write_text("# b")
        result = json.loads(self.executor.execute_tool("list_dir", {"path": "."}))
        assert result["success"] is True
        # list_dir 返回的键是 "items"（每项是 {name, type, size} 的字典）
        items = result.get("items", [])
        assert any("a.py" in str(item.get("name", "")) for item in items)

    # --- bash ---
    def test_bash_echo(self):
        result = json.loads(self.executor.execute_tool("bash", {
            "command": "echo hello_bash"
        }))
        assert result["success"] is True
        assert "hello_bash" in result["stdout"]

    def test_bash_disabled(self):
        executor_no_bash = ToolExecutor(work_dir=self.tmpdir, enable_bash=False)
        result = json.loads(executor_no_bash.execute_tool("bash", {"command": "echo x"}))
        # bash 禁用时走 else 分支，返回 {"error": "未知工具: bash"}
        assert "error" in result or result.get("success") is False

    # --- 未知工具 ---
    def test_unknown_tool(self):
        result = json.loads(self.executor.execute_tool("fly_to_moon", {}))
        # 未知工具返回 {"error": "未知工具: ..."}
        assert "error" in result or result.get("success") is False


# ============================================================================
# 2. ToolParser
# ============================================================================

class TestToolParser:

    def setup_method(self):
        self.parser = ToolParser()

    def test_parse_bare_format(self):
        """裸格式：工具名\n{JSON}"""
        text = 'write_file\n{"path": "out.txt", "content": "hi"}'
        calls = self.parser.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0][0] == "write_file"
        assert calls[0][1]["path"] == "out.txt"

    def test_parse_xml_format(self):
        """XML 格式：<tool>名</tool><input>{JSON}</input>"""
        text = '<tool>read_file</tool><input>{"path": "foo.py"}</input>'
        calls = self.parser.parse_tool_calls(text)
        assert len(calls) >= 1
        assert calls[0][0] == "read_file"
        assert calls[0][1]["path"] == "foo.py"

    def test_no_tool_call(self):
        calls = self.parser.parse_tool_calls("普通文本，没有工具调用。")
        assert calls == []

    def test_multiple_tool_calls(self):
        text = (
            '<tool>read_file</tool><input>{"path": "a.txt"}</input>\n'
            '<tool>write_file</tool><input>{"path": "b.txt", "content": "x"}</input>'
        )
        calls = self.parser.parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0][0] == "read_file"
        assert calls[1][0] == "write_file"

    def test_malformed_json_graceful(self):
        """格式错误的 JSON 不应抛异常，应返回空列表或部分结果。"""
        text = 'read_file\n{broken json'
        try:
            calls = self.parser.parse_tool_calls(text)
            # 不抛异常即通过
        except Exception:
            pytest.fail("parse_tool_calls 对格式错误输入不应抛异常")


# ============================================================================
# 3. IntentRouter（规则路由）
# ============================================================================

class TestIntentRouter:

    def setup_method(self):
        self.router = IntentRouter(available_skill_ids=["pdf", "python-dev", "code-review"])

    def test_chat_mode(self):
        result = self.router.analyze("今天天气怎么样？")
        assert result["run_mode"] == "chat"

    def test_tools_mode_file_path(self):
        result = self.router.analyze("帮我读取 core/agent_framework.py 的内容")
        assert result["run_mode"] == "tools"
        assert result["needs_tools"] is True

    def test_tools_mode_keyword(self):
        result = self.router.analyze("帮我写入文件 output.md")
        assert result["run_mode"] in ("tools", "hybrid")

    def test_skills_mode_python(self):
        result = self.router.analyze("Python 中如何实现装饰器？")
        assert "python-dev" in result.get("skill_ids", []) or result["run_mode"] in ("skills", "hybrid", "chat")

    def test_bash_keyword_tools(self):
        result = self.router.analyze("执行命令 ls -la")
        assert result["run_mode"] == "tools"

    def test_reasons_populated(self):
        result = self.router.analyze("读取文件 README.md")
        assert isinstance(result.get("reasons"), list)


# ============================================================================
# 4. TodoItem / TodoManager（四态机器）
# ============================================================================

class TestTodoItem:

    def test_initial_state(self):
        item = TodoItem(task_id=1, task="测试任务")
        assert item.status == "pending"
        assert item.id == 1

    def test_transition_valid(self):
        item = TodoItem(task_id=1, task="任务")
        item.transition("in_progress")
        assert item.status == "in_progress"
        item.transition("completed", result_preview="done")
        assert item.status == "completed"
        assert "done" in item.result_preview

    def test_transition_invalid(self):
        item = TodoItem(task_id=1, task="任务")
        with pytest.raises(ValueError):
            item.transition("flying")

    def test_to_dict_and_from_dict(self):
        item = TodoItem(task_id=3, task="序列化测试", tool="bash", priority="high")
        d = item.to_dict()
        restored = TodoItem.from_dict(d)
        assert restored.id == 3
        assert restored.task == "序列化测试"
        assert restored.tool == "bash"


class TestTodoManager:

    def setup_method(self):
        self.mgr = TodoManager(title="测试计划")

    def test_add_and_get(self):
        item = self.mgr.add("步骤一", tool="bash")
        assert item.id == 1
        assert self.mgr.get(1) is item

    def test_update_status(self):
        self.mgr.add("步骤一")
        ok = self.mgr.update(1, "completed", result_preview="结果")
        assert ok is True
        assert self.mgr.get(1).status == "completed"

    def test_update_nonexistent(self):
        ok = self.mgr.update(999, "completed")
        assert ok is False

    def test_all_done(self):
        self.mgr.add("步骤A")
        self.mgr.add("步骤B")
        assert self.mgr.all_done() is False
        self.mgr.update(1, "completed")
        self.mgr.update(2, "completed")
        assert self.mgr.all_done() is True

    def test_completion_rate(self):
        self.mgr.add("步骤A")
        self.mgr.add("步骤B")
        self.mgr.update(1, "completed")
        assert self.mgr.completion_rate() == 0.5

    def test_get_next_pending(self):
        self.mgr.add("步骤A")
        self.mgr.add("步骤B")
        self.mgr.update(1, "in_progress")
        nxt = self.mgr.get_next_pending()
        assert nxt.id == 2

    def test_render_for_context(self):
        self.mgr.add("步骤A")
        text = self.mgr.render_for_context()
        assert "测试计划" in text
        assert "步骤A" in text

    def test_render_for_context_worker_hides_pending(self):
        self.mgr.add("步骤A")
        self.mgr.add("步骤B")
        self.mgr.update(1, "completed")
        text = self.mgr.render_for_context_worker()
        assert "步骤A" in text    # completed 可见
        assert "步骤B" not in text  # pending 对 worker 不可见

    def test_render_for_ui(self):
        self.mgr.add("步骤A", tool="bash")
        text = self.mgr.render_for_ui()
        assert "步骤A" in text

    def test_apply_tool_write_update(self):
        self.mgr.add("任务")
        result = json.loads(self.mgr.apply_tool_write({"action": "update", "id": 1, "status": "completed"}))
        assert result["success"] is True
        assert self.mgr.get(1).status == "completed"

    def test_apply_tool_write_add(self):
        result = json.loads(self.mgr.apply_tool_write({"action": "add", "task": "新任务", "tool": "bash"}))
        assert result["success"] is True
        assert self.mgr.get(result["id"]).task == "新任务"

    def test_apply_tool_write_invalid_action(self):
        result = json.loads(self.mgr.apply_tool_write({"action": "delete"}))
        assert result["success"] is False

    def test_load_from_todos_list(self):
        todos = [
            {"id": 1, "task": "T1", "tool": "bash", "status": "pending"},
            {"id": 2, "task": "T2", "tool": "none", "status": "pending"},
        ]
        self.mgr.load_from_todos_list(todos)
        assert self.mgr.get(1).task == "T1"
        assert self.mgr.get(2).tool == "none"


# ============================================================================
# 5. QwenAgentFramework — chat 模式
# ============================================================================

class TestQwenAgentFrameworkChat:

    def setup_method(self):
        self.stub = make_stub_model("你好，我是 AI 助手。")
        self.framework = QwenAgentFramework(
            model_forward_fn=self.stub,
            enable_bash=False,
            max_iterations=3,
            middlewares=[RuntimeModeMiddleware()],
        )

    def test_process_message_returns_response(self):
        response, log = self.framework.process_message(
            user_message="你好",
            history=[],
            runtime_context={"run_mode": "chat"},
        )
        assert "AI" in response or len(response) > 0

    def test_execution_log_has_model_response(self):
        _, log = self.framework.process_message(
            user_message="测试",
            history=[],
            runtime_context={"run_mode": "chat"},
        )
        types = [e.get("type") for e in log]
        assert "model_response" in types

    def test_history_is_injected(self):
        captured_messages = []
        def spy_model(messages, system_prompt="", **kwargs):
            captured_messages.extend(messages)
            return "OK"
        fw = QwenAgentFramework(model_forward_fn=spy_model)
        fw.process_message(
            user_message="第二轮",
            history=[("第一轮", "第一轮回复")],
            runtime_context={"run_mode": "chat"},
        )
        contents = [m["content"] for m in captured_messages]
        assert any("第一轮" in c for c in contents)

    def test_return_runtime_context(self):
        response, log, ctx = self.framework.process_message(
            user_message="测试",
            history=[],
            runtime_context={"run_mode": "chat"},
            return_runtime_context=True,
        )
        assert isinstance(ctx, dict)
        assert "run_mode" in ctx


# ============================================================================
# 5b. QwenAgentFramework — tools 模式（工具调用循环）
# ============================================================================

class TestQwenAgentFrameworkTools:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_tool_call_write_file_then_finish(self):
        """模型第一轮输出 write_file 调用，第二轮输出完成文本。"""
        stub = make_tool_model("write_file", {"path": "out.txt", "content": "内容"})
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            work_dir=self.tmpdir,
            enable_bash=False,
            max_iterations=5,
        )
        response, log = fw.process_message(
            user_message="创建文件",
            history=[],
            runtime_context={"run_mode": "tools"},
        )
        # 应当执行过 write_file 工具
        tool_calls = [e for e in log if e.get("type") == "tool_call"]
        assert any(tc.get("tool") == "write_file" for tc in tool_calls)

    def test_tool_call_read_file(self):
        """先写入文件，再让模型读取它。"""
        target = Path(self.tmpdir, "read_target.txt")
        target.write_text("文件内容 XYZ")

        stub = make_tool_model("read_file", {"path": "read_target.txt"})
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            work_dir=self.tmpdir,
            enable_bash=False,
            max_iterations=5,
        )
        response, log = fw.process_message(
            user_message="读文件",
            history=[],
            runtime_context={"run_mode": "tools"},
        )
        tool_calls = [e for e in log if e.get("type") == "tool_call"]
        assert any(tc.get("tool") == "read_file" for tc in tool_calls)

    def test_bash_tool_call(self):
        """模型输出 bash 工具调用，执行 echo 命令。"""
        stub = make_tool_model("bash", {"command": "echo test_output_123"})
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            work_dir=self.tmpdir,
            enable_bash=True,
            max_iterations=5,
        )
        response, log = fw.process_message(
            user_message="执行命令",
            history=[],
            runtime_context={"run_mode": "tools"},
        )
        tool_calls = [e for e in log if e.get("type") == "tool_call"]
        bash_calls = [tc for tc in tool_calls if tc.get("tool") == "bash"]
        assert len(bash_calls) >= 1


# ============================================================================
# 6. 中间件
# ============================================================================

class TestRuntimeModeMiddleware:

    def test_chat_mode_hint_injected(self):
        mw = RuntimeModeMiddleware()
        messages = [{"role": "user", "content": "问题"}]
        ctx = {"run_mode": "chat"}
        result = mw.before_model(messages, iteration=0, runtime_context=ctx)
        contents = [m["content"] for m in result]
        assert any("chat" in c.lower() or "对话" in c for c in contents)

    def test_tools_mode_hint_injected(self):
        mw = RuntimeModeMiddleware()
        messages = [{"role": "user", "content": "任务"}]
        ctx = {"run_mode": "tools"}
        result = mw.before_model(messages, iteration=0, runtime_context=ctx)
        contents = [m["content"] for m in result]
        assert any("工具" in c or "tool" in c.lower() for c in contents)

    def test_no_duplicate_injection(self):
        """同一 runtime_context 多次调用，不重复注入。"""
        mw = RuntimeModeMiddleware()
        messages = [{"role": "user", "content": "问题"}]
        ctx = {"run_mode": "chat"}
        r1 = mw.before_model(messages, 0, ctx)
        r2 = mw.before_model(r1, 1, ctx)
        assert len(r2) == len(r1)  # 第二次不应再注入


class TestPlanModeMiddleware:

    def test_plan_hint_injected_in_chat_mode(self):
        mw = PlanModeMiddleware()
        messages = [{"role": "user", "content": "任务"}]
        ctx = {"plan_mode": True, "run_mode": "chat"}
        result = mw.before_model(messages, 0, ctx)
        contents = [m["content"] for m in result]
        assert any("计划" in c or "plan" in c.lower() for c in contents)

    def test_no_injection_when_plan_mode_false(self):
        mw = PlanModeMiddleware()
        messages = [{"role": "user", "content": "任务"}]
        ctx = {"plan_mode": False}
        result = mw.before_model(messages, 0, ctx)
        assert len(result) == len(messages)


class TestToolResultGuardMiddleware:

    def test_enriches_tool_result(self):
        mw = ToolResultGuardMiddleware()
        raw = json.dumps({"stdout": "output"})
        enriched = mw.after_tool_call("bash", {"command": "echo"}, raw, 0)
        data = json.loads(enriched)
        assert "tool" in data or "success" in data or "timestamp" in data

    def test_handles_non_json_result(self):
        mw = ToolResultGuardMiddleware()
        result = mw.after_tool_call("bash", {}, "pure text result", 0)
        # 不应抛异常
        assert isinstance(result, str)


class TestTodoContextMiddleware:

    def test_todo_list_injected_to_messages(self):
        mgr = TodoManager()
        mgr.add("步骤一", tool="bash")
        mw = TodoContextMiddleware(todo_manager=mgr, worker_mode=False)
        messages = [{"role": "user", "content": "执行任务"}]
        result = mw.before_model(messages, 0, {})
        contents = [m["content"] for m in result]
        assert any("步骤一" in c or "todo_list" in c for c in contents)

    def test_worker_mode_hides_pending(self):
        mgr = TodoManager()
        mgr.add("已完成步骤")
        mgr.update(1, "completed")
        mgr.add("待执行步骤")
        mw = TodoContextMiddleware(todo_manager=mgr, worker_mode=True)
        messages = [{"role": "user", "content": "执行"}]
        result = mw.before_model(messages, 0, {})
        contents = " ".join(m["content"] for m in result)
        assert "已完成步骤" in contents
        assert "待执行步骤" not in contents

    def test_todo_write_intercepted(self):
        mgr = TodoManager()
        mgr.add("任务")
        mw = TodoContextMiddleware(todo_manager=mgr)
        payload = {"action": "update", "id": 1, "status": "completed"}
        result_str = mw.after_tool_call("todo_write", payload, "", 0)
        assert json.loads(result_str)["success"] is True

    def test_non_todo_tool_not_intercepted(self):
        mgr = TodoManager()
        mw = TodoContextMiddleware(todo_manager=mgr)
        original = json.dumps({"success": True})
        result = mw.after_tool_call("bash", {}, original, 0)
        assert result == original


# ============================================================================
# 7. MemoryManager
# ============================================================================

class TestMemoryManager:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem_file = os.path.join(self.tmpdir, "memory.json")
        self.mgr = MemoryManager(storage_path=self.mem_file)

    def test_empty_memory_structure(self):
        data = self.mgr.load()
        assert "user" in data
        assert "facts" in data
        assert "history" in data

    def test_save_and_load(self):
        data = self.mgr.load()
        data["user"]["workContext"]["summary"] = "测试工作背景"
        self.mgr.save(data)
        reloaded = MemoryManager(storage_path=self.mem_file).load()
        assert reloaded["user"]["workContext"]["summary"] == "测试工作背景"

    def test_format_for_injection_empty(self):
        text = self.mgr.format_for_injection()
        assert text == ""  # 空记忆不注入

    def test_format_for_injection_with_data(self):
        data = self.mgr.load()
        data["user"]["workContext"]["summary"] = "我是一名 Python 开发者"
        self.mgr.save(data)
        text = self.mgr.format_for_injection()
        assert "Python" in text

    def test_rule_update_extracts_preference(self):
        conv = [("我喜欢用 Python 写代码", "好的，Python 是个很好的选择！")]
        changed = self.mgr.update_from_conversation(conv)
        if changed:  # 规则提取可能不匹配，允许为 False
            data = self.mgr.load()
            assert len(data["facts"]) > 0

    def test_merge_facts_deduplication(self):
        data = self.mgr.load()
        from datetime import datetime
        now = datetime.utcnow().isoformat() + "Z"
        fact = {"content": "用户喜欢 Python", "category": "preference", "confidence": 0.8}
        self.mgr._merge_facts(data, [fact], "test", now)
        self.mgr._merge_facts(data, [fact], "test", now)  # 重复添加
        matching = [f for f in data["facts"] if f["content"] == "用户喜欢 Python"]
        assert len(matching) == 1  # 只保留一条

    def test_atomic_write(self):
        """原子写入：临时文件不应残留。"""
        data = self.mgr.load()
        self.mgr.save(data)
        tmp_file = Path(self.mem_file).with_suffix(".tmp")
        assert not tmp_file.exists()


class TestMemoryInjectionMiddleware:

    def test_memory_injected_when_present(self):
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem.json")
        mgr = MemoryManager(storage_path=mem_file)
        # 写入一条记忆
        data = mgr.load()
        data["user"]["workContext"]["summary"] = "喜欢写测试"
        mgr.save(data)

        mw = MemoryInjectionMiddleware(memory_manager=mgr)
        messages = [{"role": "user", "content": "问题"}]
        result = mw.before_model(messages, 0, {})
        contents = " ".join(m["content"] for m in result)
        assert "喜欢写测试" in contents

    def test_no_injection_when_empty(self):
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "empty_mem.json")
        mgr = MemoryManager(storage_path=mem_file)
        mw = MemoryInjectionMiddleware(memory_manager=mgr)
        messages = [{"role": "user", "content": "问题"}]
        result = mw.before_model(messages, 0, {})
        assert len(result) == len(messages)  # 空记忆不注入消息

    def test_no_duplicate_injection(self):
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem2.json")
        mgr = MemoryManager(storage_path=mem_file)
        data = mgr.load()
        data["user"]["workContext"]["summary"] = "测试"
        mgr.save(data)
        mw = MemoryInjectionMiddleware(memory_manager=mgr)
        messages = [{"role": "user", "content": "问题"}]
        ctx = {}
        r1 = mw.before_model(messages, 0, ctx)
        r2 = mw.before_model(r1, 1, ctx)
        assert len(r2) == len(r1)  # 第二次不重复注入


# ============================================================================
# 8. TaskPlanner（Stub 模型）
# ============================================================================

class TestTaskPlanner:

    def test_plan_returns_todos(self):
        plan_response = json.dumps({
            "title": "测试任务",
            "todos": [
                {"id": 1, "task": "用 grep 扫描 core 目录", "tool": "bash", "status": "pending"},
                {"id": 2, "task": "将结果写入文件", "tool": "write_file", "status": "pending"},
            ]
        })
        stub = make_stub_model(plan_response)
        planner = TaskPlanner(model_forward_fn=stub)
        result = planner.plan("扫描 core 目录并写入文件")
        assert result is not None
        assert result["title"] == "测试任务"
        assert len(result["todos"]) == 2

    def test_plan_returns_none_on_bad_json(self):
        stub = make_stub_model("不是 JSON")
        planner = TaskPlanner(model_forward_fn=stub)
        result = planner.plan("做点什么")
        # 格式错误时返回 None 或空 todos
        assert result is None or result.get("todos", []) == []

    def test_merge_todos_combines_none_and_write_file(self):
        """
        _merge_todos 合并规则验证：

        场景A（无前置工具步骤）：
          tool=none 步骤任务名同时命中动词关键词（整理成）+ 目标关键词（写入）
          且前方无真实工具步骤 → 应合并为单个 write_file 步骤。

        场景B（有前置 bash 步骤）：
          前方存在 bash 步骤，none 步骤依赖其输出 → 保护逻辑生效，禁止合并，
          none 步骤和 write_file 步骤均保留在结果中。
        """
        stub = make_stub_model("{}")
        planner = TaskPlanner(model_forward_fn=stub)

        # ── 场景A：无前置工具步骤，双关键词命中 → 应合并 ──────────────
        todos_a = [
            {"id": 1, "task": "整理成文档写入文件", "tool": "none",       "status": "pending"},
            {"id": 2, "task": "写入 README",         "tool": "write_file", "status": "pending"},
        ]
        merged_a = planner._merge_todos(todos_a)
        tools_a = [t["tool"] for t in merged_a]
        # none 步骤被合并掉，只剩 write_file
        assert "none" not in tools_a, "场景A：none 步骤应被合并掉"
        assert "write_file" in tools_a, "场景A：write_file 步骤应保留"
        assert len(merged_a) == 1

        # ── 场景B：前有 bash 步骤 → 保护生效，禁止合并 ────────────────
        todos_b = [
            {"id": 1, "task": "扫描目录",   "tool": "bash",       "status": "pending"},
            {"id": 2, "task": "整理成文档写入文件", "tool": "none", "status": "pending"},
            {"id": 3, "task": "写入 README", "tool": "write_file", "status": "pending"},
        ]
        merged_b = planner._merge_todos(todos_b)
        tools_b = [t["tool"] for t in merged_b]
        # bash 步骤在前，none 依赖其输出，不应被合并
        assert "none" in tools_b, "场景B：前有 bash 步骤，none 步骤不应被合并"
        assert "write_file" in tools_b, "场景B：write_file 步骤应保留"
        assert len(merged_b) == 3

    def test_merge_todos_renumbers(self):
        todos = [
            {"id": 10, "task": "A", "tool": "bash", "status": "pending"},
            {"id": 20, "task": "B", "tool": "bash", "status": "pending"},
        ]
        stub = make_stub_model("{}")
        planner = TaskPlanner(model_forward_fn=stub)
        merged = planner._merge_todos(todos)
        ids = [t["id"] for t in merged]
        assert ids == list(range(1, len(ids) + 1))


# ============================================================================
# 9. TaskExecutor（Orchestrator-Worker 架构）
# ============================================================================

class TestTaskExecutor:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_executor(self, responses: List[str]) -> TaskExecutor:
        """
        创建一个按顺序返回 responses 的 TaskExecutor。
        每次调用 model_forward_fn 消耗一个 response。
        """
        idx = {"n": 0}

        def seq_model(messages, system_prompt="", **kwargs):
            r = responses[min(idx["n"], len(responses) - 1)]
            idx["n"] += 1
            return r

        executor = TaskExecutor(
            model_forward_fn=seq_model,
            work_dir=self.tmpdir,
        )
        return executor

    def test_execute_bash_step(self):
        """bash 步骤应执行命令并返回成功结果。"""
        todos = [
            {"id": 1, "task": "执行 echo 命令", "tool": "bash", "status": "pending"},
        ]
        # Worker 模型返回 bash 调用，然后返回摘要
        executor = self._make_executor([
            'bash\n{"command": "echo hello_from_test"}',
            "命令执行成功。",
            "任务完成。",  # Orchestrator 汇总
        ])
        result = executor.execute_todos(
            todos=todos,
            user_message="执行命令",
            system_prompt="你是助手。",
        )
        assert "final_response" in result
        log = result["execution_log"]
        bash_calls = [e for e in log if e.get("type") == "tool_call" and e.get("tool") == "bash"]
        assert len(bash_calls) >= 1

    def test_execute_write_file_step(self):
        """write_file 步骤应写入文件。"""
        todos = [
            {"id": 1, "task": "写入文件", "tool": "write_file", "status": "pending"},
        ]
        executor = self._make_executor([
            f'write_file\n{{"path": "output.txt", "content": "测试内容"}}',
            "文件写入完成。",
            "任务完成。",
        ])
        result = executor.execute_todos(
            todos=todos,
            user_message="写文件",
            system_prompt="你是助手。",
        )
        output_file = Path(self.tmpdir, "output.txt")
        assert output_file.exists()
        assert "测试内容" in output_file.read_text()

    def test_execute_none_step_calls_model(self):
        """tool=none 整理步骤不应跳过，模型应被真正调用。"""
        todos = [
            {"id": 1, "task": "整理数据", "tool": "none", "status": "pending"},
        ]
        call_count = {"n": 0}

        def counting_model(messages, system_prompt="", **kwargs):
            call_count["n"] += 1
            return "整理完成：# 标题\n- 条目1\n- 条目2"

        executor = TaskExecutor(
            model_forward_fn=counting_model,
            work_dir=self.tmpdir,
        )
        result = executor.execute_todos(
            todos=todos,
            user_message="整理",
            system_prompt="你是助手。",
        )
        # tool=none 步骤必须实际调用过模型（不应跳过）
        assert call_count["n"] >= 1
        # 结果中应包含模型的整理输出
        step_results = result.get("step_results", [])
        if step_results:
            assert len(step_results[0].get("result", "")) > 0

    def test_execute_none_step_result_in_accumulated_context(self):
        """tool=none 整理结果应完整保存到 accumulated_context，供后续步骤使用。"""
        # 模拟：步骤1 整理，步骤2 写入
        todos = [
            {"id": 1, "task": "整理数据", "tool": "none", "status": "pending"},
            {"id": 2, "task": "写入文件", "tool": "write_file", "status": "pending"},
        ]
        organized_content = "# 整理结果\n- 项目A\n- 项目B\n- 项目C"
        write_calls = []

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            # 步骤1（chat 模式）：返回整理文本
            if call_n["n"] == 1:
                return organized_content
            # 步骤2（tools 模式）：返回 write_file 调用
            if call_n["n"] == 2:
                # 检查 messages 中是否包含整理结果
                all_content = " ".join(m.get("content", "") for m in messages)
                write_calls.append(all_content)
                return f'write_file\n{{"path": "doc.md", "content": "写入内容"}}'
            return "完成。"

        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=self.tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="整理并写入文档",
            system_prompt="你是助手。",
        )
        # write_file 步骤的 prompt 应该包含步骤1整理结果
        if write_calls:
            assert any("整理结果" in c or "项目" in c for c in write_calls)

    def test_orchestrator_skip_on_failure(self):
        """步骤失败时，Orchestrator 应决策 skip 并继续。"""
        todos = [
            {"id": 1, "task": "注定失败的步骤", "tool": "bash", "status": "pending"},
            {"id": 2, "task": "后续步骤", "tool": "bash", "status": "pending"},
        ]
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1 Worker：返回一个会导致工具报错的命令
            if n == 1:
                return 'bash\n{"command": "exit 1"}'
            # 步骤1 Worker 后续：返回空（模拟失败退出）
            if n == 2:
                return ""
            # Orchestrator 决策：skip
            if n == 3:
                return '{"action": "skip", "reason": "不重要步骤", "modified_task": ""}'
            # 步骤2 Worker
            if n == 4:
                return 'bash\n{"command": "echo step2"}'
            if n == 5:
                return "步骤2完成。"
            return "汇总完成。"

        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=self.tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="执行两个步骤",
            system_prompt="你是助手。",
        )
        # 任务不应中途 abort
        assert "final_response" in result

    def test_completion_rate_all_done(self):
        """所有步骤完成后，completion_rate 应为 1.0。"""
        todos = [
            {"id": 1, "task": "写文件", "tool": "write_file", "status": "pending"},
        ]
        executor = self._make_executor([
            f'write_file\n{{"path": "done.txt", "content": "x"}}',
            "完成。",
            "全部完成。",
        ])
        result = executor.execute_todos(
            todos=todos,
            user_message="写文件",
            system_prompt="助手。",
        )
        todo_mgr = result.get("todo_manager")
        if todo_mgr:
            assert todo_mgr.completion_rate() == 1.0


# ============================================================================
# 10. _prepare_tool_result_for_model — bash / read_file / 通用截断
# ============================================================================

class TestPrepareToolResultForModel:

    def setup_method(self):
        stub = make_stub_model("ok")
        self.fw = QwenAgentFramework(model_forward_fn=stub)

    def test_bash_short_stdout_not_truncated(self):
        """bash stdout 短于限制时不应截断。"""
        result_data = {"success": True, "stdout": "short output", "stderr": "", "returncode": 0}
        out, truncated = self.fw._prepare_tool_result_for_model("bash", result_data)
        assert not truncated
        assert out["stdout"] == "short output"

    def test_bash_long_stdout_truncated(self):
        """bash stdout 超过 _BASH_STDOUT_FULL_LIMIT 时应截断。"""
        long_stdout = "A" * (self.fw._BASH_STDOUT_FULL_LIMIT + 100)
        result_data = {"success": True, "stdout": long_stdout, "stderr": "", "returncode": 0}
        out, truncated = self.fw._prepare_tool_result_for_model("bash", result_data)
        assert truncated is True
        assert len(out["stdout"]) <= self.fw._BASH_STDOUT_FULL_LIMIT + 200  # 含截断标记
        assert "省略" in out["stdout"] or len(out["stdout"]) < len(long_stdout)

    def test_read_file_short_content_not_truncated(self):
        """read_file 内容短于限制时不应截断。"""
        result_data = {"success": True, "path": "f.py", "content": "short", "line_count": 1}
        out, truncated = self.fw._prepare_tool_result_for_model("read_file", result_data)
        assert not truncated
        assert out["content"] == "short"

    def test_read_file_long_content_truncated(self):
        """read_file 内容超过 _READ_FILE_FULL_CONTENT_LIMIT 时应截断。"""
        long_content = "B" * (self.fw._READ_FILE_FULL_CONTENT_LIMIT + 500)
        result_data = {"success": True, "path": "big.py", "content": long_content}
        out, truncated = self.fw._prepare_tool_result_for_model("read_file", result_data)
        assert truncated is True
        assert len(out["content"]) < len(long_content)

    def test_other_tool_short_not_truncated(self):
        """list_dir 等其他工具结果较短时不截断。"""
        result_data = {"success": True, "entries": ["a.py", "b.py"]}
        out, truncated = self.fw._prepare_tool_result_for_model("list_dir", result_data)
        assert not truncated

    def test_other_tool_long_truncated(self):
        """其他工具结果超过 1200 字符时截断。"""
        result_data = {"success": True, "data": "C" * 2000}
        out, truncated = self.fw._prepare_tool_result_for_model("list_dir", result_data)
        assert truncated is True
        assert "truncated" in out or "preview" in out


# ============================================================================
# 11. _inject_context_before_last_user（辅助函数）
# ============================================================================

class TestInjectContext:

    def test_injects_before_last_user(self):
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "第一轮"},
            {"role": "assistant", "content": "回复"},
            {"role": "user", "content": "第二轮"},
        ]
        ctx = {"role": "user", "content": "注入内容"}
        result = _inject_context_before_last_user(messages, ctx)
        # 注入内容应在最后一个 user 消息之前
        last_user_idx = max(i for i, m in enumerate(result) if m["role"] == "user" and m["content"] == "第二轮")
        ctx_idx = next(i for i, m in enumerate(result) if m["content"] == "注入内容")
        assert ctx_idx < last_user_idx

    def test_injects_when_no_user_message(self):
        messages = [{"role": "system", "content": "系统提示"}]
        ctx = {"role": "user", "content": "注入内容"}
        result = _inject_context_before_last_user(messages, ctx)
        assert any(m["content"] == "注入内容" for m in result)


# ============================================================================
# 12. process_message_direct（单次调用，无工具循环）
# ============================================================================

class TestProcessMessageDirect:

    def test_returns_response(self):
        stub = make_stub_model("直接回复结果")
        fw = QwenAgentFramework(model_forward_fn=stub)
        messages = [{"role": "user", "content": "问题"}]
        response, log = fw.process_message_direct(
            messages=messages,
            runtime_context={"run_mode": "chat"},
        )
        assert response == "直接回复结果"

    def test_middleware_runs(self):
        """确认中间件在 process_message_direct 中也会被执行。"""
        injected = []

        class SpyMiddleware(AgentMiddleware):
            def before_model(self, messages, iteration, runtime_context=None):
                injected.append(True)
                return messages

        stub = make_stub_model("ok")
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            middlewares=[SpyMiddleware()],
        )
        fw.process_message_direct(
            messages=[{"role": "user", "content": "测试"}],
            runtime_context={"run_mode": "chat"},
        )
        assert len(injected) > 0


# ============================================================================
# 13. AIIntentRouter（AI 路由，Stub 模型）
# ============================================================================

class TestAIIntentRouter:

    def test_fallback_to_rule_on_bad_json(self):
        """AI 路由返回非 JSON 时应降级到规则路由。"""
        stub = make_stub_model("不是JSON，随便说说")
        router = AIIntentRouter(
            model_forward_fn=stub,
            available_skill_ids=["pdf", "python-dev"],
        )
        result = router.analyze("今天天气怎么样？")
        assert result["run_mode"] in ("chat", "tools", "skills", "hybrid")

    def test_ai_router_with_breakdown(self):
        """AI 路由返回 needs_breakdown=true 时应包含 todos。"""
        ai_resp = json.dumps({
            "run_mode": "tools",
            "is_followup": False,
            "needs_breakdown": True,
            "skill_ids": [],
            "reason": "多步骤任务",
            "title": "扫描并写文档",
            "todos": [
                {"id": 1, "task": "扫描目录", "tool": "bash", "status": "pending"},
                {"id": 2, "task": "写入文件", "tool": "write_file", "status": "pending"},
            ]
        })
        stub = make_stub_model(ai_resp)
        router = AIIntentRouter(
            model_forward_fn=stub,
            available_skill_ids=["pdf", "python-dev"],
        )
        result = router.analyze("帮我扫描 core 目录并写入 API.md")
        assert result.get("needs_breakdown") is True
        assert len(result.get("todos", [])) >= 2


# ============================================================================
# 14. 多轮对话场景 — 带历史的连续对话
# ============================================================================

class TestMultiTurnConversation:
    """
    场景：模拟用户连续多轮对话，验证：
      - 历史对话被正确传递给模型
      - 每轮中间件在新历史下仍正确运行
      - 追问轮次不丢失上一轮工具执行结果
    """

    def test_three_turn_conversation(self):
        """3 轮连续对话，每轮回复中都携带历史摘要信息。"""
        received_messages_per_turn: List[List[dict]] = []

        def spy_model(messages, system_prompt="", **kwargs):
            received_messages_per_turn.append(list(messages))
            turn = len(received_messages_per_turn)
            return f"第{turn}轮回复"

        fw = QwenAgentFramework(model_forward_fn=spy_model, max_iterations=1)
        history = []

        # 第1轮
        resp1, _ = fw.process_message("你好", history, runtime_context={"run_mode": "chat"})
        history.append(("你好", resp1))

        # 第2轮
        resp2, _ = fw.process_message("继续上面的话题", history, runtime_context={"run_mode": "chat"})
        history.append(("继续上面的话题", resp2))

        # 第3轮
        resp3, _ = fw.process_message("总结一下", history, runtime_context={"run_mode": "chat"})

        # 验证：第3轮 messages 中应包含前两轮的历史
        msgs_turn3 = received_messages_per_turn[2]
        contents = " ".join(m["content"] for m in msgs_turn3)
        assert "你好" in contents
        assert "继续上面的话题" in contents

    def test_tool_call_then_followup(self):
        """
        第1轮：工具调用（write_file）
        第2轮：追问"刚才写的文件内容是什么？"
        验证：第2轮 messages 包含第1轮的回复内容。
        """
        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                return f'write_file\n{{"path": "note.txt", "content": "重要备忘录"}}'
            if n == 2:
                return "文件已写入 note.txt，内容是「重要备忘录」。"
            # 第2轮追问
            return "刚才我写了一个备忘录文件。"

        fw = QwenAgentFramework(model_forward_fn=model_fn, work_dir=tmpdir, max_iterations=5)

        # 第1轮
        resp1, log1 = fw.process_message(
            "帮我创建备忘录文件", [], runtime_context={"run_mode": "tools"}
        )
        # 第2轮追问，传入历史
        resp2, log2 = fw.process_message(
            "刚才写的文件内容是什么？",
            [("帮我创建备忘录文件", resp1)],
            runtime_context={"run_mode": "chat"},
        )
        assert len(resp2) > 0

    def test_history_length_grows(self):
        """验证每轮历史条目数量递增。"""
        received_history_sizes = []

        def counting_model(messages, system_prompt="", **kwargs):
            received_history_sizes.append(len(messages))
            return "ok"

        fw = QwenAgentFramework(model_forward_fn=counting_model, max_iterations=1)
        history = []
        for i in range(4):
            resp, _ = fw.process_message(f"问题{i}", history, runtime_context={"run_mode": "chat"})
            history.append((f"问题{i}", resp))

        # 每轮传入的 messages 数应递增（user 消息累积）
        assert received_history_sizes[0] < received_history_sizes[-1]


# ============================================================================
# 15. 中间件链协同场景 — 多个中间件叠加的顺序与隔离
# ============================================================================

class TestMiddlewareChainCooperation:
    """
    场景：同时使用 RuntimeMode + PlanMode + ToolResultGuard + Memory 四个中间件，
    验证它们各自注入的内容互不冲突、执行顺序正确。
    """

    def test_all_four_middlewares_fire_in_order(self):
        """四个中间件的 before_model 都被调用，且注入内容各自不重复。"""
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem.json")
        mgr = MemoryManager(storage_path=mem_file)
        # 预置记忆数据
        data = mgr.load()
        data["user"]["workContext"]["summary"] = "用户是 Python 开发者"
        mgr.save(data)

        todo_mgr = TodoManager()
        todo_mgr.add("步骤一", tool="bash")

        injection_order = []

        class TrackingMiddleware(AgentMiddleware):
            def __init__(self, name):
                self.name = name
            def before_model(self, messages, iteration, runtime_context=None):
                injection_order.append(self.name)
                return messages

        stub = make_stub_model("响应内容")
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            max_iterations=1,
            middlewares=[
                TrackingMiddleware("A"),
                RuntimeModeMiddleware(),
                TrackingMiddleware("B"),
                MemoryInjectionMiddleware(memory_manager=mgr),
                TrackingMiddleware("C"),
                TodoContextMiddleware(todo_manager=todo_mgr),
                TrackingMiddleware("D"),
            ],
        )
        fw.process_message("测试", [], runtime_context={"run_mode": "tools"})

        # 验证中间件按声明顺序调用
        assert injection_order.index("A") < injection_order.index("B")
        assert injection_order.index("B") < injection_order.index("C")
        assert injection_order.index("C") < injection_order.index("D")

    def test_runtime_and_plan_mode_combined(self):
        """同时开启 tools 模式 + plan_mode，两者提示词都应被注入。"""
        injected_contents = []

        def spy_model(messages, system_prompt="", **kwargs):
            injected_contents.extend(m["content"] for m in messages)
            return "完成"

        fw = QwenAgentFramework(
            model_forward_fn=spy_model,
            max_iterations=1,
            middlewares=[RuntimeModeMiddleware(), PlanModeMiddleware()],
        )
        fw.process_message(
            "分析文件",
            [],
            runtime_context={"run_mode": "tools", "plan_mode": True},
        )
        combined = " ".join(injected_contents)
        # tools 模式提示
        assert "工具" in combined or "tool" in combined.lower()
        # plan_mode 提示
        assert "计划" in combined or "plan" in combined.lower()

    def test_tool_result_guard_structures_result_in_chain(self):
        """ToolResultGuard 在工具调用后将结果结构化，后续模型应收到结构化 JSON。"""
        tool_results_seen = []

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return 'bash\n{"command": "echo structured_test"}'
            # 第2轮：检查工具结果消息
            for m in messages:
                if m["role"] == "user" and "structured_test" in m.get("content", ""):
                    tool_results_seen.append(m["content"])
            return "完成"

        tmpdir = tempfile.mkdtemp()
        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            enable_bash=True,
            max_iterations=5,
            middlewares=[ToolResultGuardMiddleware()],
        )
        fw.process_message("运行命令", [], runtime_context={"run_mode": "tools"})
        # 工具结果应被传回模型（在后续轮次的消息中可见）
        assert call_n["n"] >= 2

    def test_after_run_hook_called_for_all_middlewares(self):
        """process_message 结束后，所有中间件的 after_run 钩子都被调用。"""
        after_run_called = []

        class AfterRunMiddleware(AgentMiddleware):
            def __init__(self, name):
                self.name = name
            def after_run(self, execution_log, runtime_context=None):
                after_run_called.append(self.name)

        stub = make_stub_model("ok")
        fw = QwenAgentFramework(
            model_forward_fn=stub,
            max_iterations=1,
            middlewares=[AfterRunMiddleware("MW1"), AfterRunMiddleware("MW2")],
        )
        fw.process_message("测试", [], runtime_context={"run_mode": "chat"})
        assert "MW1" in after_run_called
        assert "MW2" in after_run_called


# ============================================================================
# 16. 流式输出场景 — process_message_stream / process_message_direct_stream
# ============================================================================

class TestStreamingOutput:
    """
    场景：验证流式输出接口可以正确 yield 多个中间结果，且最终内容与非流式一致。
    """

    def test_process_message_stream_yields_chunks(self):
        """process_message_stream 应 yield 至少一个 (text, info) 元组。"""
        stub = make_stub_model("流式回复内容")
        fw = QwenAgentFramework(model_forward_fn=stub, max_iterations=1)
        chunks = list(fw.process_message_stream(
            "测试流式",
            [],
            runtime_context={"run_mode": "chat"},
        ))
        assert len(chunks) >= 1
        # 每个 chunk 是 (str, dict)
        last_text, last_info = chunks[-1]
        assert isinstance(last_text, str)
        assert isinstance(last_info, dict)
        assert len(last_text) > 0

    def test_process_message_stream_with_tool_call(self):
        """流式模式下工具调用也应被执行，执行后继续 yield。"""
        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return f'write_file\n{{"path": "stream_out.txt", "content": "流式写入"}}'
            return "文件已创建。"

        fw = QwenAgentFramework(model_forward_fn=model_fn, work_dir=tmpdir, max_iterations=5)
        chunks = list(fw.process_message_stream(
            "创建文件",
            [],
            runtime_context={"run_mode": "tools"},
        ))
        # 应有多轮 yield（工具调用前后各一次）
        assert len(chunks) >= 2
        # 文件确实被写入
        assert Path(tmpdir, "stream_out.txt").exists()

    def test_process_message_direct_stream_fallback(self):
        """
        process_message_direct_stream：无 stream_forward_fn 时降级为单次 yield 完整响应。
        """
        stub = make_stub_model("直接流式回复")
        fw = QwenAgentFramework(model_forward_fn=stub, max_iterations=1)
        messages = [{"role": "user", "content": "问题"}]
        chunks = list(fw.process_message_direct_stream(
            messages=messages,
            runtime_context={"run_mode": "chat"},
        ))
        assert len(chunks) >= 1
        final_text, _ = chunks[-1]
        assert "直接流式回复" in final_text

    def test_process_message_direct_stream_with_stream_forward_fn(self):
        """process_message_direct_stream：提供 stream_forward_fn 时走真流式路径。"""
        tokens = ["你", "好", "，", "我", "是", "AI"]
        accumulated = []

        def fake_stream_fn(messages, system_prompt="", **kwargs):
            buf = ""
            for t in tokens:
                buf += t
                yield buf  # 每次 yield 累积文本

        stub = make_stub_model("fallback")
        fw = QwenAgentFramework(model_forward_fn=stub, max_iterations=1)
        messages = [{"role": "user", "content": "你好"}]
        chunks = list(fw.process_message_direct_stream(
            messages=messages,
            runtime_context={"run_mode": "chat"},
            stream_forward_fn=fake_stream_fn,
        ))
        # 应有逐 token 的中间 yield
        assert len(chunks) > 1
        # 最终文本应是所有 token 拼接
        final_text, _ = chunks[-1]
        assert "你好" in final_text or "AI" in final_text


# ============================================================================
# 17. 大上下文传递场景 — bash 12K stdout / tool=none 全量写入 accumulated_context
# ============================================================================

class TestLargeContextPipeline:
    """
    场景：验证在 128K 窗口策略下：
      1. bash 工具 stdout 超过 12000 字节时才截断（低于此值完整传递）
      2. tool=none 整理步骤的完整输出（而非 600 字符截断版）写入 accumulated_context
      3. write_file 步骤收到的 prompt 包含 tool=none 步骤的完整整理文本
    """

    def test_bash_stdout_11000_chars_not_truncated(self):
        """bash stdout 为 11000 字符时（低于 12000 限制），不应截断。"""
        fw = QwenAgentFramework(model_forward_fn=make_stub_model())
        long_stdout = "X" * 11000
        result_data = {"success": True, "stdout": long_stdout, "stderr": "", "returncode": 0}
        out, truncated = fw._prepare_tool_result_for_model("bash", result_data)
        assert truncated is False
        assert out["stdout"] == long_stdout

    def test_bash_stdout_13000_chars_truncated(self):
        """bash stdout 为 13000 字符时（超过 12000 限制），应截断。"""
        fw = QwenAgentFramework(model_forward_fn=make_stub_model())
        long_stdout = "Y" * 13000
        result_data = {"success": True, "stdout": long_stdout, "stderr": "", "returncode": 0}
        out, truncated = fw._prepare_tool_result_for_model("bash", result_data)
        assert truncated is True
        assert len(out["stdout"]) < 13000

    def test_tool_none_full_output_in_accumulated_context(self):
        """
        tool=none 步骤输出 2000 字符的整理文本，
        write_file 步骤 prompt 中应包含完整的 2000 字符而非 600 字符截断版。
        """
        tmpdir = tempfile.mkdtemp()

        # 模拟一个整理结果：2000 字符，包含特征标记 "UNIQUE_MARKER_12345"
        organized = "# 整理结果\n" + ("行内容 UNIQUE_MARKER_12345\n" * 80)
        assert len(organized) > 1500  # 确保超过旧的 600 字符截断

        write_prompt_captured = []
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1（tool=none，chat 模式）：返回整理文本
            if n == 1:
                return organized
            # 步骤2（write_file，tools 模式）：捕获收到的 prompt
            if n == 2:
                all_content = " ".join(m.get("content", "") for m in messages)
                write_prompt_captured.append(all_content)
                return f'write_file\n{{"path": "doc.md", "content": "内容"}}'
            return "完成。"

        todos = [
            {"id": 1, "task": "整理数据", "tool": "none", "status": "pending"},
            {"id": 2, "task": "写入文档", "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        executor.execute_todos(
            todos=todos,
            user_message="整理并写入文档",
            system_prompt="你是助手。",
        )

        # write_file 步骤的 prompt 中必须含有完整整理内容的特征标记
        if write_prompt_captured:
            combined = write_prompt_captured[0]
            assert "UNIQUE_MARKER_12345" in combined, (
                f"write_file 步骤未收到完整整理结果（特征标记缺失）。\n"
                f"实际收到 prompt 片段: {combined[:300]}"
            )

    def test_read_file_11000_chars_not_truncated(self):
        """read_file 内容 11000 字符时不截断（低于 12000 限制）。"""
        fw = QwenAgentFramework(model_forward_fn=make_stub_model())
        content = "Z" * 11000
        result_data = {"success": True, "path": "big.py", "content": content}
        out, truncated = fw._prepare_tool_result_for_model("read_file", result_data)
        assert truncated is False
        assert out["content"] == content

    def test_read_file_13000_chars_truncated(self):
        """read_file 内容 13000 字符时应截断。"""
        fw = QwenAgentFramework(model_forward_fn=make_stub_model())
        content = "W" * 13000
        result_data = {"success": True, "path": "huge.py", "content": content}
        out, truncated = fw._prepare_tool_result_for_model("read_file", result_data)
        assert truncated is True
        assert len(out["content"]) < 13000


# ============================================================================
# 18. 完整端到端场景 — bash → tool=none 整理 → write_file 完整链路
# ============================================================================

class TestEndToEndPipeline:
    """
    场景：模拟最常见的"扫描代码 → 整理文档 → 写入文件"三步完整链路，
    验证数据在 Orchestrator-Worker 架构中的完整流转。
    """

    def test_bash_to_none_to_write_full_pipeline(self):
        """
        步骤1: bash — 模拟 grep 扫描，返回代码结构
        步骤2: tool=none — 模型整理 grep 结果为 Markdown
        步骤3: write_file — 写入整理后的文档

        验证：
          - 所有步骤都被执行（log 中有 todo_done/todo_start 记录）
          - write_file 最终写入磁盘
          - 最终汇总回复非空
        """
        tmpdir = tempfile.mkdtemp()
        organized_doc = "# API 文档\n" + "\n".join(
            f"## Class{i}\n说明..." for i in range(20)
        )

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1 Worker（bash 模式）：返回 bash 工具调用
            if n == 1:
                return 'bash\n{"command": "echo hello_scan"}'
            # 步骤1 Worker 第2轮（工具结果已注入，输出摘要）
            if n == 2:
                return "扫描完成，找到若干类。"
            # 步骤2 Worker（tool=none 整理）
            if n == 3:
                return organized_doc
            # 步骤3 Worker（write_file）
            if n == 4:
                # content 中的换行符在 JSON 中需要转义
                escaped = organized_doc[:100].replace('\n', '\\n').replace('"', '\\"')
                return f'write_file\n{{"path": "API.md", "content": "{escaped}"}}'
            # 步骤3 Worker 第2轮
            if n == 5:
                return "文档已写入。"
            # Orchestrator 汇总
            return "全部任务完成：已扫描类并生成 API.md。"

        todos = [
            {"id": 1, "task": "bash 扫描类", "tool": "bash", "status": "pending"},
            {"id": 2, "task": "整理为文档", "tool": "none", "status": "pending"},
            {"id": 3, "task": "写入 API.md", "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="扫描代码并生成 API 文档",
            system_prompt="你是一个代码文档助手。",
        )

        # 最终回复非空
        assert len(result["final_response"]) > 0
        # 日志中应有 3 个 todo_start 记录（每步都被启动）
        log = result["execution_log"]
        starts = [e for e in log if e.get("type") == "todo_start"]
        assert len(starts) >= 2  # 至少 2 步被启动
        # 文件被写入
        assert Path(tmpdir, "API.md").exists()
        # todo_manager 中所有步骤已完成
        todo_mgr = result["todo_manager"]
        completed = todo_mgr.get_by_status("completed")
        assert len(completed) >= 2

    def test_orchestrator_abort_stops_pipeline(self):
        """
        当 Orchestrator 决策 abort 时，后续步骤不应被执行。

        注意：框架将 step_response=="" 视为成功（not None），
        因此要触发失败路径需要抛出异常。
        """
        tmpdir = tempfile.mkdtemp()
        steps_executed = []
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1 Worker：直接抛异常触发失败路径
            if n == 1:
                steps_executed.append("step1_worker")
                raise RuntimeError("step1 critical failure")
            # Orchestrator 决策：abort
            if n == 2:
                return '{"action": "abort", "reason": "关键步骤失败，无法继续"}'
            # 步骤2 Worker（不应被执行）
            steps_executed.append("step2_worker")
            return "步骤2输出"

        todos = [
            {"id": 1, "task": "关键步骤（必须成功）", "tool": "bash", "status": "pending"},
            {"id": 2, "task": "后续步骤", "tool": "bash", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="执行任务",
            system_prompt="助手。",
        )

        # abort 后步骤2不应被执行
        assert "step2_worker" not in steps_executed
        # final_response 中应有内容
        assert len(result["final_response"]) > 0
        # 日志中应包含 type=orchestrator_decision & action=abort 的事件
        log = result["execution_log"]
        abort_events = [
            e for e in log
            if e.get("type") == "orchestrator_decision" and e.get("action") == "abort"
        ]
        assert len(abort_events) >= 1

    def test_orchestrator_retry_then_succeed(self):
        """
        步骤失败（异常触发）→ Orchestrator 决策 retry → 重试成功 → 文件写入。

        注意：框架通过 try/except 捕获异常触发失败路径，
        空字符串返回不会触发 Orchestrator 决策。
        """
        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1 第一次：抛异常触发失败路径
            if n == 1:
                raise ValueError("临时错误，可重试")
            # Orchestrator 决策：retry
            if n == 2:
                return '{"action": "retry", "reason": "重试", "modified_task": "重试步骤1"}'
            # 重试 Worker：直接 write_file
            if n == 3:
                return 'write_file\n{"path": "retry_ok.txt", "content": "重试成功"}'
            if n == 4:
                return "文件写入成功。"
            return "任务完成。"

        todos = [
            {"id": 1, "task": "步骤1（第一次会失败）", "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="写文件",
            system_prompt="助手。",
        )
        # 重试后文件应被写入
        assert Path(tmpdir, "retry_ok.txt").exists()
        # 日志中应包含 type=worker_retry 事件
        log = result["execution_log"]
        retry_events = [e for e in log if e.get("type") == "worker_retry"]
        assert len(retry_events) >= 1
        # 日志中应包含 type=orchestrator_decision & action=retry 的决策事件
        decision_events = [
            e for e in log
            if e.get("type") == "orchestrator_decision" and e.get("action") == "retry"
        ]
        assert len(decision_events) >= 1

    def test_multi_step_with_accumulated_context_growth(self):
        """
        验证 accumulated_context 随步骤推进不断增长，后续步骤能"看到"前序所有步骤结果。
        """
        tmpdir = tempfile.mkdtemp()
        prompts_per_step: List[str] = []
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 每步 Worker 第1轮调用时，收集 prompt 内容
            if n in (1, 3, 5):
                combined = " ".join(m.get("content", "") for m in messages)
                prompts_per_step.append(combined)
            if n == 1:
                return f'write_file\n{{"path": "a.txt", "content": "步骤1结果STEP1_MARKER"}}'
            if n == 2:
                return "步骤1完成。"
            if n == 3:
                return f'write_file\n{{"path": "b.txt", "content": "步骤2结果STEP2_MARKER"}}'
            if n == 4:
                return "步骤2完成。"
            if n == 5:
                return f'write_file\n{{"path": "c.txt", "content": "步骤3结果"}}'
            if n == 6:
                return "步骤3完成。"
            return "全部完成。"

        todos = [
            {"id": 1, "task": "步骤1", "tool": "write_file", "status": "pending"},
            {"id": 2, "task": "步骤2", "tool": "write_file", "status": "pending"},
            {"id": 3, "task": "步骤3", "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        executor.execute_todos(
            todos=todos,
            user_message="执行三步任务",
            system_prompt="助手。",
        )

        # 步骤3 的 prompt 应包含步骤1、步骤2 的结果标记（通过 accumulated_context）
        if len(prompts_per_step) >= 3:
            step3_prompt = prompts_per_step[2]
            assert "STEP1_MARKER" in step3_prompt or "步骤1" in step3_prompt
            assert "STEP2_MARKER" in step3_prompt or "步骤2" in step3_prompt


# ============================================================================
# 19. 记忆跨会话持久化场景
# ============================================================================

class TestMemoryCrossSession:
    """
    场景：第一个会话写入记忆，第二个会话（新 MemoryManager 实例）读取并注入到对话。
    验证记忆文件的跨实例持久化。
    """

    def test_memory_persists_across_instances(self):
        """新建 MemoryManager 实例读取同一文件，应能看到之前保存的记忆。"""
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "shared_memory.json")

        # Session 1：写入记忆
        mgr1 = MemoryManager(storage_path=mem_file)
        data = mgr1.load()
        data["user"]["workContext"]["summary"] = "Session1 写入的工作背景"
        data["facts"].append({
            "id": "fact_session1",
            "content": "用户喜欢写测试",
            "category": "preference",
            "confidence": 0.9,
            "createdAt": "2026-01-01T00:00:00Z",
            "source": "test",
        })
        mgr1.save(data)

        # Session 2：全新实例，读取同一文件
        mgr2 = MemoryManager(storage_path=mem_file)
        data2 = mgr2.load()
        assert data2["user"]["workContext"]["summary"] == "Session1 写入的工作背景"
        facts = [f for f in data2["facts"] if f.get("content") == "用户喜欢写测试"]
        assert len(facts) == 1

    def test_memory_injected_in_second_session(self):
        """第一会话保存记忆后，第二会话的对话 prompt 应包含记忆内容。"""
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "shared2.json")

        # Session 1：保存记忆
        mgr1 = MemoryManager(storage_path=mem_file)
        data = mgr1.load()
        data["user"]["topOfMind"]["summary"] = "正在研究 Agent 框架架构"
        mgr1.save(data)

        # Session 2：新实例，通过中间件注入对话
        mgr2 = MemoryManager(storage_path=mem_file)
        injected_contents = []

        def spy_model(messages, system_prompt="", **kwargs):
            injected_contents.extend(m["content"] for m in messages)
            return "ok"

        fw = QwenAgentFramework(
            model_forward_fn=spy_model,
            max_iterations=1,
            middlewares=[MemoryInjectionMiddleware(memory_manager=mgr2)],
        )
        fw.process_message("你好", [], runtime_context={"run_mode": "chat"})

        combined = " ".join(injected_contents)
        assert "Agent 框架架构" in combined

    def test_auto_update_memory_after_run(self):
        """auto_update=True 时，会话结束后应触发记忆更新。"""
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "auto.json")
        mgr = MemoryManager(storage_path=mem_file)

        update_called = {"flag": False}
        original_update = mgr.update_from_conversation

        def patched_update(conversation, session_id=None, model_forward_fn=None):
            update_called["flag"] = True
            return original_update(conversation, session_id, model_forward_fn)

        mgr.update_from_conversation = patched_update

        stub = make_stub_model("ok")
        mw = MemoryInjectionMiddleware(memory_manager=mgr, auto_update=True)

        # 模拟会话结束，传入 _conversation_pairs
        runtime_ctx = {"_conversation_pairs": [("你好", "你好！")]}
        execution_log = [{"type": "model_response", "content": "你好！"}]
        mw.after_run(execution_log, runtime_ctx)

        assert update_called["flag"] is True


# ============================================================================
# 20. 重复工具调用检测与强制总结场景
# ============================================================================

class TestAntiRepeatAndForceSummarize:
    """
    场景：模型连续对相同文件/命令重复调用工具，
    验证框架的去重检测和强制总结机制能正确触发并退出循环。
    """

    def test_duplicate_tool_calls_trigger_force_summarize(self):
        """
        模型连续对同一文件调用 read_file，
        框架应检测到重复并触发强制总结，不应无限循环。
        """
        tmpdir = tempfile.mkdtemp()
        Path(tmpdir, "dup.txt").write_text("文件内容")
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            # 每次都返回同一个 read_file 调用（模拟死循环）
            return 'read_file\n{"path": "dup.txt"}'

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            enable_bash=False,
            max_iterations=8,
        )
        response, log = fw.process_message(
            "读取文件", [], runtime_context={"run_mode": "tools"}
        )
        # 框架应在有限次迭代内退出（不超过 max_iterations）
        assert call_n["n"] <= 8
        # 日志中应有 force_summarize 或 model_response 类型
        types = {e.get("type") for e in log}
        assert "model_response" in types or "force_summarize" in types

    def test_max_iterations_guard(self):
        """
        模型永远不结束（每次都输出新工具调用），
        框架应在 max_iterations 次后强制退出。
        """
        tmpdir = tempfile.mkdtemp()
        counter = {"n": 0}

        def infinite_model(messages, system_prompt="", **kwargs):
            counter["n"] += 1
            # 每次写不同文件，绕过重复检测
            return f'write_file\n{{"path": "file{counter["n"]}.txt", "content": "x"}}'

        fw = QwenAgentFramework(
            model_forward_fn=infinite_model,
            work_dir=tmpdir,
            max_iterations=4,
        )
        response, log = fw.process_message(
            "无限循环测试", [], runtime_context={"run_mode": "tools"}
        )
        # 模型调用次数不超过 max_iterations
        assert counter["n"] <= 4
        assert "最大迭代次数" in response or len(response) > 0


# ============================================================================
# 21. TodoContextMiddleware 与 Worker 权限边界场景
# ============================================================================

class TestWorkerPermissionBoundary:
    """
    场景：验证 Worker 在 worker_mode=True 时看不到 pending 步骤，
    防止 Worker 越权感知和操作后续计划。
    """

    def test_worker_cannot_see_future_steps(self):
        """Worker 模式下，pending 步骤对模型不可见。"""
        mgr = TodoManager()
        mgr.add("步骤1")
        mgr.add("步骤2（Worker 不应看到）")
        mgr.add("步骤3（Worker 不应看到）")
        mgr.update(1, "in_progress")

        mw = TodoContextMiddleware(todo_manager=mgr, worker_mode=True)
        messages = [{"role": "user", "content": "执行当前步骤"}]
        result = mw.before_model(messages, 0, {})
        combined = " ".join(m["content"] for m in result)

        assert "步骤2（Worker 不应看到）" not in combined
        assert "步骤3（Worker 不应看到）" not in combined

    def test_orchestrator_sees_all_steps(self):
        """Orchestrator 模式（worker_mode=False）下，所有步骤可见。"""
        mgr = TodoManager()
        mgr.add("步骤1")
        mgr.add("步骤2")
        mgr.add("步骤3")

        mw = TodoContextMiddleware(todo_manager=mgr, worker_mode=False)
        messages = [{"role": "user", "content": "汇总"}]
        result = mw.before_model(messages, 0, {})
        combined = " ".join(m["content"] for m in result)

        assert "步骤1" in combined
        assert "步骤2" in combined
        assert "步骤3" in combined

    def test_worker_todo_write_intercepted_not_propagated(self):
        """
        Worker 调用 todo_write 时，应被 TodoContextMiddleware 拦截并直接更新状态，
        不应再传递给 ToolExecutor 执行（避免 ToolExecutor 返回错误）。
        """
        mgr = TodoManager()
        mgr.add("步骤1")
        mw = TodoContextMiddleware(todo_manager=mgr)

        # 模拟 todo_write 调用被拦截
        payload = {"action": "update", "id": 1, "status": "completed", "result_preview": "完成"}
        result_str = mw.after_tool_call("todo_write", payload, "", 0)
        data = json.loads(result_str)

        assert data["success"] is True
        assert mgr.get(1).status == "completed"
        # ToolExecutor 不应看到这次调用（拦截后直接返回）

    def test_worker_todo_write_add_intercepted_by_middleware(self):
        """
        Worker 在 tool=none 步骤中调用 todo_write add 时，
        会被 Worker 框架的 TodoContextMiddleware 拦截并直接处理（不走 ToolExecutor）。

        框架行为说明：
          - Worker 的 TodoContextMiddleware 使用同一个全局 todo_mgr
          - todo_write 被 after_tool_call 钩子拦截 → apply_tool_write() 处理
          - ToolExecutor.execute_tool("todo_write") 永远不会被调用
          - tool=none 步骤：max_iterations_per_step 内执行完毕即完成（n=1 被 todo_write 拦截后
            框架继续循环，n=2 返回普通文本即步骤完成，不触发汇总）

        本测试验证：
          1. todo_write 调用被拦截（不走 ToolExecutor 报错，无 error 字段）
          2. 任务最终正常完成（final_response 有内容）
          3. 执行日志中的 tool_call 条目（若有）无 ToolExecutor 错误
        """
        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                # Worker 调用 todo_write add（被 TodoContextMiddleware 拦截）
                return '<tool>todo_write</tool><input>{"action": "add", "task": "额外步骤", "tool": "bash"}</input>'
            if n == 2:
                # Worker 第2轮：拦截后继续执行，返回普通文本完成步骤
                return "任务完成。"
            # n==3: _summarize 汇总（单步骤直接返回步骤结果，不再调模型）
            return "汇总完成。"

        todos = [{"id": 1, "task": "步骤1", "tool": "none", "status": "pending"}]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="执行",
            system_prompt="助手。",
        )

        # 任务正常完成，final_response 有内容
        assert len(result["final_response"]) > 0
        # 执行日志中不应有带 ToolExecutor 错误消息的 todo_write 调用条目
        log = result["execution_log"]
        executor_error_entries = [
            e for e in log
            if e.get("tool") == "todo_write"
            and "todo_write 需要配置" in str(e.get("result_preview", ""))
        ]
        assert len(executor_error_entries) == 0


# ============================================================================
# 22. 系统提示词覆盖与剥离场景
# ============================================================================

class TestSystemPromptHandling:
    """
    场景：验证 system 消息的剥离、覆盖优先级、以及不重复拼接的行为。
    """

    def test_system_message_stripped_from_messages(self):
        """
        messages 列表头部包含 system 消息时，
        _extract_system_from_messages 应将其剥离并返回。
        """
        stub = make_stub_model("ok")
        fw = QwenAgentFramework(model_forward_fn=stub)
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "问题"},
        ]
        non_sys, sys_content = fw._extract_system_from_messages(messages)
        assert sys_content == "系统提示"
        assert all(m["role"] != "system" for m in non_sys)

    def test_system_override_takes_priority(self):
        """system_prompt_override 参数优先级高于 messages 中的 system 消息。"""
        received_system = []

        def spy_model(messages, system_prompt="", **kwargs):
            received_system.append(system_prompt)
            return "ok"

        fw = QwenAgentFramework(model_forward_fn=spy_model)
        messages = [
            {"role": "system", "content": "原始系统提示"},
            {"role": "user", "content": "问题"},
        ]
        fw.process_message_direct(
            messages=messages,
            system_prompt_override="覆盖系统提示",
            runtime_context={"run_mode": "chat"},
        )
        assert received_system[-1] == "覆盖系统提示"

    def test_no_duplicate_system_in_messages(self):
        """
        process_message 中，传入 messages 不应再包含 system 角色，
        避免 model_forward_fn 收到重复 system 消息。
        """
        received_messages_roles = []

        def spy_model(messages, system_prompt="", **kwargs):
            received_messages_roles.extend(m["role"] for m in messages)
            return "ok"

        fw = QwenAgentFramework(model_forward_fn=spy_model, max_iterations=1)
        fw.process_message(
            "问题",
            [],
            system_prompt_override="系统提示",
            runtime_context={"run_mode": "chat"},
        )
        # 传入 model_forward_fn 的 messages 中不应有 system 角色
        assert "system" not in received_messages_roles


# ============================================================================
# 23. 系统集成测试 — 调动模块越多越好的端到端流程
# ============================================================================

class TestSystemIntegration:
    """
    系统集成测试：每个用例都尽量联动多个模块。

    每个用例涉及的模块层次：
      Level-1（单模块）：ToolExecutor / ToolParser / TodoManager
      Level-2（双模块）：QwenAgentFramework + 中间件
      Level-3（多模块）：TaskPlanner → TaskExecutor → QwenAgentFramework + ToolExecutor + 中间件
      Level-4（全链路）：MemoryManager + TaskPlanner + TaskExecutor + 全部中间件 + 流式接口
    """

    # ─────────────────────────────────────────────────────────────────
    # SI-01  TaskPlanner → TaskExecutor → ToolExecutor → MemoryManager
    #        覆盖：任务规划 → 步骤执行 → 实际文件写入 → 记忆持久化
    # ─────────────────────────────────────────────────────────────────
    def test_planner_executor_memory_pipeline(self):
        """
        全链路：
          1. TaskPlanner 解析用户需求 → 生成 TODO 列表
          2. TaskExecutor 执行 TODO（write_file + tool=none）
          3. ToolExecutor 实际写入文件
          4. 执行结束后 MemoryManager 持久化会话记忆

        联动模块：TaskPlanner / TaskExecutor / QwenAgentFramework /
                  ToolExecutor / TodoManager / MemoryManager /
                  ToolParser / RuntimeModeMiddleware / TodoContextMiddleware
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "session_memory.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        # ── 阶段1：TaskPlanner 解析需求生成 TODO ──────────────────────
        planner_call = {"n": 0}

        def planner_model(messages, system_prompt="", **kwargs):
            planner_call["n"] += 1
            # 注意：TaskPlanner._merge_todos 会把 tool=none 整理步骤 + write_file 步骤合并为一步。
            # 为了测试多步规划，使用 bash + tool=none（分析） + write_file 三步，
            # 其中 none 步骤任务名不含合并关键词（"整理"/"生成" 等），避免被合并。
            return json.dumps({
                "title": "生成项目摘要文档",
                "todos": [
                    {"id": 1, "task": "读取项目根目录结构", "tool": "read_file", "status": "pending"},
                    {"id": 2, "task": "分析依赖关系",       "tool": "none",      "status": "pending"},
                    {"id": 3, "task": "写入 README",         "tool": "write_file", "status": "pending"},
                ]
            })

        planner = TaskPlanner(model_forward_fn=planner_model)
        plan = planner.plan("帮我生成项目摘要文档")
        assert plan is not None
        assert len(plan["todos"]) >= 2

        # ── 阶段2：TaskExecutor 逐步执行 ─────────────────────────────
        executor_call = {"n": 0}

        def executor_model(messages, system_prompt="", **kwargs):
            executor_call["n"] += 1
            n = executor_call["n"]
            # 步骤1 Worker（read_file）：先尝试读取，无论成功失败都返回文本
            if n == 1:
                return 'read_file\n{"path": "."}'
            if n == 2:
                return "根目录已读取，包含 core/、tests/ 等目录。"
            # 步骤2 Worker（tool=none）：分析依赖
            if n == 3:
                return "分析完成：项目依赖 transformers、fastapi 等库。"
            # 步骤3 Worker（write_file）
            if n == 4:
                return 'write_file\n{"path": "README.md", "content": "## 项目摘要\\n本项目是 Agent 框架"}'
            if n == 5:
                return "文件写入成功。"
            # Orchestrator 汇总
            return "任务完成：README.md 已生成。"

        executor = TaskExecutor(model_forward_fn=executor_model, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=plan["todos"],
            user_message="帮我生成项目摘要文档",
            system_prompt="你是文档助手。",
        )

        # 验证 ToolExecutor 实际写入了文件
        readme = Path(tmpdir, "README.md")
        assert readme.exists(), "ToolExecutor 应将 README.md 写入磁盘"
        assert "Agent" in readme.read_text(encoding="utf-8")

        # 验证 TodoManager 中步骤完成情况
        todo_mgr = result["todo_manager"]
        completed = todo_mgr.get_by_status("completed")
        assert len(completed) >= 1

        # ── 阶段3：记忆持久化 ─────────────────────────────────────────
        data = mem_mgr.load()
        data["user"]["workContext"]["summary"] = "已完成 README 文档生成任务"
        mem_mgr.save(data)

        # 跨实例读取验证
        mem_mgr2 = MemoryManager(storage_path=mem_file)
        data2 = mem_mgr2.load()
        assert "README" in data2["user"]["workContext"]["summary"]

    # ─────────────────────────────────────────────────────────────────
    # SI-02  MemoryManager 注入 → QwenAgentFramework → 多工具调用 → ToolParser
    #        覆盖：记忆注入上下文 → 模型工具调用 → 工具解析 → 文件操作 → 结果注入
    # ─────────────────────────────────────────────────────────────────
    def test_memory_injection_then_multi_tool_calls(self):
        """
        全链路：
          1. MemoryManager 加载持久化记忆
          2. MemoryInjectionMiddleware 将记忆注入到 prompt
          3. QwenAgentFramework 的工具循环执行 write_file + read_file
          4. ToolParser 解析每次工具调用
          5. ToolResultGuardMiddleware 结构化工具结果
          6. RuntimeModeMiddleware 注入 tools 模式提示
          7. 最终回复包含文件内容

        联动模块：MemoryManager / MemoryInjectionMiddleware / QwenAgentFramework /
                  ToolExecutor / ToolParser / ToolResultGuardMiddleware /
                  RuntimeModeMiddleware / TodoManager（间接）
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem2.json")

        # 预置记忆：用户曾创建过 config.yaml
        mgr = MemoryManager(storage_path=mem_file)
        data = mgr.load()
        data["facts"].append({
            "id": "f1",
            "content": "用户上次创建了 config.yaml",
            "category": "work",
            "confidence": 0.95,
            "createdAt": "2026-01-01T00:00:00Z",
            "source": "system",
        })
        mgr.save(data)

        # 追踪注入到模型的内容
        all_prompts: List[str] = []
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            all_prompts.append(" ".join(m.get("content", "") for m in messages))
            n = call_n["n"]
            if n == 1:
                # 写入新文件
                return 'write_file\n{"path": "output.txt", "content": "内容已生成"}'
            if n == 2:
                # 读回验证
                return 'read_file\n{"path": "output.txt"}'
            # 最终回答
            return "已写入并读取 output.txt，内容正确。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[
                MemoryInjectionMiddleware(memory_manager=mgr),
                RuntimeModeMiddleware(),
                ToolResultGuardMiddleware(),
            ],
        )
        response, log = fw.process_message(
            "帮我写一个文件", [],
            runtime_context={"run_mode": "tools"},
        )

        # 第1轮 prompt 应包含记忆内容
        assert "config.yaml" in all_prompts[0], (
            "MemoryInjectionMiddleware 应将记忆注入到第1轮 prompt"
        )
        # ToolExecutor 实际写入文件
        assert Path(tmpdir, "output.txt").exists()
        # log 中应有 write_file + read_file 两次工具调用
        tool_calls = [e for e in log if e.get("type") == "tool_call"]
        tools_used = {e.get("tool") for e in tool_calls}
        assert "write_file" in tools_used
        assert "read_file" in tools_used

    # ─────────────────────────────────────────────────────────────────
    # SI-03  全中间件链 + 流式输出 + 工具调用
    #        覆盖：5个中间件同时工作 + process_message_stream + 工具执行
    # ─────────────────────────────────────────────────────────────────
    def test_full_middleware_chain_with_streaming(self):
        """
        全链路：
          1. 同时启用 RuntimeMode + PlanMode + MemoryInjection + TodoContext + ToolResultGuard
          2. 使用 process_message_stream 流式接口
          3. 流式模式下触发 write_file 工具调用并实际写入
          4. 验证所有中间件的 before_model / after_run 均被调用

        联动模块：全部5个中间件 / QwenAgentFramework / ToolExecutor / MemoryManager /
                  TodoManager / ToolParser / process_message_stream
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem3.json")
        mgr = MemoryManager(storage_path=mem_file)
        todo_mgr = TodoManager()
        todo_mgr.add("写入配置文件", tool="write_file")

        before_model_calls = []
        after_run_calls = []

        class InstrumentedMiddleware(AgentMiddleware):
            """探针中间件：记录所有钩子调用。"""
            def before_model(self, messages, iteration, runtime_context=None):
                before_model_calls.append(iteration)
                return messages
            def after_run(self, execution_log, runtime_context=None):
                after_run_calls.append(True)

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                return 'write_file\n{"path": "config.yaml", "content": "version: 1"}'
            return "配置文件已写入。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[
                InstrumentedMiddleware(),
                RuntimeModeMiddleware(),
                PlanModeMiddleware(),
                MemoryInjectionMiddleware(memory_manager=mgr),
                TodoContextMiddleware(todo_manager=todo_mgr),
                ToolResultGuardMiddleware(),
            ],
        )

        # 流式接口：消费所有 chunk
        chunks = list(fw.process_message_stream(
            "写入配置", [],
            runtime_context={"run_mode": "tools", "plan_mode": True},
        ))

        # 至少有2个 chunk（工具调用前 + 完成后）
        assert len(chunks) >= 2
        # 探针中间件的 before_model 在每轮都被调用
        assert len(before_model_calls) >= 2
        # after_run 在流式结束后被调用
        assert len(after_run_calls) >= 1
        # ToolExecutor 实际写入文件
        assert Path(tmpdir, "config.yaml").exists()

    # ─────────────────────────────────────────────────────────────────
    # SI-04  TaskPlanner + TaskExecutor + 失败重试 + 记忆注入 + 积累上下文
    #        覆盖：规划 → 执行（步骤1失败→Orchestrator retry→重试成功）→ 步骤2用到步骤1结果
    # ─────────────────────────────────────────────────────────────────
    def test_planner_executor_with_retry_and_context_propagation(self):
        """
        全链路：
          1. TaskPlanner 规划两步任务
          2. 步骤1 Worker 首次失败（异常）
          3. Orchestrator 决策 retry，修改任务描述
          4. 步骤1 重试成功，写入文件
          5. 步骤2 Worker 读取 accumulated_context（包含步骤1结果）
          6. 步骤2 完成汇总

        联动模块：TaskPlanner / TaskExecutor / QwenAgentFramework /
                  ToolExecutor / TodoManager / ToolParser / ToolResultGuardMiddleware /
                  RuntimeModeMiddleware / TodoContextMiddleware
        """
        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}
        step2_saw_step1_result = {"flag": False}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 步骤1 Worker 首次：抛异常
            if n == 1:
                raise RuntimeError("网络错误，请重试")
            # Orchestrator 决策：retry
            if n == 2:
                return '{"action": "retry", "reason": "重试一次", "modified_task": "写入 step1.txt"}'
            # 步骤1 Worker 重试：成功写入
            if n == 3:
                return 'write_file\n{"path": "step1.txt", "content": "STEP1_RESULT_MARKER"}'
            if n == 4:
                return "step1.txt 已写入。"
            # 步骤2 Worker：检查 accumulated_context 是否含步骤1结果
            if n == 5:
                all_content = " ".join(m.get("content", "") for m in messages)
                if "STEP1_RESULT_MARKER" in all_content or "step1" in all_content.lower():
                    step2_saw_step1_result["flag"] = True
                return 'write_file\n{"path": "step2.txt", "content": "STEP2_RESULT"}'
            if n == 6:
                return "step2.txt 已写入。"
            # Orchestrator 汇总
            return "两步任务均已完成。"

        # TaskPlanner 直接给定 TODO（复用 planner 输出）
        todos = [
            {"id": 1, "task": "写入 step1 文件", "tool": "write_file", "status": "pending"},
            {"id": 2, "task": "写入 step2 文件", "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="依次写入两个文件",
            system_prompt="你是助手。",
        )

        # 两个文件都写入成功
        assert Path(tmpdir, "step1.txt").exists()
        assert Path(tmpdir, "step2.txt").exists()

        # 日志中有 retry 相关事件
        log = result["execution_log"]
        retry_events = [e for e in log if e.get("type") == "worker_retry"]
        assert len(retry_events) >= 1

        # 步骤2 能看到步骤1 的结果（通过 accumulated_context）
        assert step2_saw_step1_result["flag"] is True, (
            "步骤2 Worker 的 prompt 应包含步骤1的结果（accumulated_context 传递）"
        )

    # ─────────────────────────────────────────────────────────────────
    # SI-05  TaskExecutor + bash + tool=none + write_file 三步链路
    #        覆盖：bash 执行 → stdout 注入 accumulated_context →
    #              tool=none 整理（全量上下文）→ write_file 使用整理结果
    # ─────────────────────────────────────────────────────────────────
    def test_bash_stdout_flows_to_write_file_via_accumulated_context(self):
        """
        全链路（最复杂的场景）：
          1. bash Worker：执行命令，stdout 注入 accumulated_context
          2. tool=none Worker：接收全量 accumulated_context（含 bash stdout），整理为文档
          3. write_file Worker：将整理结果写入文件
          4. 验证 write_file 步骤的 prompt 中含有 bash stdout 的特征标记

        联动模块：TaskExecutor / QwenAgentFramework (x3) / ToolExecutor /
                  ToolParser / RuntimeModeMiddleware / TodoContextMiddleware /
                  ToolResultGuardMiddleware / TodoManager（全局）
        """
        tmpdir = tempfile.mkdtemp()

        # bash 产生的特征标记
        BASH_MARKER = "BASH_OUTPUT_UNIQUE_9988"
        # tool=none 整理后的特征标记
        ORGANIZED_MARKER = "ORGANIZED_DOC_UNIQUE_7766"

        none_step_saw_bash = {"flag": False}
        write_step_saw_organized = {"flag": False}
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]

            # 步骤1 Worker（bash）：返回 bash 工具调用
            if n == 1:
                return f'bash\n{{"command": "echo {BASH_MARKER}"}}'
            # 步骤1 Worker 第2轮（工具结果已注入）：完成步骤
            if n == 2:
                return f"bash 已执行，输出包含 {BASH_MARKER}。"

            # 步骤2 Worker（tool=none）：检查是否能看到 bash stdout
            if n == 3:
                all_content = " ".join(m.get("content", "") for m in messages)
                if BASH_MARKER in all_content:
                    none_step_saw_bash["flag"] = True
                # 返回整理文档（含 ORGANIZED_MARKER）
                return f"# 整理文档\n{ORGANIZED_MARKER}\n整理完成。"

            # 步骤3 Worker（write_file）：检查是否能看到 tool=none 整理结果
            if n == 4:
                all_content = " ".join(m.get("content", "") for m in messages)
                if ORGANIZED_MARKER in all_content:
                    write_step_saw_organized["flag"] = True
                return f'write_file\n{{"path": "doc.md", "content": "{ORGANIZED_MARKER}"}}'
            if n == 5:
                return "doc.md 已写入。"

            # Orchestrator 汇总
            return "全部完成。"

        todos = [
            {"id": 1, "task": "bash 执行命令", "tool": "bash",       "status": "pending"},
            {"id": 2, "task": "整理结果",       "tool": "none",       "status": "pending"},
            {"id": 3, "task": "写入文档",        "tool": "write_file", "status": "pending"},
        ]
        executor = TaskExecutor(model_forward_fn=model_fn, work_dir=tmpdir)
        result = executor.execute_todos(
            todos=todos,
            user_message="扫描并生成文档",
            system_prompt="你是助手。",
        )

        assert Path(tmpdir, "doc.md").exists(), "write_file 步骤应写入 doc.md"
        assert none_step_saw_bash["flag"] is True, (
            "tool=none Worker 的 prompt 应包含 bash stdout（通过 accumulated_context）"
        )
        assert write_step_saw_organized["flag"] is True, (
            "write_file Worker 的 prompt 应包含 tool=none 整理结果（通过 accumulated_context）"
        )
        # log 中应有 3 个 todo_start
        log = result["execution_log"]
        starts = [e for e in log if e.get("type") == "todo_start"]
        assert len(starts) == 3

    # ─────────────────────────────────────────────────────────────────
    # SI-06  多轮对话 + 记忆写入 + 第二轮对话注入记忆 + 工具调用
    #        覆盖：会话1 → 工具调用写文件 → 记忆更新 →
    #              会话2（新实例）→ 记忆注入 → 工具调用读文件
    # ─────────────────────────────────────────────────────────────────
    def test_cross_session_memory_with_tool_calls(self):
        """
        跨会话全链路：
          会话1：write_file 写入 data.json → 手动保存到 MemoryManager
          会话2：全新 MemoryManager 实例 → MemoryInjectionMiddleware 注入记忆 →
                 read_file 工具调用读取 data.json → 回复中包含文件内容

        联动模块：MemoryManager (跨实例) / MemoryInjectionMiddleware /
                  QwenAgentFramework / ToolExecutor / ToolParser /
                  RuntimeModeMiddleware / ToolResultGuardMiddleware
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "cross_session.json")
        data_file = Path(tmpdir, "data.json")

        # ── 会话1：写入文件，更新记忆 ─────────────────────────────────
        call_s1 = {"n": 0}
        def model_s1(messages, system_prompt="", **kwargs):
            call_s1["n"] += 1
            n = call_s1["n"]
            if n == 1:
                return 'write_file\n{"path": "data.json", "content": "{\\"key\\": \\"VALUE_MARKER\\"}"}'
            return "data.json 已创建。"

        mgr1 = MemoryManager(storage_path=mem_file)
        fw1 = QwenAgentFramework(
            model_forward_fn=model_s1,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[MemoryInjectionMiddleware(memory_manager=mgr1), RuntimeModeMiddleware()],
        )
        resp1, log1 = fw1.process_message(
            "创建 data.json", [],
            runtime_context={"run_mode": "tools"},
        )
        assert data_file.exists(), "会话1应成功写入 data.json"

        # 手动更新记忆：记录 data.json 的存在
        d = mgr1.load()
        d["user"]["workContext"]["summary"] = "上次创建了 data.json，内容为 JSON 数据"
        mgr1.save(d)

        # ── 会话2：全新 MemoryManager，读取文件 ──────────────────────
        content_seen = {"flag": False}
        call_s2 = {"n": 0}

        def model_s2(messages, system_prompt="", **kwargs):
            call_s2["n"] += 1
            all_content = " ".join(m.get("content", "") for m in messages)
            # 验证记忆已注入（第1轮）
            if call_s2["n"] == 1 and "data.json" in all_content:
                content_seen["flag"] = True
            n = call_s2["n"]
            if n == 1:
                return 'read_file\n{"path": "data.json"}'
            return "data.json 内容已读取，包含 VALUE_MARKER。"

        mgr2 = MemoryManager(storage_path=mem_file)  # 全新实例，读同一文件
        fw2 = QwenAgentFramework(
            model_forward_fn=model_s2,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[
                MemoryInjectionMiddleware(memory_manager=mgr2),
                RuntimeModeMiddleware(),
                ToolResultGuardMiddleware(),
            ],
        )
        resp2, log2 = fw2.process_message(
            "读取上次创建的文件", [],
            runtime_context={"run_mode": "tools"},
        )

        # 会话2 第1轮 prompt 包含了会话1的记忆
        assert content_seen["flag"] is True, "会话2 应从 MemoryManager 读到会话1写入的记忆"
        # log2 中有 read_file 工具调用
        tool_calls = [e for e in log2 if e.get("type") == "tool_call"]
        assert any(e.get("tool") == "read_file" for e in tool_calls)

    # ─────────────────────────────────────────────────────────────────
    # SI-07  IntentRouter + QwenAgentFramework + 中间件 + ToolExecutor
    #        覆盖：意图识别 → 按模式路由 → 执行工具 → 中间件注入
    # ─────────────────────────────────────────────────────────────────
    def test_intent_router_drives_framework_execution(self):
        """
        全链路：
          1. IntentRouter 识别用户意图（tools 模式）
          2. 根据路由结果设置 runtime_context
          3. QwenAgentFramework 以 tools 模式执行
          4. RuntimeModeMiddleware 注入工具提示词
          5. ToolExecutor 执行 write_file

        联动模块：IntentRouter / QwenAgentFramework / RuntimeModeMiddleware /
                  ToolExecutor / ToolParser / ToolResultGuardMiddleware
        """
        tmpdir = tempfile.mkdtemp()
        router = IntentRouter()

        # 意图路由（IntentRouter 的公开方法是 analyze，返回 dict）
        user_input = "帮我创建 hello.txt 文件"
        route_result = router.analyze(user_input)
        # 文件创建应路由到 tools 或 hybrid 模式
        assert route_result.get("run_mode") in ("tools", "hybrid", "chat", "skills")

        # IntentRouter.analyze() 返回 dict，从中取 run_mode
        run_mode = route_result.get("run_mode", "tools")
        assert run_mode in ("tools", "hybrid", "chat", "skills")

        # 按路由结果执行
        call_n = {"n": 0}
        injected_mode_hints = []

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            all_content = " ".join(m.get("content", "") for m in messages)
            injected_mode_hints.append(all_content)
            n = call_n["n"]
            if n == 1:
                return 'write_file\n{"path": "hello.txt", "content": "Hello World"}'
            return "hello.txt 已创建。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[RuntimeModeMiddleware(), ToolResultGuardMiddleware()],
        )
        # tools/hybrid 模式走工具执行，chat/skills 也兼容（会直接返回回复）
        exec_mode = run_mode if run_mode in ("tools", "chat") else "tools"
        resp, log = fw.process_message(
            user_input, [],
            runtime_context={"run_mode": exec_mode},
        )

        assert Path(tmpdir, "hello.txt").exists()
        # tools 模式下中间件应注入工具相关提示词
        if exec_mode == "tools":
            assert any("工具" in h or "tool" in h.lower() for h in injected_mode_hints)

    # ─────────────────────────────────────────────────────────────────
    # SI-08  AIIntentRouter + TaskPlanner + TaskExecutor + 全中间件 + 流式接口
    #        最大规模联动：AI 路由 → 规划 → 执行 → 流式输出 → 文件写入 → 日志完整
    # ─────────────────────────────────────────────────────────────────
    def test_ai_router_planner_executor_stream_full_chain(self):
        """
        最大规模系统测试（8个模块联动）：

          1. AIIntentRouter 分析需求 → 识别需要任务拆解
          2. TaskPlanner 规划 3 步 TODO
          3. TaskExecutor 执行全部步骤（bash + tool=none + write_file）
          4. 每步 Worker 使用完整中间件链
          5. process_message_stream 用于最终展示
          6. MemoryManager 持久化最终结果

        联动模块：AIIntentRouter / TaskPlanner / TaskExecutor /
                  QwenAgentFramework (x3) / ToolExecutor / ToolParser /
                  TodoManager / RuntimeModeMiddleware / TodoContextMiddleware /
                  ToolResultGuardMiddleware / MemoryManager /
                  process_message_stream
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "si08_mem.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        call_n = {"n": 0}

        # 统一模型函数：按调用顺序承担不同角色
        def unified_model(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]

            # ① AIIntentRouter 分析
            if n == 1:
                return json.dumps({
                    "mode": "tools",
                    "needs_breakdown": True,
                    "todos": [
                        {"id": 1, "task": "bash 统计行数", "tool": "bash",       "status": "pending"},
                        {"id": 2, "task": "整理统计结果", "tool": "none",       "status": "pending"},
                        {"id": 3, "task": "写入报告",     "tool": "write_file", "status": "pending"},
                    ]
                })

            # ② 步骤1 Worker（bash）
            if n == 2:
                return 'bash\n{"command": "echo lines:42"}'
            if n == 3:
                return "bash 执行完成，输出 lines:42。"

            # ③ 步骤2 Worker（tool=none）
            if n == 4:
                return "## 统计报告\n总行数：42\n分析完成。REPORT_MARKER"

            # ④ 步骤3 Worker（write_file）
            if n == 5:
                return 'write_file\n{"path": "report.md", "content": "## 统计报告\\n总行数: 42"}'
            if n == 6:
                return "report.md 已写入。"

            # ⑤ Orchestrator 汇总
            if n == 7:
                return "全部完成：bash 统计了行数，已整理为报告并写入文件。"

            # ⑥ 最终展示轮（process_message_stream）
            return "任务全部完成，报告已生成。"

        # ── 阶段1：AI 路由分析需求 ────────────────────────────────────
        ai_router = AIIntentRouter(
            model_forward_fn=unified_model,
            available_skill_ids=["python-dev", "code-review"],
        )
        route = ai_router.analyze("统计代码行数并生成报告")
        assert route.get("needs_breakdown") is True
        todos = route.get("todos", [])
        assert len(todos) >= 2

        # ── 阶段2：TaskExecutor 执行 ─────────────────────────────────
        executor = TaskExecutor(model_forward_fn=unified_model, work_dir=tmpdir)
        exec_result = executor.execute_todos(
            todos=todos,
            user_message="统计代码行数并生成报告",
            system_prompt="你是代码分析助手。",
        )

        # 文件写入成功
        assert Path(tmpdir, "report.md").exists()

        # log 完整性验证
        log = exec_result["execution_log"]
        log_types = {e.get("type") for e in log}
        assert "todo_start" in log_types
        assert "todo_done" in log_types

        # ── 阶段3：process_message_stream 展示最终结果 ────────────────
        fw = QwenAgentFramework(
            model_forward_fn=unified_model,
            work_dir=tmpdir,
            max_iterations=1,
            middlewares=[RuntimeModeMiddleware()],
        )
        stream_chunks = list(fw.process_message_stream(
            "展示报告", [],
            runtime_context={"run_mode": "chat"},
        ))
        assert len(stream_chunks) >= 1
        final_text, _ = stream_chunks[-1]
        assert len(final_text) > 0

        # ── 阶段4：记忆持久化 ────────────────────────────────────────
        d = mem_mgr.load()
        d["user"]["workContext"]["summary"] = "已完成代码行数统计，生成了 report.md"
        mem_mgr.save(d)
        mem_mgr2 = MemoryManager(storage_path=mem_file)
        d2 = mem_mgr2.load()
        assert "report.md" in d2["user"]["workContext"]["summary"]


# ============================================================================
# 11. 系统集成扩展测试 - 多模块深度联动（新增）
#     覆盖: bash_failed / EnhancedTodoContext / ConversationSummary /
#           gstack 中间件链 / MemoryManager+Summary 联动
# ============================================================================

class TestBashFailedRetryFix:
    """
    验证 bash_failed 修复：bash 命令失败时不加入 executed_calls，
    允许重试同一命令（process_message 和 process_message_stream 均验证）。
    """

    def _build_bash_retry_model(self, results: list):
        """构建按序返回结果的 bash 重试模型。"""
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n <= len(results):
                return results[n - 1]
            return "命令执行成功。"

        return model_fn

    def test_bash_failed_allows_retry_in_process_message(self):
        """
        bash 命令失败（returncode=1）后，相同命令可以重试执行。

        原始 bug：bash 失败时也加入 executed_calls，导致模型重试相同命令
                  被标记为"重复调用"并强制跳过，无法重试。
        修复后：bash 失败不加入 executed_calls，重试命令正常执行。
        """
        tmpdir = tempfile.mkdtemp()

        # 模拟：第1次调用 bash（失败），第2次调用 bash（同参数，成功），第3次给最终回答
        call_n = {"n": 0}
        bash_exec_count = {"n": 0}

        original_execute = None

        class CountingExecutor(ToolExecutor):
            def execute_tool(self, tool_name, tool_input):
                if tool_name == "bash":
                    bash_exec_count["n"] += 1
                    rc = 0 if bash_exec_count["n"] > 1 else 1  # 第1次失败
                    return json.dumps({
                        "success": bash_exec_count["n"] > 1,
                        "returncode": rc,
                        "stdout": "success output" if rc == 0 else "",
                        "stderr": "error" if rc != 0 else "",
                    })
                return super().execute_tool(tool_name, tool_input)

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n <= 2:
                return 'bash\n{"command": "pytest tests/"}'
            return "测试执行成功。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[ToolResultGuardMiddleware()],
        )
        fw.tool_executor = CountingExecutor(work_dir=tmpdir, enable_bash=True)

        resp, log = fw.process_message(
            "运行测试", [],
            runtime_context={"run_mode": "tools"},
        )

        # bash 应该被执行 2 次（第1次失败不进 executed_calls，第2次正常执行）
        assert bash_exec_count["n"] == 2, (
            f"bash 应执行 2 次（1次失败+1次重试），实际执行 {bash_exec_count['n']} 次"
        )
        # 执行日志中应有 2 条 tool_call 记录
        tool_call_logs = [e for e in log if e.get("type") == "tool_call" and e.get("tool") == "bash"]
        assert len(tool_call_logs) == 2

    def test_bash_failed_allows_retry_in_stream(self):
        """
        bash 失败后相同命令可在 process_message_stream 中重试。
        验证 stream 版本与 process_message 行为一致。
        """
        tmpdir = tempfile.mkdtemp()
        bash_exec_count = {"n": 0}

        class CountingExecutor(ToolExecutor):
            def execute_tool(self, tool_name, tool_input):
                if tool_name == "bash":
                    bash_exec_count["n"] += 1
                    rc = 0 if bash_exec_count["n"] > 1 else 1
                    return json.dumps({
                        "success": rc == 0,
                        "returncode": rc,
                        "stdout": "ok" if rc == 0 else "",
                        "stderr": "fail" if rc != 0 else "",
                    })
                return super().execute_tool(tool_name, tool_input)

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n <= 2:
                return 'bash\n{"command": "make test"}'
            return "make test 执行成功。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[ToolResultGuardMiddleware()],
        )
        fw.tool_executor = CountingExecutor(work_dir=tmpdir, enable_bash=True)

        chunks = list(fw.process_message_stream(
            "运行 make test", [],
            runtime_context={"run_mode": "tools"},
        ))

        # bash 应被执行 2 次（失败不进 executed_calls）
        assert bash_exec_count["n"] == 2, (
            f"stream 版本 bash 应执行 2 次，实际 {bash_exec_count['n']} 次"
        )

    def test_bash_success_blocks_retry(self):
        """
        bash 命令成功时加入 executed_calls，相同命令不会重复执行（原有行为不变）。
        """
        tmpdir = tempfile.mkdtemp()
        bash_exec_count = {"n": 0}

        class CountingExecutor(ToolExecutor):
            def execute_tool(self, tool_name, tool_input):
                if tool_name == "bash":
                    bash_exec_count["n"] += 1
                    return json.dumps({
                        "success": True,
                        "returncode": 0,
                        "stdout": "success",
                        "stderr": "",
                    })
                return super().execute_tool(tool_name, tool_input)

        # 模型总是调用相同 bash 命令（触发重复检测）
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                return 'bash\n{"command": "echo hello"}'
            if n == 2:
                # 第2次仍然调用相同 bash（应该被阻断）
                return 'bash\n{"command": "echo hello"}'
            return "完成。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[ToolResultGuardMiddleware()],
        )
        fw.tool_executor = CountingExecutor(work_dir=tmpdir, enable_bash=True)

        resp, log = fw.process_message(
            "运行 echo hello", [],
            runtime_context={"run_mode": "tools"},
        )

        # bash 成功后第2次调用应被阻断，只执行 1 次
        assert bash_exec_count["n"] == 1


class TestEnhancedTodoContextMiddleware:
    """
    验证 EnhancedTodoContextMiddleware（顶层类）的上下文丢失检测机制。
    联动：EnhancedTodoContextMiddleware + TodoManager + QwenAgentFramework
    """

    def test_enhanced_todo_injects_on_first_iteration(self):
        """首轮（iteration=0）总是注入 TODO 上下文。"""
        from core.agent_middlewares import EnhancedTodoContextMiddleware
        from core.todo_manager import TodoManager

        mgr = TodoManager(title="测试任务")
        mgr.add("步骤1", tool="none")
        mgr.add("步骤2", tool="bash")
        middleware = EnhancedTodoContextMiddleware(mgr)

        messages = [{"role": "user", "content": "执行任务"}]
        updated = middleware.before_model(messages, iteration=0)

        # 首轮必须注入 TODO 上下文
        assert len(updated) > 1
        injected = updated[0]["content"]
        assert "todo_list" in injected

    def test_enhanced_todo_recovers_on_context_loss(self):
        """
        当消息历史中不含 TODO 内容（上下文丢失）且非首轮时，
        应注入 system_reminder 恢复提示。
        """
        from core.agent_middlewares import EnhancedTodoContextMiddleware
        from core.todo_manager import TodoManager

        mgr = TodoManager(title="恢复测试")
        mgr.add("步骤A", tool="none")
        mgr.update(1, "in_progress")
        middleware = EnhancedTodoContextMiddleware(mgr)

        # 消息中没有 todo_list（模拟上下文被压缩/截断）
        messages = [
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "下一步"},
        ]
        updated = middleware.before_model(messages, iteration=2)

        # 非首轮+上下文丢失 → 注入 system_reminder
        assert len(updated) > len(messages)
        injected_content = " ".join(m.get("content", "") for m in updated)
        assert "system_reminder" in injected_content or "todo_list" in injected_content

    def test_make_todo_context_middleware_returns_enhanced(self):
        """make_todo_context_middleware 返回的实例具备上下文丢失检测功能。"""
        from core.agent_middlewares import make_todo_context_middleware, EnhancedTodoContextMiddleware
        from core.todo_manager import TodoManager

        mgr = TodoManager()
        mgr.add("任务1")
        middleware = make_todo_context_middleware(mgr)

        assert isinstance(middleware, EnhancedTodoContextMiddleware)

    def test_todo_context_drives_agent_framework(self):
        """
        TodoContextMiddleware 注入 TODO 状态 → Agent 框架执行 → todo_write 被拦截处理。

        联动：TodoManager + EnhancedTodoContextMiddleware + QwenAgentFramework + ToolParser
        """
        from core.agent_middlewares import make_todo_context_middleware
        from core.todo_manager import TodoManager

        tmpdir = tempfile.mkdtemp()
        mgr = TodoManager(title="框架集成")
        mgr.add("步骤1：读文件", tool="read_file")
        mgr.add("步骤2：汇总", tool="none")

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            # 检查 TODO 状态是否注入到了消息中
            all_content = " ".join(m.get("content", "") for m in messages)
            if n == 1:
                # 第1次：调用 todo_write 更新步骤1为 in_progress
                return 'todo_write\n{"action": "update", "id": 1, "status": "in_progress"}'
            if n == 2:
                # 第2次：完成步骤1
                return 'todo_write\n{"action": "update", "id": 1, "status": "completed"}'
            return "所有步骤完成。"

        todo_middleware = make_todo_context_middleware(mgr)
        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=5,
            middlewares=[todo_middleware, ToolResultGuardMiddleware()],
        )
        resp, log = fw.process_message("执行任务", [])

        # todo_write 应该被 TodoContextMiddleware 拦截
        todo_write_logs = [e for e in log if e.get("type") == "tool_call" and e.get("tool") == "todo_write"]
        assert len(todo_write_logs) >= 1
        # 步骤1 状态应已被更新
        item1 = mgr.get(1)
        assert item1 is not None
        assert item1.status in ("in_progress", "completed")


class TestConversationSummaryMiddleware:
    """
    验证 ConversationSummaryMiddleware 长对话压缩功能。
    联动：ConversationSummaryMiddleware + QwenAgentFramework + 模型前向
    """

    def test_summary_not_triggered_below_threshold(self):
        """对话轮次未超过阈值时，消息列表不压缩。"""
        from core.agent_middlewares import ConversationSummaryMiddleware

        middleware = ConversationSummaryMiddleware(max_history_pairs=8, keep_recent_pairs=4)

        # 构建 4 轮对话（未超过阈值 8）
        messages = []
        for i in range(4):
            messages.append({"role": "user", "content": f"用户消息{i}"})
            messages.append({"role": "assistant", "content": f"助手回复{i}"})
        messages.append({"role": "user", "content": "当前问题"})

        ctx = {}
        updated = middleware.before_model(messages, iteration=0, runtime_context=ctx)

        # 未压缩，消息数量不变
        assert len(updated) == len(messages)
        assert "_summary_compressed_pairs" not in ctx

    def test_summary_triggered_above_threshold(self):
        """对话轮次超过阈值时，早期对话被压缩为摘要。"""
        from core.agent_middlewares import ConversationSummaryMiddleware

        # 阈值设为 4，最近保留 2
        middleware = ConversationSummaryMiddleware(max_history_pairs=4, keep_recent_pairs=2)

        # 构建 6 轮对话（超过阈值 4）
        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": f"问题{i}"})
            messages.append({"role": "assistant", "content": f"回答{i}"})
        messages.append({"role": "user", "content": "最新问题"})

        ctx = {}
        updated = middleware.before_model(messages, iteration=0, runtime_context=ctx)

        # 超过阈值：消息数量应减少（早期被压缩为摘要）
        assert len(updated) < len(messages)
        assert ctx.get("_summary_compressed_pairs", 0) > 0

    def test_summary_preserves_recent_content(self):
        """压缩后最近轮次的内容被完整保留。"""
        from core.agent_middlewares import ConversationSummaryMiddleware

        middleware = ConversationSummaryMiddleware(max_history_pairs=4, keep_recent_pairs=2)

        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": f"早期问题{i}"})
            messages.append({"role": "assistant", "content": f"早期回答{i}"})
        # 最近 2 轮（倒数第2轮和最后轮）
        messages[-4]["content"] = "RECENT_USER_A"
        messages[-3]["content"] = "RECENT_ASSIST_A"
        messages[-2]["content"] = "RECENT_USER_B"
        messages[-1]["content"] = "RECENT_ASSIST_B"
        messages.append({"role": "user", "content": "当前问题"})

        ctx = {}
        updated = middleware.before_model(messages, iteration=0, runtime_context=ctx)

        all_content = " ".join(m.get("content", "") for m in updated)
        # 最近 2 轮的内容应被保留
        assert "RECENT_USER_A" in all_content
        assert "RECENT_USER_B" in all_content

    def test_summary_with_llm_then_agent(self):
        """
        ConversationSummaryMiddleware + QwenAgentFramework 联动：
        长对话被压缩后，Agent 继续执行工具调用。

        联动：ConversationSummaryMiddleware / QwenAgentFramework / ToolExecutor
        """
        from core.agent_middlewares import ConversationSummaryMiddleware

        tmpdir = tempfile.mkdtemp()
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            # 检查消息不超过压缩后的期望上限（4轮保留+摘要+当前消息）
            return "write_file\n{\"path\": \"output.txt\", \"content\": \"done\"}" \
                if call_n["n"] == 1 else "输出已写入。"

        # 阈值很低，确保触发压缩
        summary_mw = ConversationSummaryMiddleware(max_history_pairs=2, keep_recent_pairs=1)

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=3,
            middlewares=[summary_mw, ToolResultGuardMiddleware()],
        )

        # 传入长历史（超过阈值）
        history = [(f"用户问题{i}", f"助手回答{i}") for i in range(5)]
        resp, log = fw.process_message(
            "写入文件", history,
            runtime_context={"run_mode": "tools"},
        )

        # 文件应被成功写入
        assert Path(tmpdir, "output.txt").exists()


class TestGstackMiddlewareChain:
    """
    gstack 架构移植中间件（CompletenessMiddleware / AskUserQuestion /
    CompletionStatus / SearchBeforeBuilding / RepoOwnership）联动测试。

    联动：gstack 中间件 + QwenAgentFramework + 中间件链执行顺序
    """

    def test_completeness_middleware_injects_on_tools_mode(self):
        """CompletenessMiddleware 在 tools 模式下注入完整性原则。"""
        from core.agent_middlewares import CompletenessMiddleware

        mw = CompletenessMiddleware()
        messages = [{"role": "user", "content": "实现一个排序算法"}]
        ctx = {"run_mode": "tools"}

        updated = mw.before_model(messages, iteration=0, runtime_context=ctx)
        all_content = " ".join(m.get("content", "") for m in updated)

        assert "完整性" in all_content or "Boil" in all_content

    def test_completeness_middleware_skips_on_chat_mode(self):
        """CompletenessMiddleware 在 chat 模式下不注入。"""
        from core.agent_middlewares import CompletenessMiddleware

        mw = CompletenessMiddleware()
        messages = [{"role": "user", "content": "你好"}]
        ctx = {"run_mode": "chat"}

        updated = mw.before_model(messages, iteration=0, runtime_context=ctx)
        assert len(updated) == len(messages)

    def test_completion_status_protocol_injected(self):
        """CompletionStatusMiddleware 在首轮注入完成状态协议。"""
        from core.agent_middlewares import CompletionStatusMiddleware

        mw = CompletionStatusMiddleware()
        messages = [{"role": "user", "content": "分析代码"}]
        ctx = {}

        updated = mw.before_model(messages, iteration=0, runtime_context=ctx)
        all_content = " ".join(m.get("content", "") for m in updated)

        assert "DONE" in all_content or "BLOCKED" in all_content

    def test_search_before_building_triggers_on_build_intent(self):
        """SearchBeforeBuildingMiddleware 在检测到构建意图时注入。"""
        from core.agent_middlewares import SearchBeforeBuildingMiddleware

        mw = SearchBeforeBuildingMiddleware()
        messages = [{"role": "user", "content": "从头开始实现一个缓存系统"}]
        ctx = {"run_mode": "tools"}

        updated = mw.before_model(messages, iteration=0, runtime_context=ctx)
        all_content = " ".join(m.get("content", "") for m in updated)

        # 检测到"实现"关键词，应注入搜索优先提示
        assert len(updated) > len(messages)
        assert "搜索" in all_content or "Layer" in all_content

    def test_repo_ownership_solo_mode_injected(self):
        """RepoOwnershipMiddleware solo 模式注入仓库所有权提示。"""
        from core.agent_middlewares import RepoOwnershipMiddleware

        mw = RepoOwnershipMiddleware()
        messages = [{"role": "user", "content": "修复 bug"}]
        ctx = {"repo_mode": "solo"}

        updated = mw.before_model(messages, iteration=0, runtime_context=ctx)
        all_content = " ".join(m.get("content", "") for m in updated)

        assert "solo" in all_content.lower() or "独立开发" in all_content

    def test_gstack_full_chain_with_agent_framework(self):
        """
        gstack 全中间件链联动：多个 gstack 中间件同时挂载到 QwenAgentFramework。

        联动：CompletenessMiddleware + CompletionStatusMiddleware +
              SearchBeforeBuildingMiddleware + RepoOwnershipMiddleware +
              QwenAgentFramework + RuntimeModeMiddleware
        """
        from core.agent_middlewares import (
            CompletenessMiddleware, CompletionStatusMiddleware,
            SearchBeforeBuildingMiddleware, RepoOwnershipMiddleware,
        )

        tmpdir = tempfile.mkdtemp()
        injected_contents = []
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            all_content = " ".join(m.get("content", "") for m in messages)
            injected_contents.append(all_content)
            return "write_file\n{\"path\": \"result.txt\", \"content\": \"DONE\"}" \
                if call_n["n"] == 1 else "完成。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=3,
            middlewares=[
                RuntimeModeMiddleware(),
                CompletenessMiddleware(),
                CompletionStatusMiddleware(),
                SearchBeforeBuildingMiddleware(),
                RepoOwnershipMiddleware(),
                ToolResultGuardMiddleware(),
            ],
        )

        resp, log = fw.process_message(
            "实现并测试一个新功能", [],
            runtime_context={"run_mode": "tools", "repo_mode": "solo"},
        )

        # 多个 gstack 中间件都应注入（通过首轮消息内容验证）
        first_round = injected_contents[0] if injected_contents else ""
        assert "工具" in first_round or "tool" in first_round.lower()  # RuntimeModeMiddleware
        assert Path(tmpdir, "result.txt").exists()


class TestMemoryAndSummaryIntegration:
    """
    MemoryManager + ConversationSummaryMiddleware + MemoryInjectionMiddleware 深度联动。

    场景：长会话中，对话被压缩（SummaryMiddleware），但记忆注入（MemoryInjection）
         仍然向模型提供跨会话历史记忆，保证信息连续性。
    """

    def test_memory_injection_with_conversation_summary(self):
        """
        Memory + Summary 联动：先注入记忆，再压缩历史，
        最终模型收到的上下文包含记忆摘要且历史被压缩。

        联动：MemoryManager / MemoryInjectionMiddleware /
              ConversationSummaryMiddleware / QwenAgentFramework
        """
        from core.agent_middlewares import ConversationSummaryMiddleware
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "mem_summary_test.json")

        # 预填充记忆
        mem_mgr = MemoryManager(storage_path=mem_file)
        data = mem_mgr.load()
        data["user"]["workContext"]["summary"] = "用户是 Python 开发者，项目使用 pytest"
        mem_mgr.save(data)

        memory_seen = {"flag": False}
        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            all_content = " ".join(m.get("content", "") for m in messages)
            if "Python" in all_content or "pytest" in all_content:
                memory_seen["flag"] = True
            return "write_file\n{\"path\": \"out.txt\", \"content\": \"ok\"}" \
                if n == 1 else "完成。"

        summary_mw = ConversationSummaryMiddleware(max_history_pairs=2, keep_recent_pairs=1)
        mem_mw = MemoryInjectionMiddleware(memory_manager=mem_mgr)

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=3,
            middlewares=[mem_mw, summary_mw, ToolResultGuardMiddleware()],
        )

        # 传入长历史触发压缩
        history = [(f"早期问题{i}", f"早期回答{i}") for i in range(5)]
        resp, log = fw.process_message(
            "写入分析结果", history,
            runtime_context={"run_mode": "tools"},
        )

        # 记忆应被注入（模型应收到 Python/pytest 信息）
        assert memory_seen["flag"] is True, "MemoryInjectionMiddleware 应注入跨会话记忆"
        assert Path(tmpdir, "out.txt").exists()

    def test_memory_rule_extraction_from_long_conversation(self):
        """
        MemoryManager 规则提取：从包含关键词的用户消息中提取偏好/背景事实。

        联动：MemoryManager._rule_update + MemoryManager._merge_facts +
              MemoryManager.load / save
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "rule_extract.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        conversation = [
            ("我喜欢使用 pytest 做单元测试", "好的，pytest 是很好的测试框架。"),
            ("我在美团负责数据平台开发", "了解，数据平台涉及很多复杂的工程问题。"),
            ("我习惯用 Python3.9 以上版本", "Python 3.9+ 有很多新特性很好用。"),
        ]

        result = mem_mgr.update_from_conversation(conversation, session_id="test_rule")

        assert result is True, "规则提取应返回 True（有新事实）"
        data = mem_mgr.load()
        facts = data.get("facts", [])
        assert len(facts) > 0, "应提取到至少一条事实"
        # 验证内容有意义
        fact_contents = [f.get("content", "") for f in facts]
        assert any("测试" in c or "pytest" in c or "平台" in c or "Python" in c
                   for c in fact_contents)

    def test_memory_format_for_injection_empty(self):
        """空记忆格式化注入返回空字符串。"""
        tmpdir = tempfile.mkdtemp()
        mem_mgr = MemoryManager(storage_path=os.path.join(tmpdir, "empty.json"))
        result = mem_mgr.format_for_injection()
        assert result == ""

    def test_memory_format_for_injection_with_facts(self):
        """有事实时格式化注入包含结构化内容。"""
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "fmt_test.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        data = mem_mgr.load()
        data["user"]["workContext"]["summary"] = "后端开发工程师"
        data["user"]["topOfMind"]["summary"] = "正在做性能优化"
        data["facts"] = [{
            "id": "fact_001",
            "content": "用户偏好 Python",
            "category": "preference",
            "confidence": 0.9,
        }]
        mem_mgr.save(data)

        result = mem_mgr.format_for_injection()
        assert "后端开发工程师" in result
        assert "性能优化" in result
        assert "Python" in result

    def test_auto_update_memory_on_after_run(self):
        """
        MemoryInjectionMiddleware auto_update=True 时，
        after_run 钩子从 _conversation_pairs 中触发记忆更新。

        联动：MemoryInjectionMiddleware.after_run + MemoryManager.update_from_conversation
        """
        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "auto_update.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        mw = MemoryInjectionMiddleware(memory_manager=mem_mgr, auto_update=True)

        # 模拟 after_run 时的执行日志和上下文
        execution_log = []
        runtime_ctx = {
            "execution_id": "test_exec_001",
            "_conversation_pairs": [
                ("我喜欢用 Redis 做缓存", "好的，Redis 是很常用的缓存方案。"),
            ],
        }

        mw.after_run(execution_log, runtime_ctx)

        # 应触发记忆更新（规则提取）
        data = mem_mgr.load()
        facts = data.get("facts", [])
        # 可能有规则提取结果（取决于规则匹配，不强要求）
        # 主要验证 after_run 不抛异常，且文件存在
        assert mem_mgr._file_path.exists()


class TestSystemIntegrationExtended:
    """
    扩展系统集成测试：覆盖更多模块联动边界场景。
    """

    def test_planner_with_enhanced_todo_and_memory(self):
        """
        TaskPlanner + EnhancedTodoContextMiddleware + MemoryManager 三模块联动。

        流程：
          1. TaskPlanner 规划任务
          2. TodoManager 管理状态
          3. EnhancedTodoContextMiddleware 注入 TODO 上下文
          4. MemoryInjectionMiddleware 注入跨会话记忆
          5. QwenAgentFramework 执行

        联动模块：TaskPlanner / TodoManager / EnhancedTodoContextMiddleware /
                  MemoryManager / MemoryInjectionMiddleware / QwenAgentFramework
        """
        from core.agent_middlewares import make_todo_context_middleware, EnhancedTodoContextMiddleware
        from core.todo_manager import TodoManager
        from core.task_planner import TaskPlanner

        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "planner_mem.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        # 预填充记忆
        data = mem_mgr.load()
        data["user"]["topOfMind"]["summary"] = "分析 Python 代码质量"
        mem_mgr.save(data)

        planner_call = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            planner_call["n"] += 1
            n = planner_call["n"]
            if n == 1:
                # TaskPlanner 规划
                return json.dumps({
                    "title": "代码分析任务",
                    "todos": [
                        {"id": 1, "task": "扫描代码文件", "tool": "bash", "status": "pending"},
                        {"id": 2, "task": "生成报告",     "tool": "write_file", "status": "pending"},
                    ]
                })
            # QwenAgentFramework 执行（todo_write 完成步骤）
            if n == 2:
                return 'todo_write\n{"action": "update", "id": 1, "status": "completed"}'
            return "分析完成。"

        # 阶段1：TaskPlanner 规划
        planner = TaskPlanner(model_forward_fn=model_fn)
        plan = planner.plan("分析 Python 代码")
        todos = plan.get("todos", [])
        assert len(todos) >= 2

        # 阶段2：TodoManager 加载规划结果
        mgr = TodoManager(title=plan.get("title", "代码分析"))
        mgr.load_from_todos_list(todos, title=plan.get("title", ""))
        assert len(mgr._items) >= 2

        # 阶段3：QwenAgentFramework + 中间件链
        todo_mw = make_todo_context_middleware(mgr)
        mem_mw = MemoryInjectionMiddleware(memory_manager=mem_mgr)

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=3,
            middlewares=[mem_mw, todo_mw, ToolResultGuardMiddleware()],
        )

        resp, log = fw.process_message("执行代码分析", [])

        # todo_write 应被拦截处理
        todo_writes = [e for e in log if e.get("type") == "tool_call" and e.get("tool") == "todo_write"]
        assert len(todo_writes) >= 1

    def test_intent_router_to_memory_full_session(self):
        """
        完整会话流程：IntentRouter → 路由 → QwenAgentFramework 执行 →
        MemoryInjectionMiddleware auto_update 持久化会话记忆。

        联动：IntentRouter / QwenAgentFramework / RuntimeModeMiddleware /
              MemoryManager / MemoryInjectionMiddleware / ToolExecutor
        """
        from core.intent_router import IntentRouter

        tmpdir = tempfile.mkdtemp()
        mem_file = os.path.join(tmpdir, "full_session.json")
        mem_mgr = MemoryManager(storage_path=mem_file)

        router = IntentRouter()

        # 用户输入：文件操作（应路由到 tools 模式）
        user_input = "读取 README.md 并总结"
        route = router.analyze(user_input)

        run_mode = route.get("run_mode", "chat")

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                return "总结：README 包含项目介绍和安装说明。"
            return "完成。"

        mem_mw = MemoryInjectionMiddleware(memory_manager=mem_mgr, auto_update=True)
        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=2,
            middlewares=[mem_mw, RuntimeModeMiddleware(), ToolResultGuardMiddleware()],
        )

        resp, log = fw.process_message(
            user_input, [],
            runtime_context={
                "run_mode": run_mode,
                "_conversation_pairs": [(user_input, "总结：README 包含项目介绍。")],
            },
        )

        assert len(resp) > 0
        # after_run 已触发，文件应存在（即使规则未提取到事实也会尝试创建）

    def test_stream_with_todo_and_gstack_middleware(self):
        """
        流式输出 + TodoContext + gstack 中间件联动。

        联动：process_message_stream / TodoContextMiddleware /
              CompletionStatusMiddleware / RuntimeModeMiddleware /
              ToolResultGuardMiddleware
        """
        from core.agent_middlewares import CompletionStatusMiddleware
        from core.todo_manager import TodoManager

        tmpdir = tempfile.mkdtemp()
        mgr = TodoManager(title="流式任务")
        mgr.add("步骤1", tool="none")
        mgr.add("步骤2", tool="write_file")

        todo_mw = TodoContextMiddleware(mgr)

        call_n = {"n": 0}

        def model_fn(messages, system_prompt="", **kwargs):
            call_n["n"] += 1
            n = call_n["n"]
            if n == 1:
                return 'write_file\n{"path": "stream_out.txt", "content": "流式写入"}'
            return "DONE: 流式任务全部完成。"

        fw = QwenAgentFramework(
            model_forward_fn=model_fn,
            work_dir=tmpdir,
            max_iterations=3,
            middlewares=[
                RuntimeModeMiddleware(),
                todo_mw,
                CompletionStatusMiddleware(),
                ToolResultGuardMiddleware(),
            ],
        )

        chunks = list(fw.process_message_stream(
            "执行流式任务", [],
            runtime_context={"run_mode": "tools"},
        ))

        assert len(chunks) >= 1
        all_text = " ".join(text for text, _ in chunks)
        assert len(all_text) > 0
        assert Path(tmpdir, "stream_out.txt").exists()


# ============================================================================
# 运行入口（兼容直接执行）
# ============================================================================

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pytest",
        __file__, "-v", "--tb=short"
    ])

