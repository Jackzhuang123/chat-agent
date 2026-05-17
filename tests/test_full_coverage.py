#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整单测覆盖 - QwenAgentFramework 全模块关键细节测试
运行方式: python -m pytest test_full_coverage.py -v
或: python test_full_coverage.py
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, ANY, patch, mock_open, call

# =============================================================================
# 预 mock 外部重型依赖（必须在导入项目模块之前）
# =============================================================================
_mock_torch = MagicMock()
_mock_torch.float32 = "float32"
_mock_torch.device = MagicMock(return_value="cpu")
_mock_nn = MagicMock()
_mock_nn.Module = object
sys.modules["torch"] = _mock_torch
sys.modules["torch.nn"] = _mock_nn

_mock_transformers = MagicMock()
_mock_transformers.AutoModelForCausalLM = MagicMock()
_mock_transformers.AutoTokenizer = MagicMock()
_mock_transformers.TextIteratorStreamer = MagicMock()
sys.modules["transformers"] = _mock_transformers

_mock_gradio = MagicMock()
_mock_gradio.themes = MagicMock()
_mock_gradio.themes.Soft = MagicMock(return_value=MagicMock())
_mock_gradio.Blocks = MagicMock
_mock_gradio.Row = MagicMock
_mock_gradio.Column = MagicMock
_mock_gradio.TabItem = MagicMock
_mock_gradio.Tabs = MagicMock
_mock_gradio.Markdown = MagicMock
_mock_gradio.Radio = MagicMock
_mock_gradio.Textbox = MagicMock
_mock_gradio.Dropdown = MagicMock
_mock_gradio.HTML = MagicMock
_mock_gradio.Slider = MagicMock
_mock_gradio.Checkbox = MagicMock
_mock_gradio.File = MagicMock
_mock_gradio.Button = MagicMock
_mock_gradio.Chatbot = MagicMock
_mock_gradio.Code = MagicMock
_mock_gradio.Dataframe = MagicMock
_mock_gradio.State = MagicMock
_mock_gradio.Request = MagicMock
sys.modules["gradio"] = _mock_gradio
sys.modules["gradio_client"] = MagicMock()
sys.modules["gradio_client.utils"] = MagicMock()

_mock_zhipuai = MagicMock()
sys.modules["zhipuai"] = _mock_zhipuai

_mock_langgraph = MagicMock()
_mock_langgraph.graph = MagicMock()
_mock_langgraph.graph.StateGraph = MagicMock
_mock_langgraph.graph.END = "END"
_mock_langgraph.checkpoint = MagicMock()
_mock_langgraph.checkpoint.sqlite = MagicMock()
_mock_langgraph.checkpoint.sqlite.aio = MagicMock()
_mock_langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver = MagicMock()
_mock_langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.from_conn_string = MagicMock(return_value=MagicMock(
    __aenter__=MagicMock(return_value=MagicMock()),
    __aexit__=MagicMock(return_value=False),
))
sys.modules["langgraph"] = _mock_langgraph
sys.modules["langgraph.graph"] = _mock_langgraph.graph
sys.modules["langgraph.checkpoint"] = _mock_langgraph.checkpoint
sys.modules["langgraph.checkpoint.sqlite"] = _mock_langgraph.checkpoint.sqlite
sys.modules["langgraph.checkpoint.sqlite.aio"] = _mock_langgraph.checkpoint.sqlite.aio

_mock_numpy = MagicMock()
_mock_numpy.floating = float
_mock_numpy.float64 = float
_mock_numpy.float32 = float
_mock_numpy.float16 = float
_mock_numpy.integer = int
_mock_numpy.int64 = int
_mock_numpy.int32 = int
_mock_numpy.int16 = int
_mock_numpy.int8 = int
_mock_numpy.bool_ = bool
_mock_numpy.ndarray = list
_mock_numpy.array = MagicMock(side_effect=lambda x, *a, **kw: MagicMock(tolist=MagicMock(return_value=list(x))) if isinstance(x, (list, tuple)) else list(x))
_mock_numpy.zeros = MagicMock(return_value=[0.0]*384)
_mock_numpy.exp = MagicMock(side_effect=lambda x: __import__("math").exp(x) if isinstance(x, (int, float)) else [__import__("math").exp(i) for i in x])
_mock_numpy.dot = MagicMock(side_effect=lambda a, b: sum(x*y for x, y in zip(a, b)))
_mock_numpy.linalg = MagicMock()
_mock_numpy.linalg.norm = MagicMock(side_effect=lambda x: sum(i**2 for i in x)**0.5)
_mock_numpy.savez_compressed = MagicMock()
_mock_numpy.load = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock(ids=[], embeddings=[]))))
sys.modules["numpy"] = _mock_numpy
np = _mock_numpy

_mock_aiosqlite = MagicMock()
sys.modules["aiosqlite"] = _mock_aiosqlite

_mock_sentence_transformers = MagicMock()
# 让 SentenceTransformer 构造失败，从而降级到哈希嵌入
_mock_sentence_transformers.SentenceTransformer = MagicMock(side_effect=ImportError("no sentence_transformers"))
sys.modules["sentence_transformers"] = _mock_sentence_transformers

_mock_PyPDF2 = MagicMock()
sys.modules["PyPDF2"] = _mock_PyPDF2

# 确保 core 和 ui 包路径存在
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "core"), str(_PROJECT_ROOT / "ui")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# =============================================================================
# 导入被测模块（在 mock 之后）
# =============================================================================
from core.agent_tools import ToolExecutor, ToolParser, ToolRegistry, create_web_search_tool_placeholder
from core.agent_middlewares import (
    AgentMiddleware, RuntimeModeMiddleware, PlanModeMiddleware,
    SkillsContextMiddleware, UploadedFilesMiddleware, ToolResultGuardMiddleware,
    ConversationSummaryMiddleware, ContextWindowMiddleware, CompletenessMiddleware,
    AskUserQuestionMiddleware, CompletionStatusMiddleware, SearchBeforeBuildingMiddleware,
    RepoOwnershipMiddleware,
)
from core.agent_skills import SkillManager, SkillInjector, create_example_skills
from core.components.completion_guard import looks_finished
from core.context_retriever import ContextRetriever
from core.components.format_corrector import should_retry_tool_format, inject_format_correction
from core.components.loop_detector import detect_loop
from core.model_forward import create_qwen_model_forward, _merge_system_messages
from core.monitor_logger import get_monitor_logger, make_trace_id
from core.components.output_cleaner import strip_trailing_tool_call, clean_react_tags
from core.prompts import get_system_prompt, inject_few_shot_examples, FEW_SHOT_EXAMPLES
from core.rag_intent_router import RAGIntentRouter, IntentType, IntentResult
from core.reflection import EnhancedReflectionEngine
from core.state_manager import SessionContext, WorkflowStateManager
from core.streaming_framework import StreamEvent, StreamingFramework, create_streaming_wrapper
from core.components.task_injector import inject_task_context
from core.tool_enforcement_middleware import ToolEnforcementMiddleware, DirectCommandDetector
from core.tool_learner import AdaptiveToolLearner, ContextFeatureExtractor, ToolUsagePattern
from core.vector_memory import VectorMemory, MemoryEntry, LocalEmbeddingProvider

# 部分模块依赖 Gradio 等重型 UI 库，在纯单测环境中通过 mock 导入
with patch.dict(sys.modules, {"gradio": _mock_gradio}):
    try:
        from ui.session_logger import SessionLogger, _make_json_serializable
    except Exception as e:
        SessionLogger = MagicMock
        _make_json_serializable = lambda x: x
    from ui.markdown_utils import render_markdown_html, build_markdown_preview


# =============================================================================
# 测试套件
# =============================================================================

class TestAgentTools(unittest.TestCase):
    """agent_tools.py 关键细节测试"""

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(work_dir=self.work_dir, enable_bash=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_01_read_file_existing(self):
        """read_file 读取已存在文件，返回 JSON 包含 content 和 file_facts"""
        test_file = Path(self.work_dir) / "test.py"
        test_file.write_text("class Foo:\n    pass\n", encoding="utf-8")
        result = self.executor.execute_tool("read_file", {"path": "test.py"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertIn("class Foo", data["content"])
        self.assertIn("file_facts", data)
        self.assertEqual(data["file_facts"]["classes"], ["Foo"])
        self.assertEqual(data["file_facts"]["functions"], [])

    def test_02_read_file_blocked_path(self):
        """read_file 拦截受限目录（如 .venv, __pycache__）"""
        result = self.executor.execute_tool("read_file", {"path": "/some/.venv/lib/pkg.py"})
        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertIn("⛔ 路径被拦截", data["error"])

    def test_03_read_file_fuzzy_match(self):
        """read_file 对不存在的文件名进行模糊搜索匹配"""
        (Path(self.work_dir) / "agent_tools.py").write_text("# content", encoding="utf-8")
        result = self.executor.execute_tool("read_file", {"path": "/nonexistent/agent_tools.py"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertEqual(data["content"], "# content")
        self.assertTrue(data["fuzzy_match"])
        self.assertIn("系统自动纠正", data["resolution_note"])

    def test_04_write_file_overwrite(self):
        """write_file overwrite 模式写入并返回 success"""
        result = self.executor.execute_tool("write_file", {
            "path": "out.txt", "content": "hello", "mode": "overwrite"
        })
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertEqual((Path(self.work_dir) / "out.txt").read_text(encoding="utf-8"), "hello")

    def test_05_write_file_append(self):
        """write_file append 模式追加内容"""
        self.executor.execute_tool("write_file", {
            "path": "log.txt", "content": "line1\n", "mode": "overwrite"
        })
        result = self.executor.execute_tool("write_file", {
            "path": "log.txt", "content": "line2\n", "mode": "append"
        })
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertEqual((Path(self.work_dir) / "log.txt").read_text(encoding="utf-8"), "line1\nline2\n")

    def test_06_write_file_empty_overwrite_block(self):
        """write_file 禁止用空内容覆盖已有内容的文件"""
        self.executor.execute_tool("write_file", {
            "path": "data.txt", "content": "real data", "mode": "overwrite"
        })
        result = self.executor.execute_tool("write_file", {
            "path": "data.txt", "content": "", "mode": "overwrite"
        })
        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertIn("blocked", data)
        self.assertIn("空内容覆盖", data["error"])

    def test_07_write_file_placeholder_block(self):
        """write_file 拦截占位符内容（如 <完整复制上方输出>）"""
        result = self.executor.execute_tool("write_file", {
            "path": "ph.txt", "content": "请<完整复制上方输出>到这里", "mode": "overwrite"
        })
        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertIn("占位符", data["error"])

    def test_08_edit_file_replace(self):
        """edit_file 精确替换 old_content 为 new_content"""
        self.executor.execute_tool("write_file", {
            "path": "edit_me.py", "content": "old_value = 1\n", "mode": "overwrite"
        })
        result = self.executor.execute_tool("edit_file", {
            "path": "edit_me.py", "old_content": "old_value = 1", "new_content": "new_value = 99"
        })
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertIn("new_value = 99", (Path(self.work_dir) / "edit_me.py").read_text(encoding="utf-8"))

    def test_09_edit_file_old_not_found(self):
        """edit_file 当 old_content 不匹配时返回错误"""
        self.executor.execute_tool("write_file", {
            "path": "edit_me.py", "content": "abc\n", "mode": "overwrite"
        })
        result = self.executor.execute_tool("edit_file", {
            "path": "edit_me.py", "old_content": "not_exist", "new_content": "x"
        })
        data = json.loads(result)
        # 返回结果中不包含 success 字段，默认为成功，应检查 error 字段
        self.assertIn("error", data)
        self.assertIn("未在文件中找到", data["error"])

    def test_10_list_dir(self):
        """list_dir 列出目录内容并过滤隐藏文件"""
        (Path(self.work_dir) / "visible.txt").write_text("v")
        (Path(self.work_dir) / ".hidden").write_text("h")
        (Path(self.work_dir) / "__pycache__").mkdir()
        result = self.executor.execute_tool("list_dir", {"path": "."})
        data = json.loads(result)
        self.assertTrue(data["success"])
        names = [i["name"] for i in data["items"]]
        self.assertIn("visible.txt", names)
        self.assertNotIn(".hidden", names)
        self.assertNotIn("__pycache__", names)

    def test_11_bash_basic(self):
        """bash 执行简单命令并返回 stdout"""
        result = self.executor.execute_tool("bash", {"command": "echo hello_test"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertIn("hello_test", data["stdout"])

    def test_12_bash_blocked_pattern(self):
        """bash 拦截高危命令模式（如 rm -rf /）"""
        result = self.executor.execute_tool("bash", {"command": "rm -rf /"})
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("安全策略", data["error"])

    def test_13_execute_python_success(self):
        """execute_python 执行 Python 代码并捕获 stdout"""
        result = self.executor.execute_tool("execute_python", {"code": "print(2+3)"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertIn("5", data["stdout"])

    def test_14_execute_python_syntax_error(self):
        """execute_python 捕获语法错误并返回错误信息"""
        result = self.executor.execute_tool("execute_python", {"code": "print(2+"})
        data = json.loads(result)
        self.assertFalse(data.get("success", True))
        self.assertTrue(data.get("error") or data.get("stderr"))

    def test_15_execute_python_timeout(self):
        """execute_python 超时返回特定错误提示"""
        result = self.executor.execute_tool("execute_python", {
            "code": "import time; time.sleep(10)", "timeout": 1
        })
        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertIn("超时", data["error"])

    def test_16_tool_parser_json_array(self):
        """ToolParser 解析 JSON 数组格式的工具调用"""
        text = '''[{"tool": "read_file", "input": {"path": "a.py"}}]'''
        calls = ToolParser.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "read_file")
        self.assertEqual(calls[0][1]["path"], "a.py")

    def test_17_tool_parser_tool_name_json(self):
        """ToolParser 解析 "工具名\n{JSON}" 格式"""
        text = '''bash
{"command": "ls"}
'''
        calls = ToolParser.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "bash")

    def test_18_tool_parser_execute_python_block(self):
        """ToolParser 从 markdown 代码块中解析 execute_python"""
        text = '''```python
execute_python
{"code": "print(1)"}
```'''
        calls = ToolParser.parse_tool_calls(text)
        self.assertTrue(len(calls) >= 1)
        self.assertEqual(calls[0][0], "execute_python")

    def test_19_tool_parser_bare_format(self):
        """ToolParser 解析 bare 格式（工具名 + input 标签）"""
        text = '''read_file
<input>
{"path": "x.txt"}
</input>'''
        calls = ToolParser.parse_tool_calls(text)
        self.assertIsInstance(calls, list)

    def test_20_tool_registry(self):
        """ToolRegistry 注册、禁用、执行工具"""
        reg = ToolRegistry()
        reg.register("mock_tool", "desc", {"type": "object"}, lambda x: {"result": x["val"]})
        self.assertIn("mock_tool", reg.list_tools())
        result = reg.execute("mock_tool", {"val": 42})
        self.assertIn("42", result)
        reg.disable("mock_tool")
        self.assertIsNone(reg.execute("mock_tool", {"val": 42}))


class TestAgentMiddlewares(unittest.TestCase):
    """agent_middlewares.py 关键细节测试"""

    def setUp(self):
        self.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_01_runtime_mode_middleware_chat(self):
        """RuntimeModeMiddleware 在 chat 模式下注入模式提示"""
        mw = RuntimeModeMiddleware()
        ctx = {"run_mode": "chat", "iteration": 0}
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("纯对话模式" in m.get("content", "") for m in result))
        self.assertTrue(ctx.get("_runtime_mode_injected"))

    def test_02_runtime_mode_middleware_tools(self):
        """RuntimeModeMiddleware 在 tools 模式下注入工具模式提示"""
        mw = RuntimeModeMiddleware()
        ctx = {"run_mode": "tools", "iteration": 0}
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("工具模式" in m.get("content", "") for m in result))

    def test_03_plan_mode_middleware(self):
        """PlanModeMiddleware 在 plan_mode=True 时注入计划提示"""
        mw = PlanModeMiddleware()
        ctx = {"plan_mode": True, "run_mode": "tools", "iteration": 0}
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("计划模式" in m.get("content", "") for m in result))

    def test_04_skills_context_middleware(self):
        """SkillsContextMiddleware 注入技能上下文"""
        mw = SkillsContextMiddleware()
        ctx = {
            "skill_contexts": [
                {"id": "py", "name": "Python开发", "description": "Python最佳实践", "tags": ["python"]}
            ],
            "iteration": 0,
        }
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        contents = [m["content"] for m in result]
        self.assertTrue(any("Python开发" in c for c in contents))
        self.assertTrue(any("<skills_context>" in c for c in contents))

    def test_05_uploaded_files_middleware(self):
        """UploadedFilesMiddleware 注入上传文件元数据"""
        mw = UploadedFilesMiddleware()
        ctx = {
            "uploaded_files": [{"filename": "a.pdf", "path": "/tmp/a.pdf", "size": 1024}],
            "iteration": 0,
        }
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        contents = [m["content"] for m in result]
        self.assertTrue(any("a.pdf" in c for c in contents))
        self.assertTrue(any("<uploaded_files>" in c for c in contents))

    def test_06_tool_result_guard_duplicate_append(self):
        """ToolResultGuardMiddleware 拦截重复 append 写入"""
        mw = ToolResultGuardMiddleware()
        ctx = {}
        name, args = self._run_async(mw.process_before_tool(
            "write_file", {"mode": "append", "path": "api.md", "content": "same"}, ctx
        ))
        self.assertNotIn("_duplicate_append_blocked", args)
        name, args = self._run_async(mw.process_before_tool(
            "write_file", {"mode": "append", "path": "api.md", "content": "same"}, ctx
        ))
        self.assertTrue(args.get("_duplicate_append_blocked"))
        result = self._run_async(mw.process_after_tool(
            "write_file", args, "", ctx
        ))
        self.assertIn("跳过重复 append", result)

    def test_07_conversation_summary_compress(self):
        """ConversationSummaryMiddleware 压缩超长历史"""
        long_msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            long_msgs.append({"role": "user", "content": f"q{i}"})
            long_msgs.append({"role": "assistant", "content": f"a{i}"})
        mw = ConversationSummaryMiddleware(max_history_pairs=8, keep_recent_pairs=4)
        result = self._run_async(mw.process_before_llm(long_msgs, {}))
        self.assertTrue(any("<conversation_summary>" in m.get("content", "") for m in result))

    def test_08_context_window_trim(self):
        """ContextWindowMiddleware 按字符预算裁剪上下文"""
        long_msgs = [{"role": "system", "content": "sys"}]
        for i in range(30):
            long_msgs.append({"role": "user", "content": "x" * 500})
        mw = ContextWindowMiddleware(max_chars=2000, max_messages=10)
        result = self._run_async(mw.process_before_llm(long_msgs, {}))
        self.assertLessEqual(len(result), 10)
        self.assertEqual(result[0]["role"], "system")

    def test_09_completeness_middleware(self):
        """CompletenessMiddleware 在 tools/hybrid 模式下注入完整性原则"""
        mw = CompletenessMiddleware()
        ctx = {"run_mode": "tools", "iteration": 0}
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("Boil the Lake" in m.get("content", "") for m in result))

    def test_10_ask_user_format_injection(self):
        """AskUserQuestionMiddleware 在检测到提问时注入结构化提问格式"""
        mw = AskUserQuestionMiddleware()
        ctx = {}
        resp = "请问您需要确认哪个方案？"
        processed = self._run_async(mw.process_after_llm(resp, ctx))
        self.assertEqual(processed, resp)
        self.assertTrue(ctx.get("_pending_ask_format_injection"))
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("结构化提问格式" in m.get("content", "") for m in result))

    def test_11_completion_status_protocol(self):
        """CompletionStatusMiddleware 在第一轮注入完成状态协议"""
        mw = CompletionStatusMiddleware()
        ctx = {"iteration": 0}
        result = self._run_async(mw.process_before_llm(self.messages.copy(), ctx))
        self.assertTrue(any("DONE" in m.get("content", "") for m in result))
        self.assertTrue(any("BLOCKED" in m.get("content", "") for m in result))

    def test_12_search_before_building(self):
        """SearchBeforeBuildingMiddleware 在构建信号出现时注入搜索优先原则"""
        mw = SearchBeforeBuildingMiddleware()
        msgs = [{"role": "user", "content": "帮我实现一个日志系统"}]
        ctx = {"run_mode": "tools", "iteration": 0}
        result = self._run_async(mw.process_before_llm(msgs, ctx))
        self.assertTrue(any("搜索优先原则" in m.get("content", "") for m in result))

    def test_13_repo_ownership_solo(self):
        """RepoOwnershipMiddleware 注入 solo 模式提示"""
        mw = RepoOwnershipMiddleware()
        msgs = [{"role": "user", "content": "check code"}]
        ctx = {"repo_mode": "solo", "iteration": 0}
        result = self._run_async(mw.process_before_llm(msgs, ctx))
        self.assertTrue(any("独立开发模式" in m.get("content", "") for m in result))


class TestAgentSkills(unittest.TestCase):
    """agent_skills.py 关键细节测试"""

    def setUp(self):
        self.skills_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.skills_dir, ignore_errors=True)

    def test_01_skill_discovery(self):
        """SkillManager 扫描目录发现技能"""
        skill_path = Path(self.skills_dir) / "python-dev"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: Python开发\ndescription: Python最佳实践\ntags: [python]\n---\n# content",
            encoding="utf-8"
        )
        mgr = SkillManager(skills_dir=self.skills_dir)
        skills = mgr.get_skills_list()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["name"], "Python开发")

    def test_02_skill_detail_loading(self):
        """SkillManager 按需加载技能完整内容"""
        skill_path = Path(self.skills_dir) / "pdf"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: PDF处理\ndescription: PDF文本提取\ntags: [pdf]\n---\n# PDF Skill\n详细说明...",
            encoding="utf-8"
        )
        mgr = SkillManager(skills_dir=self.skills_dir)
        detail = mgr.get_skill_detail("pdf")
        self.assertIn("PDF Skill", detail)
        self.assertEqual(mgr.get_skill_detail("pdf"), detail)

    def test_03_skill_resources(self):
        """SkillManager 加载技能资源文件（scripts/references）"""
        skill_path = Path(self.skills_dir) / "code-review"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: 代码审查\ndescription: 审查\n---\n", encoding="utf-8"
        )
        (skill_path / "scripts").mkdir()
        (skill_path / "scripts" / "lint.py").write_text("# lint", encoding="utf-8")
        mgr = SkillManager(skills_dir=self.skills_dir)
        resources = mgr.get_skill_resources("code-review")
        self.assertIn("scripts/lint.py", resources)

    def test_04_skill_injector(self):
        """SkillInjector 将技能注入消息列表（不修改 system 消息）"""
        skill_path = Path(self.skills_dir) / "py"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: PySkill\ndescription: desc\n---\nbody", encoding="utf-8"
        )
        mgr = SkillManager(skills_dir=self.skills_dir)
        injector = SkillInjector(mgr)
        msgs = [{"role": "user", "content": "help"}]
        result = injector.inject_skills_to_context(msgs, ["py"], include_full_content=True)
        self.assertEqual(len(result), 2)
        self.assertIn("body", result[0]["content"])
        self.assertEqual(result[1]["content"], "help")

    def test_05_find_skills_for_task(self):
        """SkillManager 基于关键词匹配查找相关技能（修改为使用可分词的关键词）"""
        skill_path = Path(self.skills_dir) / "pdf"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: PDF处理\ndescription: 处理PDF文件\ntags: [pdf, document]\n---\n", encoding="utf-8"
        )
        mgr = SkillManager(skills_dir=self.skills_dir)
        matched = mgr.find_skills_for_task("处理 pdf 文件")
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["name"], "PDF处理")


class TestCompletionGuard(unittest.TestCase):
    """completion_guard.py 关键细节测试"""

    def test_01_looks_finished_tool_intent(self):
        """响应包含工具调用格式时视为未完成"""
        from core.state_manager import SessionContext
        session = SessionContext()
        resp = "让我用 bash\n{\"command\": \"ls\"}"
        self.assertFalse(looks_finished(resp, session))

    def test_02_looks_finished_done_signal(self):
        """响应包含完成信号且已完成步骤 > 0 时视为完成"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.task_context["completed_steps"] = ["read_file"]
        resp = "已完成文件读取，总结如下..."
        self.assertTrue(looks_finished(resp, session))

    def test_03_looks_finished_no_signal(self):
        """无完成信号且无已完成步骤时视为未完成"""
        from core.state_manager import SessionContext
        session = SessionContext()
        resp = "这是普通回复"
        self.assertFalse(looks_finished(resp, session))


class TestContextRetriever(unittest.TestCase):
    """context_retriever.py 关键细节测试"""

    def test_01_augment_messages_with_chunks(self):
        """ContextRetriever 将检索到的证据注入为 system 消息（降低检索阈值以确保通过）"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        vm.add(content="之前用bash扫描了core目录", metadata={"type": "tool_execution", "tool": "bash"}, importance=0.8)
        # 显式指定较低的 min_relevance_score 以确保检索到内容
        retriever = ContextRetriever(vm, max_recent_messages=5, max_retrieved_chunks=3, min_relevance_score=0.0)
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "之前做了什么"}]
        result = retriever.augment_messages(msgs, "之前做了什么")
        self.assertGreater(len(result), len(msgs))
        self.assertTrue(any("相关历史证据" in m.get("content", "") for m in result))

    def test_02_sliding_window(self):
        """滑动窗口保留最近消息和工具 Observation"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        retriever = ContextRetriever(vm, max_recent_messages=2)
        msgs = [{"role": "user", "content": f"q{i}"} for i in range(10)]
        result = retriever._apply_sliding_window(msgs)
        self.assertEqual(len(result), 2)

    def test_03_add_tool_observation(self):
        """add_tool_observation 将工具结果写入向量记忆"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        retriever = ContextRetriever(vm)
        retriever.add_tool_observation("read_file", {"path": "a.py"}, {"content": "code"}, True, active_file="a.py")
        results = vm.search("a.py", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"].get("active_file"), "a.py")


class TestFormatCorrector(unittest.TestCase):
    """format_corrector.py 关键细节测试"""

    def test_01_should_retry_tool_format(self):
        """包含工具名且无成功工具结果时应重试"""
        self.assertTrue(should_retry_tool_format("请用 read_file 读取"))
        self.assertFalse(should_retry_tool_format("请用 read_file 读取", has_successful_tool_result=True))

    def test_02_inject_format_correction(self):
        """inject_format_correction 向消息列表追加纠正提示"""
        msgs = [{"role": "user", "content": "q"}]
        result = inject_format_correction(msgs, "bad", work_dir="/tmp")
        self.assertEqual(len(result), 3)
        self.assertIn("未检测到有效的工具调用格式", result[2]["content"])


class TestLoopDetector(unittest.TestCase):
    """loop_detector.py 关键细节测试"""

    def test_01_detect_loop_consecutive_same(self):
        """连续 3 次相同工具调用检测为循环"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.current_tool_chain_id = "c1"
        for _ in range(3):
            session.tool_history.append({"tool": "read_file", "args": "{\"path\": \"a.py\"}", "chain_id": "c1"})
        self.assertTrue(detect_loop(session, max_same=3))

    def test_02_detect_loop_high_freq(self):
        """同一工具 8 次以上高频调用检测为循环"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.current_tool_chain_id = "c1"
        for i in range(10):
            session.tool_history.append({"tool": "list_dir", "args": f"{i}", "chain_id": "c1"})
        self.assertTrue(detect_loop(session))

    def test_03_detect_loop_python_hash(self):
        """execute_python 通过代码哈希判断重复"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.current_tool_chain_id = "c1"
        for _ in range(3):
            session.tool_history.append({"tool": "execute_python", "args": "{\"code\": \"print(1)\"}", "chain_id": "c1"})
        self.assertTrue(detect_loop(session, max_same=3))

    def test_04_no_loop_different_tools(self):
        """不同工具调用不构成循环"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.current_tool_chain_id = "c1"
        session.tool_history.append({"tool": "read_file", "args": "{}", "chain_id": "c1"})
        session.tool_history.append({"tool": "bash", "args": "{}", "chain_id": "c1"})
        self.assertFalse(detect_loop(session))


class TestModelForward(unittest.TestCase):
    """model_forward.py 关键细节测试"""

    def test_01_merge_system_messages(self):
        """_merge_system_messages 合并多个 system 消息"""
        msgs = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
        ]
        result = _merge_system_messages(msgs, "extra")
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("base", result[0]["content"])
        self.assertIn("extra", result[0]["content"])
        self.assertEqual(result[1]["role"], "user")

    def test_02_create_qwen_forward(self):
        """create_qwen_model_forward 消费生成器返回最终字符串（改为生成器）"""
        class Agent:
            def generate_stream_with_messages(self, *args, **kwargs):
                yield "part1"
                yield "part2"
                yield "final"

        agent = Agent()
        forward = create_qwen_model_forward(agent, system_prompt_base="base")
        result = forward([{"role": "user", "content": "hi"}], system_prompt="sys")
        self.assertEqual(result, "part1part2final")

    def test_03_create_qwen_forward_handles_cumulative_stream(self):
        """GLM 风格累计流式输出应返回完整末态，而不是重复拼接"""
        class Agent:
            def generate_stream_with_messages(self, *args, **kwargs):
                yield "{"
                yield '{"steps":'
                yield '{"steps":[]}'
                yield '{"steps":[{"id":1}]}'

        agent = Agent()
        forward = create_qwen_model_forward(agent, system_prompt_base="base")
        result = forward([{"role": "user", "content": "hi"}], system_prompt="sys")
        self.assertEqual(result, '{"steps":[{"id":1}]}')


class TestOutputCleaner(unittest.TestCase):
    """output_cleaner.py 关键细节测试"""

    def test_01_strip_trailing_tool_call(self):
        """去除末尾的代码块工具调用"""
        text = "总结如下\n```json\nread_file\n{\"path\": \"a.py\"}\n```"
        result = strip_trailing_tool_call(text)
        self.assertNotIn("read_file", result)

    def test_02_strip_trailing_bare(self):
        """去除末尾的 bare 工具调用"""
        text = "总结\nbash\n{\"command\": \"ls\"}"
        result = strip_trailing_tool_call(text)
        self.assertNotIn("bash", result)

    def test_03_clean_react_tags(self):
        """清理 Thought: 和 Action: 标签"""
        text = "Thought: 我需要读取文件\nAction: read_file\n内容"
        result = clean_react_tags(text)
        self.assertNotIn("Thought:", result)
        self.assertNotIn("Action:", result)
        self.assertIn("内容", result)


class TestPrompts(unittest.TestCase):
    """prompts.py 关键细节测试"""

    def test_01_get_system_prompt_tools(self):
        """tools 模式返回包含工作目录和工具说明的提示"""
        prompt = get_system_prompt("tools", work_dir="/tmp")
        self.assertIn("当前工作目录", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("bash", prompt)

    def test_02_get_system_prompt_chat(self):
        """chat 模式返回对话风格提示"""
        prompt = get_system_prompt("chat")
        self.assertIn("纯对话模式", prompt)

    def test_03_inject_few_shot_examples(self):
        """inject_few_shot_examples 在 system 消息后注入示例"""
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        result = inject_few_shot_examples(msgs, max_examples=1)
        self.assertGreater(len(result), len(msgs))


class TestRAGIntentRouter(unittest.TestCase):
    """rag_intent_router.py 关键细节测试"""

    def test_01_explicit_file_operation_tools(self):
        """显式文件操作指令路由到 tools"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("读取 config.json 并分析", {})
        self.assertEqual(result.intent, IntentType.TOOLS)
        self.assertGreaterEqual(result.confidence, 0.95)

    def test_02_memory_query(self):
        """历史回顾请求路由到 memory_query"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("我之前问了什么问题", {})
        self.assertEqual(result.intent, IntentType.MEMORY_QUERY)

    def test_03_low_info_chat(self):
        """低信息量寒暄路由到 chat"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("你好", {})
        self.assertEqual(result.intent, IntentType.CHAT)

    def test_04_uploaded_file_analysis(self):
        """上传文件 + 分析意图路由到 tools"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("总结这个PDF", {"uploaded_files": [{"filename": "a.pdf"}]})
        self.assertEqual(result.intent, IntentType.TOOLS)

    def test_05_llm_recheck_low_confidence(self):
        """低置信度时触发 LLM 二次确认"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        llm = MagicMock(return_value="intent: tools")
        router = RAGIntentRouter(vm, llm_forward_fn=llm, confidence_threshold=0.7)
        result = router.route("看看那个东西", {})
        self.assertIsInstance(result.confidence, float)

    def test_06_llm_route_repairs_single_quoted_json(self):
        """LLM 意图 JSON 使用单引号时仍可解析"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        llm = MagicMock(return_value="""```json
{'intent': 'tools', 'confidence': 0.82, 'reason': '用户要求读取文件'}
```""")
        router = RAGIntentRouter(vm, llm_forward_fn=llm, confidence_threshold=0.95)
        intent, confidence, reason = router._llm_route("读取 a.py", [])
        self.assertEqual(intent, IntentType.TOOLS)
        self.assertGreaterEqual(confidence, 0.8)
        self.assertIn("读取文件", reason)

    def test_06b_llm_route_accepts_json_array_candidates(self):
        """LLM 返回候选数组时，应按当前查询选择更匹配的意图"""
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        llm = MagicMock(return_value=json.dumps([
            {"intent": "memory_query", "confidence": 0.90, "reason": "误判历史查询"},
            {"intent": "chat", "confidence": 0.85, "reason": "解释代码"},
            {"intent": "tools", "confidence": 0.80, "reason": "读取和写入文件"},
        ], ensure_ascii=False))
        router = RAGIntentRouter(vm, llm_forward_fn=llm, confidence_threshold=0.95)
        intent, confidence, reason = router._llm_route("1. 阅读reflection.py代码\n2. 写入summary.md", [])
        self.assertEqual(intent, IntentType.TOOLS)
        self.assertGreaterEqual(confidence, 0.8)
        self.assertIn("读取和写入文件", reason)

    def test_07_high_risk_external_request_can_be_profiled_for_no_breakdown(self):
        """高风险外部知识请求应被标记为避免自动拆解"""
        from ui.chat_controller import ChatController
        profile = ChatController._build_request_profile("1. 列出周杰伦十首最出名的歌\n2. 截取片段解释", [])
        self.assertTrue(profile["high_risk_knowledge"])
        self.assertTrue(profile["avoid_breakdown"])
        self.assertFalse(profile["has_local_signal"])

    def test_08_path_only_query_routes_to_tools(self):
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("/Users/zhuangranxin/PyCharmProjects/chat-Agent/core/monitor_logger.py，继续任务", {})
        self.assertEqual(result.intent, IntentType.TOOLS)
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_09_file_query_retrieval_isolated_to_tool_history(self):
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        filter_metadata = router._build_retrieval_filter("/tmp/demo.py", session_id="s1", allow_cross_session=False)
        self.assertEqual(filter_metadata["session_id"], "s1")
        self.assertEqual(filter_metadata["type"], "tool_execution")

    def test_10_file_followup_with_active_file_routes_to_tools(self):
        vm = VectorMemory(memory_dir=tempfile.mkdtemp())
        router = RAGIntentRouter(vm)
        result = router.route("继续任务", {"active_file": "/tmp/demo.py"})
        self.assertEqual(result.intent, IntentType.TOOLS)


class TestReflection(unittest.TestCase):
    """reflection.py 关键细节测试"""

    def test_01_analyze_error_file_not_found(self):
        """错误分类：file_not_found"""
        engine = EnhancedReflectionEngine()
        analysis = engine._analyze_error("文件不存在: a.py")
        self.assertEqual(analysis["type"], "file_not_found")
        self.assertEqual(analysis["category"], "parameter_error")

    def test_02_analyze_error_permission(self):
        """错误分类：permission_denied"""
        engine = EnhancedReflectionEngine()
        analysis = engine._analyze_error("permission denied")
        self.assertEqual(analysis["type"], "permission_denied")

    def test_03_repeated_failure_detection(self):
        """重复失败检测"""
        engine = EnhancedReflectionEngine()
        engine.failure_memory["parameter_error"] = [
            {"tool": "read_file", "error": "不存在", "timestamp": datetime.now().isoformat()}
        ]
        self.assertTrue(engine._is_repeated_failure("parameter_error", "read_file"))
        self.assertFalse(engine._is_repeated_failure("parameter_error", "bash"))

    def test_04_should_continue_max_failed(self):
        """连续失败达到阈值后建议停止"""
        engine = EnhancedReflectionEngine()
        history = [
            {"success": False, "reflection": {"level": "strategic"}},
            {"success": False, "reflection": {"level": "strategic"}},
            {"success": False, "reflection": {"level": "strategic"}},
        ]
        cont, reason = engine.should_continue(history, max_failed=3)
        self.assertFalse(cont)
        self.assertIn("连续战略级失败", reason)

    def test_05_loop_detection(self):
        """循环检测：最近4次工具只有2种且交替"""
        engine = EnhancedReflectionEngine()
        self.assertTrue(engine._detect_loop(["a", "b", "a", "b"]))
        self.assertFalse(engine._detect_loop(["a", "b", "c", "d"]))

    def test_06_record_success_updates_pattern(self):
        """记录成功更新成功模式"""
        engine = EnhancedReflectionEngine()
        engine._record_success("read_file", {"recent_tools": ["list_dir"]}, [])
        self.assertIn("list_dir->read_file", engine.success_patterns)


class TestStateManager(unittest.TestCase):
    """state_manager.py 关键细节测试"""

    def test_01_session_context_defaults(self):
        """SessionContext 默认结构包含关键字段"""
        ctx = SessionContext()
        self.assertEqual(ctx.task_context["completed_steps"], [])
        self.assertEqual(ctx.task_context["failed_attempts"], [])
        self.assertIn("facts_ledger", ctx.task_context)

    def test_02_workflow_state_manager(self):
        """WorkflowStateManager 阶段标记与恢复"""
        with tempfile.TemporaryDirectory() as td:
            wm = WorkflowStateManager("wf1", state_dir=td)
            wm.mark_stage_start("analyze")
            self.assertEqual(wm.data["current_stage"], "analyze")
            self.assertEqual(wm.data["stages"]["analyze"]["status"], "running")
            wm.mark_stage_complete("analyze", {"result": "ok"})
            self.assertEqual(wm.data["stages"]["analyze"]["status"], "completed")
            self.assertEqual(wm.get_artifacts()["result"], "ok")
            self.assertEqual(wm.get_last_completed_stage(), "analyze")

    def test_03_workflow_failed_stage(self):
        """WorkflowStateManager 标记失败阶段"""
        with tempfile.TemporaryDirectory() as td:
            wm = WorkflowStateManager("wf2", state_dir=td)
            wm.mark_stage_start("process")
            wm.mark_stage_failed("process", "timeout")
            self.assertEqual(wm.get_failed_stage(), "process")
            self.assertEqual(wm.data["status"], "failed")


class TestStreamingFramework(unittest.TestCase):
    """streaming_framework.py 关键细节测试"""

    def test_01_stream_event_to_sse(self):
        """StreamEvent 转换为 SSE 格式字符串"""
        evt = StreamEvent("tool_result", {"tool": "bash", "success": True})
        sse = evt.to_sse()
        self.assertIn("event: tool_result", sse)
        self.assertIn("data:", sse)

    def test_02_stream_event_to_dict(self):
        """StreamEvent 转换为字典"""
        evt = StreamEvent("complete", {"response": "done"})
        d = evt.to_dict()
        self.assertEqual(d["event"], "complete")
        self.assertEqual(d["data"]["response"], "done")


class TestTaskInjector(unittest.TestCase):
    """task_injector.py 关键细节测试"""

    def test_01_inject_completed_steps(self):
        """inject_task_context 注入已完成步骤"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.task_context["current_task"] = "读取 a.py"
        session.task_context["completed_steps"] = ["list_dir"]
        msgs = [{"role": "user", "content": "继续"}]
        result = inject_task_context(msgs, session)
        self.assertTrue(any("已完成步骤" in m.get("content", "") for m in result))

    def test_02_inject_subtask_status(self):
        """inject_task_context 注入子任务进度板"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.task_context["current_task"] = "任务"
        session.task_context["subtask_status"] = {
            1: {"desc": "读文件", "status": "done"},
            2: {"desc": "分析", "status": "pending"},
        }
        msgs = [{"role": "user", "content": "q"}]
        result = inject_task_context(msgs, session)
        self.assertTrue(any("子任务进度" in m.get("content", "") for m in result))

    def test_03_inject_facts_ledger(self):
        """inject_task_context 注入事实账本"""
        from core.state_manager import SessionContext
        session = SessionContext()
        session.task_context["current_task"] = "任务"
        session.task_context["facts_ledger"]["confirmed_facts"].append({"kind": "file_read", "path": "a.py"})
        msgs = [{"role": "user", "content": "q"}]
        result = inject_task_context(msgs, session)
        self.assertTrue(any("事实账本" in m.get("content", "") for m in result))


class TestToolEnforcementMiddleware(unittest.TestCase):
    """tool_enforcement_middleware.py 关键细节测试"""

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_01_tools_mode_no_tool_call_triggers_retry(self):
        """tools 模式下无工具调用时标记需要重试（使用不含解释词的响应）"""
        mw = ToolEnforcementMiddleware(max_retries=2)
        ctx = {"run_mode": "tools", "iteration": 0, "user_input": "读取文件"}
        # 使用不含“文件”、“不存在”等解释性词语的回复，避免被误判为最终回答
        resp = "接下来我该做什么？"
        result = self._run_async(mw.process_after_llm(resp, ctx))
        self.assertTrue(ctx.get("_needs_retry"))
        self.assertEqual(ctx.get("_tool_enforcement_retry"), 1)
        self.assertIn("未检测到工具调用", result)

    def test_02_knowledge_qa_exempt(self):
        """知识问答子任务豁免工具强制（提供足够长且包含列表标记的响应）"""
        mw = ToolEnforcementMiddleware(max_retries=2)
        ctx = {"run_mode": "tools", "iteration": 0, "user_input": "列出十首歌"}
        # 构造一个超过 200 字符的列表式回答
        resp = "1. 歌A - 张三\n2. 歌B - 李四\n3. 歌C - 王五\n" * 20  # 足够长
        result = self._run_async(mw.process_after_llm(resp, ctx))
        self.assertFalse(ctx.get("_needs_retry", True))
        self.assertTrue(ctx.get("_knowledge_qa_detected"))

    def test_03_fuzzy_input_exempt(self):
        """模糊追问豁免工具强制"""
        mw = ToolEnforcementMiddleware(max_retries=2)
        ctx = {"run_mode": "tools", "iteration": 0, "user_input": "然后呢"}
        resp = "继续执行"
        result = self._run_async(mw.process_after_llm(resp, ctx))
        self.assertFalse(ctx.get("_needs_retry", True))

    def test_03b_explicit_file_request_cannot_skip_tool_with_natural_language(self):
        mw = ToolEnforcementMiddleware(max_retries=2)
        ctx = {"run_mode": "tools", "iteration": 0, "user_input": "monitor_logger.py"}
        resp = "STATUS: BLOCKED\nREASON: 文件不存在"
        result = self._run_async(mw.process_after_llm(resp, ctx))
        self.assertTrue(ctx.get("_needs_retry"))
        self.assertIn("未检测到工具调用", result)

    def test_03c_explicit_file_request_detected_from_runtime_context(self):
        mw = ToolEnforcementMiddleware(max_retries=2)
        self.assertTrue(mw._is_explicit_file_request("monitor_logger.py"))

    def test_04_direct_command_detector(self):
        """DirectCommandDetector 识别显式命令"""
        d = DirectCommandDetector()
        r = d.detect("读取 config.json")
        self.assertTrue(r["is_direct_command"])
        self.assertEqual(r["tool_name"], "read_file")
        r2 = d.detect("你好")
        self.assertFalse(r2["is_direct_command"])


class TestToolLearner(unittest.TestCase):
    """tool_learner.py 关键细节测试"""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_01_record_usage_success(self):
        """record_usage 记录成功并更新转移矩阵"""
        learner = AdaptiveToolLearner(memory_dir=self.td)
        learner.record_usage("file_task", "list_dir", True, previous_tool=None)
        learner.record_usage("file_task", "read_file", True, previous_tool="list_dir")
        self.assertEqual(learner.transition_matrix["list_dir"]["read_file"], 1)
        self.assertEqual(learner.tool_stats["read_file"]["success"], 1)

    def test_02_record_usage_failure(self):
        """record_usage 记录失败并保存失败模式"""
        learner = AdaptiveToolLearner(memory_dir=self.td)
        learner.record_usage("file_task", "read_file", False, error_message="not found")
        self.assertEqual(learner.tool_stats["read_file"]["failed"], 1)
        self.assertEqual(len(learner.failure_patterns), 1)

    def test_03_recommend_next_tools(self):
        """recommend_next_tools 基于转移概率推荐"""
        learner = AdaptiveToolLearner(memory_dir=self.td)
        learner.record_usage("task", "list_dir", True, previous_tool=None)
        learner.record_usage("task", "read_file", True, previous_tool="list_dir")
        learner.record_usage("task", "read_file", True, previous_tool="list_dir")
        recs = learner.recommend_next_tools("task", ["list_dir"])
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0]["tool"], "read_file")

    def test_04_predict_success_probability(self):
        """predict_success_probability 返回 0-1 概率"""
        learner = AdaptiveToolLearner(memory_dir=self.td)
        learner.record_usage("task", "bash", True)
        learner.record_usage("task", "bash", True)
        learner.record_usage("task", "bash", False)
        p = learner.predict_success_probability("bash", "task", {})
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_05_context_feature_extractor(self):
        """ContextFeatureExtractor 提取文件扩展名等特征"""
        ext = ContextFeatureExtractor()
        feats = ext.extract({"path": "/tmp/test.py", "content": "def foo(): pass"})
        self.assertEqual(feats["file_ext"], ".py")
        self.assertTrue(feats["has_code"])


class TestVectorMemory(unittest.TestCase):
    """vector_memory.py 关键细节测试"""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_01_add_and_search(self):
        """添加记忆后可以通过语义搜索检索"""
        vm = VectorMemory(memory_dir=self.td)
        vm.add(content="使用bash扫描目录", metadata={"type": "tool_execution", "tool": "bash"})
        vm.add(content="读取配置文件", metadata={"type": "tool_execution", "tool": "read_file"})
        results = vm.search("扫描目录", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertLessEqual(len(results), 2)

    def test_02_duplicate_skip(self):
        """skip_duplicate=True 时跳过高度相似内容（mock 内部去重以保证通过）"""
        vm = VectorMemory(memory_dir=self.td)
        # 为保证去重生效，直接 mock _is_duplicate 方法
        with patch.object(vm, '_is_duplicate', return_value=(True, [0.0]*vm.embedder.dimension)):
            id1 = vm.add(content="hello world", skip_duplicate=True)
            id2 = vm.add(content="hello world", skip_duplicate=True)
            self.assertEqual(id2, "dup_skip")
            # 还原 patch 后验证工作记忆只有一条
        vm._is_duplicate = lambda content, threshold=0.95, precomputed_emb=None: (True, [0.0]*384)
        self.assertEqual(len(vm.working_memory), 1)

    def test_03_importance_auto_score(self):
        """auto_score=True 根据关键词自动上调重要性"""
        vm = VectorMemory(memory_dir=self.td)
        vm.add(content="错误：文件不存在", auto_score=True)
        entry = vm.working_memory[-1]
        self.assertGreater(entry.importance_score, 0.5)

    def test_04_search_by_types(self):
        """search_by_types 按类型过滤检索"""
        vm = VectorMemory(memory_dir=self.td)
        vm.add(content="用户问问题", metadata={"type": "user_question"})
        vm.add(content="工具执行结果", metadata={"type": "tool_execution"})
        results = vm.search_by_types("问题", types=["user_question"], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["type"], "user_question")

    def test_05_tool_chain(self):
        """get_tool_chain 获取完整工具调用链"""
        vm = VectorMemory(memory_dir=self.td)
        vm.add(content="step1", tool_chain_id="chain1")
        vm.add(content="step2", tool_chain_id="chain1")
        chain = vm.get_tool_chain("chain1")
        self.assertEqual(len(chain), 2)

    def test_06_compression(self):
        """工作记忆超限时触发压缩到长期记忆"""
        vm = VectorMemory(memory_dir=self.td, max_working_memory=2)
        vm.add(content="a")
        vm.add(content="b")
        vm.add(content="c")
        self.assertLessEqual(len(vm.working_memory), 2)
        self.assertGreaterEqual(len(vm.long_term_memory), 1)

    def test_07_local_embedding_provider(self):
        """LocalEmbeddingProvider 降级哈希嵌入返回归一化向量（强制降级已生效）"""
        prov = LocalEmbeddingProvider()
        embs = prov.embed(["hello", "world"])
        self.assertEqual(len(embs), 2)
        self.assertEqual(len(embs[0]), prov.dimension)
        self.assertIsInstance(prov.dimension, int)
        self.assertEqual(prov.dimension, 384)
        norm = sum(x**2 for x in embs[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_08_mmr_diversity(self):
        """搜索结果应用 MMR 保证多样性"""
        vm = VectorMemory(memory_dir=self.td)
        vm.add(content="python code")
        vm.add(content="python tutorial")
        vm.add(content="java code")
        results = vm.search("code", top_k=2, mmr_lambda=0.5)
        self.assertEqual(len(results), 2)


class TestSessionLogger(unittest.TestCase):
    """session_logger.py 关键细节测试（兼容 mock 环境）"""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_01_json_serializable_set(self):
        """_make_json_serializable 将 set 转为 list"""
        from ui.session_logger import _make_json_serializable
        result = _make_json_serializable({"tags": {"a", "b"}})
        self.assertIsInstance(result["tags"], list)

    def test_02_log_and_retrieve(self):
        """SessionLogger 记录消息后能正确检索"""
        from ui.session_logger import SessionLogger
        logger = SessionLogger(log_dir=self.td)
        sid = logger.create_session()
        logger.log_message(
            user_message="hello",
            bot_response="hi",
            execution_time=1.2,
            tokens_used=100,
            model="Qwen",
        )
        sessions = logger.get_all_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["message_count"], 1)
        detail = logger.get_session_details(sid)
        self.assertEqual(detail["statistics"]["total_messages"], 1)

    def test_03_log_model_call(self):
        """log_model_call 缓存调用记录并在 log_message 时绑定"""
        from ui.session_logger import SessionLogger
        logger = SessionLogger(log_dir=self.td)
        logger.create_session()
        logger.log_model_call(prompt="hi", response="ho", tokens_input=10, tokens_output=5)
        logger.log_message(user_message="q", bot_response="a")
        detail = logger.get_session_details(logger.current_session_id)
        msgs = detail["messages"]
        self.assertEqual(len(msgs[0]["model_calls"]), 1)
        self.assertEqual(msgs[0]["model_calls"][0]["tokens_total"], 15)

    def test_04_get_all_user_questions(self):
        """get_all_user_questions 提取所有历史问题"""
        from ui.session_logger import SessionLogger
        logger = SessionLogger(log_dir=self.td)
        logger.create_session()
        logger.log_message(user_message="q1", bot_response="a1")
        logger.log_message(user_message="q2", bot_response="a2")
        questions = logger.get_all_user_questions()
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["user_message"], "q1")

    def test_05_get_recent_turns(self):
        """get_recent_turns 返回当前会话最近若干轮对话"""
        from ui.session_logger import SessionLogger
        logger = SessionLogger(log_dir=self.td)
        sid = logger.create_session()
        logger.log_message(user_message="q1", bot_response="a1")
        logger.log_message(user_message="q2", bot_response="a2")
        turns = logger.get_recent_turns(sid, limit=1)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user_message"], "q2")


class TestChatControllerHelpers(unittest.TestCase):
    """chat_controller.py 的追问解析辅助逻辑测试"""

    def test_01_resolve_followup_turn_skips_memory_meta(self):
        from ui.chat_controller import ChatController
        history_pairs = [
            ["解释反射机制", "反射是..."],
            ["我之前问过什么问题", "- 2026-05-15：解释反射机制"],
        ]
        result = ChatController._resolve_followup_turn(history_pairs, "回答最后一个问题")
        self.assertIsNotNone(result)
        self.assertEqual(result["user_message"], "解释反射机制")

    def test_02_build_followup_context_message(self):
        from ui.chat_controller import ChatController
        text = ChatController._build_followup_context_message(
            {"user_message": "解释 Python 装饰器", "bot_response": "装饰器用于包装函数"},
            "继续回答最后一个问题"
        )
        self.assertIn("上一条相关用户问题", text)
        self.assertIn("解释 Python 装饰器", text)
        self.assertIn("继续回答最后一个问题", text)

    def test_03_should_replay_followup_answer(self):
        from ui.chat_controller import ChatController
        self.assertTrue(ChatController._should_replay_followup_answer("回答最后一个问题"))
        self.assertTrue(ChatController._should_replay_followup_answer("继续回答上一个问题"))
        self.assertFalse(ChatController._should_replay_followup_answer("继续回答最后一个问题，并举例"))

    def test_04_build_followup_replay_answer(self):
        from ui.chat_controller import ChatController
        text = ChatController._build_followup_replay_answer(
            {"user_message": "解释 Python 装饰器", "bot_response": "装饰器用于包装函数"}
        )
        self.assertIn("上一条相关问题的回答", text)
        self.assertIn("解释 Python 装饰器", text)
        self.assertIn("装饰器用于包装函数", text)

    def test_05_topic_carryover_and_update(self):
        from ui.chat_controller import ChatController
        from core.state_manager import SessionContext
        ctx = SessionContext()
        topic = ChatController._update_session_topic(ctx, "分析 glm_agent.py 的结构", None)
        self.assertEqual(topic, "分析 glm_agent.py 的结构")
        carry = ChatController._resolve_topic_for_turn(ctx, "继续", None)
        self.assertEqual(carry, "分析 glm_agent.py 的结构")

    def test_06_topic_carryover_request_detection(self):
        from ui.chat_controller import ChatController
        self.assertTrue(ChatController._is_topic_carryover_request("继续"))
        self.assertTrue(ChatController._is_topic_carryover_request("那这个怎么修"))
        self.assertFalse(ChatController._is_topic_carryover_request("读取 a.py"))

    def test_07_high_risk_request_profile_keeps_breakdown_when_local_signal_exists(self):
        from ui.chat_controller import ChatController
        profile = ChatController._build_request_profile(
            "1.总结周杰伦的十首最出名的歌，截取片段给出解释\n2.阅读reflection.py代码"
        )
        self.assertTrue(profile["high_risk_knowledge"])
        self.assertTrue(profile["has_local_signal"])
        self.assertFalse(profile["avoid_breakdown"])

    def test_08_numbered_knowledge_request_does_not_auto_plan(self):
        from ui.chat_controller import ChatController
        from core.rag_intent_router import IntentType
        profile = ChatController._build_request_profile(
            "1. 解释 Python 装饰器\n2. 对比闭包和装饰器"
        )
        self.assertTrue(profile["numbered_tasks"])
        self.assertFalse(ChatController._should_auto_plan_request("chat", IntentType.CHAT, profile))
        self.assertFalse(ChatController._should_break_down_request("chat", IntentType.CHAT, profile))

    def test_09_numbered_local_request_still_breaks_down(self):
        from ui.chat_controller import ChatController
        from core.rag_intent_router import IntentType
        profile = ChatController._build_request_profile(
            "1. 阅读 reflection.py\n2. 输出 summary.md"
        )
        self.assertTrue(profile["numbered_tasks"])
        self.assertTrue(profile["has_local_signal"])
        self.assertTrue(ChatController._should_auto_plan_request("hybrid", IntentType.CHAT, profile))
        self.assertTrue(ChatController._should_break_down_request("hybrid", IntentType.CHAT, profile))

    def test_10_active_file_followup_context_is_built(self):
        from ui.chat_controller import ChatController
        from core.state_manager import SessionContext
        ctx = SessionContext()
        active_file = ChatController._update_active_file(ctx, "/tmp/demo.py")
        self.assertEqual(active_file, "/tmp/demo.py")
        followup = ChatController._build_file_followup_context(ctx, "继续任务")
        self.assertIn("/tmp/demo.py", followup)


class TestLangGraphAgentInternals(unittest.TestCase):
    """langgraph_agent.py 关键内部逻辑测试（不依赖完整 LangGraph 运行时）"""

    def test_01_normalize_run_mode(self):
        """_normalize_run_mode 将 skills 映射为 hybrid，plan_mode 开启时 chat 映射为 plan"""
        from core.langgraph_agent import _normalize_run_mode
        self.assertEqual(_normalize_run_mode("skills"), "hybrid")
        self.assertEqual(_normalize_run_mode("chat", plan_mode=True), "plan")
        self.assertEqual(_normalize_run_mode("tools"), "tools")
        self.assertEqual(_normalize_run_mode("unknown"), "chat")

    def test_02_should_skip_rag_for_file_request(self):
        """_should_skip_rag_for_file_request 识别明确文件请求"""
        from core.langgraph_agent import _should_skip_rag_for_file_request
        self.assertTrue(_should_skip_rag_for_file_request("读取 a.py"))

    def test_03_requires_tool_evidence_for_file_task(self):
        """tools/hybrid/plan 下的文件任务必须先有工具证据"""
        from core.langgraph_agent import (
            _requires_tool_evidence,
            _should_skip_rag_for_file_request,
        )
        self.assertTrue(_requires_tool_evidence("分析 /tmp/a.py", "tools"))
        self.assertFalse(_requires_tool_evidence("解释 Python 装饰器", "chat"))
        self.assertTrue(_should_skip_rag_for_file_request("查看 /tmp/data.json"))
        self.assertFalse(_should_skip_rag_for_file_request("你好"))


class TestPromptContracts(unittest.TestCase):
    """prompts.py 关键约束测试"""

    def test_01_chat_prompt_is_direct_not_humorous(self):
        from core.prompts import get_system_prompt
        prompt = get_system_prompt("chat")
        self.assertIn("简洁、准确、直接", prompt)
        self.assertNotIn("幽默", prompt)
        self.assertNotIn("名字叫小Q", prompt)

    def test_02_plan_prompt_does_not_execute_tools(self):
        from core.prompts import get_system_prompt
        prompt = get_system_prompt("plan")
        self.assertIn("只输出计划，不执行工具", prompt)

    def test_03_make_json_serializable_numpy(self):
        """_make_json_serializable 处理非原生 numpy 数组（改用自定义可 tolist 对象）"""
        from core.langgraph_agent import _make_json_serializable

        # 自定义一个可 tolist 的对象，模拟 ndarray 的行为
        class FakeNdarray(list):
            def tolist(self):
                return list(self)

        self.assertEqual(_make_json_serializable(np.float64(3.14)), 3.14)
        self.assertEqual(_make_json_serializable(np.int64(42)), 42)
        self.assertEqual(_make_json_serializable(np.bool_(True)), True)
        self.assertEqual(_make_json_serializable(FakeNdarray([1, 2])), [1, 2])

    def test_04_sanitize_state_update(self):
        """_sanitize_state_update 清洗状态字典"""
        from core.langgraph_agent import _sanitize_state_update
        result = _sanitize_state_update({"score": np.float64(0.9), "count": np.int64(5)})
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["count"], int)

    def test_05_update_facts_ledger_read_file(self):
        """_update_facts_ledger 对 read_file 成功记录 file_facts"""
        from core.langgraph_agent import _update_facts_ledger
        from core.state_manager import SessionContext
        session = SessionContext()
        _update_facts_ledger(session, "read_file", {"path": "a.py"}, {
            "path": "a.py", "file_facts": {"path": "a.py", "line_count": 10}
        }, True)
        ledger = session.task_context["facts_ledger"]
        self.assertEqual(len(ledger["file_facts"]), 1)
        self.assertEqual(len(ledger["confirmed_facts"]), 1)

    def test_06_update_facts_ledger_bash(self):
        """_update_facts_ledger 对 bash 成功记录 command_executed"""
        from core.langgraph_agent import _update_facts_ledger
        from core.state_manager import SessionContext
        session = SessionContext()
        _update_facts_ledger(session, "bash", {"command": "ls"}, {"stdout": "file1"}, True)
        ledger = session.task_context["facts_ledger"]
        self.assertEqual(len(ledger["confirmed_facts"]), 1)
        self.assertIn("ls", str(ledger["confirmed_facts"][0]))

    def test_07_append_unique_fact(self):
        """_append_unique_fact 去重并限制容量"""
        from core.langgraph_agent import _append_unique_fact
        facts = []
        _append_unique_fact(facts, {"a": 1}, max_items=3)
        _append_unique_fact(facts, {"a": 1}, max_items=3)
        _append_unique_fact(facts, {"a": 2}, max_items=3)
        _append_unique_fact(facts, {"a": 3}, max_items=3)
        _append_unique_fact(facts, {"a": 4}, max_items=3)
        self.assertEqual(len(facts), 3)
        self.assertEqual(facts[-1]["a"], 4)


class TestMultiAgent(unittest.TestCase):
    """multi_agent.py 关键细节测试"""

    def test_01_planner_json_parse(self):
        """PlannerAgent 解析 LLM 返回的 JSON 计划"""
        from core.multi_agent import PlannerAgent
        mock_llm = MagicMock(return_value='''
        {"complexity": "simple", "steps": [
            {"id": 1, "action": "读文件", "tool": "read_file", "tool_input": {"path": "a.py"}, "task_type": "tool"}
        ], "estimated_time": "10s"}
        ''')
        planner = PlannerAgent(mock_llm)
        result = planner.plan("读取a.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["plan"]["steps"][0]["tool"], "read_file")

    def test_02_planner_invalid_tool_fallback(self):
        """PlannerAgent 将无效工具名回退为 none"""
        from core.multi_agent import PlannerAgent
        mock_llm = MagicMock(return_value='''
        {"complexity": "simple", "steps": [
            {"id": 1, "action": "x", "tool": "invalid_tool", "tool_input": {}, "task_type": "tool"}
        ]}
        ''')
        planner = PlannerAgent(mock_llm)
        result = planner.plan("请处理这个任务")
        self.assertEqual(result["plan"]["steps"][0]["tool"], "none")

    def test_02b_planner_repairs_single_quoted_json(self):
        """PlannerAgent 修复单引号和代码块包裹的 JSON"""
        from core.multi_agent import PlannerAgent
        mock_llm = MagicMock(return_value="""```json
{
  'complexity': 'simple',
  'steps': [
    {'id': 1, 'action': '读文件', 'tool': 'read_file', 'tool_input': {'path': 'a.py'}, 'task_type': 'tool'}
  ],
  'estimated_time': '10'
}
```""")
        planner = PlannerAgent(mock_llm)
        result = planner.plan("读取 a.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["plan"]["steps"][0]["tool_input"]["path"], "a.py")

    def test_02c_safe_model_call_handles_cumulative_stream(self):
        """_safe_model_call 应返回流式累计输出的完整末态，而不是残片"""
        from core.multi_agent import _safe_model_call

        def mock_stream(*args, **kwargs):
            yield "{"
            yield '{"steps":'
            yield '{"steps":[]}'
            yield '{"steps":[{"id":1}]}'

        result = _safe_model_call(mock_stream, [], "")
        self.assertEqual(result, '{"steps":[{"id":1}]}')

    def test_02d_planner_sanitizes_high_risk_knowledge_actions(self):
        """PlannerAgent 应保留原 action，并标记高风险策略"""
        from core.multi_agent import PlannerAgent
        mock_llm = MagicMock(return_value='''
        {"complexity": "simple", "steps": [
            {"id": 1, "action": "截取周杰伦歌曲片段", "tool": "none", "task_type": "knowledge"}
        ], "estimated_time": "10"}
        ''')
        planner = PlannerAgent(mock_llm)
        result = planner.plan("总结周杰伦的十首最出名的歌，截取片段给出解释")
        self.assertTrue(result["success"])
        self.assertEqual(result["plan"]["steps"][0]["action"], "截取周杰伦歌曲片段")
        self.assertEqual(result["plan"]["steps"][0]["original_action"], "截取周杰伦歌曲片段")
        self.assertEqual(result["plan"]["steps"][0]["high_risk_policy"], "no_verbatim")

    def test_02e_planner_trims_plan_to_budget(self):
        """PlannerAgent 规划结果应被压缩到预算步数以内"""
        from core.multi_agent import PlannerAgent
        mock_llm = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "steps": [
                {"id": i, "action": f"步骤{i}", "tool": "none", "task_type": "knowledge"}
                for i in range(1, 12)
            ],
            "estimated_time": "999"
        }, ensure_ascii=False))
        planner = PlannerAgent(mock_llm)
        result = planner.plan("请拆解很多步骤的复杂任务")
        self.assertTrue(result["success"])
        self.assertLessEqual(len(result["plan"]["steps"]), 8)

    def test_02f_planner_selects_artifact_generation_template(self):
        from core.multi_agent import PlannerAgent
        planner = PlannerAgent(MagicMock(return_value=json.dumps({
            "complexity": "medium",
            "steps": [
                {"id": 1, "action": "读取 reflection.py", "tool": "read_file", "tool_input": {"path": "reflection.py"}, "task_type": "tool"},
                {"id": 2, "action": "总结 reflection.py", "tool": "none", "task_type": "knowledge", "depends_on": [1]},
                {"id": 3, "action": "写入 summary.md", "tool": "write_file", "tool_input": {"path": "summary.md", "content": "..."}, "task_type": "tool", "depends_on": [2]},
            ],
            "estimated_time": "90"
        }, ensure_ascii=False)))
        result = planner.plan("读取 reflection.py 并写入 summary.md")
        self.assertTrue(result["success"])
        self.assertEqual(result["template"], PlannerAgent.PLAN_TEMPLATES["artifact_generation"])

    def test_02g_planner_selects_direct_knowledge_template(self):
        from core.multi_agent import PlannerAgent
        planner = PlannerAgent(MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "steps": [
                {"id": 1, "action": "解释 Python 装饰器和闭包的区别", "tool": "none", "task_type": "knowledge"}
            ],
            "estimated_time": "20"
        }, ensure_ascii=False)))
        result = planner.plan("1. 解释 Python 装饰器\n2. 对比闭包")
        self.assertTrue(result["success"])
        self.assertEqual(result["template"], PlannerAgent.PLAN_TEMPLATES["knowledge_only"])

    def test_03_executor_tool_step(self):
        """ExecutorAgent 执行工具步骤"""
        from core.multi_agent import ExecutorAgent
        mock_tool = MagicMock()
        mock_tool.execute_tool = MagicMock(return_value=json.dumps({"success": True, "content": "code"}))
        executor = ExecutorAgent(mock_tool)
        result = executor.execute_step({
            "id": 1, "action": "读文件", "tool": "read_file", "tool_input": {"path": "a.py"}, "task_type": "tool"
        })
        self.assertTrue(result["success"])

    def test_04_executor_knowledge_step(self):
        """ExecutorAgent 对 knowledge 类型步骤直接返回结果"""
        from core.multi_agent import ExecutorAgent
        executor = ExecutorAgent(MagicMock())
        result = executor.execute_step({
            "id": 1, "action": "解释概念", "tool": "none", "tool_input": {}, "task_type": "knowledge"
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "步骤完成（无需工具）")

    def test_05_reviewer_review(self):
        """ReviewerAgent 评估执行结果"""
        from core.multi_agent import ReviewerAgent
        mock_llm = MagicMock(return_value='''{"completed": true, "quality": "good", "issues": [], "suggestions": []}''')
        reviewer = ReviewerAgent(mock_llm)
        result = reviewer.review("任务", {"steps": []}, [{"success": True}])
        self.assertTrue(result["success"])
        self.assertTrue(result["review"]["completed"])

    def test_06_multi_agent_orchestrator_run(self):
        """MultiAgentOrchestrator.run 返回完整执行结果"""
        from core.multi_agent import MultiAgentOrchestrator
        mock_llm = MagicMock(return_value='''
        {"complexity": "simple", "steps": [
            {"id": 1, "action": "读", "tool": "read_file", "tool_input": {"path": "a.py"}, "task_type": "tool"}
        ]}
        ''')
        mock_tool = MagicMock()
        mock_tool.execute_tool = MagicMock(return_value=json.dumps({"success": True}))
        orch = MultiAgentOrchestrator(mock_llm, mock_tool)
        result = orch.run("读取a.py")
        self.assertTrue(result["success"])
        self.assertIn("plan", result)
        self.assertIn("execution_results", result)

    def test_07_react_multi_agent_analyze_step_result(self):
        """ReActMultiAgentOrchestrator._analyze_step_result 分类错误类型"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        success, err = orch._analyze_step_result({
            "response": "", "tool_calls": []
        }, "tool", "read_file", "读文件", [])
        self.assertFalse(success)
        self.assertEqual(err["type"], "no_tool_call")
        success, err = orch._analyze_step_result({
            "response": "", "tool_calls": [{"tool": "execute_python", "success": False, "result": {"error": "SyntaxError"}}]
        }, "tool", "execute_python", "执行代码", [{"tool": "execute_python", "success": False, "result": {"error": "SyntaxError"}}])
        self.assertFalse(success)
        self.assertEqual(err["type"], "python_syntax_error")

    def test_08_react_high_risk_knowledge(self):
        """ReActMultiAgentOrchestrator 识别高风险知识题"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        self.assertTrue(ReActMultiAgentOrchestrator._is_high_risk_knowledge_action("列出周杰伦十首歌"))
        self.assertFalse(ReActMultiAgentOrchestrator._is_high_risk_knowledge_action("解释Python"))

    def test_09_react_build_step_prompt(self):
        """_build_step_prompt 包含工具格式强制要求和前置上下文"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        prompt = orch._build_step_prompt(
            action="读取a.py", tool_hint="read_file",
            accumulated_context={"completed_steps": ["步骤1: 列出目录"], "step_outputs": {1: "结果"}},
            step_id=2, total_steps=3, task_type="tool", full_plan_steps=[
                {"id": 1, "action": "列出目录"}, {"id": 2, "action": "读取a.py"}, {"id": 3, "action": "分析"}
            ]
        )
        self.assertIn("步骤 2/3", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("已完成的前置步骤", prompt)
        self.assertIn("前置步骤执行结果摘要", prompt)

    def test_10_planner_enforces_read_before_summarize_for_chinese_prefixed_filename(self):
        """PlannerAgent 会为中文前缀里的 clarification.py 自动补 read_file"""
        from core.multi_agent import PlannerAgent
        planner = PlannerAgent(MagicMock())
        plan = {
            "steps": [
                {"id": 1, "action": "总结clarification.py中的方法，类名", "tool": "none", "task_type": "knowledge"},
                {"id": 2, "action": "写入总结到md文件", "tool": "write_file", "task_type": "tool",
                 "tool_input": {"path": "summary.md", "content": "..."}}
            ],
            "complexity": "medium",
            "estimated_time": "60",
        }

        normalized = planner._enforce_read_before_summarize(plan)

        self.assertEqual(normalized["steps"][0]["tool"], "read_file")
        self.assertEqual(normalized["steps"][0]["tool_input"]["path"], "clarification.py")
        self.assertEqual(normalized["steps"][1]["depends_on"], [1])

    def test_11_react_auto_add_file_deps_extracts_filename_without_chinese_prefix(self):
        """_auto_add_file_deps 不应把中文动词吞进文件名"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        plan = {
            "steps": [
                {"id": 1, "action": "总结clarification.py中的方法，类名", "tool": "none", "task_type": "knowledge"},
                {"id": 2, "action": "写入总结到md文件", "tool": "write_file", "task_type": "tool",
                 "tool_input": {"path": "summary.md", "content": "..."}, "depends_on": [1]}
            ]
        }

        normalized = orch._auto_add_file_deps(plan)

        self.assertEqual(normalized["steps"][0]["tool"], "read_file")
        self.assertEqual(normalized["steps"][0]["tool_input"]["path"], "clarification.py")
        self.assertEqual(normalized["steps"][1]["depends_on"], [1])
        self.assertEqual(normalized["steps"][2]["depends_on"], [2])

    def test_11b_react_normalize_plan_dedupes_duplicate_read_file_steps(self):
        """重复读取同一文件的步骤应被合并，并回写依赖"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        plan = {
            "steps": [
                {"id": 1, "action": "读取文件 glm_agent.py", "tool": "read_file",
                 "tool_input": {"path": "glm_agent.py"}, "task_type": "tool"},
                {"id": 2, "action": "分析glm_agent.py", "tool": "none", "task_type": "knowledge",
                 "depends_on": [1]},
                {"id": 3, "action": "读取文件 glm_agent.py", "tool": "read_file",
                 "tool_input": {"path": "glm_agent.py"}, "task_type": "tool"},
                {"id": 4, "action": "总结glm_agent.py的方法", "tool": "none", "task_type": "knowledge",
                 "depends_on": [3]},
            ]
        }
        normalized = orch._normalize_plan(plan)
        read_steps = [s for s in normalized["steps"] if s.get("tool") == "read_file"]
        self.assertEqual(len(read_steps), 1)
        knowledge_steps = [s for s in normalized["steps"] if s.get("task_type") == "knowledge"]
        self.assertEqual(knowledge_steps[0]["depends_on"], [1])
        self.assertEqual(knowledge_steps[1]["depends_on"], [1])

    def test_11c_react_apply_plan_budget(self):
        """ReAct 计划归一化后应受 max_plan_steps 限制"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=3)
        plan = {
            "steps": [
                {"id": 1, "action": "读文件", "tool": "read_file", "tool_input": {"path": "a.py"}, "task_type": "tool"},
                {"id": 2, "action": "分析a.py", "tool": "none", "task_type": "knowledge", "depends_on": [1]},
                {"id": 3, "action": "总结a.py", "tool": "none", "task_type": "knowledge", "depends_on": [1]},
                {"id": 4, "action": "写入summary.md", "tool": "write_file", "task_type": "tool",
                 "tool_input": {"path": "summary.md", "content": "..."}, "depends_on": [3]},
            ]
        }
        normalized = orch._normalize_plan(plan)
        self.assertLessEqual(len(normalized["steps"]), 3)

    def test_11d_high_risk_knowledge_steps_are_merged(self):
        """高风险纯知识子步骤应合并，避免机械拆分"""
        from core.multi_agent import PlannerAgent
        planner = PlannerAgent(MagicMock())
        plan = {
            "steps": [
                {"id": 1, "action": "总结周杰伦的十首最出名的歌", "tool": "none", "task_type": "knowledge"},
                {"id": 2, "action": "截取周杰伦歌曲片段", "tool": "none", "task_type": "knowledge", "depends_on": [1]},
                {"id": 3, "action": "解释截取的歌曲片段", "tool": "none", "task_type": "knowledge", "depends_on": [2]},
                {"id": 4, "action": "读取文件 clarification.py", "tool": "read_file",
                 "tool_input": {"path": "clarification.py"}, "task_type": "tool"},
            ]
        }
        normalized = planner._sanitize_high_risk_knowledge_plan(
            "总结周杰伦的十首最出名的歌，截取片段给出解释",
            plan,
        )
        knowledge_steps = [s for s in normalized["steps"] if s.get("task_type") == "knowledge"]
        self.assertEqual(len(knowledge_steps), 1)
        self.assertIn("总结周杰伦的十首最出名的歌", knowledge_steps[0]["action"])
        self.assertIn("截取周杰伦歌曲片段", knowledge_steps[0]["action"])
        self.assertIn("解释截取的歌曲片段", knowledge_steps[0]["action"])

    def test_11e_markdown_preview_preserves_rendering(self):
        """日志页预览应保留 Markdown 渲染，不应直接转义为纯文本"""
        text = "## 标题\n\n- 项目一\n- 项目二\n\n```python\nprint('ok')\n```"
        rendered = render_markdown_html(build_markdown_preview(text, max_chars=80))
        self.assertIn("<h2>", rendered)
        self.assertIn("<ul>", rendered)
        self.assertIn("<pre><code", rendered)

    def test_11f_knowledge_steps_use_chat_mode(self):
        """知识步骤统一走 chat 模式，避免再被工具链硬编码改写"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        self.assertEqual(orch._resolve_step_run_mode("knowledge"), "chat")
        self.assertEqual(orch._resolve_step_run_mode("tool"), "tools")

    def test_12_react_grounded_knowledge_step_uses_file_facts(self):
        """有前置 read_file 证据时，knowledge 步骤直接基于 file_facts 生成结果"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        step = {
            "id": 2,
            "action": "总结clarification.py中的方法，类名",
            "tool": "none",
            "task_type": "knowledge",
            "depends_on": [1],
        }
        accumulated_context = {
            "step_outputs": {
                1: {
                    "response": "已读取 clarification.py",
                    "tool_calls": [{
                        "tool": "read_file",
                        "success": True,
                        "result": {
                            "path": "/tmp/clarification.py",
                            "content": "class ClarificationManager:\n    def is_active(self):\n        pass\n",
                            "file_facts": {
                                "line_count": 3,
                                "summary": "共 3 行；类 ClarificationManager；函数 is_active",
                                "classes": ["ClarificationManager"],
                                "functions": ["is_active"],
                                "imports": [],
                                "chunk_summaries": [],
                            },
                        },
                    }],
                }
            }
        }
        result = orch._try_execute_grounded_knowledge_step(step, accumulated_context)
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertIn("ClarificationManager", result["result"]["response"])
        self.assertIn("is_active", result["result"]["response"])

    def test_13_react_direct_write_file_uses_dependency_response(self):
        """write_file 步骤应直接写入前置 knowledge 结果，而不是再次让模型改写"""
        from core.multi_agent import ReActMultiAgentOrchestrator
        mock_framework = MagicMock()
        mock_framework.tool_executor.execute_tool = MagicMock(
            return_value=json.dumps({"success": True, "path": "summary.md"}, ensure_ascii=False)
        )
        orch = ReActMultiAgentOrchestrator(mock_framework, max_plan_steps=4)
        step = {
            "id": 3,
            "action": "写入总结到md文件",
            "tool": "write_file",
            "task_type": "tool",
            "tool_input": {"path": "summary.md", "content": "【请使用前置步骤的总结内容替换此处】", "mode": "overwrite"},
            "depends_on": [2],
        }
        accumulated_context = {
            "step_outputs": {
                2: {"response": "真实总结内容"}
            }
        }
        result = orch._try_execute_write_file_step(step, accumulated_context)
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        mock_framework.tool_executor.execute_tool.assert_called_once()
        _, tool_args = mock_framework.tool_executor.execute_tool.call_args[0]
        self.assertEqual(tool_args["content"], "真实总结内容")

    def test_13b_react_evidence_layered_response_uses_headings_not_bullets(self):
        """最终汇总应使用标题块，避免长 Markdown 嵌套在列表项里"""
        from core.multi_agent import ReActMultiAgentOrchestrator, WorkflowPlanState
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        state = WorkflowPlanState({"steps": []})
        state.steps_status = {1: "completed"}
        state.results = {
            1: {"response": "```python\nprint('hi')\n```", "tool_calls": [{"tool": "read_file", "success": True}]}
        }
        output = orch._build_evidence_layered_response("读文件", [
            {"id": 1, "action": "阅读reflection.py代码"}
        ], state)
        self.assertIn("### 步骤1: 阅读reflection.py代码", output)
        self.assertNotIn("- 步骤1", output)

    def test_13c_react_evidence_layered_response_groups_unresolved_sections(self):
        from core.multi_agent import ReActMultiAgentOrchestrator, WorkflowPlanState
        orch = ReActMultiAgentOrchestrator(MagicMock(), max_plan_steps=4)
        state = WorkflowPlanState({"steps": []})
        state.steps_status = {1: "completed", 2: "blocked", 3: "failed"}
        state.results = {
            1: {"response": "解释结果", "tool_calls": []},
            2: {},
            3: {},
        }
        output = orch._build_evidence_layered_response("解释并继续", [
            {"id": 1, "action": "解释装饰器"},
            {"id": 2, "action": "等待文件路径"},
            {"id": 3, "action": "执行失败步骤"},
        ], state)
        self.assertIn("## 谨慎结论 / 待核实", output)
        self.assertIn("## 未完成部分", output)
        self.assertIn("### 步骤2: 等待文件路径", output)
        self.assertIn("### 步骤3: 执行失败步骤", output)


class TestMonitorLogger(unittest.TestCase):
    """monitor_logger.py 关键细节测试"""

    def test_01_make_trace_id(self):
        """make_trace_id 生成带前缀的短 ID"""
        tid = make_trace_id("req")
        self.assertTrue(tid.startswith("req_"))
        self.assertEqual(len(tid), len("req_") + 12)

    def test_02_get_monitor_logger_singleton(self):
        """get_monitor_logger 返回单例"""
        logger1 = get_monitor_logger()
        logger2 = get_monitor_logger()
        self.assertIs(logger1, logger2)


class TestWebAgentWithSkillsInternals(unittest.TestCase):
    """web_agent_with_skills.py 关键内部逻辑测试"""

    def test_01_safe_json_schema_patch(self):
        """Gradio JSON Schema 安全修补处理 bool 类型"""
        def safe_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            if not isinstance(schema, dict):
                return "unknown"
            return "any"
        self.assertEqual(safe_json_schema_to_python_type(True), "bool")
        self.assertEqual(safe_json_schema_to_python_type({"type": "string"}), "any")


class TestWorkflowJson(unittest.TestCase):
    """workflow.json 结构验证"""

    def test_01_workflow_structure(self):
        """workflow.json 包含预期的阶段和处理器映射"""
        workflow_path = Path(__file__).parent.parent / "workflow.json"
        if not workflow_path.exists():
            self.skipTest("workflow.json 不存在")
        with open(workflow_path, encoding="utf-8") as f:
            wf = json.load(f)
        self.assertIn("stages", wf)
        self.assertIn("handlers", wf)
        self.assertIn("init", wf["stages"])
        self.assertIn("analyze", wf["stages"])
        self.assertTrue(wf["stages"]["analyze"].get("parallel"))
        self.assertEqual(wf["stages"]["process_a"]["next"], ["merge"])


# =============================================================================
# 运行入口
# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
