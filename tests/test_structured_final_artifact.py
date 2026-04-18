import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_core_module(module_name: str, relative_path: str):
    if "core" not in sys.modules:
        pkg = types.ModuleType("core")
        pkg.__path__ = [str(PROJECT_ROOT / "core")]
        sys.modules["core"] = pkg

    full_name = f"core.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


monitor_logger = _load_core_module("monitor_logger", "core/monitor_logger.py")
state_manager = _load_core_module("state_manager", "core/state_manager.py")
agent_tools = _load_core_module("agent_tools", "core/agent_tools.py")
multi_agent = _load_core_module("multi_agent", "core/multi_agent.py")
loop_detector = _load_core_module("components.loop_detector", "core/components/loop_detector.py")
format_corrector = _load_core_module("components.format_corrector", "core/components/format_corrector.py")

ToolExecutor = agent_tools.ToolExecutor
ReActMultiAgentOrchestrator = multi_agent.ReActMultiAgentOrchestrator
SessionContext = state_manager.SessionContext
detect_loop = loop_detector.detect_loop
should_retry_tool_format = format_corrector.should_retry_tool_format


class StructuredArtifactTests(unittest.TestCase):
    def test_read_file_returns_chunk_summaries_and_file_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.py"
            file_path.write_text(
                "import os\n\n"
                "class Demo:\n"
                "    def method_one(self):\n"
                "        return 1\n\n"
                "def top_level():\n"
                "    return Demo()\n",
                encoding="utf-8",
            )
            executor = ToolExecutor(work_dir=tmpdir)
            result = json.loads(executor._read_file(str(file_path)))

        self.assertTrue(result["success"])
        self.assertIn("file_facts", result)
        self.assertEqual(result["file_facts"]["path"], str(file_path))
        self.assertIn("Demo", result["file_facts"]["classes"])
        self.assertIn("top_level", result["file_facts"]["functions"])
        self.assertTrue(result["file_facts"]["chunk_summaries"])

    def test_validator_rejects_untrusted_fact_and_missing_latest_read(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        trusted_bundle = {
            "confirmed_facts": ["已读取文件: /tmp/sample.py"],
            "file_evidence": ["文件: /tmp/sample.py；行数 10；类 Demo；函数 top_level"],
            "failed_attempts": ["read_file 失败: missing.py | 文件不存在"],
            "unresolved_questions": ["missing.py 尚未找到"],
            "citations": ["/tmp/sample.py"],
            "latest_successful_read_file": "/tmp/sample.py",
        }
        artifact = {
            "answer": "已完成读取。",
            "confirmed_facts": ["未受信事实"],
            "file_evidence": ["文件: /tmp/other.py"],
            "failed_attempts": [],
            "unresolved_questions": [],
            "citations": [],
        }
        issues = orchestrator._validate_final_answer_artifact(artifact, trusted_bundle, step_results=[])

        self.assertTrue(any("confirmed_facts" in issue for issue in issues))
        self.assertTrue(any("遗漏最近成功的 read_file" in issue for issue in issues))
        self.assertTrue(any("failed_attempts 为空" in issue for issue in issues))

    def test_finalize_artifact_renders_from_template(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        artifact = orchestrator._finalize_artifact({
            "answer": "完成文件读取与总结。",
            "confirmed_facts": ["已读取文件: /tmp/sample.py"],
            "file_evidence": ["文件: /tmp/sample.py；行数 10"],
            "failed_attempts": [],
            "unresolved_questions": [],
            "citations": ["/tmp/sample.py"],
        })

        self.assertIn("完成文件读取与总结。", artifact["final_response"])
        self.assertIn("已确认：", artifact["final_response"])
        self.assertIn("文件证据：", artifact["final_response"])
        self.assertEqual(artifact["final_facts"], ["已读取文件: /tmp/sample.py"])


class TrustedBundleTests(unittest.TestCase):
    def test_build_trusted_bundle_uses_ledger_file_facts(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        session = SessionContext()
        session.task_context["facts_ledger"] = {
            "confirmed_facts": [{"kind": "file_read", "path": "/tmp/sample.py", "tool": "read_file"}],
            "file_facts": [{
                "path": "/tmp/sample.py",
                "line_count": 5,
                "classes": ["Demo"],
                "functions": ["top_level"],
                "summary": "共 5 行；类 Demo；函数 top_level",
            }],
            "failed_actions": [],
            "open_questions": [],
        }
        step_results = [{
            "tool_calls": [{
                "tool": "read_file",
                "success": True,
                "args": {"path": "sample.py"},
                "result": {"path": "/tmp/sample.py"},
            }]
        }]

        bundle = orchestrator._build_trusted_evidence_bundle(session, step_results, review_result={})

        self.assertIn("已读取文件: /tmp/sample.py", bundle["confirmed_facts"])
        self.assertTrue(any("/tmp/sample.py" in item for item in bundle["file_evidence"]))
        self.assertEqual(bundle["latest_successful_read_file"], "/tmp/sample.py")


class LoopDetectionTests(unittest.TestCase):
    def test_detect_loop_scopes_to_current_chain(self):
        session = SessionContext()
        session.current_tool_chain_id = "chain-b"
        session.tool_history = [
            {"tool": "bash", "args": '{"command":"a"}', "chain_id": "chain-a"},
            {"tool": "bash", "args": '{"command":"a"}', "chain_id": "chain-a"},
            {"tool": "bash", "args": '{"command":"a"}', "chain_id": "chain-a"},
            {"tool": "read_file", "args": '{"path":"x.py"}', "chain_id": "chain-b"},
        ]

        self.assertFalse(detect_loop(session))


class KnowledgeGuardTests(unittest.TestCase):
    def test_high_risk_knowledge_response_is_rejected_when_overconfident(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        react_result = {
            "response": (
                "1. 《青花瓷》\n片段：“素胚勾勒出青花笔锋浓转淡”\n含义：...\n"
                "2. 《七里香》\n片段：“窗外的麻雀在电线杆上多嘴”\n含义：...\n"
                "3. 《东风破》\n片段：“断桥残雪独自愁”\n含义：...\n"
                "4. 《菊花台》\n片段：“菊花台上一壶酒”\n含义：...\n"
                "5. 《夜曲》\n片段：“月光下的凤尾竹”\n含义：...\n"
            )
        }
        success, error = orchestrator._analyze_step_result(
            react_result=react_result,
            task_type="knowledge",
            tool_hint="none",
            action="给出周杰伦最出名的十首歌，截取片段并解释含义",
            tool_calls=[],
        )

        self.assertFalse(success)
        self.assertEqual(error["type"], "knowledge_unverified")

    def test_high_risk_knowledge_response_with_tail_disclaimer_is_still_rejected(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        react_result = {
            "response": (
                "1. 《青花瓷》\n片段：“素胚勾勒出青花笔锋浓转”\n含义：...\n"
                "2. 《七里香》\n片段：“窗外的麻雀，在电线杆上多嘴”\n含义：...\n"
                "3. 《东风破》\n片段：“一盏离愁，孤单伫立在窗口”\n含义：...\n"
                "以上仅供参考，具体细节可能存在偏差。"
            )
        }
        success, error = orchestrator._analyze_step_result(
            react_result=react_result,
            task_type="knowledge",
            tool_hint="none",
            action="列出周杰伦最出名的十首歌，截取片段并解释含义",
            tool_calls=[],
        )

        self.assertFalse(success)
        self.assertEqual(error["type"], "knowledge_unverified")

    def test_high_risk_knowledge_response_with_caution_is_allowed(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        react_result = {
            "response": (
                "可以先给高层概括：周杰伦常被提及的代表作通常包括《青花瓷》《七里香》《东风破》等。"
                "但“最出名十首”的精确名单和逐字片段需要核对资料，以下细节均为（待核实），不宜直接当作准确信息。"
            )
        }
        success, error = orchestrator._analyze_step_result(
            react_result=react_result,
            task_type="knowledge",
            tool_hint="none",
            action="给出周杰伦最出名的十首歌，截取片段并解释含义",
            tool_calls=[],
        )

        self.assertTrue(success)
        self.assertEqual(error, {})

    def test_hard_repair_final_artifact_replaces_untrusted_file_evidence(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        trusted_bundle = {
            "confirmed_facts": ["已读取文件: /tmp/sample.py"],
            "file_evidence": ["文件: /tmp/sample.py；行数 10；类 Demo；函数 top_level"],
            "failed_attempts": [],
            "unresolved_questions": [],
            "citations": ["/tmp/sample.py"],
            "latest_successful_read_file": "/tmp/sample.py",
            "latest_successful_read_fact": "文件: /tmp/sample.py；行数 10；类 Demo；函数 top_level",
        }
        artifact = {
            "answer": "已成功完成所有任务。",
            "confirmed_facts": ["已读取文件: /tmp/sample.py"],
            "file_evidence": ["/tmp/sample.py"],
            "failed_attempts": [],
            "unresolved_questions": [],
            "citations": ["/tmp/sample.py"],
        }
        repaired = orchestrator._hard_repair_final_artifact(
            artifact=artifact,
            trusted_bundle=trusted_bundle,
            step_results=[],
            validation_issues=["file_evidence 包含未受信的文件结论: /tmp/sample.py"],
        )

        self.assertEqual(
            repaired["file_evidence"],
            ["文件: /tmp/sample.py；行数 10；类 Demo；函数 top_level"],
        )
        self.assertEqual(repaired["citations"], ["/tmp/sample.py"])


class FormatCorrectionTests(unittest.TestCase):
    def test_should_not_retry_format_after_successful_tool_result(self):
        response = "已成功扫描 core/ 目录并提取类和方法，结果已重定向写入 API.md 文件。"
        self.assertFalse(should_retry_tool_format(response, has_successful_tool_result=True))

    def test_should_retry_format_before_any_successful_tool_result(self):
        response = 'bash\n{"command": "grep -rn test ."}'
        self.assertTrue(should_retry_tool_format(response, has_successful_tool_result=False))


class TaskResultTemplateTests(unittest.TestCase):
    def test_build_task_results_prefers_result_summary_over_status(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        step_results = [
            {
                "action": "用 bash 扫描 core/ 目录提取类和方法并重定向写入 API.md",
                "success": True,
                "response": "成功执行了bash命令。",
                "tool_calls": [{
                    "tool": "bash",
                    "success": True,
                    "args": {"command": "grep -Ern '^(class|def) ' core/ > API.md"},
                    "result": {"stdout": ""},
                }],
            },
            {
                "action": "用 read_file 读取 session_analyzer.py 代码",
                "success": True,
                "response": "这是一个基于 Gradio 的高级会话分析页面。",
                "tool_calls": [{
                    "tool": "read_file",
                    "success": True,
                    "args": {"path": "session_analyzer.py"},
                    "result": {"path": "/tmp/session_analyzer.py"},
                }],
            },
        ]

        task_results = orchestrator._build_task_results(step_results)

        self.assertIn("结果已写入 API.md。", task_results[0])
        self.assertIn("这是一个基于 Gradio 的高级会话分析页面。", task_results[1])

    def test_render_final_response_template_includes_task_results_section(self):
        orchestrator = object.__new__(ReActMultiAgentOrchestrator)
        artifact = orchestrator._finalize_artifact({
            "answer": "任务已完成。",
            "task_results": ["1. 任务A -> 结果A", "2. 任务B -> 结果B"],
            "confirmed_facts": [],
            "file_evidence": [],
            "failed_attempts": [],
            "unresolved_questions": [],
            "citations": [],
        })

        self.assertIn("任务结果：", artifact["final_response"])
        self.assertIn("任务A -> 结果A", artifact["final_response"])
        self.assertIn("任务B -> 结果B", artifact["final_response"])


if __name__ == "__main__":
    unittest.main()
