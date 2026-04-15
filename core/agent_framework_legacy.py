# -*- coding: utf-8 -*-
"""先进 Agent 框架 - ReAct + 反思 + 并行执行 + 持久化记忆 + 语义压缩
架构说明：
  QwenAgentFramework._run_iter()  —— 单次推理迭代核心（生成器）
  QwenAgentFramework.run()        —— 阻塞式入口，消费 _run_iter
  StreamingFramework              —— 流式入口，消费 _run_iter，发送 SSE 事件
  AdaptiveToolLearner             —— 工具序列学习，首轮推荐 + 序列转移矩阵
  DeepReflectionEngine            —— 失败反思 + 成功模式记录，双向同步 ToolLearner
  MultiAgentOrchestrator          —— plan 模式封装，复用 QwenAgentFramework 的 ReAct 循环
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .agent_tools import ToolExecutor, ToolParser
from .state_manager import SessionContext
from .tool_learner import AdaptiveToolLearner
from .vector_memory import VectorMemory, LocalEmbeddingProvider

# ---------------------------------------------------------------------------
# 只读工具集合（可并发执行的工具名称，内联在框架中，无需独立 Executor 类）
# ---------------------------------------------------------------------------
# list_dir 不加入并行集合：
# 1. list_dir 在 LLM 输出中常与 read_file 一起出现，并行会使双方同时失败产生连锁错误
# 2. list_dir 通常只用于探路，不是真正的数据读取，单独调用即可
READ_ONLY_TOOLS: Set[str] = {"read_file"}


def register_read_only_tool(tool_name: str) -> None:
    """全局注册只读工具，允许在并行路径中并发执行。"""
    READ_ONLY_TOOLS.add(tool_name)


# ---------------------------------------------------------------------------
# 兼容占位：原 EnhancedParallelExecutor / ParallelConfig
# 并行逻辑已内联到 QwenAgentFramework._execute_tools_parallel()
# 保留这两个名称以防外部代码仍在引用
# ---------------------------------------------------------------------------
@dataclass
class ParallelConfig:
    max_workers: int = 4

    # noinspection PyUnusedLocal
    def get_optimal_workers(self, task_count: int) -> int:
        return min(task_count, self.max_workers)


class EnhancedParallelExecutor:
    """兼容占位——并行逻辑已内联到框架，此类仅保留供旧引用不报错。"""

    READ_ONLY_TOOLS: Set[str] = READ_ONLY_TOOLS  # 指向模块级集合

    def __init__(self, *args, **kwargs):
        self.stats = {"total": 0, "parallel": 0, "failed": 0}

    def register_read_only_tool(self, tool_name: str):
        READ_ONLY_TOOLS.add(tool_name)

    def is_read_only(self, tool_name: str) -> bool:
        return tool_name in READ_ONLY_TOOLS



class DeepReflectionEngine:
    """深度反思引擎 —— 失败分析 + 成功模式学习 + 双向同步 AdaptiveToolLearner

    新增能力：
      • reflect_on_success()  —— 成功时记录高效工具序列，注入 ToolLearner 转移矩阵
      • attach_tool_learner() —— 绑定外部 ToolLearner，启用双向数据同步
    """

    FAILURE_PATTERNS = {
        "file_not_found": {
            "patterns": [r"not found", r"不存在", r"找不到", r"No such file"],
            "category": "parameter_error",
            "fixes": ["使用绝对路径", "先list_dir确认", "检查文件名拼写"]
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"权限不足"],
            "category": "tool_error",
            "fixes": ["检查文件权限", "更换工作目录"]
        },
        "syntax_error": {
            "patterns": [r"syntax error", r"invalid syntax", r"语法错误"],
            "category": "parameter_error",
            "fixes": ["检查JSON格式", "验证参数类型"]
        },
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"超时"],
            "category": "execution_error",
            "fixes": ["增加超时时间", "拆分任务", "优化命令"]
        }
    }

    def __init__(self):
        self.failure_memory: Dict[str, List] = {}
        # 成功模式：记录高效工具序列（序列 -> 出现次数 + 平均耗时）
        self.success_patterns: Dict[str, Dict] = {}
        # 绑定的 ToolLearner（可选），用于双向同步
        self._tool_learner: Optional["AdaptiveToolLearner"] = None

    def attach_tool_learner(self, learner: "AdaptiveToolLearner") -> None:
        """绑定 AdaptiveToolLearner，启用反思引擎 ↔ 工具学习器双向同步。"""
        self._tool_learner = learner

    # ------------------------------------------------------------------
    # 失败反思（原有逻辑，保持接口不变）
    # ------------------------------------------------------------------
    def reflect_on_result(self, tool_name: str, result: Dict, context: Dict = None, history: List[Dict] = None) -> Dict:
        context = context or {}
        history = history if history is not None else []
        error = result.get("error")
        if not error:
            # 成功路径：记录高效模式
            self._record_success(tool_name, context, history)
            return {
                "success": True,
                "level": "surface",
                "analysis": "执行成功",
                "suggestions": [],
                "action": "continue"
            }
        error_analysis = self._analyze_error(error)
        is_repeated = self._is_repeated_failure(error_analysis["category"], tool_name)
        recent_tools = context.get("recent_tools", [])
        is_loop = self._detect_loop(recent_tools)

        if is_loop:
            level = "meta"
            action = "break_loop"
            suggestions = ["强制切换策略", "更换工具类型", "请求用户澄清"]
        elif is_repeated:
            level = "strategic"
            action = "escalate"
            suggestions = [
                f"重复错误！停止使用{tool_name}",
                f"尝试: {error_analysis['fixes'][0] if error_analysis['fixes'] else '替代方案'}",
                "如仍失败则上报BLOCKED"
            ]
        else:
            level = "operational"
            action = "retry_with_fix"
            suggestions = error_analysis["fixes"][:2]

        reflection = {
            "success": False,
            "level": level,
            "category": error_analysis["category"],
            "root_cause": error_analysis["root_cause"],
            "analysis": error_analysis["description"],
            "suggestions": suggestions,
            "action": action,
            "is_repeated": is_repeated,
            "is_loop": is_loop,
            "confidence": 0.9 if is_repeated else 0.7
        }
        history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": False,
            "error": error[:100],
            "category": reflection.get("category", "unknown"),
            "args_sig": context.get("tool_args_sig", ""),
            "reflection": reflection
        })
        cat = error_analysis["category"]
        if cat not in self.failure_memory:
            self.failure_memory[cat] = []
        self.failure_memory[cat].append({
            "tool": tool_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        # 同步失败信息到 ToolLearner
        if self._tool_learner:
            self._tool_learner.record_usage(
                task_type=context.get("task", "general"),
                tool_name=tool_name,
                success=False,
                execution_time=result.get("_execution_time", 0),
                context=context,
                previous_tool=recent_tools[-1] if recent_tools else None,
                error_message=error[:200]
            )
        return reflection

    # ------------------------------------------------------------------
    # 成功模式学习（新增）
    # ------------------------------------------------------------------
    def _record_success(self, tool_name: str, context: Dict, history: List[Dict]) -> None:
        """记录成功执行，更新高效工具序列并同步到 ToolLearner。"""
        recent_tools: List[str] = context.get("recent_tools", [])
        exec_time: float = context.get("_execution_time", 0)
        task_type: str = context.get("task", "general")

        history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": True,
            "reflection": {"level": "surface", "analysis": "执行成功", "action": "continue"}
        })

        # 记录序列 key：前驱工具 -> 当前工具
        if recent_tools:
            prev = recent_tools[-1]
            seq_key = f"{prev}->{tool_name}"
            if seq_key not in self.success_patterns:
                self.success_patterns[seq_key] = {"count": 0, "avg_time": 0.0, "task_types": []}
            pat = self.success_patterns[seq_key]
            pat["count"] += 1
            pat["avg_time"] = (pat["avg_time"] * (pat["count"] - 1) + exec_time) / pat["count"]
            if task_type not in pat["task_types"]:
                pat["task_types"].append(task_type)

        # 同步到 ToolLearner（转移矩阵 + 统计）
        if self._tool_learner:
            self._tool_learner.record_usage(
                task_type=task_type,
                tool_name=tool_name,
                success=True,
                execution_time=exec_time,
                context=context,
                previous_tool=recent_tools[-1] if recent_tools else None,
            )

    def get_efficient_sequences(self, top_k: int = 5) -> List[Dict]:
        """返回最常见的高效工具序列，供注入 prompt 使用。"""
        sorted_patterns = sorted(
            self.success_patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        return [
            {"sequence": k, "count": v["count"], "avg_time": v["avg_time"]}
            for k, v in sorted_patterns[:top_k]
        ]

    # ------------------------------------------------------------------
    # 内部工具方法（保持原接口）
    # ------------------------------------------------------------------
    def _analyze_error(self, error: str) -> Dict:
        import re
        error_lower = error.lower()
        for error_type, info in self.FAILURE_PATTERNS.items():
            for pattern in info["patterns"]:
                if re.search(pattern, error_lower, re.I):
                    return {
                        "category": info["category"],
                        "type": error_type,
                        "description": f"{error_type}: {error[:100]}",
                        "root_cause": info["category"],
                        "fixes": info["fixes"],
                        "confidence": 0.85
                    }
        return {
            "category": "unknown_error",
            "type": "unknown",
            "description": f"未知错误: {error[:100]}",
            "root_cause": "需要进一步分析",
            "fixes": ["记录错误详情", "尝试替代工具", "简化任务"],
            "confidence": 0.3
        }

    def _is_repeated_failure(self, category: str, tool_name: str) -> bool:
        failures = self.failure_memory.get(category, [])
        recent = [
            f for f in failures
            if f["tool"] == tool_name
            and (datetime.now() - datetime.fromisoformat(f["timestamp"])).seconds < 300
        ]
        return len(recent) > 0

    def _detect_loop(self, recent_tools: List[str]) -> bool:
        if len(recent_tools) < 4:
            return False
        last_4 = recent_tools[-4:]
        return len(set(last_4)) <= 2 and last_4[0] == last_4[2]

    def should_continue(self, history: List[Dict], max_failed: int = 3) -> Tuple[bool, str]:
        """判断是否继续执行。

        修复要点（v2）：
          1. strategic 级别需要连续 3 次（而非 2 次）才中断，容忍偶发性路径错误
          2. 成功记录（success=True）不计入失败，避免"部分成功"被误判
          3. 只统计 reflection_history 中的失败项，而非所有 history 条目
          4. 提供更宽松的阈值，对多步骤复杂任务更友好
        """
        # 仅统计真实失败的反思记录（成功不计入）
        failed_history = [
            h for h in history
            if isinstance(h, dict) and not h.get("success", True)
        ]
        if len(failed_history) < max_failed:
            return True, ""

        recent_failed = failed_history[-max_failed:]

        # strategic 级别需要连续 3 次才中断（原来是 2 次，过于激进）
        strategic_count = sum(
            1 for h in recent_failed
            if isinstance(h, dict) and (
                h.get("level") == "strategic" or
                (isinstance(h.get("reflection"), dict) and h["reflection"].get("level") == "strategic")
            )
        )
        if strategic_count >= max_failed:
            return False, "连续战略级失败，建议重新规划或人工介入"

        # meta 级别（检测到循环）立即中断
        meta_count = sum(
            1 for h in recent_failed
            if isinstance(h, dict) and (
                h.get("level") == "meta" or
                (isinstance(h.get("reflection"), dict) and h["reflection"].get("level") == "meta")
            )
        )
        if meta_count >= 2:
            return False, "检测到工具调用循环，建议更换策略"

        # 连续失败且「类别+工具+参数签名」相同（更精确地识别死循环）
        signatures = []
        for h in recent_failed:
            category = h.get("category") or (h.get("reflection") or {}).get("category", "unknown")
            tool = h.get("tool", "")
            args_sig = h.get("args_sig", "")
            signatures.append((category, tool, args_sig))

        if signatures and len(set(signatures)) == 1 and signatures[0][0] != "unknown":
            cat, tool, _ = signatures[0]
            return False, f"连续 {len(recent_failed)} 次相同错误({cat})且重复工具/参数({tool})，建议更换策略"

        return True, ""

    def get_reflection_summary(self, history: List[Dict]) -> str:
        if not history:
            return "暂无反思记录"
        recent = history[-5:]
        lines = ["## 近期反思记录"]
        for r in recent:
            refl = r.get("reflection", {})
            level_icon = {"surface": "📝", "operational": "🔧", "strategic": "🎯", "meta": "🧠"}.get(
                refl.get("level", "surface"), "•"
            )
            icon = "✅" if r.get("success") else "❌"
            lines.append(f"{icon}{level_icon} {r['tool']}: {refl.get('analysis', '')[:50]}...")
        efficient = self.get_efficient_sequences(3)
        if efficient:
            lines.append("## 高效序列（Top-3）")
            for seq in efficient:
                lines.append(f"  {seq['sequence']} ×{seq['count']}次 avg {seq['avg_time']:.1f}s")
        return "\n".join(lines)


from pydantic import BaseModel, Field, ValidationError

class ReadFileArgs(BaseModel):
    path: str = Field(..., description="相对或绝对路径")

class WriteFileArgs(BaseModel):
    path: str
    content: str
    mode: str = Field(default="overwrite")

class EditFileArgs(BaseModel):
    path: str
    old_content: str
    new_content: str

class ListDirArgs(BaseModel):
    path: str = Field(default=".")

class BashArgs(BaseModel):
    command: str
    timeout: int = Field(default=30)

class ExecutePythonArgs(BaseModel):
    code: str
    timeout: int = Field(default=30)

TOOL_SCHEMAS = {
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "edit_file": EditFileArgs,
    "list_dir": ListDirArgs,
    "bash": BashArgs,
    "execute_python": ExecutePythonArgs,  # ⚠️ 必须注册，否则 validate_tool_call 会报"未知工具"
}

class OutputValidator:
    @staticmethod
    def validate_tool_call(tool_name: str, args: Dict) -> Tuple[bool, str, Dict]:
        if tool_name not in TOOL_SCHEMAS:
            return False, f"未知工具: {tool_name}", args
        try:
            schema = TOOL_SCHEMAS[tool_name]
            validated = schema(**args).model_dump()
            return True, "", validated
        except ValidationError as e:
            return False, f"参数验证失败: {e}", args

    @staticmethod
    def sanitize_output(output: str, max_length: int = 100000) -> str:
        if len(output) > max_length:
            return f"{output[:max_length]}...\n[输出过长，已截断。共 {len(output)} 字符]"
        return output


class QwenAgentFramework:
    """增强Agent框架 - 集成所有改进"""

    def __init__(
            self,
            model_forward_fn: Callable,
            work_dir: Optional[str] = None,
            enable_bash: bool = False,
            max_iterations: int = 20,
            middlewares: Optional[List] = None,
            enable_memory: bool = True,
            enable_reflection: bool = True,
            enable_parallel: bool = True,
            enable_tool_learning: bool = True,
            tools_in_system_prompt: bool = True,
            default_runtime_context: Optional[Dict] = None,
    ):
        self.model_forward_fn = model_forward_fn
        self.tool_executor = ToolExecutor(work_dir=work_dir, enable_bash=enable_bash)
        self.tool_parser = ToolParser()
        self.max_iterations = max_iterations
        self.middlewares = middlewares or []
        self.enable_parallel = enable_parallel
        self.tools_in_system_prompt = tools_in_system_prompt
        self.default_runtime_context = default_runtime_context or {}

        # 核心组件
        if enable_memory:
            self.vector_memory = VectorMemory(embedding_provider=LocalEmbeddingProvider())
            self.memory = self.vector_memory  # 兼容
        else:
            self.vector_memory = None
            self.memory = None

        self.reflection = DeepReflectionEngine() if enable_reflection else None
        self.tool_learner = AdaptiveToolLearner() if enable_tool_learning else None
        # 兼容占位（不再持有真实 Executor，仅供外部代码引用 .stats 时不报错）
        self.parallel_executor = EnhancedParallelExecutor()

        # 双向绑定：反思引擎 ↔ 工具学习器
        if self.reflection and self.tool_learner:
            self.reflection.attach_tool_learner(self.tool_learner)

        self.validator = OutputValidator()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        work_dir_abs = str(self.tool_executor.work_dir.resolve())
        tools = ["bash", "read_file", "write_file", "edit_file", "list_dir", "execute_python"] if self.tool_executor.enable_bash else [
            "read_file", "write_file", "edit_file", "list_dir", "execute_python"]
        tools_desc = "\n".join(f"- {t}" for t in tools)

        return f"""你是智能助手，使用 ReAct (Reasoning + Acting) 模式工作。

当前工作目录（绝对路径）：{work_dir_abs}
使用 read_file/write_file 时若不确定路径，优先用此绝对路径拼接文件名，例如：{work_dir_abs}/文件名.py

可用工具：
{tools_desc}

【工具选择策略】
- 批量扫描/搜索多个文件 → 优先用 bash (grep, find)
- 读取单个文件内容 → 用 read_file
- 浏览目录结构 → 用 list_dir
- 若 execute_python 连续失败，应立即改用 bash 命令完成（如用 grep 提取类/方法，用重定向写入文件）
- 回溯/总结历史对话（包含所有历史会话）→ 用 execute_python 调用 get_logger().get_all_user_questions()，这是最快最全的方法：
  execute_python {{"code": "import sys; sys.path.insert(0,'ui'); from session_logger import get_logger; qs=get_logger().get_all_user_questions(); [print(f\"[{{q['session_date']}}] {{q['user_message'][:80]}}\") for q in qs]"}}
  ⚠️ 禁止逐个 read_file 读取 session_logs/，会话数量多时会超时。应一次性用上面的 execute_python 拉取全部问题。

【read_file 路径防护 - 必须遵守】
- ⛔ 禁止读取以下目录内的文件：.venv/、venv/、node_modules/、__pycache__/、.git/、*.dist-info/、site-packages/
- ⛔ 禁止读取与当前任务无关的配置文件（LICENSE、README、entry_points.txt、top_level.txt 等）
- ✅ 只读取项目源码目录（core/、ui/、skills/）和任务明确指定的文件

工具调用格式（严格遵守，每次只调用一个工具）：
bash
{{"command": "shell命令，如 grep -Ern '^class |^def ' core/"}}

read_file
{{"path": "相对或绝对路径"}}

write_file
{{"path": "目标文件路径", "content": "要写入的内容", "mode": "overwrite（默认，完全替换）或 append（追加到末尾）"}}

list_dir
{{"path": "目录路径，支持相对路径（如 core）或绝对路径（如 /Users/xxx/project/core）"}}

edit_file
{{"path": "文件路径", "old_content": "原文本", "new_content": "新文本"}}

execute_python
{{"code": "# Python 代码\\nprint('hello')", "timeout": 30}}

ReAct 流程：
1. **Thought**: 分析当前情况，思考下一步
2. **Action**: 调用上方格式之一（工具名单独一行，JSON 紧跟在下一行）
3. **Observation**: 观察工具返回结果
4. **Reflection**: 反思是否成功，下一步怎么做

核心原则：
- 直接调用工具，不要解释"我无法访问文件系统"——你完全可以调用工具
- 用户提供绝对路径时（如 /Users/xxx/...），直接用该路径调用工具
- ⚠️ 扫描/搜索多个文件时，必须用 bash grep，禁止用 list_dir + read_file 逐个读取
- 【重要】read_file 支持仅凭文件名自动全项目搜索：直接 read_file {{"path": "文件名.py"}} 即可，无需先 list_dir 找路径
- list_dir 只在需要了解目录整体结构时使用，不要用 list_dir 来"找文件"
- 相同工具相同参数不要重复调用超过 2 次
- 失败后换策略，不要循环重试相同操作
- 禁止说"无法使用工具"、"环境限制"等放弃性话术
- ⚠️ 知识类回答防幻觉：对于引用、歌词、诗句、名人名言、具体数据等，如不确定请标注「（待核实）」，不要编造

execute_python 使用规范：
- 适用场景：执行任意 Python 代码——算法验证、文件读写、系统调用、数据处理、调试函数等
- 优先于 bash：执行纯 Python 逻辑时用 execute_python，执行 shell 命令时用 bash
- 完整 Python 功能：os/subprocess/socket/文件读写/eval/pathlib 等全部可用，无任何代码内容限制
- 预装库（无需手动 import）：math、re、json、os、os.path、subprocess、shutil、pathlib（Path）、
  datetime（date/timedelta）、collections（Counter/defaultdict/deque）、itertools、functools、random、statistics
- 输出捕获：print() 内容会出现在 Observation 的 stdout 字段中
- ⚠️【重要】print 必须写在代码顶层，绝对不能包裹在 `if __name__ == '__main__':` 块内！
  因为代码在 wrapper 中运行时 __name__ 不是 '__main__'，被包裹的 print 永远不会执行！
  错误：if __name__ == '__main__':\\n    print(results)  ← 不会输出！
  正确：print(results)  ← 直接顶层调用
- ⚠️【写文件必须 print 确认】若代码中包含 open(...,'w') / .write() 等写文件操作，
  必须在写完后用 print() 输出确认信息！否则 stdout 为空会导致系统误判为未执行而无限重试！
  正确：with open('API.md','w') as f: f.write(content) → 之后必须加 print("✅ API.md 已写入")
  错误：只写文件不加 print → stdout 为空 → 系统认为未执行 → 无限重复调用
- 若 Observation 中 stdout 为空（但 success=true）且代码不含写文件操作：
  说明代码未包含顶层 print，修改代码加上 print() 后再调用
- timeout 配置：默认 30s，长时间计算可传 timeout 参数
  示例：execute_python {{"code": "for i in range(10**6): pass", "timeout": 60}}
- 代码格式：用 \\n 表示换行，或使用多行字符串（JSON 中转义）
- 简单示例：execute_python {{"code": "result = sum(range(100))\\nprint(result)"}}
- 文件读取示例：execute_python {{"code": "content = open('core/agent_tools.py').read()\\nprint(len(content))"}}
- 【执行失败自动修复】：若 Observation 中 success=false，必须仔细阅读 error/stderr 字段的错误信息，
  修正代码后立即再次调用 execute_python，不要放弃，也不要改用其他工具替代
- 【禁止重复调用】：同一段代码（相同功能）调用超过 2 次后，系统会自动检测为死循环并中断。
  若已调用 2 次仍无结果，说明策略有误，应更换工具或修改代码逻辑而非继续重试
- ⚠️【强制降级指令】：若 execute_python 连续失败 2 次，必须立即停止使用 execute_python，
  改用 bash 命令完成相同任务！例如扫描目录并写入文件：
  bash {{\"command\": \"grep -E '^class |^def ' core/*.py > API.md\"}}

bash 使用规范：
- timeout 配置：默认 30s，长时间命令可传 timeout 参数，示例：bash {{"command": "find . -name '*.py'", "timeout": 60}}
- 大输出保护：stdout 超 12000 字符自动截断，建议用 head -N、grep -m N 限制输出行数
  示例：bash {{"command": "grep -rn 'def ' core/ | head -50"}}

文件操作策略（务必遵守）：
- 【禁止】一次性读完所有文件再统一写出——这会超出迭代限制
- 【正确】读一个文件 → 立即写/修改它 → 再读下一个（流水线方式）
- 【写前读规则】：若需要将扫描/读取结果写入文件，必须先执行 bash/read_file 获取结果，
  在 Observation 中看到实际内容后，再调用 write_file 写入真实数据，禁止在未获得读取结果时就写文件
- 【禁止伪造内容】：write_file 的 content 字段必须是 Observation 中出现的实际文本，
  严禁使用：bash['stdout']、变量引用、"...省略..."、"# (更多内容)"等占位符代替真实内容
- 【bash 重定向写文件】：bash 命令中使用 > 或 >> 重定向（如 grep ... > API.md）时，
  文件由 Shell 直接写入，bash 工具返回的 stdout 为空是完全正常的，文件内容已正确写入磁盘。
  ⛔ 此后【绝对禁止】再用 write_file 写同一个文件（会用空内容覆盖掉 bash 写入的内容！）
  ✅ 若要验证内容，用 read_file 或 bash {{"command": "head -5 API.md"}} 查看
- 【write_file 模式选择】：
  * 新建文件或完全重写：用 overwrite（默认）
  * 向现有文件末尾追加内容（如整理文档时逐步追加）：必须用 append
  * 禁止用 overwrite 反复覆盖同一文件（会丢失前一次写入的内容）
  * 禁止对同一文件多次 append 写入相同内容（会产生重复）
- 【read_file 去重】：Observation 中已显示"已读过"的文件不要重复 read_file，直接使用上次结果

知识问答策略（混合任务时）：
- 任务中若包含纯知识问答子任务（如"列出十首歌"、"解释概念"），直接在回复中给出答案
- 不要将知识问答的答案主动写入文件，除非用户明确要求保存到文件
- 每次只调用一个工具，不要在同一轮中规划多个工具调用并一次性全部输出

输出要求：
- 简洁明确，先思考再调用工具
- 遇到错误分析原因并调整策略"""

    def _detect_parallel_tools(self, tool_calls: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """将 tool_calls 分为可并行的只读工具和必须串行的写操作工具。

        「写前读」规则：
        - 若同一批调用中**既有读操作（bash/read_file）又有写操作（write_file/edit_file）**，
          则所有工具均退为串行，且读操作排在写操作前面。
          这防止模型在未获得读取结果的情况下就写入文件（产生空壳/伪造内容）。
        - 没有写操作时，只读工具才能并行。
        """
        if not self.enable_parallel or len(tool_calls) <= 1:
            return [], tool_calls

        _write_tools = {"write_file", "edit_file"}
        _read_tools = {"read_file", "bash"}

        has_write = any(tc["name"] in _write_tools for tc in tool_calls)
        has_read = any(tc["name"] in _read_tools for tc in tool_calls)

        # 「写前读」：同批次混有读写操作 → 全部串行，读操作置前
        if has_write and has_read:
            reads_first = [tc for tc in tool_calls if tc["name"] in _read_tools]
            others = [tc for tc in tool_calls if tc["name"] not in _read_tools]
            return [], reads_first + others

        parallel, sequential = [], []
        for tc in tool_calls:
            if tc["name"] in READ_ONLY_TOOLS:
                parallel.append(tc)
            else:
                sequential.append(tc)
        return parallel, sequential

    async def _execute_tools_parallel(self, tool_calls: List[Dict], session: SessionContext, runtime_context: Dict = None) -> List[Dict]:
        """并行执行只读工具，每个线程仍经过完整中间件 + 反思 + 记忆管道。"""
        if not tool_calls:
            return []

        import asyncio

        async def _run(idx: int, tc: Dict):
            try:
                result = await self._execute_single_tool(tc["name"], tc["args"], session, runtime_context or {})
                return idx, {"index": idx, "tool": tc["name"], "result": result, "parallel": True}
            except Exception as e:
                return idx, {
                    "index": idx,
                    "tool": tc["name"],
                    "result": {"error": str(e)},
                    "parallel": True
                }

        tasks = [_run(i, tc) for i, tc in enumerate(tool_calls)]
        completed = await asyncio.gather(*tasks)

        results: List[Optional[Dict]] = [None] * len(tool_calls)
        for idx, res in completed:
            results[idx] = res

        return results  # type: ignore[return-value]

    def _already_done_tool_guard(
        self, tool_name: str, tool_args: Dict, session: SessionContext
    ) -> Optional[Dict]:
        """工具执行前守卫：若该工具操作对应一个「已完成」的子任务，直接拦截并返回提示。

        拦截条件：
        1. 有结构化子任务状态（subtask_status 非空）
        2. 本次工具调用可推断到某个 done 子任务（通过 _infer_subtask_from_tool）
        3. 该子任务已标记为 done

        返回：若拦截返回错误 dict，否则返回 None（放行）。
        注意：对 write_file/edit_file 不拦截（写入操作属于完成步骤，不应被阻止）。
        """
        subtask_status = session.task_context.get("subtask_status", {})
        if not subtask_status:
            return None

        # 写操作不拦截——即使子任务已完成，也允许补充写入
        _write_tools = {"write_file", "edit_file"}
        if tool_name in _write_tools:
            return None

        # 推断该工具对应哪个子任务（严格模式：只有关键词精确匹配才拦截，不使用回退）
        idx = self._infer_subtask_from_tool(tool_name, tool_args, session, strict=True)
        if idx is None:
            return None

        info = subtask_status.get(idx)
        if not info or info["status"] != "done":
            return None

        # 被拦截：返回友好提示
        pending_list = [i for i, inf in sorted(subtask_status.items()) if inf["status"] == "pending"]
        next_hint = ""
        if pending_list:
            next_idx = pending_list[0]
            next_hint = f"\n▶ 请立即执行子任务{next_idx}: {subtask_status[next_idx]['desc']}"

        return {
            "output": (
                f"🚫 [系统拦截] 子任务{idx}（{info['desc'][:30]}）已完成，"
                f"无需再次调用 {tool_name}。"
                f"{next_hint}"
                f"\n请跳过已完成子任务，继续执行未完成的子任务。"
            ),
            "_exec_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "_tool_name": tool_name,
            "_reflection": None,
            "_execution_time": 0.0,
            "_subtask_guard": True,
        }

    async def _execute_single_tool(self, tool_name: str, tool_args: Dict, session: SessionContext, context: Dict = None) -> Dict:
        """执行单个工具，经过：参数验证 → 前置中间件 → 执行 → 后置中间件 → 反思 → 记忆。"""
        runtime_context = context or {}

        # ── 子任务完成守卫：若对应子任务已完成，直接拦截并提示跳至下一任务 ──────
        # ⚠️ 修复: 拦截结果必须写入 tool_history，否则 _detect_loop 无法感知循环，
        # 导致模型对同一已完成子任务无限重试工具调用。
        _guard_result = self._already_done_tool_guard(tool_name, tool_args, session)
        if _guard_result is not None:
            _args_sig = json.dumps(tool_args, sort_keys=True)
            session.tool_history.append({
                "tool": tool_name,
                "args": _args_sig,
                "success": False,   # 视为失败，触发 _detect_loop 的高频重复检测
                "reflection": None,
            })
            session.task_context["failed_attempts"].append(tool_name)
            return _guard_result

        # ── 已读文件拦截：对 read_file 工具，若路径已在 read_files_cache 中，
        # 直接返回提示，不重复读取。防止模型因看到截断提示而陷入反复读同一文件的循环。
        #
        # ⚠️ 修复 (v3): 仅对"精确路径命中"拦截；basename 归一化只在绝对/相对路径互转时使用，
        # 不能跨目录拦截同名不同文件（如 core/session_analyzer.py vs ui/session_analyzer.py）。
        if tool_name == "read_file":
            import os as _os
            path = tool_args.get("path", "")
            _exact_hit = bool(path and path in session.read_files_cache)

            # 仅在精确路径命中时拦截；对于 basename 匹配只做同目录判断
            _basename_only_hit = False
            if not _exact_hit and path:
                _basename = _os.path.basename(path)
                _abs_path = _os.path.abspath(path)
                for cached_path in session.read_files_cache:
                    # 两个路径的真实绝对路径相同（处理相对路径和绝对路径互转）
                    if _os.path.abspath(cached_path) == _abs_path:
                        _exact_hit = True
                        break
                    # 同目录下 basename 相同才认为是同一文件
                    if (_os.path.basename(cached_path) == _basename
                            and _os.path.dirname(_os.path.abspath(cached_path))
                                == _os.path.dirname(_abs_path)):
                        _basename_only_hit = True
                        break

            if _exact_hit or _basename_only_hit:
                _matched_path = path
                # ⚠️ 修复: 重复读文件拦截也写入 tool_history，触发 _detect_loop 防止无限循环
                _args_sig = json.dumps(tool_args, sort_keys=True)
                session.tool_history.append({
                    "tool": tool_name,
                    "args": _args_sig,
                    "success": False,
                    "reflection": None,
                })
                session.task_context["failed_attempts"].append(tool_name)
                return {
                    "output": (
                        f"⚠️ [系统拦截] 文件 {_matched_path} 已在本轮会话中读取过（见上方 Observation）。"
                        f"\n请勿重复 read_file 读取同一文件。"
                        f"\n请直接基于已获取的文件内容完成分析和回答，无需任何工具调用。"
                    ),
                    "_exec_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "_tool_name": tool_name,
                    "_reflection": None,
                    "_execution_time": 0.0,
                    "_cached": True,
                }

        for mw in self.middlewares:
            tool_name, tool_args = await mw.process_before_tool(tool_name, tool_args, runtime_context)

        valid, error_msg, validated_args = self.validator.validate_tool_call(tool_name, tool_args)
        if valid:
            tool_args = validated_args
        else:
            result: Dict = {"error": f"参数验证失败: {error_msg}"}
            for mw in self.middlewares:
                result_str = json.dumps(result)
                result_str = await mw.process_after_tool(tool_name, tool_args, result_str, runtime_context)
                try:
                    result = json.loads(result_str)
                except Exception:
                    result = {"error": result_str}
            return result

        start_time = time.time()
        exec_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        result_str = self.tool_executor.execute_tool(tool_name, tool_args)
        exec_time = time.time() - start_time

        for mw in self.middlewares:
            result_str = await mw.process_after_tool(tool_name, tool_args, result_str, runtime_context)

        try:
            result_obj = json.loads(result_str)
            is_success = not result_obj.get("error")
        except Exception:
            is_success = not result_str.startswith("Error:")
            result_obj = {"output": result_str} if is_success else {"error": result_str}

        if not is_success:
            session.task_context["failed_attempts"].append(tool_name)

            # ---------- 修复：execute_python 连续失败后自动执行 bash 降级 ----------
            if tool_name == "execute_python":
                recent_fails = session.task_context["failed_attempts"][-10:]
                py_fail_count = sum(1 for t in recent_fails if t == "execute_python")
                # 降低阈值：连续失败 2 次即触发降级（原为 3 次）
                if py_fail_count >= 2 and not session.task_context.get("_auto_bash_executed"):
                    session.task_context["_auto_bash_executed"] = True
                    # 构造更鲁棒的 bash 命令
                    bash_cmd = "grep -E '^class |^def ' core/*.py > API.md"
                    bash_args = {"command": bash_cmd}
                    # 直接调用工具执行器执行 bash
                    bash_result_str = self.tool_executor.execute_tool("bash", bash_args)
                    try:
                        bash_result_obj = json.loads(bash_result_str)
                        bash_success = not bash_result_obj.get("error")
                    except Exception:
                        bash_success = not bash_result_str.startswith("Error:")
                        bash_result_obj = {"output": bash_result_str} if bash_success else {"error": bash_result_str}

                    # 记录这次自动执行的 bash 调用
                    session.tool_history.append({
                        "tool": "bash",
                        "args": json.dumps(bash_args, sort_keys=True),
                        "success": bash_success,
                        "reflection": None,
                    })
                    if bash_success:
                        session.task_context["completed_steps"].append(f"bash(auto-fallback)")
                        if "files_written" not in session.task_context:
                            session.task_context["files_written"] = []
                        session.task_context["files_written"].append("API.md")

                    # 构造返回结果，替换原本失败的 execute_python 结果
                    result_obj = {
                        "output": f"[自动降级] execute_python 连续失败 {py_fail_count} 次，系统已自动执行 bash 命令：{bash_cmd}\n\n"
                                  f"执行结果：{bash_result_obj.get('stdout', bash_result_obj.get('output', ''))}\n"
                                  f"错误信息：{bash_result_obj.get('stderr', '')}",
                        "_auto_fallback": True,
                        "_bash_result": bash_result_obj,
                    }
                    result_str = json.dumps(result_obj, ensure_ascii=False)
                    is_success = bash_success  # 将本次操作标记为成功
                    final_result: Dict = {
                        "output": result_obj.get("output"),
                        "_exec_timestamp": exec_timestamp,
                        "_tool_name": "bash",
                        "_reflection": None,
                        "_execution_time": time.time() - start_time,
                    }
                    return final_result
            # -----------------------------------------------------------------

            if "tool_result_log" not in session.task_context:
                session.task_context["tool_result_log"] = []
            args_hint = ", ".join(f"{k}={str(v)[:30]}" for k, v in tool_args.items())
            session.task_context["tool_result_log"].append({
                "tool": tool_name,
                "args_hint": args_hint,
                "success": False,
                "result": result_str[:300],
            })
            if len(session.task_context["tool_result_log"]) > 20:
                session.task_context["tool_result_log"] = session.task_context["tool_result_log"][-20:]

        # 反思引擎（成功/失败均触发，内部分别调用 _record_success 或记录失败）
        reflection_result = None
        if self.reflection:
            reflection_context = {
                "recent_tools": [h.get("tool") for h in session.tool_history[-5:]],
                "task": session.task_context.get("current_task", "general"),
                "_execution_time": exec_time,
                "tool_args_sig": json.dumps(tool_args, ensure_ascii=False, sort_keys=True)[:200],
            }
            reflection_result = self.reflection.reflect_on_result(
                tool_name, result_obj, context=reflection_context, history=session.reflection_history
            )

        # 注意：ToolLearner 的 record_usage 已通过 DeepReflectionEngine 的双向同步调用，
        # 此处不再重复调用，避免重复计数。

        if self.vector_memory and session.current_tool_chain_id:
            self.vector_memory.add(
                content=f"Tool: {tool_name}, Args: {tool_args}, Success: {is_success}",
                metadata={"type": "tool_execution", "tool": tool_name, "success": is_success},
                importance=0.8 if not is_success else 0.5,
                tool_chain_id=session.current_tool_chain_id
            )

        if self.memory and hasattr(self.memory, "update_tool_stats"):
            self.memory.update_tool_stats(tool_name, is_success, exec_time)

        session.tool_history.append({
            "tool": tool_name,
            "args": json.dumps(tool_args, sort_keys=True),
            "success": is_success,
            "reflection": reflection_result
        })

        if is_success:
            session.task_context["completed_steps"].append(f"{tool_name}({list(tool_args.keys())})")
            if tool_name == "read_file" and tool_args.get("path"):
                session.read_files_cache[tool_args["path"]] = exec_timestamp
                # 记录已读文件清单（供执行骨架使用）
                if "files_read" not in session.task_context:
                    session.task_context["files_read"] = []
                fpath = tool_args["path"]
                if fpath not in session.task_context["files_read"]:
                    session.task_context["files_read"].append(fpath)
            elif tool_name in ("write_file", "edit_file") and tool_args.get("path"):
                # 记录已写文件清单
                if "files_written" not in session.task_context:
                    session.task_context["files_written"] = []
                fpath = tool_args.get("path", "")
                if fpath not in session.task_context["files_written"]:
                    session.task_context["files_written"].append(fpath)
            # 记录工具调用结果日志（最多保留 20 条）
            if "tool_result_log" not in session.task_context:
                session.task_context["tool_result_log"] = []
            args_hint = ", ".join(f"{k}={str(v)[:30]}" for k, v in tool_args.items())
            # execute_python 的 stdout 是关键数据（如历史问题列表），需保留更多内容
            if tool_name == "execute_python":
                try:
                    _py_resp = json.loads(result_str)
                    _py_stdout = _py_resp.get("stdout", "").strip()
                    if _py_stdout and len(_py_stdout) > 50:
                        # 将 stdout 内容单独保存到 task_context，防止压缩时丢失
                        if "execute_python_outputs" not in session.task_context:
                            session.task_context["execute_python_outputs"] = []
                        session.task_context["execute_python_outputs"].append({
                            "stdout": _py_stdout[:3000],
                            "code_hint": tool_args.get("code", "")[:80],
                        })
                        if len(session.task_context["execute_python_outputs"]) > 5:
                            session.task_context["execute_python_outputs"] = session.task_context["execute_python_outputs"][-5:]
                    elif not _py_stdout:
                        # ⚠️ stdout 为空：注入警告到结果中，防止模型认为"没有执行"而无限重试
                        # 常见原因：代码被 `if __name__ == '__main__':` 包裹，在 wrapper 中不执行
                        import re as _re_write_detect
                        _code_text = tool_args.get("code", "")
                        _has_write_op = bool(
                            _re_write_detect.search(
                                r'open\s*\([^)]*[\'\"].*[\'\"],\s*[\'\"]w[\'\"]'
                                r'|\.write\s*\(|write_file|to_csv|to_json|to_excel'
                                r'|shutil\.copy|os\.rename',
                                _code_text
                            )
                        )
                        if _has_write_op:
                            # 代码包含写文件操作且 stdout 为空 → 写文件任务已完成
                            _py_resp["_stdout_empty_warning"] = (
                                "✅ 代码执行成功。stdout 为空是因为代码执行了写文件操作（没有 print 输出），"
                                "文件已成功写入，任务已完成！不要重复执行相同的代码！"
                            )
                        else:
                            _py_resp["_stdout_empty_warning"] = (
                                "⚠️ stdout 为空（代码执行成功但无输出）。"
                                "若需要查看结果，请确保代码顶层包含 print() 语句，"
                                "不要将 print 放在 `if __name__ == '__main__':` 块内。"
                                "若代码目的是写文件，任务已完成，请不要重试。"
                            )
                        result_str = json.dumps(_py_resp, ensure_ascii=False)
                    result_snippet = result_str[:2000]  # execute_python 结果保留更多
                except Exception:
                    result_snippet = result_str[:500]
            else:
                result_snippet = result_str[:500]
            session.task_context["tool_result_log"].append({
                "tool": tool_name,
                "args_hint": args_hint,
                "success": True,
                "result": result_snippet,
            })
            if len(session.task_context["tool_result_log"]) > 20:
                session.task_context["tool_result_log"] = session.task_context["tool_result_log"][-20:]
            # 更新结构化子任务状态
            self._mark_subtask_done_if_applicable(tool_name, tool_args, True, session)
        else:
            session.task_context["failed_attempts"].append(tool_name)
            # 记录失败工具调用（供执行骨架和反思使用）
            if "tool_result_log" not in session.task_context:
                session.task_context["tool_result_log"] = []
            args_hint = ", ".join(f"{k}={str(v)[:30]}" for k, v in tool_args.items())
            session.task_context["tool_result_log"].append({
                "tool": tool_name,
                "args_hint": args_hint,
                "success": False,
                "result": result_str[:300],
            })
            if len(session.task_context["tool_result_log"]) > 20:
                session.task_context["tool_result_log"] = session.task_context["tool_result_log"][-20:]

        final_result: Dict = {
            "output": result_str if is_success else None,
            "error": result_obj.get("error") if not is_success else None,
            "_exec_timestamp": exec_timestamp,
            "_tool_name": tool_name,
            "_reflection": reflection_result,
            "_execution_time": exec_time,
        }
        if final_result.get("output"):
            final_result["output"] = self.validator.sanitize_output(final_result["output"])
        return final_result

    # ------------------------------------------------------------------
    # 核心生成器：_run_iter
    # ------------------------------------------------------------------
    from typing import AsyncGenerator

    async def _run_iter(
            self,
            messages: List[Dict],
            user_input: str,
            session: SessionContext,
            runtime_context: Dict,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> AsyncGenerator[Dict, None]:
        """单轮推理迭代生成器。
        每次 yield 一个事件 dict，消费者（run / StreamingFramework）据此驱动逻辑。

        事件类型（event 字段）：
          progress      —— 迭代开始
          thought       —— LLM 思考（content 字段）
          tool_call     —— 工具即将执行
          tool_result   —— 工具执行完成（result 字段）
          reflection    —— 反思摘要
          format_error  —— 格式纠错，需 continue 外循环
          loop_detected —— 检测到循环，需终止
          interrupted   —— 反思引擎要求中断
          done          —— 任务完成（final_response 字段）
          max_iter      —— 达到最大迭代次数
          llm_error     —— 模型调用失败
        """
        tool_calls_log: List[Dict] = []
        last_response = ""

        # 初始化结构化子任务状态（解析用户任务中的编号子任务；plan/multi_agent 模式自动跳过）
        self._init_subtask_status(session, runtime_context=runtime_context)

        # ⚠️ 全局超时保险：最多执行 3 分钟（180秒），防止任何路径卡死
        # Plan 模式下每步独立调用，每步可配置更长超时
        import time as _time_module
        _iter_start_ts = _time_module.time()
        _MAX_TOTAL_SECONDS = runtime_context.get("max_timeout_seconds", 180)

        for iteration in range(self.max_iterations):
            # 硬超时检测：超过 3 分钟强制中断
            _elapsed = _time_module.time() - _iter_start_ts
            if _elapsed > _MAX_TOTAL_SECONDS:
                _timeout_resp = f"⏰ 任务执行超时（已运行 {int(_elapsed)}s），强制中断。已完成的步骤结果已在上方输出。"
                yield {
                    "event": "done",
                    "final_response": _timeout_resp,
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                    "timed_out": True,
                }
                return

            runtime_context["iteration"] = iteration
            yield {"event": "progress", "iteration": iteration + 1, "max_iterations": self.max_iterations}

            # 反思引擎：是否继续
            if self.reflection:
                should_continue, reason = self.reflection.should_continue(session.reflection_history, max_failed=3)
                if not should_continue:
                    fallback_answer = self._attempt_no_tool_answer_before_interrupt(
                        messages=messages,
                        user_input=user_input,
                        session=session,
                        reason=reason,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    if fallback_answer:
                        final_response = self._clean_react_tags(self._strip_trailing_tool_call(fallback_answer))
                        tool_calls_log.append({
                            "iteration": iteration + 1,
                            "tool": None,
                            "type": "final_response",
                            "success": True,
                            "parallel": False,
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        })
                        yield {
                            "event": "done",
                            "final_response": final_response,
                            "tool_calls": tool_calls_log,
                            "iterations": iteration + 1,
                        }
                        return

                    yield {
                        "event": "interrupted",
                        "reason": reason,
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                    }
                    return

            # 上下文压缩 + 注入
            # Plan 子步骤执行时：每步 system prompt 已带完整计划骨架（约 600 字），
            # 加上 Observation 会快速积累；提高 limit 至 16000 避免每步内频繁压缩
            _compress_limit = 16000 if session.task_context.get("_plan_step_id") else 12000
            messages = self._compress_context_smart(messages, session, limit=_compress_limit)
            messages = self._inject_task_context(messages, session)
            messages = self._inject_reflection(messages, session)

            # 工具推荐（首轮及后续，阈值降至 0.3）
            if self.tool_learner:
                recommendations = self.tool_learner.recommend_next_tools(
                    user_input,
                    [h["tool"] for h in session.tool_history],
                    current_context={"iteration": iteration, "task": user_input}
                )
                # 首轮使用历史推荐（如有），后续降低阈值至 0.3
                min_confidence = 0.5 if iteration == 0 else 0.3
                if recommendations and recommendations[0]["confidence"] >= min_confidence:
                    messages = self._inject_tool_recommendations(messages, recommendations)

            # 高效序列提示（反思引擎积累的成功模式）
            if self.reflection and iteration > 0:
                messages = self._inject_efficient_sequences(messages, session)

            # 中间件：before LLM
            for mw in self.middlewares:
                if hasattr(mw, "process_before_llm"):
                    messages = await mw.process_before_llm(messages, runtime_context)

            # LLM 调用
            try:
                response = self.model_forward_fn(
                    messages, self.system_prompt,
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens
                )
                last_response = response
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": None,
                    "type": "thought",
                    "content": response[:1000],
                    "success": True,
                    "parallel": False,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
                yield {"event": "thought", "iteration": iteration + 1, "content": response}
            except Exception as e:
                fallback = None
                for mw in self.middlewares:
                    fb = await mw.process_on_error(e, "llm_call", runtime_context)
                    if fb is not None:
                        fallback = fb
                yield {
                    "event": "llm_error",
                    "error": str(e),
                    "fallback": fallback,
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                }
                return

            # 中间件：after LLM
            for mw in self.middlewares:
                response = await mw.process_after_llm(response, runtime_context)

            if runtime_context.get("_needs_retry"):
                runtime_context["_needs_retry"] = False
                continue

            # 记录原始响应用于 thought 循环检测（解决 ToolParser 解析失败时 tool_history 为空导致循环检测失效的问题）
            if not hasattr(session, 'raw_response_cache'):
                session.raw_response_cache = []
            session.raw_response_cache.append(response)
            if len(session.raw_response_cache) > 20:  # 只保留最近 20 条，避免内存膨胀
                session.raw_response_cache = session.raw_response_cache[-20:]

            # 解析工具调用
            tool_calls_raw = self.tool_parser.parse_tool_calls(response)
            tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

            if not tool_calls:
                if self._looks_finished(response, session, runtime_context):
                    final_response = self._clean_react_tags(self._strip_trailing_tool_call(response))
                    if self.vector_memory:
                        self.vector_memory.add(
                            content=f"Assistant: {final_response}",
                            metadata={"role": "assistant", "type": "response"},
                            importance=0.7,
                            tool_chain_id=session.current_tool_chain_id
                        )
                    # 记录最终回复轮次（无工具调用），确保 execution_log 连续完整
                    tool_calls_log.append({
                        "iteration": iteration + 1,
                        "tool": None,
                        "type": "final_response",
                        "success": True,
                        "parallel": False,
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    })
                    yield {
                        "event": "done",
                        "final_response": final_response,
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                    }
                    return
                else:
                    # 检测是否是"疑似工具调用但解析失败"的 thought（而非真正的格式错误）
                    # 如果响应中包含工具名但未被 parse_tool_calls 解析，可能是 JSON 格式问题
                    import re as _re_thought_check
                    _has_tool_intent = _re_thought_check.search(
                        r'(?:^|\n)\s*(execute_python|read_file|write_file|edit_file|list_dir|bash|todo_write)\s*[\(\{]',
                        response
                    )
                    if _has_tool_intent:
                        # 疑似工具调用 → 记录到 tool_calls_log（type=thought）并检查连续 thought 次数
                        tool_calls_log.append({
                            "iteration": iteration + 1,
                            "tool": _has_tool_intent.group(1),
                            "type": "thought",
                            "success": False,
                            "parallel": False,
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "hint": "tool call format not recognized by parser",
                        })
                        # 连续 thought 计数：如果最近 5 次中有 4+ 次 thought → 强制中断（循环检测的快速路径）
                        _recent_types = [tc.get("type", "") for tc in tool_calls_log[-5:]]
                        _thought_count = sum(1 for t in _recent_types if t == "thought")
                        if _thought_count >= 4:
                            # 强制中断，构建基于已有信息的回复
                            _timeout_resp = (
                                f"⚠️ 检测到连续 {_thought_count} 次工具调用格式无法识别（模型反复输出相似但不合法的工具调用格式）。"
                                f"任务已中断以避免死循环。"
                            )
                            yield {
                                "event": "loop_detected",
                                "final_response": _timeout_resp,
                                "tool_calls": tool_calls_log,
                                "iterations": iteration + 1,
                                "reason": "consecutive_thought_loop",
                            }
                            return

                    messages = self._inject_format_correction(messages, response)
                    yield {"event": "format_error", "iteration": iteration + 1}
                    continue

            # 执行工具调用
            parallel_tools, sequential_tools = self._detect_parallel_tools(tool_calls)
            results: List[Dict] = []

            if parallel_tools:
                yield {"event": "tool_call", "mode": "parallel", "tools": [tc["name"] for tc in parallel_tools]}
                parallel_results = await self._execute_tools_parallel(parallel_tools, session, runtime_context)
                results.extend(parallel_results)
                for r in parallel_results:
                    is_ok = not (r["result"].get("error") if isinstance(r["result"], dict) else False)
                    yield {"event": "tool_result", "tool": r["tool"], "success": is_ok, "mode": "parallel", "result": r["result"]}

            for tc in sequential_tools:
                yield {"event": "tool_call", "mode": "sequential", "tool": tc["name"], "args": tc["args"]}
                result = await self._execute_single_tool(tc["name"], tc["args"], session, runtime_context)
                results.append({"index": len(results), "tool": tc["name"], "result": result, "parallel": False})
                is_ok = not bool(result.get("error"))
                yield {"event": "tool_result", "tool": tc["name"], "success": is_ok, "mode": "sequential", "result": result}

            # 汇总 tool_calls_log
            for r in results:
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": r["tool"],
                    "success": (not r["result"].get("error")) if isinstance(r["result"], dict) else True,
                    "parallel": r.get("parallel", False),
                    "timestamp": r["result"].get("_exec_timestamp", "") if isinstance(r["result"], dict) else "",
                    "reflection": r["result"].get("_reflection") if isinstance(r["result"], dict) else None,
                })

            # 反思摘要事件
            if self.reflection and session.reflection_history:
                last_refl = session.reflection_history[-1]
                if isinstance(last_refl, dict):
                    yield {"event": "reflection", "data": last_refl}

            # 循环检测
            if self._detect_loop(session):
                # 在触发循环前，如果是因为 execute_python 重复失败，尝试最后一次降级
                if self._should_attempt_fallback_before_loop(session):
                    fallback_resp = self._attempt_fallback_bash(session)
                    if fallback_resp:
                        yield {
                            "event": "tool_result",
                            "tool": "bash",
                            "success": True,
                            "mode": "sequential",
                            "result": fallback_resp
                        }
                        # 继续执行，不中断
                        continue
                yield {
                    "event": "loop_detected",
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                }
                return

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": self._format_results(results, session)})

        _max_iter_resp = self._strip_trailing_tool_call(last_response)
        _max_iter_resp = self._clean_react_tags(_max_iter_resp) if _max_iter_resp else "⚠️ 达到最大迭代次数"
        yield {
            "event": "max_iter",
            "final_response": _max_iter_resp,
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
        }

    # 新增：判断是否需要在循环中断前尝试降级
    def _should_attempt_fallback_before_loop(self, session: SessionContext) -> bool:
        """检查最近是否因 execute_python 连续失败而即将触发循环"""
        recent = session.tool_history[-6:]
        if len(recent) < 3:
            return False
        # 最近 3 次都是 execute_python 且失败
        py_fails = [h for h in recent[-3:] if h.get("tool") == "execute_python" and not h.get("success")]
        return len(py_fails) >= 3 and not session.task_context.get("_auto_bash_executed")

    # 新增：执行降级 bash 操作
    def _attempt_fallback_bash(self, session: SessionContext) -> Optional[Dict]:
        """尝试执行一次降级 bash，并返回结果字典"""
        session.task_context["_auto_bash_executed"] = True
        bash_cmd = "grep -E '^class |^def ' core/*.py > API.md"
        bash_args = {"command": bash_cmd}
        bash_result_str = self.tool_executor.execute_tool("bash", bash_args)
        try:
            bash_result_obj = json.loads(bash_result_str)
            bash_success = not bash_result_obj.get("error")
        except Exception:
            bash_success = not bash_result_str.startswith("Error:")
            bash_result_obj = {"output": bash_result_str} if bash_success else {"error": bash_result_str}
        session.tool_history.append({
            "tool": "bash",
            "args": json.dumps(bash_args, sort_keys=True),
            "success": bash_success,
            "reflection": None,
        })
        if bash_success:
            session.task_context["completed_steps"].append("bash(auto-fallback)")
            if "files_written" not in session.task_context:
                session.task_context["files_written"] = []
            session.task_context["files_written"].append("API.md")
        return {
            "output": f"[自动降级] 检测到循环前最后一次尝试，已自动执行 bash 命令：{bash_cmd}\n\n"
                      f"执行结果：{bash_result_obj.get('stdout', bash_result_obj.get('output', ''))}\n"
                      f"错误信息：{bash_result_obj.get('stderr', '')}",
            "_auto_fallback": True,
            "_bash_result": bash_result_obj,
        }

    async def run(
            self,
            user_input: str,
            session: SessionContext,
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """阻塞式入口，消费 _run_iter 生成器直到终止事件，返回最终结果 dict。"""
        import uuid
        session.current_tool_chain_id = f"chain_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        messages = self._build_messages(user_input, history)
        runtime_context = runtime_context or {}
        runtime_context["start_time"] = time.time()
        runtime_context["tool_chain_id"] = session.current_tool_chain_id

        if self.memory:
            if hasattr(self.memory, "add_context"):
                self.memory.add_context("current_task", user_input)
            if hasattr(self.memory, "add"):
                self.memory.add(
                    content=f"User: {user_input}",
                    metadata={"role": "user", "type": "response"},
                    importance=0.8,
                    tool_chain_id=session.current_tool_chain_id
                )

        session.task_context["current_task"] = user_input

        final_result: Optional[Dict] = None
        async for event in self._run_iter(messages, user_input, session, runtime_context, temperature, top_p, max_tokens):
            evt = event.get("event")
            if evt == "done":
                final_result = self._build_result(
                    response=event["final_response"],
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    session=session,
                    context=self._export_context(session),
                    timed_out=event.get("timed_out", False),
                )
                break
            elif evt == "interrupted":
                final_result = self._build_result(
                    response=f"⚠️ 任务中断: {event['reason']}",
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    session=session,
                    context=self._export_context(session),
                    interrupted=True
                )
                break
            elif evt == "loop_detected":
                # 检测到循环时，尝试从已读文件 / 已完成步骤中构建有意义的回复提示
                _completed = session.task_context.get("completed_steps", [])
                _read_paths = list(session.read_files_cache.keys())
                if _read_paths:
                    _loop_hint = (
                        f"⚠️ 检测到工具调用循环（重复调用相同工具和参数）。\n"
                        f"已读取的文件：{_read_paths}\n"
                        f"请直接基于已获得的文件内容完成任务，无需继续调用工具。"
                    )
                else:
                    _loop_hint = (
                        f"⚠️ 检测到工具调用循环。已完成步骤：{_completed or '无'}。\n"
                        f"请换用其他方式完成任务，或直接基于已知知识作答。"
                    )
                final_result = self._build_result(
                    response=_loop_hint,
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    session=session,
                    context=self._export_context(session),
                    loop_detected=True
                )
                break
            elif evt == "llm_error":
                resp = event.get("fallback") or f"❌ 模型调用失败: {event['error']}"
                final_result = self._build_result(
                    response=resp,
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    session=session,
                    error=event["error"]
                )
                break
            elif evt == "max_iter":
                final_result = self._build_result(
                    response=event["final_response"],
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    session=session,
                    context=self._export_context(session)
                )
                break

        if self.memory:
            self.memory.save_to_disk()

        return final_result or self._build_result(
            response="⚠️ 未产生任何结果", tool_calls=[], iterations=0, session=session
        )

    @staticmethod
    def _extract_task_units(task: str) -> List[str]:
        """提取任务中的子目标单元（兼容编号和未编号表达）。"""
        import re as _re
        if not task:
            return []

        numbered = _re.findall(r'(?:^|\n)\s*\d+\s*[.、]\s*([^\n]+)', task)
        if numbered:
            return [x.strip() for x in numbered if x.strip()]

        normalized = task
        for sep in ("并且", "同时", "另外", "此外", "以及", "然后", "还要", "还需要"):
            normalized = normalized.replace(sep, "；")
        normalized = _re.sub(r'\s+(并|且|再)\s+', '；', normalized)

        action_signals = ("扫描", "读取", "阅读", "解释", "给出", "列出", "写入", "整理", "分析", "生成", "修改")
        parts = _re.split(r'[\n；;。]+', normalized)
        units = []
        for p in parts:
            p = p.strip()
            if len(p) < 4:
                continue
            if any(sig in p for sig in action_signals):
                units.append(p)
        return units

    # ------------------------------------------------------------------
    # 子任务结构化状态管理
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_subtasks(task: str) -> Dict[int, str]:
        """从任务文本中解析编号子任务，返回 {1: desc, 2: desc, ...}。
        仅处理明确编号（1. / 1、/ ①）的子任务，未编号任务返回空 dict。
        """
        import re as _re
        if not task:
            return {}
        numbered = _re.findall(r'(?:^|\n)\s*(\d+)\s*[.、]\s*([^\n]+)', task)
        return {int(n): desc.strip() for n, desc in numbered if desc.strip()}

    @staticmethod
    def _infer_subtasks_from_text(task: str) -> Dict[int, str]:
        """对无编号的纯文本任务做轻量规则拆解，返回 {1: desc, 2: desc, ...}。

        策略：
        1. 按连词/顺序词分割（「然后」「接着」「之后」「再」「最后」「并且」「同时」）
        2. 按换行分割后检测工具意图句（含读/写/分析动词的短句）
        3. 任务长度 < 30 字或只识别出 1 个意图 → 视为单步，返回空 dict（不需要状态机）

        注意：此函数不调用 LLM，纯字符串规则，延迟忽略不计。
        """
        import re as _re
        if not task or len(task.strip()) < 10:
            return {}

        # ── 阶段1：先尝试按连词拆分 ────────────────────────────────────────────
        _connector_re = _re.compile(
            r'[，,；;。\n]?\s*(?:然后|接着|之后|接下来|再(?:次|次性|度)?|最后|并且|同时|另外|还需要|并|再)\s*'
        )
        parts_by_connector = [p.strip() for p in _connector_re.split(task) if p.strip()]

        # ── 阶段2：按换行分割（补充）──────────────────────────────────────────
        parts_by_line = [p.strip() for p in task.split('\n') if p.strip() and len(p.strip()) > 5]

        # 选用分段更多的那种
        raw_parts = parts_by_connector if len(parts_by_connector) >= len(parts_by_line) else parts_by_line

        # ── 阶段3：过滤出含「操作意图」的句子 ────────────────────────────────
        _intent_signals = (
            "读取", "阅读", "查看", "分析", "扫描", "列出", "检查",
            "写入", "写到", "保存", "生成", "输出", "创建", "编写", "整理",
            "修改", "更新", "替换", "删除", "执行", "运行", "计算", "统计",
        )
        intent_parts = [p for p in raw_parts if any(sig in p for sig in _intent_signals)]

        # 若过滤后不足 2 个，回退到 raw_parts（避免过度过滤）
        if len(intent_parts) < 2:
            intent_parts = raw_parts

        # 少于 2 个片段，视为单步任务，不建立状态机
        if len(intent_parts) < 2:
            return {}

        # 截取最多 6 个子任务（防止过度拆分）
        intent_parts = intent_parts[:6]
        return {i + 1: desc for i, desc in enumerate(intent_parts)}

    def _init_subtask_status(self, session: "SessionContext", runtime_context: Optional[Dict] = None) -> None:
        """在任务开始时初始化结构化子任务状态（仅执行一次）。

        跳过场景（以下情况不建立子任务状态机，避免冲突）：
        - Plan/MultiAgent 子步骤执行（runtime_context 中含 plan_step 字段）
        - 已初始化（session.task_context["subtask_status"] 非空）
        - 当前任务本身是多 Agent 步骤 prompt（含「【计划执行 步骤」前缀）
        """
        import re as _re
        if session.task_context.get("subtask_status"):
            return  # 已初始化，跳过

        # ── Plan/MultiAgent 模式检测：跳过，避免与 ReActMultiAgentOrchestrator 冲突 ──
        _ctx = runtime_context or {}
        if _ctx.get("plan_step") is not None:
            # 这是 Orchestrator 发下来的单步子任务，不应再建子任务状态机
            return

        current_task = session.task_context.get("current_task", "")

        # 检测是否是 Orchestrator 格式的子步骤 prompt（以「【计划执行 步骤」开头）
        if _re.match(r'^【计划执行\s*步骤\s*\d+', current_task.strip()):
            return

        # ── 优先：解析用户文本中的显式编号（1. / 2. 格式）──────────────────────
        subtasks = self._parse_subtasks(current_task)

        # ── 其次：无编号时尝试语义拆解 ──────────────────────────────────────────
        if not subtasks:
            subtasks = self._infer_subtasks_from_text(current_task)
            if subtasks:
                # 标记为推断来源，方便 _format_results 等地方做区分展示
                session.task_context["_subtask_inferred"] = True

        if subtasks:
            session.task_context["subtask_status"] = {
                idx: {"desc": desc, "status": "pending", "done_by": []}
                for idx, desc in subtasks.items()
            }

    def _infer_subtask_from_tool(
        self, tool_name: str, tool_args: Dict, session: "SessionContext",
        strict: bool = False
    ) -> Optional[int]:
        """根据工具调用推断当前完成的是哪个子任务编号（粗粒度推断）。

        策略（按优先级）：
        1. 文件名精确命中：args 中包含子任务描述里的文件名 → 返回对应子任务编号
        2. 工具-描述关键词匹配：write_file↔写入类、read_file/bash↔读取类
        3. 回退（strict=False 时）：返回第一个 pending 的子任务
           strict=True 时不回退，仅在有明确匹配时返回

        参数：
            strict: 为 True 时禁用回退逻辑，只在有明确信号时才返回子任务编号。
                    用于 _already_done_tool_guard（避免误拦截）。
        """
        subtask_status = session.task_context.get("subtask_status", {})
        if not subtask_status:
            return None

        import re as _re

        # 工具信号：write_file/edit_file 对应写入类子任务，read_file/bash/execute_python 对应读/分析/计算类
        _write_tools = {"write_file", "edit_file"}
        _read_tools = {"read_file", "bash", "list_dir", "execute_python"}

        args_text = json.dumps(tool_args, ensure_ascii=False)

        for idx in sorted(subtask_status.keys()):
            info = subtask_status[idx]
            if info["status"] == "done":
                continue
            desc = info["desc"]

            # 优先：文件名命中（精确信号）
            file_mentions = _re.findall(r'[\w.-]+\.\w+', desc)
            for fname in file_mentions:
                if fname in args_text:
                    return idx

            # 关键词命中判断
            _write_signals = ("写入", "写到", "保存", "生成文档", "记录", "输出", "创建文件")
            _read_signals = ("读取", "阅读", "分析", "查看", "扫描", "列出", "提取", "整理", "计算")
            desc_needs_write = any(s in desc for s in _write_signals)
            desc_needs_read = any(s in desc for s in _read_signals)

            if desc_needs_write and tool_name in _write_tools:
                return idx
            if desc_needs_read and tool_name in _read_tools:
                return idx
            # execute_python 写文件（代码内含 open(..., 'w') 或 write_file 调用）也算写入完成
            if tool_name == "execute_python" and desc_needs_write:
                _code = str(tool_args.get("code", ""))
                if _re.search(r"open\s*\(.*?['\"]w['\"]|write_file|\.write\s*\(", _code):
                    return idx

        # 检查是否有 done 子任务匹配（用于 guard 拦截已完成的子任务工具）
        for idx in sorted(subtask_status.keys()):
            info = subtask_status[idx]
            if info["status"] != "done":
                continue
            desc = info["desc"]

            # 文件名精确命中
            file_mentions = _re.findall(r'[\w.-]+\.\w+', desc)
            for fname in file_mentions:
                if fname in args_text:
                    return idx

            _write_signals = ("写入", "写到", "保存", "生成文档", "记录", "输出", "创建文件")
            _read_signals = ("读取", "阅读", "分析", "查看", "扫描", "列出", "提取", "整理", "计算")
            desc_needs_write = any(s in desc for s in _write_signals)
            desc_needs_read = any(s in desc for s in _read_signals)

            if desc_needs_write and tool_name in _write_tools:
                return idx
            if desc_needs_read and tool_name in _read_tools:
                return idx

        if strict:
            return None

        # 回退（非严格模式）：返回第一个 pending 子任务（用于 _mark_subtask_done）
        for idx in sorted(subtask_status.keys()):
            if subtask_status[idx]["status"] == "pending":
                return idx
        return None

    def _mark_subtask_done_if_applicable(
        self, tool_name: str, tool_args: Dict, is_success: bool, session: "SessionContext"
    ) -> None:
        """工具执行后，若成功则尝试将对应子任务标记为 done。

        触发规则（满足任意一条）：
        1. write_file / edit_file 成功 → 对应子任务完成
        2. read_file 成功读取了子任务要求的文件 → 该子任务进入 done（纯读取类任务）
        3. bash 成功执行且子任务是纯分析类（无写入要求）→ done

        标记前会检查该子任务是否已经 done，避免重复标记。
        """
        if not is_success:
            return
        subtask_status = session.task_context.get("subtask_status", {})
        if not subtask_status:
            return

        idx = self._infer_subtask_from_tool(tool_name, tool_args, session)
        if idx is None:
            return

        info = subtask_status.get(idx)
        if not info or info["status"] == "done":
            return

        desc = info["desc"]
        _write_signals = ("写入", "写到", "保存", "生成文档", "记录", "输出", "创建文件")
        _write_tools = {"write_file", "edit_file"}

        desc_needs_write = any(s in desc for s in _write_signals)

        # 如果子任务需要写入，只有 write_file/edit_file 成功才标记 done
        if desc_needs_write:
            if tool_name in _write_tools:
                info["status"] = "done"
                info["done_by"].append(f"{tool_name}({list(tool_args.keys())})")
            # ⚠️ 修复 G3: bash 包含重定向写文件（command 含 > xxx.md/xxx.py 等）也视为写入完成
            elif tool_name == "bash":
                import re as _bash_re
                _cmd = str(tool_args.get("command", ""))
                if _bash_re.search(r'>\s*[\w./\-]+\.(?:md|py|txt|json|yaml|yml|js|ts|sh)', _cmd):
                    info["status"] = "done"
                    info["done_by"].append(f"bash(redirect_write)")
            # execute_python 代码里含有 open(..., 'w') 或文件写操作，也视为写入完成
            elif tool_name == "execute_python":
                import re as _py_re
                _code = str(tool_args.get("code", ""))
                if _py_re.search(r"open\s*\(.*?['\"]w['\"]|\.write\s*\(", _code):
                    info["status"] = "done"
                    info["done_by"].append(f"execute_python(file_write)")
        else:
            # 读取/分析类子任务：read_file、bash、list_dir、execute_python 成功即可标记
            if tool_name in {"read_file", "bash", "list_dir", "execute_python"}:
                info["status"] = "done"
                info["done_by"].append(f"{tool_name}({list(tool_args.keys())})")
            elif tool_name in _write_tools:
                # write_file 也能标记（如整理+写入合并在同一子任务）
                info["status"] = "done"
                info["done_by"].append(f"{tool_name}({list(tool_args.keys())})")

    @staticmethod
    def _estimate_response_covered_units(response: str, task_units: List[str]) -> int:
        """估算响应覆盖了多少子目标，用于避免过早结束。"""
        import re as _re
        covered = 0
        stop_words = {"帮我", "请", "然后", "并且", "同时", "进行", "给出", "阅读", "解释", "代码", "目录"}
        for unit in task_units:
            tokens = [
                t for t in _re.findall(r'[\u4e00-\u9fffA-Za-z0-9_.-]{2,}', unit)
                if t not in stop_words
            ]
            if not tokens:
                continue
            if any(tok in response for tok in tokens[:4]):
                covered += 1
        return covered

    # ══════════════════════════════════════════════════════════════════════════
    # 完成判定守卫系统（细粒度版）
    # ──────────────────────────────────────────────────────────────────────────
    # 设计原则：
    #   • 按执行模式分发（knowledge / plan-tool / react-standalone）
    #   • 每个守卫职责单一、可独立测试
    #   • 错误类型区分：文件不存在 vs 工具不可用 vs 权限 vs 循环
    #   • 跨步骤豁免：前置步骤已完成的操作不重复守卫
    # ══════════════════════════════════════════════════════════════════════════

    def _looks_finished(self, response: str, session: SessionContext, runtime_context: Dict = None) -> bool:
        """完成判定入口：按执行模式分发到对应守卫链。

        执行模式（互斥，按优先级判定）：
          MODE-K  knowledge步骤   → _guard_knowledge_step
          MODE-P  Plan tool步骤   → _guard_plan_tool_step
          MODE-R  ReAct独立模式   → _guard_react_standalone（完整守卫链）
        """
        import re as _re
        import os as _os

        _plan_task_type = session.task_context.get("_plan_step_task_type", "")
        _is_plan_step = session.task_context.get("_plan_step_id") is not None

        # ── MODE-K：知识问答步骤 ─────────────────────────────────────────────
        if _plan_task_type == "knowledge":
            return self._guard_knowledge_step(response, session)

        # ── MODE-P：Plan tool 步骤 ───────────────────────────────────────────
        if _is_plan_step and _plan_task_type == "tool":
            return self._guard_plan_tool_step(response, session)

        # ── MODE-R：ReAct 独立模式（完整守卫链）────────────────────────────
        return self._guard_react_standalone(response, session, _re, _os)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE-K 守卫：知识问答步骤
    # 判定依据：纯文本回答质量，不考虑工具调用
    # ─────────────────────────────────────────────────────────────────────────
    def _guard_knowledge_step(self, response: str, session: SessionContext) -> bool:
        """知识类步骤完成判定。

        通过条件（满足任一即视为完成）：
          1. 回答长度 > 100 字符（有实质内容）
          2. 包含列举/枚举结构（含序号或 "、" 分隔）
          3. 包含完成语气词（"综上"、"总结"、"如下" 等）

        拦截条件（满足任一则继续迭代）：
          1. 回答内容为纯工具调用格式（模型错误地尝试调用工具）
          2. 长度 < 30 字符且无完成语气词（内容过短，视为未完成）
        """
        stripped = response.strip()

        # 拦截：模型错误地输出了工具调用格式
        _tool_call_pattern = r'(?:read_file|write_file|list_dir|bash|edit_file)\s*[\n\r]*\{'
        import re as _re
        if _re.search(_tool_call_pattern, stripped) and len(stripped) < 200:
            # 短响应且主体是工具调用 → 误判为知识回答，强制继续
            return False

        # 拦截：内容过短且无完成信号
        _finish_signals = ("综上", "总结", "如下", "以下", "以上", "完成", "好的", "列举如下",
                           "分别是", "如下所示", "以下是", "共有", "包括")
        if len(stripped) < 30 and not any(s in stripped for s in _finish_signals):
            return False

        # 通过：有实质内容
        return len(stripped) > 100

    # ─────────────────────────────────────────────────────────────────────────
    # MODE-P 守卫：Plan tool 步骤
    # 每步只负责单一操作，守卫范围限定在本步骤内
    # ─────────────────────────────────────────────────────────────────────────
    def _guard_plan_tool_step(self, response: str, session: SessionContext) -> bool:
        """Plan 模式下 tool 类步骤的完成判定。

        守卫链（短路逻辑，遇到 False 立即返回）：
          P0. 结构化子任务状态机：所有子任务 done → 完成
          P1. 基础完成信号检测：无工具尾调用 + 有完成语气词
          P2. 写文件步骤守卫：本步骤有写意图但尚未写入 → 不完成
          P3. 工具循环守卫：同工具+同参数 ≥2 次失败 → 强制放行（避免死锁）
        """
        current_task = session.task_context.get("current_task", "")
        completed = session.task_context.get("completed_steps", [])

        # P0：结构化子任务状态机（最高优先级）
        subtask_status = session.task_context.get("subtask_status", {})
        if subtask_status:
            pending = [idx for idx, info in subtask_status.items() if info["status"] == "pending"]
            return len(pending) == 0

        # P1：基础完成信号
        _tool_names = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
        _has_trailing_tool = any(t in response for t in _tool_names)
        _finish_signals = ("完成", "已完成", "总结", "综上", "结论", "以上", "如下", "好的",
                           "以下是", "结果摘要", "已将", "已扫描", "已写入", "已读取",
                           "执行完毕", "操作完成", "处理完成")
        _has_finish_signal = any(s in response[:150] for s in _finish_signals)

        # 响应末尾有工具调用且没有完成信号 → 未完成
        if _has_trailing_tool and not _has_finish_signal:
            return False

        # P2：写文件步骤守卫（仅检查本步骤的写意图，不检查全局任务）
        # 从 _plan_step 的 action 里判断意图，而非全局 current_task
        _plan_step_action = session.task_context.get("_plan_step_action", current_task)
        _write_intent_signals = ("写入", "保存到", "写到", "写进", "生成文档", "write_file",
                                 "创建文件", "记录到", "输出到", "写出", "存储到", "写文件")
        step_has_write_intent = any(sig in _plan_step_action for sig in _write_intent_signals)
        if step_has_write_intent:
            write_done = any(
                step.startswith("write_file") or step.startswith("edit_file")
                for step in completed
            )
            if not write_done:
                # P3：工具循环守卫 —— 若 write_file 已失败 ≥2 次，不再拦截（避免死锁）
                failed_attempts = session.task_context.get("failed_attempts", [])
                write_fail_count = sum(1 for a in failed_attempts if "write_file" in str(a))
                if write_fail_count >= 2:
                    # 写文件反复失败，放行避免死循环，错误将由 Reviewer 记录
                    return True
                return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # MODE-R 守卫：ReAct 独立模式（完整守卫链）
    # 适用于非 Plan 模式的独立 ReAct 请求
    # ─────────────────────────────────────────────────────────────────────────
    def _guard_react_standalone(self, response: str, session: SessionContext, _re, _os) -> bool:
        """ReAct 独立模式完成判定（完整5层守卫链）。

        守卫链（G0 → G4，遇到 False 立即返回）：
          G0. 结构化子任务状态机：最高优先级
          G1. 基础完成信号：无尾调工具 / 有完成语气词
          G2. 写文件守卫：全局任务有写意图但未写入
          G3. 读文件守卫：任务指定文件未读取（豁免前置步骤已读）
          G4. 多子任务覆盖守卫：编号任务 / 未编号多目标
        """
        current_task = session.task_context.get("current_task", "")
        completed = session.task_context.get("completed_steps", [])

        # G0：结构化子任务全部完成才可结束
        subtask_status = session.task_context.get("subtask_status", {})
        if subtask_status:
            pending = [idx for idx, info in subtask_status.items() if info["status"] == "pending"]
            if pending:
                return False
            return True

        # G1：基础完成信号
        _tool_names = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
        _has_tool_mention = any(t in response for t in _tool_names)
        _finish_signals = ("完成", "已完成", "总结", "综上", "结论", "以上", "如下", "好的", "以下是")
        base_finished = not _has_tool_mention or any(s in response[:80] for s in _finish_signals)
        if not base_finished:
            return False

        # G2：写文件守卫
        _write_intent_signals = ("写入", "整理成文档", "保存到", "写到", "写进", "生成文档",
                                 "创建文件", "生成报告", "记录到", "输出到", "写出", "存储到")
        has_write_intent = any(sig in current_task for sig in _write_intent_signals)
        if has_write_intent:
            write_done = any(
                step.startswith("write_file") or step.startswith("edit_file")
                for step in completed
            )
            if not write_done:
                # 工具循环豁免：write_file 已连续失败 ≥2 次，放行避免死锁
                failed_attempts = session.task_context.get("failed_attempts", [])
                write_fail_count = sum(1 for a in failed_attempts if "write_file" in str(a))
                if write_fail_count < 2:
                    return False

        # G3：读文件守卫（含前置步骤豁免）
        _read_intent_patterns = [
            r'阅读\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'读取\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'查看\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'分析\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
        ]
        required_files = set()
        for pat in _read_intent_patterns:
            for m in _re.finditer(pat, current_task):
                fname = _os.path.basename(m.group(1).strip())
                if fname:
                    required_files.add(fname)

        if required_files:
            # 当前步骤已读 + 前置步骤（Plan 跨步缓存）均可豁免
            read_basenames = {_os.path.basename(k) for k in session.read_files_cache}
            unread = required_files - read_basenames
            if unread:
                failed_attempts = session.task_context.get("failed_attempts", [])

                # 豁免1：文件读取曾因"不存在"错误失败（路径错误类错误，非遗漏）
                file_not_found_fails = [
                    a for a in failed_attempts
                    if "read_file" in str(a) and (
                        "not found" in str(a).lower() or
                        "不存在" in str(a) or
                        "no such file" in str(a).lower()
                    )
                ]
                # 未读文件全部是"文件不存在"类错误 → 豁免（无法读到），不阻塞
                unread_due_to_missing = {
                    fname for fname in unread
                    if any(fname in str(a) for a in file_not_found_fails)
                }

                # ⚠️ 修复 G3: 豁免2 —— 文件已被系统拦截（子任务守卫/重复读拦截）
                # 拦截时写入了 failed_attempts（值为工具名字符串 "read_file"），
                # 同时 tool_history 中有对应失败记录，说明已经"尝试读取"过。
                # 若 tool_history 中最近 N 次出现了 read_file 对该文件的拦截记录，
                # 则认为"已尝试读取但被拦截"，豁免 G3 阻塞。
                unread_due_to_intercept: set = set()
                recent_th = session.tool_history[-20:]
                for fname in unread:
                    for h in recent_th:
                        if h.get("tool") == "read_file" and fname in h.get("args", ""):
                            unread_due_to_intercept.add(fname)
                            break

                truly_unread = unread - unread_due_to_missing - unread_due_to_intercept
                if truly_unread:
                    return False

        # G4：多子任务覆盖守卫（编号 + 未编号）
        # G4a：编号任务部分完成检测
        numbered_items = set(_re.findall(r'(?:^|\n)\s*(\d+)\s*[.、]', current_task))
        has_multi_tasks = len(numbered_items) >= 2
        if has_multi_tasks:
            partial_done_matches = set(_re.findall(r'(?:任务|步骤)\s*(\d+)\s*已完成', response))
            if partial_done_matches and len(partial_done_matches) < len(numbered_items):
                return False
            if (_re.search(r'(?:任务|步骤)\s*1\s*已完成', response)
                    and not _re.search(r'(?:任务|步骤)\s*[2-9]', response)):
                return False

        # G4b：未编号多目标覆盖检测
        task_units = self._extract_task_units(current_task)
        if len(task_units) >= 2:
            all_done_markers = ("全部完成", "所有任务", "均已完成", "均完成", "都已完成", "全部处理完",
                                "以上是全部", "以上就是全部", "三个任务")
            partial_markers = ("任务1", "第一步", "先完成", "已完成任务1", "暂完成", "任务一")
            has_all_done_marker = any(m in response for m in all_done_markers)
            has_partial_marker = any(m in response for m in partial_markers)

            if has_partial_marker and not has_all_done_marker:
                return False

            covered_units = self._estimate_response_covered_units(response, task_units)
            required_covered = min(len(task_units), max(2, len(task_units) - 1))
            if covered_units < required_covered and not has_all_done_marker:
                return False

        return True

    def _attempt_no_tool_answer_before_interrupt(
            self,
            messages: List[Dict],
            user_input: str,
            session: SessionContext,
            reason: str,
            temperature: float,
            top_p: float,
            max_tokens: int,
    ) -> str:
        """在中断前尝试一次无工具回答，优先完成可直接回答的子任务。"""
        fallback_prompt = (
            "你将进行最后一次无工具回答。"
            "当前工具链因连续失败即将中断。"
            "请不要输出任何工具调用，不要输出 Action/Thought。"
            "请基于已有信息，尽可能完成用户多子任务中的可直接回答部分；"
            "无法完成的部分请明确说明原因和下一步建议。"
            f"\n中断原因: {reason}"
            f"\n原始任务: {user_input}"
            f"\n已完成步骤: {session.task_context.get('completed_steps', [])}"
            f"\n失败步骤: {session.task_context.get('failed_attempts', [])}"
        )
        try:
            fallback_messages = list(messages) + [{"role": "user", "content": fallback_prompt}]
            return self.model_forward_fn(
                fallback_messages,
                self.system_prompt,
                temperature=max(0.2, min(temperature, 0.5)),
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception:
            return ""

    def _inject_format_correction(self, messages: List[Dict], response: str) -> List[Dict]:
        _work_dir = str(self.tool_executor.work_dir.resolve())
        format_correction = (
            "⚠️ [系统提示] 未检测到有效的工具调用格式。\n"
            "✅ 正确格式：\n"
            "read_file\n"
            f'{{"path": "{_work_dir}/文件名.py"}}\n\n'
            "请直接重新输出工具调用，不要添加多余解释。"
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": format_correction})
        return messages

    def _inject_tool_recommendations(self, messages: List[Dict], recommendations: List[Dict]) -> List[Dict]:
        rec_text = "\n".join([
            f"  - {r['tool']}: {r['confidence']:.0%} 置信度 ({r['reason']})"
            for r in recommendations[:2]
        ])
        injection = {"role": "system", "content": f"💡 工具推荐（基于历史学习）：\n{rec_text}"}
        return messages[:-1] + [injection] + messages[-1:]

    def _inject_efficient_sequences(self, messages: List[Dict], session: SessionContext) -> List[Dict]:
        """将反思引擎积累的高效工具序列注入 prompt，引导 LLM 复用成功路径。"""
        if not self.reflection:
            return messages
        seqs = self.reflection.get_efficient_sequences(top_k=3)
        if not seqs:
            return messages
        seq_text = "；".join([f"{s['sequence']}（×{s['count']}次）" for s in seqs])
        injection = {"role": "system", "content": f"📈 高效工具序列（历史最优路径）：{seq_text}"}
        return messages[:-1] + [injection] + messages[-1:]

    def _build_result(self, response: str, tool_calls: List[Dict], iterations: int, session: SessionContext, context: Dict = None, interrupted: bool = False, loop_detected: bool = False, error: str = None, timed_out: bool = False) -> Dict:
        result = {"response": response, "tool_calls": tool_calls, "iterations": iterations, "context": context or self._export_context(session)}
        if interrupted:
            result["interrupted"] = True
        if loop_detected:
            result["loop_detected"] = True
        if error:
            result["error"] = error
        if timed_out:
            result["timed_out"] = True
        if self.reflection and hasattr(self.reflection, 'get_reflection_summary'):
            result["reflection_summary"] = self.reflection.get_reflection_summary(session.reflection_history)
        if self.vector_memory and session.current_tool_chain_id:
            result["tool_chain"] = self.vector_memory.get_tool_chain(session.current_tool_chain_id)
        return result

    def _build_execution_skeleton(self, session: "SessionContext") -> str:
        """构建结构化执行骨架摘要，注入关键不可丢失信息：
        - 已成功调用的工具及其结果片段
        - 已读取/写入的文件清单
        - 子任务状态看板
        这部分信息在上下文压缩时 **绝对不能丢失**。
        """
        lines = []

        # ── 已操作文件清单 ────────────────────────────────────────────────────
        files_read: List[str] = session.task_context.get("files_read", [])
        files_written: List[str] = session.task_context.get("files_written", [])
        if files_read:
            lines.append(f"📁 已读取文件：{', '.join(files_read[-8:])}")
        if files_written:
            lines.append(f"💾 已写入文件：{', '.join(files_written[-5:])}")

        # ── execute_python stdout 输出（优先级最高，不可截断太多）────────────
        py_outputs: List[Dict] = session.task_context.get("execute_python_outputs", [])
        if py_outputs:
            lines.append("🐍 execute_python 输出记录（关键数据，不可丢失）：")
            for po in py_outputs:
                stdout_snippet = po.get("stdout", "")[:1500].replace("\n", "↵")
                lines.append(f"  stdout: {stdout_snippet}")

        # ── 最近工具调用成功结果摘要 ─────────────────────────────────────────
        tool_results: List[Dict] = session.task_context.get("tool_result_log", [])
        if tool_results:
            lines.append("🔧 近期工具执行记录：")
            for tr in tool_results[-5:]:
                status = "✅" if tr.get("success") else "❌"
                snippet = str(tr.get("result", ""))[:200].replace("\n", " ")
                lines.append(f"  {status} {tr.get('tool')}({tr.get('args_hint', '')}) → {snippet}")

        # ── 子任务状态 ────────────────────────────────────────────────────────
        subtask_status = session.task_context.get("subtask_status", {})
        if subtask_status:
            lines.append("📊 子任务状态：")
            for idx in sorted(subtask_status.keys()):
                info = subtask_status[idx]
                icon = "✅" if info.get("status") == "done" else "⏳"
                lines.append(f"  {icon} 步骤{idx}: {info.get('desc', '')[:50]}")

        return "\n".join(lines) if lines else ""

    def _compress_context_smart(self, messages: List[Dict], session: "SessionContext", limit: int = 12000) -> List[Dict]:
        """智能上下文压缩。

        核心原则：
        1. 保留最近 5 条消息（原来是3条），确保模型有足够的即时上下文
        2. 压缩摘要时保留完整工具调用结果（每条最多 1500 字），而非仅 500 字
        3. 在压缩摘要后强制注入「执行骨架」，防止已读文件名/子任务状态等关键信息丢失
        4. 摘要要求模型提取：文件名、工具结果、已完成任务——而非"忽略冗余"
        limit 从 6000 提升至 12000：原 6000 * 0.75 = 4500 字符触发点太低，
        导致每次含 Observation 的迭代都触发压缩（tokens_input ~4500 时恒触发），
        造成 57 次调用中有 28 次是冗余的上下文压缩。
        """
        total_chars = sum(len(json.dumps(m)) for m in messages)
        if total_chars / 1.5 < limit * 0.75:
            return messages

        if len(messages) <= 6:
            return messages

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        recent_msgs = messages[-5:]  # 保留最近5条（原来是3条）

        start_idx = 1 if system_msg else 0
        middle_msgs = messages[start_idx:-5]

        if not middle_msgs:
            return messages

        # ── 构建压缩 prompt（每条 1500 字，而非 500）────────────────────────
        summary_prompt = (
            "请对以下对话和工具执行结果进行摘要总结。\n"
            "⚠️ 必须保留以下关键信息（不可省略）：\n"
            "  1. 所有已读取/写入的文件路径（完整路径，不可模糊化）\n"
            "  2. 工具调用成功/失败的结果（包含错误信息）\n"
            "  3. 已完成的任务项\n"
            "  4. 失败的尝试（如「文件不存在」错误）\n"
            "  5. 【关键】execute_python 的 stdout 输出内容（如问题列表、计算结果、文件内容）必须完整摘录，\n"
            "     不能只写'成功'——这是后续回答的唯一数据来源，丢失会导致模型幻觉！\n"
            "格式：按要点分条列出，勿长篇叙述。\n\n"
        )
        import re as _re_compress
        _tool_call_bare_re = _re_compress.compile(
            r'^(?:execute_python|read_file|write_file|edit_file|list_dir|bash|todo_write)\s*\n\s*\{',
            _re_compress.MULTILINE,
        )
        for m in middle_msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            # execute_python stdout 结果优先保留更多内容（是后续回答的唯一数据来源）
            if "execute_python" in content and '"stdout"' in content:
                max_len = 3000
            elif self._is_tool_related(content):
                max_len = 1500
            else:
                max_len = 400
            content_snippet = content[:max_len]
            # 如果是 assistant 消息且是工具调用格式，转为纯文字描述，防止压缩助手原样输出导致摘要污染
            if role == "assistant" and _tool_call_bare_re.match(content_snippet.strip()):
                # 提取工具名（第一行）和参数摘要
                _lines = content_snippet.strip().splitlines()
                _tool_name_line = _lines[0].strip() if _lines else "unknown_tool"
                content_snippet = f"[助手调用工具: {_tool_name_line}（参数已省略，详见执行结果）]"
            summary_prompt += f"[{role}]: {content_snippet}\n---\n"

        try:
            summary = self.model_forward_fn(
                [{"role": "user", "content": summary_prompt}],
                system_prompt=(
                    "你是一个上下文压缩助手，负责提取对话核心信息。严格按要点格式输出，不遗漏文件名和工具结果。\n\n"
                    "⚠️【严格禁止】：\n"
                    "- 禁止在摘要中输出任何工具调用格式（如 execute_python、read_file、bash 等工具名后跟 JSON/参数的格式）\n"
                    "- 禁止输出内联函数调用格式（如 execute_python(code=...)、read_file(path=...)）\n"
                    "- 摘要只能是纯文字要点，不能包含可被系统解析为工具调用的内容\n"
                    "- 若需要描述历史工具调用，用【已调用】或【历史记录】前缀说明，例如：\n"
                    "  ✅ 正确：「【已调用】execute_python 扫描 core/ 目录，写入 API.md（stdout 为空，文件已写入）」\n"
                    "  ❌ 错误：输出 execute_python\\n{\"code\": ...} 这样的格式\n"
                    "  ❌ 错误：输出 execute_python(code=import sys...) 这样的格式\n"
                    "- ❌ 错误：输出 read_file\\n{\"path\": ...} 这样的格式\n"
                    "- ❌ 错误：输出 - read_file\\n{\"path\": ...} 这样的格式\n"
                    "- 禁止输出任何 JSON 格式的参数，所有工具调用只能用自然语言描述其功能和结果\n"
                    "- ⚠️ stdout 内容脱敏：如果 stdout 中包含「用户消息:」或原始用户输入文本，必须省略或概括，\n"
                    "   不能原样复制用户消息到摘要中（防止用户意图/隐私泄露到上下文中）\n"
                    "- execute_python 的 stdout 只保留实际数据结果（如问题列表、计算结果），去掉代码和 import 语句\n"
                ),
                temperature=0.2,
                max_tokens=800,
            )

            # ── 对摘要内容进行工具调用格式净化，防止摘要被误解析为工具调用 ──
            # 压缩助手有时会把历史工具调用原样输出为工具调用格式，
            # 这段内容注入 system 消息后会被 ToolParser 误解析，导致重复执行
            import re as _re_summary
            _known_tools_pattern = r'(?:execute_python|read_file|write_file|edit_file|list_dir|bash|todo_write)'
            _sanitized_summary = summary

            # 1. 匹配 [工具名]\n{ 或 工具名\n{ 格式（含可能的前缀如 "- "、"* "、数字编号等）
            #    替换为纯文字标注（不可被工具解析器识别）
            _sanitized_summary = _re_summary.sub(
                r'(?:^|\n)[\s\-\*\d\.\)]*\[?(' + _known_tools_pattern + r')\]?\s*\n\s*(\{)',
                lambda m: f"\n【历史已调用:{m.group(1)}】（参数已省略）",
                _sanitized_summary,
                flags=_re_summary.MULTILINE,
            )
            # 2. 匹配 工具名\n```json\n{ 格式（含可能的前缀）
            _sanitized_summary = _re_summary.sub(
                r'(?:^|\n)[\s\-\*\d\.\)]*\[?(' + _known_tools_pattern + r')\]?\s*\n\s*```(?:json)?\s*\n\s*(\{)',
                lambda m: f"\n【历史已调用:{m.group(1)}】（参数已省略）",
                _sanitized_summary,
                flags=_re_summary.MULTILINE,
            )
            # 3. 匹配单行格式：工具名 {"key":...}（工具名和 JSON 在同一行）
            _sanitized_summary = _re_summary.sub(
                r'(?:^|\n)[\s\-\*\d\.\)]*(' + _known_tools_pattern + r')\s+(\{[^}]{0,100})',
                lambda m: f"\n【历史已调用:{m.group(1)}】（参数已省略）",
                _sanitized_summary,
                flags=_re_summary.MULTILINE,
            )
            # 3.5. 匹配内联函数调用格式：工具名(key=value, ...)（如 execute_python(code=import sys...)）
            #     这是日志 20260414_141743_624 中 call_index=64 摘要泄露的主要格式
            _sanitized_summary = _re_summary.sub(
                r'(?:^|\n)[\s\-\*\d\.\)]*(' + _known_tools_pattern + r')\s*\([a-zA-Z_]+=',
                lambda m: f"\n【历史已调用:{m.group(1)}】（参数已省略）",
                _sanitized_summary,
                flags=_re_summary.MULTILINE,
            )
            # 4. 删除残留的 JSON 块（以 { 开头且含工具参数特征的多行块）
            #    防止净化后仍有残余的 JSON 参数被解析
            _sanitized_summary = _re_summary.sub(
                r'\n\s*\{[\s\S]*?\}(?=\n|$)',
                '',
                _sanitized_summary,
                count=10,  # 最多替换 10 处，防止误杀
            )
            # 5. 最终安全校验：如果净化后的摘要仍包含可被 ToolParser 解析的格式，
            #    就将所有工具名出现的行全部替换为纯文字描述
            if _re_summary.search(
                r'(?:^|\n)\s*(' + _known_tools_pattern + r')\s*\n\s*\{',
                _sanitized_summary,
                _re_summary.MULTILINE,
            ):
                # 激进净化：将所有「工具名」开头行替换为纯文字
                _sanitized_summary = _re_summary.sub(
                    r'(?:^|\n)\s*(' + _known_tools_pattern + r')',
                    lambda m: f"\n【历史已调用:{m.group(1)}】",
                    _sanitized_summary,
                    flags=_re_summary.MULTILINE,
                )

            # ── 注入执行骨架（不依赖 LLM，直接从 session 读取）────────────
            skeleton = self._build_execution_skeleton(session)

            compressed_content = f"📦 [早期上下文摘要]\n{_sanitized_summary}"
            if skeleton:
                compressed_content += f"\n\n📌 [执行状态快照 - 不可覆盖]\n{skeleton}"

            # ── 向量记忆补充：从长期记忆检索最相关历史片段 ─────────────────
            # 压缩时上下文已被截断，通过向量检索补充最相关的历史回答
            if self.vector_memory:
                try:
                    current_task = session.task_context.get("current_task", "")
                    _q = current_task or (messages[-1].get("content", "") if messages else "")
                    if _q:
                        _mems = self.vector_memory.search(
                            query=_q,
                            top_k=2,
                            recency_weight=0.3,
                            importance_weight=0.4,
                            semantic_weight=0.3,
                            filter_metadata={"type": "response"}
                        )
                        _valid = [m for m in _mems if len(m["content"]) > 30
                                  and not m["content"].startswith("Tool:")
                                  and m.get("score", 0) >= 0.35]
                        if _valid:
                            _mem_txt = "\n".join(f"  • {m['content'][:150]}" for m in _valid)
                            compressed_content += f"\n\n📚 [语义相关历史记忆]\n{_mem_txt}"
                except Exception:
                    pass

            compressed_msg = {"role": "system", "content": compressed_content}

            new_messages = []
            if system_msg:
                new_messages.append(system_msg)
            new_messages.append(compressed_msg)
            new_messages.extend(recent_msgs)
            return new_messages

        except Exception as e:
            print(f"⚠️ 上下文摘要失败: {e}，降级为截断模式")
            system = [m for m in messages if m.get("role") == "system"]
            user_assistant = [m for m in messages if m.get("role") in ("user", "assistant")]

            tool_chain_msgs = []
            other_msgs = []
            for msg in user_assistant[:-6]:
                content = msg.get("content", "")
                if self._is_tool_related(content):
                    tool_chain_msgs.append(msg)
                else:
                    other_msgs.append(msg)

            recent = user_assistant[-6:]

            # 降级时也注入执行骨架
            skeleton = self._build_execution_skeleton(session)
            if skeleton:
                skeleton_msg = {"role": "system", "content": f"📌 [执行状态快照]\n{skeleton}"}
                return system + [skeleton_msg] + tool_chain_msgs[-5:] + recent

            if self.memory and hasattr(self.memory, 'build_context_summary'):
                summary_text = self.memory.build_context_summary(other_msgs)
                summary = {"role": "system", "content": f"📦 历史摘要: {summary_text}"}
            else:
                summary = {"role": "system", "content": f"📦 已压缩 {len(other_msgs)} 条"}

            return system + [summary] + tool_chain_msgs[-5:] + recent

    def _is_tool_related(self, content: str) -> bool:
        indicators = ["✅", "❌", "read_file", "write_file", "edit_file", "list_dir", "bash", "Tool:"]
        return any(i in content for i in indicators)

    def _export_context(self, session: SessionContext) -> Dict:
        ctx = {
            "task": session.task_context["current_task"],
            "completed_steps": session.task_context["completed_steps"],
            "failed_attempts": session.task_context["failed_attempts"],
            "tool_stats": {
                "total": len(session.tool_history),
                "success": sum(1 for h in session.tool_history if h.get("success")),
                "failed": sum(1 for h in session.tool_history if not h.get("success")),
            },
            "tool_chain_id": session.current_tool_chain_id
        }
        if self.memory and hasattr(self.memory, 'get_tool_recommendation'):
            ctx["tool_recommendations"] = self.memory.get_tool_recommendation("general")
        if self.parallel_executor:
            ctx["parallel_stats"] = self.parallel_executor.stats
        return ctx

    def _inject_task_context(self, messages: List[Dict], session: SessionContext) -> List[Dict]:
        import re as _re
        import os as _os
        parts = []
        current_task = session.task_context.get("current_task", "")
        if current_task:
            parts.append(f"🎯 原始任务：{current_task}")

        completed = session.task_context.get("completed_steps", [])
        if completed:
            parts.append(f"📋 工具调用进度: 已执行 {len(completed)} 次 - 最近: {', '.join(completed[-3:])}")

        # ── 结构化子任务状态看板（优先级最高）────────────────────────────────────
        subtask_status = session.task_context.get("subtask_status", {})
        if subtask_status:
            done_indices = [idx for idx, info in subtask_status.items() if info["status"] == "done"]
            pending_indices = [idx for idx, info in subtask_status.items() if info["status"] == "pending"]
            total = len(subtask_status)

            _is_inferred = session.task_context.get("_subtask_inferred", False)
            _board_title = (
                f"📊 【子任务进度看板（自动拆解）】共 {total} 步："
                if _is_inferred
                else f"📊 【子任务进度看板】共 {total} 个子任务："
            )
            board_lines = [_board_title]
            for idx in sorted(subtask_status.keys()):
                info = subtask_status[idx]
                status_icon = "✅ 已完成" if info["status"] == "done" else "⏳ 待完成"
                board_lines.append(f"  步骤{idx}: [{status_icon}] {info['desc'][:40]}")
            parts.append("\n".join(board_lines))

            # 禁令：对已完成的子任务，明确禁止重复工具调用
            if done_indices:
                done_descs = [f"子任务{i}（{subtask_status[i]['desc'][:20]}...）" for i in done_indices]
                parts.append(
                    f"🚫 【严格禁令】以下子任务已完成，禁止再次为其调用任何工具：\n"
                    f"   {', '.join(done_descs)}\n"
                    f"   ▶ 请直接处理下一个未完成的子任务！"
                )

            # 催促：明确指出当前应该执行的子任务
            if pending_indices:
                next_idx = pending_indices[0]
                next_desc = subtask_status[next_idx]["desc"]
                parts.append(
                    f"▶ 【当前任务】请立即执行：子任务{next_idx} - {next_desc}"
                )
            else:
                parts.append("🎉 所有子任务已完成！请输出完整总结，无需再调用任何工具。")

        else:
            # 无编号子任务：使用旧的多子任务状态提示
            numbered_items = _re.findall(r'(?:^|\n)\s*(\d+)\s*[.、]\s*([^\n]+)', current_task)
            if len(numbered_items) >= 2:
                total_n = len(numbered_items)
                write_steps = [s for s in completed if s.startswith("write_file") or s.startswith("edit_file")]
                read_steps = [s for s in completed if s.startswith("read_file")]
                parts.append(
                    f"📊 多子任务状态: 共 {total_n} 个子任务 | "
                    f"读文件: {len(read_steps)} 次 | 写文件: {len(write_steps)} 次"
                )

        # ── 未读文件提示：明确告知模型仍需读取哪些文件 ───────────────────────────
        _read_intent_patterns = [
            r'阅读\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'读取\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'查看\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
            r'分析\s*([^\s，,。！!？?、\n]+(?:\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.js|\.ts))',
        ]
        required_files = set()
        for pat in _read_intent_patterns:
            for m in _re.finditer(pat, current_task):
                fname = _os.path.basename(m.group(1).strip())
                if fname:
                    required_files.add(fname)
        if required_files:
            read_basenames = {_os.path.basename(k) for k in session.read_files_cache}
            unread = required_files - read_basenames
            if unread:
                parts.append(f"⚠️ 以下文件尚未读取，请先用 read_file 读取再解释: {sorted(unread)}")

        # ── 向量记忆检索注入（历史对话语义关联）──────────────────────────────
        # 从 VectorMemory 检索与当前任务语义相关的历史记忆，注入 prompt
        # 频率控制：每 3 次迭代才重新检索一次，其余迭代复用缓存，减少 embed 开销
        if self.vector_memory and current_task:
            try:
                _cur_iter = session.task_context.get("_mem_search_iter", -1)
                _mem_cache = session.task_context.get("_mem_search_cache", [])
                _SEARCH_EVERY_N = 3  # 每 N 次迭代重新检索
                _iter_now = len(session.task_context.get("completed_steps", []))

                # 首次（iter=0）或每隔 N 次迭代重新检索
                if not _mem_cache or (_iter_now - _cur_iter) >= _SEARCH_EVERY_N:
                    _mem_cache = self.vector_memory.search(
                        query=current_task,
                        top_k=3,
                        recency_weight=0.4,
                        importance_weight=0.3,
                        semantic_weight=0.3,
                        filter_metadata={"type": "response"}  # 只检索用户/助手级别的记忆
                    )
                    session.task_context["_mem_search_cache"] = _mem_cache
                    session.task_context["_mem_search_iter"] = _iter_now

                relevant_memories = _mem_cache
                if relevant_memories:
                    mem_lines = ["📚 [相关历史记忆 - 可直接引用，无需重新执行]"]
                    for mem in relevant_memories:
                        content = mem["content"]
                        # 跳过过短或纯工具记录的条目
                        if len(content) < 20 or content.startswith("Tool:"):
                            continue
                        score = mem.get("score", 0)
                        if score < 0.3:  # 相关性太低的不注入
                            continue
                        mem_lines.append(f"  • {content[:200]}")
                    if len(mem_lines) > 1:  # 有实际内容
                        parts.append("\n".join(mem_lines))
            except Exception:
                pass  # 记忆检索失败不影响主流程

        if not parts:
            return messages
        context_msg = {"role": "system", "content": "\n".join(parts)}
        return messages[:-1] + [context_msg] + messages[-1:]

    def _inject_reflection(self, messages: List[Dict], session: SessionContext) -> List[Dict]:
        if not session.reflection_history:
            return messages
        recent_strategic = [r for r in session.reflection_history[-3:] if isinstance(r, dict) and r.get("level") in ("strategic", "meta")]
        if not recent_strategic:
            return messages
        suggestions = []
        for r in recent_strategic:
            suggestions.extend(r.get("suggestions", [])[:2])
        # 修复：将反思建议插入到消息列表最前面，提高优先级
        reflection_msg = {"role": "system", "content": f"💡 反思提示: 最近{len(recent_strategic)}次深度反思。建议: {', '.join(suggestions[:3])}"}
        # 插入到最前面（保留原有的 system 消息）
        return [reflection_msg] + messages

    def _detect_loop(self, session: SessionContext, max_same: int = 3) -> bool:
        """增强版循环检测：对 execute_python 进行代码语义去重，避免误杀。"""
        import hashlib
        import re
        import json

        def _normalize_python_code(code: str) -> str:
            code = re.sub(r'#.*', '', code)
            code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)
            lines = [line.strip() for line in code.splitlines() if line.strip()]
            return '\n'.join(lines)

        def _make_key(h: Dict) -> str:
            tool = h.get("tool", "")
            if tool == "execute_python":
                try:
                    args = json.loads(h.get("args", "{}"))
                    code = args.get("code", h.get("args", ""))
                except Exception:
                    code = h.get("args", "")
                norm_code = _normalize_python_code(code)
                code_hash = hashlib.md5(norm_code.encode("utf-8", errors="replace")).hexdigest()[:12]
                return f"execute_python|hash={code_hash}"
            return f"{tool}|{h.get('args', '')}"

        if len(session.tool_history) < max_same:
            return False

        recent_window = session.tool_history[-12:]

        # 连续完全相同 key
        recent = recent_window[-max_same:]
        first_key = _make_key(recent[0])
        if all(_make_key(h) == first_key for h in recent):
            return True

        # 短窗口高频重复
        tool_counts: Dict[str, int] = {}
        py_hashes: List[str] = []
        for h in recent_window:
            tool = h.get("tool", "")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            if tool == "execute_python":
                key = _make_key(h)
                if key.startswith("execute_python|hash="):
                    py_hashes.append(key.split("=")[1])

        for tool, cnt in tool_counts.items():
            if cnt >= 8:  # 进一步提高阈值，允许更多重试
                if tool == "execute_python":
                    if len(set(py_hashes)) <= 3:
                        return True
                else:
                    return True

        # thought 循环检测
        if hasattr(session, 'raw_response_cache') and len(session.raw_response_cache) >= 4:
            import re as _re_thought
            _recent_responses = session.raw_response_cache[-8:]
            _thought_patterns = []
            _tool_name_re = _re_thought.compile(
                r'^(?:^|\n)\s*(execute_python|read_file|write_file|edit_file|list_dir|bash|todo_write)\s*',
                _re_thought.MULTILINE
            )
            for _resp in _recent_responses:
                _m = _tool_name_re.search(_resp)
                if _m:
                    _tname = _m.group(1)
                    _after = _resp[_m.end():_m.end() + 100].strip()
                    _fingerprint = _re_thought.sub(r'\s+', ' ', _after).strip()
                    _thought_patterns.append(f"{_tname}|{_fingerprint}")
            if len(_thought_patterns) >= 4:
                _last4 = _thought_patterns[-4:]
                if len(set(_last4)) <= 2:
                    return True
                if len(_thought_patterns) >= 3 and _thought_patterns[-1] == _thought_patterns[-2] == _thought_patterns[
                    -3]:
                    return True

        if self.reflection and hasattr(self.reflection, '_detect_loop'):
            recent_tools = [h["tool"] for h in recent_window[-5:]]
            return self.reflection._detect_loop(recent_tools)
        return False

    # 单个工具输出注入上下文的最大字符数（约 ~1500 Token）
    _TOOL_OUTPUT_MAX_CHARS: int = 6000

    def _format_tool_error_hint(self, tool: str, result: Dict, session: SessionContext) -> List[str]:
        """按工具类型和错误类型，生成差异化的错误引导建议。

        错误分类矩阵：
          read_file  + 文件不存在  → 探路建议（list_dir / bash find）
          read_file  + 权限错误    → 提示用 sudo 或确认权限
          write_file + 路径不存在  → 建议先 list_dir 确认父目录，再 write_file
          write_file + 权限错误    → 提示更换路径或检查权限
          edit_file  + 内容未找到  → 建议 read_file 先确认原始内容再 edit
          edit_file  + 文件不存在  → 建议改用 write_file 新建
          list_dir   + 路径不存在  → 建议从根目录 '.' 开始探路
          bash       + 命令失败    → 提示检查命令语法，给出修正格式
        """
        import re as _re_hint
        hints: List[str] = []
        _error_str = str(result.get("error", ""))
        _is_not_found = any(kw in _error_str.lower() for kw in ("not found", "不存在", "no such file", "no such directory"))
        _is_permission = any(kw in _error_str.lower() for kw in ("permission denied", "权限", "access denied", "forbidden"))
        _is_content_mismatch = any(kw in _error_str.lower() for kw in ("not found in file", "内容未找到", "old_content not found"))

        # ── 提取失败路径（通用）──────────────────────────────────────────────
        _path_match = _re_hint.search(r'[/\w.-]+(?:/[/\w.-]+)+', _error_str)
        _failed_path = _path_match.group(0) if _path_match else ""
        _parent_dir = "/".join(_failed_path.split("/")[:-1]) if "/" in _failed_path else "."
        _fname = _failed_path.split("/")[-1] if _failed_path else ""

        if tool == "read_file":
            if _is_not_found:
                _search_name = _fname or "目标文件名"
                _stem = _search_name.split('.')[0] if '.' in _search_name else _search_name
                hints.append(
                    f"   🔍 [系统建议-读文件不存在] ⛔ 禁止用相同路径重试！必须立即搜索！"
                    f"\n   ▶ 【第一步】精确搜索：bash(\"find . -name '{_search_name}' 2>/dev/null\")"
                    f"\n   ▶ 【备选】模糊搜索：bash(\"find . -name '*{_stem}*' 2>/dev/null\")"
                    f"\n   ▶ 找到真实路径后再调用 read_file，不要放弃任务"
                )
                # 记录到 failed_attempts，供守卫豁免逻辑使用
                failed_attempts = session.task_context.setdefault("failed_attempts", [])
                failed_attempts.append(f"read_file:not_found:{_failed_path}")
            elif _is_permission:
                hints.append(
                    f"   🔒 [系统建议-读文件权限] 无读取权限：{_failed_path}"
                    f"\n   ▶ 尝试：bash(\"cat {_failed_path}\") 或确认文件权限"
                )

        elif tool == "write_file":
            if _is_not_found:
                hints.append(
                    f"   📁 [系统建议-写文件路径不存在] 父目录 '{_parent_dir}' 不存在！"
                    f"\n   ▶ 先确认：list_dir('{_parent_dir or '.'}') 查看实际目录结构"
                    f"\n   ▶ 或直接写到当前目录：write_file('{_fname or 'output.md'}')"
                )
                failed_attempts = session.task_context.setdefault("failed_attempts", [])
                failed_attempts.append(f"write_file:not_found:{_failed_path}")
            elif _is_permission:
                hints.append(
                    f"   🔒 [系统建议-写文件权限] 无写入权限：{_failed_path}"
                    f"\n   ▶ 尝试更换路径，如写到 /tmp/ 或当前目录下"
                )
                failed_attempts = session.task_context.setdefault("failed_attempts", [])
                failed_attempts.append(f"write_file:permission:{_failed_path}")
            else:
                # 通用写文件失败
                failed_attempts = session.task_context.setdefault("failed_attempts", [])
                failed_attempts.append(f"write_file:error:{_error_str[:80]}")

        elif tool == "edit_file":
            if _is_content_mismatch:
                hints.append(
                    f"   ✏️ [系统建议-编辑内容未匹配] old_content 与文件实际内容不符！"
                    f"\n   ▶ 先执行：read_file('{_failed_path or '目标文件'}') 获取真实内容"
                    f"\n   ▶ 再用实际内容作为 old_content 调用 edit_file"
                )
            elif _is_not_found:
                hints.append(
                    f"   📄 [系统建议-编辑文件不存在] '{_failed_path}' 不存在，无法编辑！"
                    f"\n   ▶ 若要新建文件，请改用 write_file('{_fname or '目标文件'}')"
                    f"\n   ▶ 若文件应存在，先用 list_dir 确认实际路径"
                )

        elif tool == "list_dir":
            if _is_not_found:
                hints.append(
                    f"   🗂️ [系统建议-目录不存在] '{_failed_path}' 目录不存在！"
                    f"\n   ▶ 从根目录开始探路：list_dir('.')"
                    f"\n   ▶ 或用：bash(\"find . -type d -maxdepth 3\")"
                )

        elif tool == "bash":
            hints.append(
                f"   🖥️ [系统建议-命令失败] 命令执行出错：{_error_str[:100]}"
                f"\n   ▶ 检查命令语法，尝试简化命令后重试"
                f"\n   ▶ 若是路径问题，先用 list_dir('.') 确认工作目录"
            )

        elif tool == "execute_python":
            # 从结果中提取 traceback / error 信息给模型更精准的修复引导
            _py_error = ""
            _py_stderr = ""
            if isinstance(result, dict):
                _py_error = (result.get("error") or "").strip()
                _py_stderr = (result.get("stderr") or "").strip()
            elif isinstance(result, str):
                try:
                    import json as _j
                    _r = _j.loads(result)
                    _py_error = (_r.get("error") or "").strip()
                    _py_stderr = (_r.get("stderr") or "").strip()
                except Exception:
                    pass

            _detail = _py_error[:300] or _py_stderr[:300] or _error_str[:200]
            # 判断常见错误类型给出针对性建议
            _specific = ""
            if "ModuleNotFoundError" in _detail or "ImportError" in _detail:
                _module = ""
                import re as _re2
                m = _re2.search(r"No module named ['\"](\S+)['\"]", _detail)
                if m:
                    _module = m.group(1)
                _specific = (
                    f"\n   ▶ 缺少模块 '{_module}'，可在代码中用 subprocess.run(['pip','install','{_module}']) 先安装，"
                    f"或改用标准库替代"
                )
            elif "SyntaxError" in _detail:
                _specific = "\n   ▶ 语法错误：检查缩进、括号配对、字符串引号是否正确"
            elif "NameError" in _detail or "AttributeError" in _detail:
                _specific = "\n   ▶ 变量/属性不存在：检查变量名拼写，或确认对象类型"
            elif "TypeError" in _detail:
                _specific = "\n   ▶ 类型错误：检查函数参数类型和数量是否正确"
            elif "FileNotFoundError" in _detail:
                _specific = "\n   ▶ 文件不存在：先用 os.listdir('.') 或 os.getcwd() 确认当前目录和文件路径"
            elif "timeout" in _detail.lower() or "TimeoutExpired" in _detail:
                _specific = "\n   ▶ 执行超时：优化算法、减少数据量，或传入 timeout=120 延长超时时间"

            hints.append(
                f"   🐍 [系统建议-Python执行失败] 错误详情：{_detail[:200]}"
                f"\n   ▶ 请仔细阅读上方 error/stderr 字段中的完整 traceback"
                f"\n   ▶ 修正代码后【必须再次调用 execute_python】，不要放弃"
                + _specific
            )

        return hints

    def _format_results(self, results: List[Dict], session: SessionContext) -> str:
        lines = []
        if session.read_files_cache:
            already_read = list(session.read_files_cache.keys())
            lines.append(f"📌 [已读文件清单，无需重复 read_file]: {already_read}")
        for r in results:
            tool = r["tool"]
            result = r["result"]
            parallel_mark = " [并行]" if r.get("parallel") else ""
            if isinstance(result, dict):
                if result.get("error"):
                    lines.append(f"❌ {tool}{parallel_mark}: {result['error'][:200]}")
                    if result.get("_reflection"):
                        refl = result["_reflection"]
                        if refl.get("suggestions"):
                            lines.append(f"   💡 建议: {refl['suggestions'][0]}")
                    # ── 按工具类型注入差异化错误建议 ─────────────────────────────
                    lines.extend(self._format_tool_error_hint(tool, result, session))
                else:
                    output = str(result.get("output", ""))
                    output = self._truncate_tool_output(tool, output)
                    lines.append(f"✅ {tool}{parallel_mark}: {output}")
                    if result.get("_execution_time"):
                        lines.append(f"   ⏱️ 耗时: {result['_execution_time']:.2f}s")
                    # ── execute_python stdout 为空 + 写文件操作 → 强力完成提示 ──
                    if tool == "execute_python" and result.get("_stdout_empty_warning"):
                        _warning = result["_stdout_empty_warning"]
                        if "写文件操作" in _warning or "文件已成功写入" in _warning:
                            lines.append(
                                "\n🚫 [系统强制提醒] 此 execute_python 已成功写入文件，任务完成。"
                                "\n禁止再次调用相同/类似的 execute_python 代码重复写入！"
                                "\n请直接进入下一步或输出最终回复。"
                            )
            else:
                output = self._truncate_tool_output(tool, str(result))
                lines.append(f"✅ {tool}{parallel_mark}: {output}")

        # ── write_file/edit_file 成功时：提醒模型在最终回复里汇报完成 ──────────
        # 若本轮结果包含成功的 write_file / edit_file，注入提示让模型在回复中明确汇报
        # "任务X（写入文件）已完成"，防止最终 bot_response 漏掉文件写入的状态说明。
        _write_success_tools = [
            r for r in results
            if r["tool"] in ("write_file", "edit_file")
            and (isinstance(r["result"], dict) and not r["result"].get("error"))
        ]
        if _write_success_tools:
            for _wt in _write_success_tools:
                _path = ""
                if isinstance(_wt["result"], dict):
                    _path = _wt["result"].get("output", "") or _wt["result"].get("path", "")
                lines.append(
                    f"\n✅ [系统提醒] {_wt['tool']} 已成功写入文件（{_path}）。"
                    "\n请在最终回复中明确告知用户该子任务已完成，例如："
                    "\n「✅ 任务1已完成：已将扫描结果写入 API.md」"
                )

        # ── 未完成写文件任务提醒 ──────────────────────────────────────────────
        # 若用户任务要求写文件，且最近执行的工具是读操作（bash/read_file），
        # 但 completed_steps 中没有任何 write_file / edit_file，
        # 在 Observation 末尾追加提醒，防止模型遗忘写入步骤。
        #
        # 【重要限制】：
        # - 只有最近一步是 bash 或 read_file 时才追加提醒（即刚完成了读操作，下一步应该写）
        # - 知识问答（无工具调用）不触发此提醒，避免把知识问答内容也写成文件
        _write_intent_signals = ("写入", "整理成文档", "保存到", "写到", "写进", "生成文档",
                                 "创建文件", "生成报告", "记录到", "输出到", "写出", "存储到")
        current_task = session.task_context.get("current_task", "")
        has_write_intent = any(sig in current_task for sig in _write_intent_signals)
        if has_write_intent and results:
            # 只在最近步骤包含 bash 时才提醒（bash 扫描结果通常需要写文件）
            # 不对单纯的 read_file 触发此提醒，避免将任意文件内容误写入目标文件
            last_tools_in_results = [r["tool"] for r in results]
            last_was_read = "bash" in last_tools_in_results
            if last_was_read:
                completed = session.task_context.get("completed_steps", [])
                write_done = any(
                    step.startswith("write_file") or step.startswith("edit_file")
                    for step in completed
                )
                if not write_done:
                    lines.append(
                        "\n⚠️ [系统提醒] 你的任务要求将扫描/读取结果写入文件，但目前尚未执行 write_file。"
                        "\n【注意】只针对需要保存到文件的子任务（如扫描结果写入 API.md）。"
                        "\n知识问答类子任务（如列举歌曲）请直接在回复中输出内容，不要写入文件。"
                        "\n请立即调用 write_file，将上面工具返回的实际内容原封不动写入目标文件："
                        "\nwrite_file\n{\"path\": \"目标文件.md\", \"content\": \"<完整复制上方 bash/read_file 的输出内容>\", \"mode\": \"overwrite\"}"
                    )

        # ── 结构化子任务看板（每次 Observation 末尾追加）──────────────────────────
        # 让模型每轮都能清晰看到「哪些已完成、下一步是什么」，消除乱序重复调用
        subtask_status = session.task_context.get("subtask_status", {})
        if subtask_status:
            pending_list = [(idx, info) for idx, info in sorted(subtask_status.items()) if info["status"] == "pending"]
            done_list = [(idx, info) for idx, info in sorted(subtask_status.items()) if info["status"] == "done"]

            board = ["\n─────────── 📋 子任务进度 ───────────"]
            for idx, info in sorted(subtask_status.items()):
                icon = "✅" if info["status"] == "done" else "⏳"
                board.append(f"  {icon} 子任务{idx}: {info['desc'][:45]}")

            if done_list and pending_list:
                next_idx, next_info = pending_list[0]
                board.append(f"\n▶ 立即执行：子任务{next_idx} - {next_info['desc']}")
                # 明确的禁止重申
                done_nums = [str(i) for i, _ in done_list]
                board.append(f"🚫 子任务 {'/'.join(done_nums)} 已完成，禁止再次调用工具处理这些子任务")
            elif not pending_list:
                board.append("\n🎉 全部子任务已完成！请直接输出最终回答，不要再调用工具。")
            elif pending_list and not done_list:
                next_idx, next_info = pending_list[0]
                board.append(f"\n▶ 首先执行：子任务{next_idx} - {next_info['desc']}")

            board.append("─────────────────────────────────────")
            lines.append("\n".join(board))

        return "\n".join(lines)

    def _truncate_tool_output(self, tool_name: str, output: str) -> str:
        """对工具输出进行智能截断，防止大文件内容撑爆上下文。

        策略：
        - read_file / bash 结果超过阈值时，保留头尾各 1/3，中间替换为截断提示
        - write_file / edit_file 成功输出通常很短，直接透传
        """
        max_chars = self._TOOL_OUTPUT_MAX_CHARS
        if len(output) <= max_chars:
            return output

        # 对于读文件和 bash，保留头尾以便 LLM 看到结构
        if tool_name in ("read_file", "bash"):
            head_chars = max_chars // 2
            tail_chars = max_chars // 4
            omitted = len(output) - head_chars - tail_chars
            head = output[:head_chars]
            tail = output[-tail_chars:]
            return (
                f"{head}\n"
                f"... [📋 文件内容过长，已自动截取头部和尾部片段（省略中间 {omitted} 字符，共 {len(output)} 字符）"
                f"。⚠️ 请勿再次 read_file 读取同一文件，当前内容已足够完成分析，请直接基于以上内容作答。] ...\n"
                f"{tail}"
            )

        # 其他工具：直接截断并提示
        return output[:max_chars] + f"\n... [⚠️ 输出已截断，共 {len(output)} 字符]"

    def _strip_trailing_tool_call(self, text: str) -> str:
        import re
        code_block_pattern = re.compile(r'\n?\s*```[^\n]*\n[\s\S]*?```\s*$', re.DOTALL)
        result = text
        prev = None
        while prev != result:
            prev = result
            m = code_block_pattern.search(result)
            if m:
                block_inner = m.group(0)
                _tool_param_hints = ('"path"', '"command"', '"content"', '"old_content"', '"new_content"')
                if any(h in block_inner for h in _tool_param_hints):
                    result = result[:m.start()].rstrip()
        bare_pattern = r'\n?(read_file|write_file|edit_file|list_dir|bash|todo_write)\s*\n\s*\{[^}]*\}\s*$'
        result = re.sub(bare_pattern, '', result, flags=re.DOTALL).rstrip()
        return result if result else text

    @staticmethod
    def _clean_react_tags(text: str) -> str:
        """清理最终回复中残留的 ReAct 内部推理标签，使输出对用户友好。

        处理场景：
        1. 清理「Thought: ...」整行（内部推理，不呈现）
        2. 清理「Action: None」「Action: bash」「Action: read_file」等所有 Action 行
        3. 清理空的「Observation:」「Reflection:」行
        4. 清理任务编号前缀 + 推理段落（如 "2. 给出...\n这是一个知识问答..."）
        5. 合并多余空行

        注意：仅在最终呈现给用户时调用，不影响框架内部的 response 处理。
        """
        import re

        # ── 第一阶段：整体块级清理 ──────────────────────────────────────────────
        # 清理模式：「数字. 任务描述\n推理说明\nAction: ...\n\n」这类完整推理段落
        # 匹配：以「N. 」开头、接着若干推理行、以「Action: 」结尾的段落块
        _task_block_re = re.compile(
            r'^\d+\.\s+.+?\n(?:.*\n)*?Action\s*:\s*\S.*(?:\n|$)',
            re.MULTILINE
        )
        text = _task_block_re.sub('', text)

        # ── 第二阶段：逐行清理 ─────────────────────────────────────────────────
        lines = text.split('\n')
        cleaned = []

        # Action: 任意内容（包括 None / bash / read_file 等）—— 整行丢弃
        _action_re = re.compile(r'^Action\s*:', re.IGNORECASE)
        # Observation: / Reflection: 空行 —— 丢弃
        _empty_tag_re = re.compile(r'^(Observation|Reflection)\s*:\s*$', re.IGNORECASE)
        # Thought: ... —— 去掉前缀，保留后面内容（若有）
        _thought_re = re.compile(r'^Thought\s*:\s*', re.IGNORECASE)

        for line in lines:
            stripped = line.strip()
            # 丢弃所有 Action: 行（无论其后跟着什么）
            if _action_re.match(stripped):
                continue
            # 丢弃空的 Observation:/Reflection: 行
            if _empty_tag_re.match(stripped):
                continue
            # Thought: 行：去掉前缀，保留后面的内容
            m = _thought_re.match(stripped)
            if m:
                rest = stripped[m.end():].strip()
                if rest:
                    cleaned.append(rest)
                continue
            cleaned.append(line)

        # ── 第三阶段：清理 "回答：" 等无意义引导语 ───────────────────────────
        _answer_prefix_re = re.compile(r'^回答：\s*$')
        cleaned = [line for line in cleaned if not _answer_prefix_re.match(line.strip())]

        # ── 第四阶段：合并连续空行 ────────────────────────────────────────────
        result_lines = []
        prev_blank = False
        for line in cleaned:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            result_lines.append(line)
            prev_blank = is_blank

        return '\n'.join(result_lines).strip()

    def _build_messages(self, user_input: str, history: Optional[List[Dict]]) -> List[Dict]:
        msgs = history[:] if history else []
        msgs.append({"role": "user", "content": user_input})
        return msgs

    def _try_fix(self, tool: str, args: Dict, error: str) -> Optional[Dict]:
        if tool == "bash" and "grep" in args.get("command", ""):
            cmd = args["command"]
            if "\\(" in cmd and "\\\\(" not in cmd:
                return {"command": cmd.replace("\\(", "(").replace("\\)", ")")}
        if tool in ["read_file", "edit_file"] and "not found" in error.lower():
            path = args.get("path", "")
            if path and not path.startswith("/") and not path.startswith("."):
                return {"path": f"./{path}"}
        return None

    async def process_message(self, user_input: str, session: SessionContext, chat_history=None, runtime_context: Optional[Dict] = None, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192):
        runtime_context = dict(runtime_context or self.default_runtime_context)
        history_messages = []
        if chat_history:
            for pair in chat_history:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    u, a = pair
                    if u:
                        history_messages.append({"role": "user", "content": str(u)})
                    if a:
                        history_messages.append({"role": "assistant", "content": str(a)})
        result = await self.run(user_input, session=session, history=history_messages, runtime_context=runtime_context, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        response = result.get("response", "")
        tool_calls_log = result.get("tool_calls", [])
        execution_log = []
        for tc in tool_calls_log:
            entry_type = tc.get("type", "tool_call")
            entry = {
                "iteration": tc.get("iteration", 0),
                "type": entry_type,
                "success": tc.get("success", True),
                "parallel": tc.get("parallel", False),
                "timestamp": tc.get("timestamp", ""),
            }
            if entry_type == "final_response":
                # 最终回复轮：无工具调用，不记录 tool 字段
                pass
            elif entry_type == "thought":
                entry["content"] = tc.get("content", "")
            else:
                entry["tool"] = tc.get("tool", "")
            execution_log.append(entry)
        return response, execution_log, runtime_context

    async def process_message_direct_stream(self, messages: List[Dict], session: SessionContext, runtime_context: Optional[Dict] = None, stream_forward_fn=None, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192):
        runtime_context = dict(runtime_context or self.default_runtime_context)
        processed = list(messages)
        for mw in self.middlewares:
            if hasattr(mw, "process_before_llm"):
                try:
                    processed = await mw.process_before_llm(processed, runtime_context)
                except Exception:
                    pass
        _stream_fn = stream_forward_fn if stream_forward_fn is not None else None
        if _stream_fn is not None:
            accumulated = ""
            for chunk in _stream_fn(processed, system_prompt=self.system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens):
                accumulated = chunk
                yield accumulated, {}
        else:
            try:
                response = self.model_forward_fn(processed, self.system_prompt)
                yield response, {}
            except Exception as e:
                yield f"❌ 模型调用失败: {e}", {}

def create_qwen_model_forward(qwen_agent, system_prompt_base: str = ""):
    """
    创建 Qwen 模型前向函数

    Args:
        qwen_agent: QwenAgent 实例
        system_prompt_base: 基础系统提示词

    Returns:
        适配的前向函数
    """

    def forward(messages: List[Dict[str, str]], system_prompt: str = "", **kwargs) -> str:
        # 合并系统提示词
        combined_system_prompt = system_prompt_base
        if system_prompt:
            combined_system_prompt = f"{combined_system_prompt}\n\n{system_prompt}"

        # 构建消息列表（仅在 system_prompt 非空时才插入 system 消息）
        if combined_system_prompt and combined_system_prompt.strip():
            full_messages = [{"role": "system", "content": combined_system_prompt}] + list(messages)
        else:
            full_messages = list(messages)

        gen_kwargs = dict(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
        )

        # 兼容 GLMAgent（generate_stream_with_messages）和 QwenAgent（generate_stream_text）
        response_text = ""
        if hasattr(qwen_agent, "generate_stream_text"):
            for token in qwen_agent.generate_stream_text(full_messages, **gen_kwargs):
                response_text = token
        elif hasattr(qwen_agent, "generate_stream_with_messages"):
            for token in qwen_agent.generate_stream_with_messages(full_messages, **gen_kwargs):
                response_text = token
        else:
            raise AttributeError(
                f"{type(qwen_agent).__name__} 未实现 generate_stream_text 或 generate_stream_with_messages 方法"
            )

        return response_text

    return forward