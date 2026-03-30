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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from .agent_tools import ToolExecutor, ToolParser
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
        self.reflection_history: List[Dict] = []
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
    def reflect_on_result(self, tool_name: str, result: Dict, context: Dict = None) -> Dict:
        context = context or {}
        error = result.get("error")
        if not error:
            # 成功路径：记录高效模式
            self._record_success(tool_name, context)
            return {
                "success": True,
                "level": "surface",
                "analysis": "执行成功",
                "suggestions": [],
                "action": "continue"
            }
        error_analysis = self._analyze_error(error, tool_name, context)
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
        self.reflection_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": False,
            "error": error[:100],
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
    def _record_success(self, tool_name: str, context: Dict) -> None:
        """记录成功执行，更新高效工具序列并同步到 ToolLearner。"""
        recent_tools: List[str] = context.get("recent_tools", [])
        exec_time: float = context.get("_execution_time", 0)
        task_type: str = context.get("task", "general")

        self.reflection_history.append({
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
    def _analyze_error(self, error: str, tool_name: str, context: Dict) -> Dict:
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

        # 连续失败且类别相同（死循环同一错误）
        categories = [
            h.get("category") or
            (h.get("reflection") or {}).get("category", "unknown")
            for h in recent_failed
        ]
        if len(set(categories)) == 1 and categories[0] != "unknown":
            return False, f"连续 {len(recent_failed)} 次相同错误({categories[0]})，建议更换策略"

        return True, ""

    def get_reflection_summary(self) -> str:
        if not self.reflection_history:
            return "暂无反思记录"
        recent = self.reflection_history[-5:]
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


class OutputValidator:
    @staticmethod
    def validate_tool_call(tool_name: str, args: Dict) -> Tuple[bool, str]:
        required_params = {
            "read_file": ["path"],
            "write_file": ["path", "content"],
            "edit_file": ["path", "old_content", "new_content"],
            "list_dir": [],
            "bash": ["command"]
        }
        if tool_name not in required_params:
            return False, f"未知工具: {tool_name}"
        missing = [p for p in required_params[tool_name] if p not in args]
        if missing:
            return False, f"缺少参数: {', '.join(missing)}"
        # 允许非字符串值（数字、布尔等），自动 str 转换，不拒绝调用
        for key, value in list(args.items()):
            if not isinstance(value, (str, type(None))):
                args[key] = str(value)
        return True, ""

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
            max_iterations: int = 50,
            middlewares: Optional[List] = None,
            enable_memory: bool = True,
            enable_reflection: bool = True,
            enable_parallel: bool = True,
            enable_tool_learning: bool = True,
            tools_in_system_prompt: bool = True,
            default_runtime_context: Optional[Dict] = None,
            parallel_config: Optional[ParallelConfig] = None,
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
        self.tool_history = []
        self.reflection_history = []
        self.read_files_cache: Dict[str, str] = {}
        self.task_context = {"current_task": None, "completed_steps": [], "failed_attempts": []}
        self.system_prompt = self._build_system_prompt()
        self._current_tool_chain_id: Optional[str] = None

    def _build_system_prompt(self) -> str:
        work_dir_abs = str(self.tool_executor.work_dir.resolve())
        tools = ["bash", "read_file", "write_file", "edit_file", "list_dir"] if self.tool_executor.enable_bash else [
            "read_file", "write_file", "edit_file", "list_dir"]
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

文件操作策略（务必遵守）：
- 【禁止】一次性读完所有文件再统一写出——这会超出迭代限制
- 【正确】读一个文件 → 立即写/修改它 → 再读下一个（流水线方式）
- 【写前读规则】：若需要将扫描/读取结果写入文件，必须先执行 bash/read_file 获取结果，
  在 Observation 中看到实际内容后，再调用 write_file 写入真实数据，禁止在未获得读取结果时就写文件
- 【禁止伪造内容】：write_file 的 content 字段必须是 Observation 中出现的实际文本，
  严禁使用：bash['stdout']、变量引用、"...省略..."、"# (更多内容)"等占位符代替真实内容
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

    def _execute_tools_parallel(self, tool_calls: List[Dict], runtime_context: Dict = None) -> List[Dict]:
        """并行执行只读工具，每个线程仍经过完整中间件 + 反思 + 记忆管道。"""
        if not tool_calls:
            return []
        n_workers = min(len(tool_calls), 4)
        results: List[Optional[Dict]] = [None] * len(tool_calls)

        def _run(idx: int, tc: Dict):
            result = self._execute_single_tool(tc["name"], tc["args"], runtime_context or {})
            return idx, {"index": idx, "tool": tc["name"], "result": result, "parallel": True}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run, i, tc): i for i, tc in enumerate(tool_calls)}
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                except Exception as e:
                    idx = futures[future]
                    results[idx] = {
                        "index": idx,
                        "tool": tool_calls[idx]["name"],
                        "result": {"error": str(e)},
                        "parallel": True
                    }
        return results  # type: ignore[return-value]

    def _execute_single_tool(self, tool_name: str, tool_args: Dict, context: Dict = None) -> Dict:
        """执行单个工具，经过：参数验证 → 前置中间件 → 执行 → 后置中间件 → 反思 → 记忆。"""
        runtime_context = context or {}

        # ── 已读文件拦截：对 read_file 工具，若路径已在 read_files_cache 中，
        # 直接返回提示，不重复读取。防止模型因看到截断提示而陷入反复读同一文件的循环。
        # 同时对相对路径 / 绝对路径做 basename 归一化匹配，防止路径写法不同绕过缓存。
        if tool_name == "read_file":
            import os as _os
            path = tool_args.get("path", "")
            _basename = _os.path.basename(path) if path else ""
            _cached_basenames = {_os.path.basename(k): k for k in self.read_files_cache}
            _exact_hit = path and path in self.read_files_cache
            _basename_hit = _basename and _basename in _cached_basenames
            if _exact_hit or _basename_hit:
                _matched_path = path if _exact_hit else _cached_basenames[_basename]
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
            tool_name, tool_args = mw.process_before_tool(tool_name, tool_args, runtime_context)

        valid, error_msg = self.validator.validate_tool_call(tool_name, tool_args)
        if not valid:
            result: Dict = {"error": f"参数验证失败: {error_msg}"}
            for mw in self.middlewares:
                result_str = json.dumps(result)
                result_str = mw.process_after_tool(tool_name, tool_args, result_str, runtime_context)
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
            result_str = mw.process_after_tool(tool_name, tool_args, result_str, runtime_context)

        try:
            result_obj = json.loads(result_str)
            is_success = not result_obj.get("error")
        except Exception:
            is_success = not result_str.startswith("Error:")
            result_obj = {"output": result_str} if is_success else {"error": result_str}

        if not is_success:
            fixed_args = self._try_fix(tool_name, tool_args, result_str)
            if fixed_args:
                result_str = self.tool_executor.execute_tool(tool_name, fixed_args)
                exec_time = time.time() - start_time
                try:
                    result_obj = json.loads(result_str)
                    is_success = not result_obj.get("error")
                except Exception:
                    is_success = not result_str.startswith("Error:")

        # 反思引擎（成功/失败均触发，内部分别调用 _record_success 或记录失败）
        reflection_result = None
        if self.reflection:
            reflection_context = {
                "recent_tools": [h.get("tool") for h in self.tool_history[-5:]],
                "task": self.task_context.get("current_task", "general"),
                "_execution_time": exec_time,
            }
            reflection_result = self.reflection.reflect_on_result(
                tool_name, result_obj, context=reflection_context
            )
            self.reflection_history.append(reflection_result)

        # 注意：ToolLearner 的 record_usage 已通过 DeepReflectionEngine 的双向同步调用，
        # 此处不再重复调用，避免重复计数。

        if self.vector_memory and self._current_tool_chain_id:
            self.vector_memory.add(
                content=f"Tool: {tool_name}, Args: {tool_args}, Success: {is_success}",
                metadata={"type": "tool_execution", "tool": tool_name, "success": is_success},
                importance=0.8 if not is_success else 0.5,
                tool_chain_id=self._current_tool_chain_id
            )

        if self.memory and hasattr(self.memory, "update_tool_stats"):
            self.memory.update_tool_stats(tool_name, is_success, exec_time)

        self.tool_history.append({
            "tool": tool_name,
            "args": json.dumps(tool_args, sort_keys=True),
            "success": is_success,
            "reflection": reflection_result
        })

        if is_success:
            self.task_context["completed_steps"].append(f"{tool_name}({list(tool_args.keys())})")
            if tool_name == "read_file" and tool_args.get("path"):
                self.read_files_cache[tool_args["path"]] = exec_timestamp
        else:
            self.task_context["failed_attempts"].append(tool_name)

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
    def _run_iter(
            self,
            messages: List[Dict],
            user_input: str,
            runtime_context: Dict,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Generator[Dict, None, None]:
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

        for iteration in range(self.max_iterations):
            runtime_context["iteration"] = iteration
            yield {"event": "progress", "iteration": iteration + 1, "max_iterations": self.max_iterations}

            # 反思引擎：是否继续
            if self.reflection:
                should_continue, reason = self.reflection.should_continue(self.reflection_history, max_failed=3)
                if not should_continue:
                    yield {
                        "event": "interrupted",
                        "reason": reason,
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                    }
                    return

            # 上下文压缩 + 注入
            messages = self._compress_context_smart(messages)
            messages = self._inject_task_context(messages)
            messages = self._inject_reflection(messages)

            # 工具推荐（首轮及后续，阈值降至 0.3）
            if self.tool_learner:
                recommendations = self.tool_learner.recommend_next_tools(
                    user_input,
                    [h["tool"] for h in self.tool_history],
                    current_context={"iteration": iteration, "task": user_input}
                )
                # 首轮使用历史推荐（如有），后续降低阈值至 0.3
                min_confidence = 0.5 if iteration == 0 else 0.3
                if recommendations and recommendations[0]["confidence"] >= min_confidence:
                    messages = self._inject_tool_recommendations(messages, recommendations)

            # 高效序列提示（反思引擎积累的成功模式）
            if self.reflection and iteration > 0:
                messages = self._inject_efficient_sequences(messages)

            # 中间件：before LLM
            for mw in self.middlewares:
                if hasattr(mw, "process_before_llm"):
                    messages = mw.process_before_llm(messages, runtime_context)

            # LLM 调用
            try:
                response = self.model_forward_fn(
                    messages, self.system_prompt,
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens
                )
                last_response = response
                yield {"event": "thought", "iteration": iteration + 1, "content": response}
            except Exception as e:
                fallback = None
                for mw in self.middlewares:
                    fb = mw.process_on_error(e, "llm_call", runtime_context)
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
                response = mw.process_after_llm(response, runtime_context)

            if runtime_context.get("_needs_retry"):
                runtime_context["_needs_retry"] = False
                continue

            # 解析工具调用
            tool_calls_raw = self.tool_parser.parse_tool_calls(response)
            tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

            if not tool_calls:
                if self._looks_finished(response, runtime_context):
                    final_response = self._clean_react_tags(self._strip_trailing_tool_call(response))
                    if self.vector_memory:
                        self.vector_memory.add(
                            content=f"Assistant: {final_response}",
                            metadata={"role": "assistant", "type": "response"},
                            importance=0.7,
                            tool_chain_id=self._current_tool_chain_id
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
                    messages = self._inject_format_correction(messages, response)
                    yield {"event": "format_error", "iteration": iteration + 1}
                    continue

            # 执行工具调用
            parallel_tools, sequential_tools = self._detect_parallel_tools(tool_calls)
            results: List[Dict] = []

            if parallel_tools:
                yield {"event": "tool_call", "mode": "parallel", "tools": [tc["name"] for tc in parallel_tools]}
                parallel_results = self._execute_tools_parallel(parallel_tools, runtime_context)
                results.extend(parallel_results)
                for r in parallel_results:
                    is_ok = not (r["result"].get("error") if isinstance(r["result"], dict) else False)
                    yield {"event": "tool_result", "tool": r["tool"], "success": is_ok, "mode": "parallel", "result": r["result"]}

            for tc in sequential_tools:
                yield {"event": "tool_call", "mode": "sequential", "tool": tc["name"], "args": tc["args"]}
                result = self._execute_single_tool(tc["name"], tc["args"], runtime_context)
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
            if self.reflection and self.reflection_history:
                last_refl = self.reflection_history[-1]
                if isinstance(last_refl, dict):
                    yield {"event": "reflection", "data": last_refl}

            # 循环检测
            if self._detect_loop():
                yield {
                    "event": "loop_detected",
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                }
                return

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": self._format_results(results)})

        _max_iter_resp = self._strip_trailing_tool_call(last_response)
        _max_iter_resp = self._clean_react_tags(_max_iter_resp) if _max_iter_resp else "⚠️ 达到最大迭代次数"
        yield {
            "event": "max_iter",
            "final_response": _max_iter_resp,
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
        }

    def run(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            runtime_context: Optional[Dict] = None,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """阻塞式入口，消费 _run_iter 生成器直到终止事件，返回最终结果 dict。"""
        self._current_tool_chain_id = f"chain_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        messages = self._build_messages(user_input, history)
        runtime_context = runtime_context or {}
        runtime_context["start_time"] = time.time()
        runtime_context["tool_chain_id"] = self._current_tool_chain_id

        if self.memory:
            if hasattr(self.memory, "add_context"):
                self.memory.add_context("current_task", user_input)
            if hasattr(self.memory, "add"):
                self.memory.add(
                    content=f"User: {user_input}",
                    metadata={"role": "user", "type": "input"},
                    importance=0.8,
                    tool_chain_id=self._current_tool_chain_id
                )

        self.task_context["current_task"] = user_input

        final_result: Optional[Dict] = None
        for event in self._run_iter(messages, user_input, runtime_context, temperature, top_p, max_tokens):
            evt = event.get("event")
            if evt == "done":
                final_result = self._build_result(
                    response=event["final_response"],
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    context=self._export_context()
                )
                break
            elif evt == "interrupted":
                final_result = self._build_result(
                    response=f"⚠️ 任务中断: {event['reason']}",
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    context=self._export_context(),
                    interrupted=True
                )
                break
            elif evt == "loop_detected":
                # 检测到循环时，尝试从已读文件 / 已完成步骤中构建有意义的回复提示
                _completed = self.task_context.get("completed_steps", [])
                _read_paths = list(self.read_files_cache.keys())
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
                    context=self._export_context(),
                    loop_detected=True
                )
                break
            elif evt == "llm_error":
                resp = event.get("fallback") or f"❌ 模型调用失败: {event['error']}"
                final_result = self._build_result(
                    response=resp,
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    error=event["error"]
                )
                break
            elif evt == "max_iter":
                final_result = self._build_result(
                    response=event["final_response"],
                    tool_calls=event["tool_calls"],
                    iterations=event["iterations"],
                    context=self._export_context()
                )
                break

        if self.memory:
            self.memory.save_to_disk()

        return final_result or self._build_result(
            response="⚠️ 未产生任何结果", tool_calls=[], iterations=0
        )

    def _looks_finished(self, response: str, runtime_context: Dict = None) -> bool:
        """判断当前响应是否是任务最终完成的回答。

        除原有「有无工具提及 + 完成词」逻辑外，增加「写文件意图守卫」：
        - 若用户原始任务中要求写入文件（含「写入」「整理成文档」「保存」等），
          但 completed_steps 中从未出现 write_file / edit_file，
          则不判定为完成，强制模型继续执行写入操作。
        """
        _tool_names = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
        _has_tool_mention = any(t in response for t in _tool_names)
        _finish_signals = ("完成", "已完成", "总结", "综上", "结论", "以上", "如下", "好的", "以下是")
        base_finished = not _has_tool_mention or any(s in response[:80] for s in _finish_signals)

        if not base_finished:
            return False

        # 写文件意图守卫：检查任务是否要求写文件但从未执行
        _write_intent_signals = ("写入", "整理成文档", "保存到", "写到", "写进", "生成文档",
                                 "创建文件", "生成报告", "记录到", "输出到", "写出", "存储到")
        current_task = self.task_context.get("current_task", "")
        has_write_intent = any(sig in current_task for sig in _write_intent_signals)

        if has_write_intent:
            completed = self.task_context.get("completed_steps", [])
            write_done = any(
                step.startswith("write_file") or step.startswith("edit_file")
                for step in completed
            )
            if not write_done:
                # 任务要求写文件但从未执行，不判定为完成
                return False

        return True

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

    def _inject_efficient_sequences(self, messages: List[Dict]) -> List[Dict]:
        """将反思引擎积累的高效工具序列注入 prompt，引导 LLM 复用成功路径。"""
        if not self.reflection:
            return messages
        seqs = self.reflection.get_efficient_sequences(top_k=3)
        if not seqs:
            return messages
        seq_text = "；".join([f"{s['sequence']}（×{s['count']}次）" for s in seqs])
        injection = {"role": "system", "content": f"📈 高效工具序列（历史最优路径）：{seq_text}"}
        return messages[:-1] + [injection] + messages[-1:]

    def _build_result(self, response: str, tool_calls: List[Dict], iterations: int, context: Dict = None, interrupted: bool = False, loop_detected: bool = False, error: str = None) -> Dict:
        result = {"response": response, "tool_calls": tool_calls, "iterations": iterations, "context": context or self._export_context()}
        if interrupted:
            result["interrupted"] = True
        if loop_detected:
            result["loop_detected"] = True
        if error:
            result["error"] = error
        if self.reflection and hasattr(self.reflection, 'get_reflection_summary'):
            result["reflection_summary"] = self.reflection.get_reflection_summary()
        if self.vector_memory and self._current_tool_chain_id:
            result["tool_chain"] = self.vector_memory.get_tool_chain(self._current_tool_chain_id)
        return result

    def _compress_context_smart(self, messages: List[Dict], limit: int = 6000) -> List[Dict]:
        total_chars = sum(len(json.dumps(m)) for m in messages)
        if total_chars / 1.5 < limit * 0.75:
            return messages
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

        if self.memory and hasattr(self.memory, 'build_context_summary'):
            summary_text = self.memory.build_context_summary(other_msgs)
            summary = {"role": "system", "content": f"📦 历史摘要: {summary_text}"}
        else:
            summary = {"role": "system", "content": f"📦 已压缩 {len(other_msgs)} 条"}

        return system + [summary] + tool_chain_msgs[-5:] + recent

    def _is_tool_related(self, content: str) -> bool:
        indicators = ["✅", "❌", "read_file", "write_file", "edit_file", "list_dir", "bash", "Tool:"]
        return any(i in content for i in indicators)

    def _export_context(self) -> Dict:
        ctx = {
            "task": self.task_context["current_task"],
            "completed_steps": self.task_context["completed_steps"],
            "failed_attempts": self.task_context["failed_attempts"],
            "tool_stats": {
                "total": len(self.tool_history),
                "success": sum(1 for h in self.tool_history if h.get("success")),
                "failed": sum(1 for h in self.tool_history if not h.get("success")),
            },
            "tool_chain_id": self._current_tool_chain_id
        }
        if self.memory and hasattr(self.memory, 'get_tool_recommendation'):
            ctx["tool_recommendations"] = self.memory.get_tool_recommendation("general")
        if self.parallel_executor:
            ctx["parallel_stats"] = self.parallel_executor.stats
        return ctx

    def _inject_task_context(self, messages: List[Dict]) -> List[Dict]:
        parts = []
        if self.task_context.get("current_task"):
            parts.append(f"🎯 原始任务：{self.task_context['current_task']}")
        if self.task_context["completed_steps"]:
            parts.append(f"📋 进度: 已完成 {len(self.task_context['completed_steps'])} 步 - {', '.join(self.task_context['completed_steps'][-3:])}")
        if not parts:
            return messages
        context_msg = {"role": "system", "content": "\n".join(parts)}
        return messages[:-1] + [context_msg] + messages[-1:]

    def _inject_reflection(self, messages: List[Dict]) -> List[Dict]:
        if not self.reflection_history:
            return messages
        recent_strategic = [r for r in self.reflection_history[-3:] if isinstance(r, dict) and r.get("level") in ("strategic", "meta")]
        if not recent_strategic:
            return messages
        suggestions = []
        for r in recent_strategic:
            suggestions.extend(r.get("suggestions", [])[:2])
        reflection_msg = {"role": "system", "content": f"💡 反思提示: 最近{len(recent_strategic)}次深度反思。建议: {', '.join(suggestions[:3])}"}
        return messages[:-1] + [reflection_msg] + messages[-1:]

    def _detect_loop(self, max_same: int = 3) -> bool:
        if len(self.tool_history) < max_same:
            return False
        recent = self.tool_history[-max_same:]
        first = recent[0]
        if all(h["tool"] == first["tool"] and h["args"] == first["args"] for h in recent):
            return True
        _call_counts: Dict[str, int] = {}
        for h in self.tool_history:
            key = f"{h['tool']}|{h['args']}"
            _call_counts[key] = _call_counts.get(key, 0) + 1
        if any(cnt >= 5 for cnt in _call_counts.values()):
            return True
        if self.reflection and hasattr(self.reflection, '_detect_loop'):
            recent_tools = [h["tool"] for h in self.tool_history[-5:]]
            return self.reflection._detect_loop(recent_tools)
        return False

    # 单个工具输出注入上下文的最大字符数（约 ~1500 Token）
    _TOOL_OUTPUT_MAX_CHARS: int = 6000

    def _format_results(self, results: List[Dict]) -> str:
        lines = []
        if self.read_files_cache:
            already_read = list(self.read_files_cache.keys())
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
                else:
                    output = str(result.get("output", ""))
                    output = self._truncate_tool_output(tool, output)
                    lines.append(f"✅ {tool}{parallel_mark}: {output}")
                    if result.get("_execution_time"):
                        lines.append(f"   ⏱️ 耗时: {result['_execution_time']:.2f}s")
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
        current_task = self.task_context.get("current_task", "")
        has_write_intent = any(sig in current_task for sig in _write_intent_signals)
        if has_write_intent and results:
            # 只在最近步骤包含 bash 时才提醒（bash 扫描结果通常需要写文件）
            # 不对单纯的 read_file 触发此提醒，避免将任意文件内容误写入目标文件
            last_tools_in_results = [r["tool"] for r in results]
            last_was_read = "bash" in last_tools_in_results
            if last_was_read:
                completed = self.task_context.get("completed_steps", [])
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

    def process_message(self, user_input: str, chat_history=None, system_prompt_override: str = "", runtime_context: Optional[Dict] = None, return_runtime_context: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192):
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
        result = self.run(user_input, history=history_messages, runtime_context=runtime_context, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
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
            else:
                entry["tool"] = tc.get("tool", "")
            execution_log.append(entry)
        return response, execution_log, runtime_context

    def process_message_direct_stream(self, messages: List[Dict], system_prompt_override: str = "", runtime_context: Optional[Dict] = None, stream_forward_fn=None, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192):
        runtime_context = dict(runtime_context or self.default_runtime_context)
        processed = list(messages)
        for mw in self.middlewares:
            if hasattr(mw, "process_before_llm"):
                try:
                    processed = mw.process_before_llm(processed, runtime_context)
                except Exception:
                    pass
        _stream_fn = stream_forward_fn if stream_forward_fn is not None else None
        if _stream_fn is not None:
            accumulated = ""
            for chunk in _stream_fn(processed, system_prompt=system_prompt_override, temperature=temperature, top_p=top_p, max_tokens=max_tokens):
                accumulated = chunk
                yield accumulated, {}
        else:
            try:
                response = self.model_forward_fn(processed, system_prompt_override or self.system_prompt)
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