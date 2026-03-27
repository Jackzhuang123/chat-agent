# -*- coding: utf-8 -*-
"""先进 Agent 框架 - ReAct + 反思 + 并行执行 + 持久化记忆 + 语义压缩"""

import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent_middlewares import AgentMiddleware
from .agent_tools import ToolExecutor, ToolParser
from .tool_learner import ToolLearner


class SessionMemory:
    """会话记忆 + 工具使用统计 + 持久化"""

    def __init__(self, memory_dir: str = ".agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.current_session = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 工具使用统计
        self.tool_stats = defaultdict(lambda: {"success": 0, "failed": 0, "avg_time": 0})

        # 跨会话记忆文件
        self.memory_file = self.memory_dir / "session_memory.pkl"
        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载历史记忆"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.tool_stats = data.get("tool_stats", self.tool_stats)
                    # 加载最近3次会话的关键信息
                    self.current_session = data.get("recent_context", {})
            except Exception:
                pass

    def save_to_disk(self):
        """保存记忆到磁盘"""
        try:
            data = {
                "tool_stats": dict(self.tool_stats),
                "recent_context": self.current_session,
                "last_update": time.time()
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def add_context(self, key: str, value: Any):
        """添加上下文"""
        self.current_session[key] = {"value": value, "timestamp": time.time()}

    def get_context(self, key: str) -> Optional[Any]:
        """获取上下文"""
        return self.current_session[key]["value"] if key in self.current_session else None

    def update_tool_stats(self, tool_name: str, success: bool, exec_time: float):
        """更新工具统计"""
        # defaultdict 自动创建，但需要先访问才能触发
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {"success": 0, "failed": 0, "avg_time": 0}

        stats = self.tool_stats[tool_name]
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

        # 更新平均时间
        total = stats["success"] + stats["failed"]
        stats["avg_time"] = (stats["avg_time"] * (total - 1) + exec_time) / total

    def get_tool_recommendation(self, task_type: str) -> List[str]:
        """基于历史推荐工具"""
        # 按成功率排序
        ranked = sorted(
            self.tool_stats.items(),
            key=lambda x: x[1]["success"] / max(1, x[1]["success"] + x[1]["failed"]),
            reverse=True
        )
        return [tool for tool, _ in ranked[:3]]

    def extract_key_info(self, messages: List[Dict]) -> Dict:
        """提取关键信息"""
        info = {"user_requests": [], "files_touched": [], "errors": []}

        for msg in messages[-10:]:
            content = msg.get("content", "")
            if msg.get("role") == "user" and len(content) < 100:
                info["user_requests"].append(content)

            if "path" in content:
                import re
                files = re.findall(r'"path":\s*"([^"]+)"', content)
                info["files_touched"].extend(files)

            if "❌" in content or "Error" in content:
                info["errors"].append(content[:100])

        info["files_touched"] = list(set(info["files_touched"]))[-5:]
        info["user_requests"] = info["user_requests"][-3:]
        info["errors"] = info["errors"][-2:]

        return info

    def build_context_summary(self, messages: List[Dict]) -> str:
        """构建上下文摘要"""
        info = self.extract_key_info(messages)
        parts = []
        if info["user_requests"]:
            parts.append(f"请求: {', '.join(info['user_requests'])}")
        if info["files_touched"]:
            parts.append(f"文件: {', '.join(info['files_touched'])}")
        if info["errors"]:
            parts.append(f"错误: {len(info['errors'])}个")
        return " | ".join(parts) if parts else "无关键信息"

    def compute_message_importance(self, messages: List[Dict]) -> List[float]:
        """计算消息重要性（简化版语义评分）"""
        scores = []
        keywords = ["error", "failed", "success", "file", "path", "tool", "result"]

        for msg in messages:
            content = msg.get("content", "").lower()
            # 基于关键词密度计算重要性
            score = sum(1 for kw in keywords if kw in content)
            # 用户消息权重更高
            if msg.get("role") == "user":
                score *= 1.5
            # 工具结果权重更高
            if "✅" in content or "❌" in content:
                score *= 1.3
            scores.append(score)

        return scores


class ReflectionEngine:
    """反思引擎 - ReAct 模式"""

    @staticmethod
    def reflect_on_result(tool_name: str, result: Dict, expected: str = None) -> Dict:
        """反思工具执行结果"""
        reflection = {
            "success": not result.get("error"),
            "analysis": "",
            "suggestions": []
        }

        if result.get("error"):
            error = result["error"]

            # 错误分类
            if "not found" in error.lower():
                reflection["analysis"] = "文件/路径不存在"
                reflection["suggestions"] = [
                    "使用 list_dir 确认路径",
                    "检查文件名拼写",
                    "使用相对路径"
                ]
            elif "permission" in error.lower():
                reflection["analysis"] = "权限不足"
                reflection["suggestions"] = ["检查文件权限", "使用其他路径"]
            elif "syntax" in error.lower() or "invalid" in error.lower():
                reflection["analysis"] = "命令语法错误"
                reflection["suggestions"] = ["检查参数格式", "参考工具文档"]
            else:
                reflection["analysis"] = "未知错误"
                reflection["suggestions"] = ["换用其他工具", "调整策略"]
        else:
            reflection["analysis"] = "执行成功"

        return reflection

    @staticmethod
    def should_continue(history: List[Dict], max_failed: int = 3) -> Tuple[bool, str]:
        """判断是否应该继续"""
        if len(history) < max_failed:
            return True, ""

        recent = history[-max_failed:]
        all_failed = all(not h.get("success", True) for h in recent)

        if all_failed:
            return False, "连续失败，建议重新规划任务"

        return True, ""


class OutputValidator:
    """输出验证器 - 确保稳定性"""

    @staticmethod
    def validate_tool_call(tool_name: str, args: Dict) -> Tuple[bool, str]:
        """验证工具调用"""
        # 必需参数检查
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

        # 参数类型检查
        for key, value in args.items():
            if not isinstance(value, str):
                return False, f"参数 {key} 必须是字符串"

        return True, ""

    @staticmethod
    def sanitize_output(output: str, max_length: int = 2000) -> str:
        """清理输出"""
        if len(output) > max_length:
            return f"{output[:max_length]}...\n[输出过长，已截断。共 {len(output)} 字符]"
        return output


class QwenAgentFramework:
    """先进 Agent 框架 - ReAct + 反思 + 并行执行 + 持久化 + 语义压缩"""

    def __init__(
        self,
        model_forward_fn: Callable,
        work_dir: Optional[str] = None,
        enable_bash: bool = False,
        max_iterations: int = 50,
        middlewares: Optional[List[AgentMiddleware]] = None,
        enable_memory: bool = True,
        enable_reflection: bool = True,
        enable_parallel: bool = True,
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
        self.memory = SessionMemory() if enable_memory else None
        self.reflection = ReflectionEngine() if enable_reflection else None
        self.validator = OutputValidator()
        # 工具学习器（持久化工具成功率，跨轮推荐）
        self.tool_learner = ToolLearner() if enable_memory else None

        # 执行历史
        self.tool_history = []
        self.reflection_history = []
        # 已读文件集合：记录本轮会话中 read_file 成功读过的路径，防止 LLM 重复读取
        self.read_files_cache: Dict[str, str] = {}  # path -> 读取时间戳

        # 任务上下文
        self.task_context = {
            "current_task": None,
            "completed_steps": [],
            "failed_attempts": [],
        }

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示 - ReAct 模式"""
        # bash优先，用于批量扫描任务
        if self.tool_executor.enable_bash:
            tools = ["bash", "read_file", "write_file", "edit_file", "list_dir"]
        else:
            tools = ["read_file", "write_file", "edit_file", "list_dir"]

        tools_desc = "\n".join(f"- {t}" for t in tools)

        # 注入当前工作目录的绝对路径，供 LLM 构造正确的文件路径
        work_dir_abs = str(self.tool_executor.work_dir.resolve())

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
{{"command": "shell命令，如 grep -rn '^class \\\\|^def ' core/"}}

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
- 【write_file 模式选择】：
  * 新建文件或完全重写：用 overwrite（默认）
  * 向现有文件末尾追加内容（如整理文档时逐步追加）：必须用 append
  * 禁止用 overwrite 反复覆盖同一文件（会丢失前一次写入的内容）
- 【read_file 去重】：Observation 中已显示"已读过"的文件不要重复 read_file，直接使用上次结果

输出要求：
- 简洁明确，先思考再调用工具
- 遇到错误分析原因并调整策略"""

    def _detect_parallel_tools(self, tool_calls: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """检测可并行执行的工具"""
        if not self.enable_parallel or len(tool_calls) <= 1:
            return [], tool_calls

        # 只读工具可并行
        parallel_tools = []
        sequential_tools = []

        for tc in tool_calls:
            tool_name = tc["name"]
            if tool_name in ["read_file", "list_dir"]:
                parallel_tools.append(tc)
            else:
                sequential_tools.append(tc)

        return parallel_tools, sequential_tools

    def _execute_tools_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """并行执行工具"""
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for tc in tool_calls:
                future = executor.submit(
                    self._execute_single_tool,
                    tc["name"],
                    tc["args"]
                )
                futures[future] = tc

            for future in as_completed(futures):
                tc = futures[future]
                try:
                    result = future.result()
                    results.append({
                        "tool": tc["name"],
                        "result": result,
                        "parallel": True
                    })
                except Exception as e:
                    results.append({
                        "tool": tc["name"],
                        "result": {"error": str(e)},
                        "parallel": True
                    })

        return results

    def _execute_single_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """执行单个工具"""
        # 验证工具调用
        valid, error_msg = self.validator.validate_tool_call(tool_name, tool_args)
        if not valid:
            return {"error": f"参数验证失败: {error_msg}"}

        # 执行工具
        start_time = time.time()
        exec_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        result_str = self.tool_executor.execute_tool(tool_name, tool_args)
        exec_time = time.time() - start_time

        result = {"output": result_str} if not result_str.startswith("Error:") else {"error": result_str}

        # 智能重试
        if result.get("error"):
            fixed_args = self._try_fix(tool_name, tool_args, result["error"])
            if fixed_args:
                result_str = self.tool_executor.execute_tool(tool_name, fixed_args)
                result = {"output": result_str} if not result_str.startswith("Error:") else {"error": result_str}
                tool_args = fixed_args

        # 反思
        if self.reflection:
            reflection = self.reflection.reflect_on_result(tool_name, result)
            self.reflection_history.append({
                "tool": tool_name,
                "success": reflection["success"],
                "analysis": reflection["analysis"],
                "suggestions": reflection["suggestions"]
            })

        # 更新统计
        if self.memory:
            self.memory.update_tool_stats(tool_name, not result.get("error"), exec_time)

        # 工具学习器：记录本次调用结果，供下次推荐
        if self.tool_learner:
            task_types = self.tool_learner.classify_task(
                self.task_context.get("current_task") or ""
            )
            task_type = task_types[0] if task_types else "通用"
            self.tool_learner.record_usage(task_type, tool_name, not result.get("error"))
            self.tool_learner.save_to_disk()

        # 记录历史
        self.tool_history.append({
            "tool": tool_name,
            "args": json.dumps(tool_args, sort_keys=True),
            "success": not result.get("error"),
        })

        # 更新任务上下文
        if not result.get("error"):
            self.task_context["completed_steps"].append(f"{tool_name}({list(tool_args.keys())})")
            # 记录已成功读过的文件路径，供 _format_results 添加去重提示
            if tool_name == "read_file" and tool_args.get("path"):
                self.read_files_cache[tool_args["path"]] = exec_timestamp
        else:
            self.task_context["failed_attempts"].append(tool_name)

        # 清理输出
        if result.get("output"):
            result["output"] = self.validator.sanitize_output(result["output"])

        # 附加真实执行时间戳和工具名，供调用方构建 execution_log
        result["_exec_timestamp"] = exec_timestamp
        result["_tool_name"] = tool_name

        return result

    def run(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """主循环 - ReAct 模式 + 并行执行

        Args:
            user_input: 用户输入
            history: 历史消息列表
            runtime_context: 运行时上下文
            temperature/top_p/max_tokens: 生成参数，透传给 model_forward_fn
        """
        messages = self._build_messages(user_input, history)
        runtime_context = runtime_context or {}
        runtime_context["start_time"] = time.time()

        if self.memory:
            self.memory.add_context("current_task", user_input)

        tool_calls_log = []
        last_response = ""

        for iteration in range(self.max_iterations):
            # 检查是否应该继续
            if self.reflection:
                should_continue, reason = self.reflection.should_continue(self.reflection_history)
                if not should_continue:
                    if self.memory:
                        self.memory.save_to_disk()
                    return {
                        "response": f"⚠️ 任务中断: {reason}",
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                        "context": self._export_context(),
                    }

            # 上下文管理
            messages = self._compress_context_smart(messages)
            messages = self._inject_task_context(messages)
            messages = self._inject_reflection(messages)

            # 中间件
            for mw in self.middlewares:
                if hasattr(mw, "process_before_llm"):
                    messages = mw.process_before_llm(messages, runtime_context)

            # 调用模型（透传生成参数）
            try:
                response = self.model_forward_fn(
                    messages,
                    self.system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                last_response = response
            except Exception as e:
                if self.memory:
                    self.memory.save_to_disk()
                return {
                    "response": f"❌ 模型调用失败: {e}",
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                }

            # 解析工具
            tool_calls_raw = self.tool_parser.parse_tool_calls(response)
            tool_calls = [{"name": name, "args": args} for name, args in tool_calls_raw]

            if not tool_calls:
                # 判断是"任务完成自然结束"还是"工具调用格式错误导致解析失败"
                # 若响应中出现已知工具名但没解析到调用，说明是格式问题，给模型一次纠错机会
                _tool_names = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
                _has_tool_mention = any(t in response for t in _tool_names)
                # 响应包含终止信号：不含工具名，或者包含明确的完成/回答词
                _finish_signals = ("完成", "已完成", "总结", "综上", "结论", "以上", "如下", "好的", "以下是")
                _looks_finished = not _has_tool_mention or any(s in response[:80] for s in _finish_signals)

                if _looks_finished:
                    # 确实完成，正常退出
                    if self.memory:
                        self.memory.save_to_disk()
                    return {
                        "response": response,
                        "tool_calls": tool_calls_log,
                        "iterations": iteration + 1,
                        "context": self._export_context(),
                    }
                else:
                    # 格式错误：工具名出现在响应里但未能解析
                    # 注入纠错提示，让模型重新用正确格式输出工具调用
                    _work_dir = str(self.tool_executor.work_dir.resolve())
                    format_correction = (
                        "⚠️ [系统提示] 未检测到有效的工具调用格式。\n"
                        "❌ 错误格式示例（不要使用）：\n"
                        '{"api": "read_file", "path": "xxx.py"}  ← 不支持 api 字段\n'
                        '```json\n{"api": "read_file", ...}\n```  ← 不要用 api 字段\n\n'
                        "✅ 正确格式（工具名单独一行，JSON 紧跟下一行，不要用代码块包裹）：\n"
                        "read_file\n"
                        f'{{\"path\": \"{_work_dir}/文件名.py\"}}\n\n'
                        f"当前工作目录：{_work_dir}\n"
                        "请使用绝对路径以确保文件能被正确找到。\n"
                        "请直接重新输出工具调用，不要添加多余解释。"
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": format_correction})
                    continue  # 继续下一次迭代，让模型重试

            # 检测可并行工具
            parallel_tools, sequential_tools = self._detect_parallel_tools(tool_calls)

            # 执行工具
            results = []

            # 并行执行只读工具
            if parallel_tools:
                results.extend(self._execute_tools_parallel(parallel_tools))

            # 串行执行写入工具
            for tc in sequential_tools:
                result = self._execute_single_tool(tc["name"], tc["args"])
                results.append({"tool": tc["name"], "result": result, "parallel": False})

            # 记录日志（_exec_timestamp 由 _execute_single_tool 在真实执行时写入）
            for r in results:
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": r["tool"],
                    "success": not r["result"].get("error"),
                    "parallel": r.get("parallel", False),
                    "timestamp": r["result"].get("_exec_timestamp", ""),
                })

            # 循环检测
            if self._detect_loop():
                if self.memory:
                    self.memory.save_to_disk()
                return {
                    "response": "⚠️ 检测到循环。\n💡 建议：\n1. 换用其他工具\n2. 重新分析问题\n3. 调整参数",
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                    "loop_detected": True,
                    "context": self._export_context(),
                }

            # 回注结果
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": self._format_results(results)})

        if self.memory:
            self.memory.save_to_disk()

        # 截断末尾残留的工具调用字符串，避免污染最终输出
        clean_response = self._strip_trailing_tool_call(last_response) if last_response else ""
        return {
            "response": clean_response or "⚠️ 达到最大迭代次数，请尝试拆分任务或减少步骤",
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
            "context": self._export_context(),
        }

    @staticmethod
    def _strip_trailing_tool_call(text: str) -> str:
        """截断响应末尾残留的工具调用字符串，只保留自然语言思考/推理部分。

        处理以下两类残留：
        1. 裸格式：工具名独占一行 + 下一行是 JSON
               接下来我将读取文件。read_file
               {"path": "core/xxx.py"}
        2. 代码块格式：```json / ```plaintext 等包裹的工具参数块（LLM 没有正确触发工具）
               **Action**: 将分析结果写入 Api.md
               ```json
               {"path": "Api.md", "content": "..."}
               ```
        """
        import re

        # ── 第一步：清除末尾的 ```lang ... ``` 代码块（可能含工具参数）──────────
        # 反复清除，直到末尾不再有代码块为止（有时模型输出多个连续代码块）
        code_block_pattern = re.compile(
            r'\n?\s*```[^\n]*\n[\s\S]*?```\s*$',
            re.DOTALL
        )
        prev = None
        result = text
        while prev != result:
            prev = result
            m = code_block_pattern.search(result)
            if m:
                # 只有代码块内容看起来像工具参数（含 "path"/"command"/"content" 等键）才截断
                block_inner = m.group(0)
                _tool_param_hints = ('"path"', '"command"', '"content"', '"old_content"', '"new_content"')
                if any(h in block_inner for h in _tool_param_hints):
                    result = result[:m.start()].rstrip()

        # ── 第二步：清除末尾裸格式「工具名\n{...}」片段 ──────────────────────
        bare_pattern = r'\n?(read_file|write_file|edit_file|list_dir|bash|todo_write)\s*\n\s*\{[^}]*\}\s*$'
        result = re.sub(bare_pattern, '', result, flags=re.DOTALL).rstrip()

        return result if result else text

    def _build_messages(self, user_input: str, history: Optional[List[Dict]]) -> List[Dict]:
        """构建消息"""
        msgs = history[:] if history else []
        msgs.append({"role": "user", "content": user_input})
        return msgs

    def _inject_task_context(self, messages: List[Dict]) -> List[Dict]:
        """注入任务上下文：包含进度 + 原始目标锚定（防止目标漂移）"""
        parts = []

        # 锚定用户原始目标（始终注入，防止长对话中 LLM 忘记真正要做什么）
        original_task = self.task_context.get("current_task")
        if original_task:
            parts.append(f"🎯 原始任务（始终牢记，不可偏离）：{original_task}")

        # 进度信息
        if self.task_context["completed_steps"]:
            parts.append(
                f"📋 进度: 已完成 {len(self.task_context['completed_steps'])} 步 - "
                f"{', '.join(self.task_context['completed_steps'][-3:])}"
            )

        if not parts:
            return messages

        context_msg = {
            "role": "system",
            "content": "\n".join(parts)
        }
        return messages[:-1] + [context_msg] + messages[-1:]

    def _inject_reflection(self, messages: List[Dict]) -> List[Dict]:
        """注入反思信息"""
        if not self.reflection_history:
            return messages

        recent_failures = [r for r in self.reflection_history[-3:] if not r["success"]]
        if recent_failures:
            suggestions = []
            for rf in recent_failures:
                suggestions.extend(rf["suggestions"])

            reflection_msg = {
                "role": "system",
                "content": f"💡 反思: 最近{len(recent_failures)}次失败。建议: {', '.join(suggestions[:3])}"
            }
            return messages[:-1] + [reflection_msg] + messages[-1:]

        return messages

    def _detect_loop(self, max_same=3) -> bool:
        """循环检测：多层次检测模型陷入无进展循环。

        检测策略（按优先级）：
        1. 连续 max_same 次相同工具+参数调用（无论成功失败） → 立即中断
        2. 全局累计：同一工具+参数在整个会话中调用超过 5 次 → 中断（防止日志中出现的 87 次死循环）
        3. 连续全部失败且相同调用 → 中断（原逻辑保留）
        """
        if not self.tool_history:
            return False

        # ── 检测1：连续 max_same 次相同调用（滑动窗口） ──────────────────
        if len(self.tool_history) >= max_same:
            recent = self.tool_history[-max_same:]
            first = recent[0]
            if all(h["tool"] == first["tool"] and h["args"] == first["args"] for h in recent):
                return True

        # ── 检测2：全局累计相同调用超过阈值（防止极端死循环） ────────────
        # 对整个历史中"相同 tool+args"出现次数进行统计，超过 5 次直接中断
        _call_counts: Dict[str, int] = {}
        for h in self.tool_history:
            key = f"{h['tool']}|{h['args']}"
            _call_counts[key] = _call_counts.get(key, 0) + 1
        if any(cnt >= 5 for cnt in _call_counts.values()):
            return True

        return False

    def _compress_context_smart(self, messages: List[Dict], limit=6000) -> List[Dict]:
        """智能压缩 - 语义重要性"""
        total_chars = sum(len(json.dumps(m)) for m in messages)
        if total_chars / 1.5 < limit * 0.75:
            return messages

        system = [m for m in messages if m.get("role") == "system"]
        user_assistant = [m for m in messages if m.get("role") in ("user", "assistant")]

        # 保留最近6条
        recent = user_assistant[-6:]
        old = user_assistant[:-6] if len(user_assistant) > 6 else []

        if not old:
            return system + recent

        # 计算重要性并保留top-K
        if self.memory:
            scores = self.memory.compute_message_importance(old)
            # 按重要性排序，保留前3条
            scored_msgs = list(zip(old, scores))
            scored_msgs.sort(key=lambda x: x[1], reverse=True)
            important_msgs = [msg for msg, _ in scored_msgs[:3]]

            summary_text = self.memory.build_context_summary(old)
            summary = {"role": "system", "content": f"📦 历史摘要: {summary_text}"}
            return system + [summary] + important_msgs + recent

        summary = {"role": "system", "content": f"📦 已压缩 {len(old)} 条"}
        return system + [summary] + recent

    def _try_fix(self, tool: str, args: Dict, error: str) -> Optional[Dict]:
        """智能重试"""
        if tool == "bash" and "grep" in args.get("command", ""):
            cmd = args["command"]
            if "\\(" in cmd and "\\\\(" not in cmd:
                return {"command": cmd.replace("\\(", "(").replace("\\)", ")")}
            if "\\class" in cmd or "\\def" in cmd:
                return {"command": cmd.replace("\\class", "class").replace("\\def", "def")}

        if tool in ["read_file", "edit_file"] and "not found" in error.lower():
            path = args.get("path", "")
            if path and not path.startswith("/") and not path.startswith("."):
                return {"path": f"./{path}"}

        return None

    def _format_results(self, results: List[Dict]) -> str:
        """格式化结果，并在头部注入已读文件清单，防止 LLM 重复读取"""
        lines = []

        # 已读文件提示：若本轮累计读过文件，在 Observation 顶部注入一次摘要
        if self.read_files_cache:
            already_read = list(self.read_files_cache.keys())
            lines.append(
                f"📌 [已读文件清单，无需重复 read_file]: {already_read}"
            )

        for r in results:
            tool = r["tool"]
            result = r["result"]
            parallel_mark = " [并行]" if r.get("parallel") else ""

            if result.get("error"):
                lines.append(f"❌ {tool}{parallel_mark}: {result['error'][:200]}")

                # 添加反思建议
                if self.reflection_history:
                    last_reflection = self.reflection_history[-1]
                    if last_reflection["suggestions"]:
                        lines.append(f"   💡 建议: {last_reflection['suggestions'][0]}")
            else:
                output = str(result.get("output", ""))
                lines.append(f"✅ {tool}{parallel_mark}: {output}")

        return "\n".join(lines)

    def _export_context(self) -> Dict:
        """导出上下文"""
        ctx = {
            "task": self.task_context["current_task"],
            "completed_steps": self.task_context["completed_steps"],
            "failed_attempts": self.task_context["failed_attempts"],
            "tool_stats": {
                "total": len(self.tool_history),
                "success": sum(1 for h in self.tool_history if h["success"]),
                "failed": sum(1 for h in self.tool_history if not h["success"]),
            }
        }

        if self.memory:
            ctx["tool_recommendations"] = self.memory.get_tool_recommendation("general")

        return ctx

    # ------------------------------------------------------------------
    # 高层接口：供 UI 层调用
    # ------------------------------------------------------------------

    def process_message(
        self,
        user_input: str,
        chat_history=None,
        system_prompt_override: str = "",
        runtime_context: Optional[Dict] = None,
        return_runtime_context: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        """同步工具模式入口：调用 ReAct 循环，返回 (response, execution_log, runtime_context)。

        Args:
            user_input: 用户消息文本
            chat_history: 历史对话（二维列表 [[user, bot], ...] 或 None）
            system_prompt_override: 覆盖系统提示
            runtime_context: 运行时上下文字典
            return_runtime_context: 是否在返回值中包含 runtime_context
            temperature/top_p/max_tokens: 生成参数（透传给 model_forward_fn）
        Returns:
            (response_str, execution_log_list, updated_runtime_context)
        """
        runtime_context = dict(runtime_context or self.default_runtime_context)

        # 将 chat_history 转换为 messages 格式
        history_messages: List[Dict] = []
        if chat_history:
            for pair in chat_history:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    u, a = pair
                    if u:
                        history_messages.append({"role": "user", "content": str(u)})
                    if a:
                        history_messages.append({"role": "assistant", "content": str(a)})

        result = self.run(
            user_input,
            history=history_messages,
            runtime_context=runtime_context,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        response = result.get("response", "")
        tool_calls_log = result.get("tool_calls", [])

        # 构建结构化执行日志（timestamp 来自 _execute_single_tool 的真实执行时刻）
        execution_log = []
        for tc in tool_calls_log:
            execution_log.append({
                "iteration": tc.get("iteration", 0),
                "type": "tool_call",
                "tool": tc.get("tool", ""),
                "success": tc.get("success", True),
                "parallel": tc.get("parallel", False),
                "timestamp": tc.get("timestamp", ""),
            })

        return response, execution_log, runtime_context

    def process_message_direct_stream(
        self,
        messages: List[Dict],
        system_prompt_override: str = "",
        runtime_context: Optional[Dict] = None,
        stream_forward_fn=None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        """流式对话模式入口：经过中间件链后调用流式生成，逐 token yield (chunk, meta)。

        适用于 chat / skills 模式（不调用工具循环），直接流式输出模型响应。

        Args:
            messages: 完整的消息列表（含 system / history）
            system_prompt_override: 覆盖系统提示（已包含在 messages 中时可不传）
            runtime_context: 运行时上下文
            stream_forward_fn: 流式生成函数 generator(messages, system_prompt, **kw)
            temperature/top_p/max_tokens: 生成参数
        Yields:
            (accumulated_text, meta_dict) 元组
        """
        runtime_context = dict(runtime_context or self.default_runtime_context)
        processed = list(messages)

        # 应用中间件 process_before_llm
        for mw in self.middlewares:
            if hasattr(mw, "process_before_llm"):
                try:
                    processed = mw.process_before_llm(processed, runtime_context)
                except Exception:
                    pass

        # 决定使用哪个流式函数
        _stream_fn = stream_forward_fn if stream_forward_fn is not None else None

        if _stream_fn is not None:
            # 使用外部流式函数：yield 累积文本
            accumulated = ""
            for chunk in _stream_fn(
                processed,
                system_prompt=system_prompt_override,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                accumulated = chunk  # chunk 是累积文本
                yield accumulated, {}
        else:
            # 降级：同步调用 model_forward_fn，一次性返回全部文本
            try:
                response = self.model_forward_fn(
                    processed,
                    system_prompt_override or self.system_prompt,
                )
                yield response, {}
            except Exception as e:
                yield f"❌ 模型调用失败: {e}", {}
