#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会话日志管理模块 - 用于持久化存储和分析对话记录
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def _make_json_serializable(obj: Any) -> Any:
    """递归地将不可 JSON 序列化的对象转换为可序列化形式。

    主要处理:
    - set → list（_executed_tool_calls、_read_file_paths 等内部追踪集合）
    - tuple → list
    - 其他不可序列化对象 → str
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # 先转换每个元素，再尝试排序（排序失败时不排序，保证不崩溃）
        items = [_make_json_serializable(v) for v in obj]
        try:
            return sorted(str(v) for v in items)
        except Exception:
            return items
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    # 基础 JSON 类型：直接返回
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # 其他不可序列化的对象（Path 等）：转为字符串
    return str(obj)


class SessionLogger:
    """会话日志管理器 - 负责日志的持久化存储和查询"""

    def __init__(self, log_dir: str = None):
        """
        初始化日志管理器

        Args:
            log_dir: 日志存储目录，如果为None则使用项目根目录下的session_logs
        """
        if log_dir is None:
            # 获取项目根目录（ui/session_logger.py 的父目录的父目录）
            project_root = Path(__file__).parent.parent
            log_dir = str(project_root / "session_logs")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id = None
        self.current_session_file = None
        # 以 session_id 为 key 缓存模型调用记录，避免多轮请求/生成器交错时相互污染
        # 结构: { session_id: [call_record, ...] }
        self._pending_model_calls: Dict[str, List[Dict[str, Any]]] = {}

    def create_session(self) -> str:
        """
        创建新的会话

        Returns:
            session_id: 会话ID (基于时间戳)
        """
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.current_session_file = self.log_dir / f"{self.current_session_id}.json"

        # 初始化会话文件
        session_data = {
            "session_id": self.current_session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "calls": [],
            "statistics": {
                "total_messages": 0,
                "total_tokens_used": 0,
                "total_duration": 0,
                "total_calls": 0
            }
        }

        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # 新会话开始时清空该 session 的待绑定记录
        self._pending_model_calls[self.current_session_id] = []

        return self.current_session_id

    def log_message(self,
                   user_message: str,
                   bot_response: str,
                   execution_time: float = 0,
                   tokens_used: int = 0,
                   model: str = "Qwen2.5-0.5B",
                   model_calls: List[Dict[str, Any]] = None,
                   runtime_context: Optional[Dict[str, Any]] = None,
                   execution_log: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        记录一条对话消息

        Args:
            user_message: 用户消息
            bot_response: 机器人回复
            execution_time: 执行时间(秒)
            tokens_used: 使用的token数（若 model_calls 有数据则以计算值为准）
            model: 使用的模型
            model_calls: 模型调用的详细记录列表
            runtime_context: 运行时上下文（模式、技能、上传文件等）
            execution_log: Agent 执行日志（模型轮次、工具调用等）
        """
        if not self.current_session_file or not self.current_session_file.exists():
            self.create_session()

        # 合并：优先使用传入的 model_calls，若没有则使用当前 session 的缓存 pending calls
        merged_calls = list(model_calls) if model_calls else []
        _sid = self.current_session_id or ""
        if not merged_calls:
            _pending_cur = self._pending_model_calls.get(_sid, [])
            merged_calls = list(_pending_cur)
        # 重置当前 session 的缓存（已绑定到本条消息）
        self._pending_model_calls[_sid] = []
        self._pending_model_calls[""] = []

        # 从 model_calls 重新计算真实 token 用量（覆盖传入值）
        if merged_calls:
            calculated_tokens = sum(
                c.get("tokens_total", c.get("tokens_input", 0) + c.get("tokens_output", 0))
                for c in merged_calls
            )
            tokens_used = max(tokens_used, calculated_tokens)

        # 对 runtime_context 做序列化安全处理：
        # agent_framework 内部使用 set 追踪已执行工具调用（_executed_tool_calls）
        # 和已读文件路径（_read_file_paths），这些 set 不能直接被 json.dump 序列化。
        safe_runtime_context = _make_json_serializable(runtime_context or {})

        message_record = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "execution_time": execution_time,
            "tokens_used": tokens_used,
            "model": model,
            "model_calls": merged_calls,
            "runtime_context": safe_runtime_context,
            "execution_log": execution_log or []
        }

        try:
            with open(self.current_session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
        except Exception as e:
            print(f"读取会话文件失败，重新创建: {e}")
            self.create_session()
            with open(self.current_session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

        session_data["messages"].append(message_record)
        session_data["statistics"]["total_messages"] += 1
        session_data["statistics"]["total_tokens_used"] += tokens_used
        session_data["statistics"]["total_duration"] += execution_time
        session_data["statistics"]["total_calls"] += len(merged_calls)

        try:
            with open(self.current_session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except TypeError as e:
            # 兜底：若仍有不可序列化字段，再次清洗后保存
            print(f"日志序列化失败（{e}），尝试深度清洗后重试")
            session_data["messages"][-1] = _make_json_serializable(message_record)
            with open(self.current_session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

    def log_model_call(self,
                      prompt: str,
                      response: str,
                      execution_time: float = 0,
                      tokens_input: int = 0,
                      tokens_output: int = 0,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      model_name: str = "Qwen2.5-0.5B") -> None:
        """
        记录一次模型调用的详细信息（实时记录中间调用）。

        调用记录先缓存到 _pending_model_calls，等 log_message 调用时统一绑定到正确的用户消息。

        Args:
            prompt: 输入提示词
            response: 模型输出
            execution_time: 执行时间
            tokens_input: 输入token数
            tokens_output: 输出token数
            temperature: 温度参数
            top_p: top_p参数
            model_name: 模型名称
        """
        _sid = self.current_session_id or ""
        _cur_pending = self._pending_model_calls.setdefault(_sid, [])
        model_call_record = {
            "call_index": len(_cur_pending) + 1,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "execution_time": execution_time,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_total": tokens_input + tokens_output,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p
            },
            "model": model_name
        }

        # 缓存在当前 session 的 pending 列表，等待 log_message 绑定
        _cur_pending.append(model_call_record)

    def log_skill_call(self,
                      skill_id: str,
                      skill_name: str,
                      input_data: Dict[str, Any],
                      output_data: Dict[str, Any],
                      execution_time: float = 0,
                      status: str = "success") -> None:
        """
        记录一次技能调用

        Args:
            skill_id: 技能ID
            skill_name: 技能名称
            input_data: 输入数据
            output_data: 输出数据
            execution_time: 执行时间
            status: 执行状态 (success/error)
        """
        if not self.current_session_file or not self.current_session_file.exists():
            self.create_session()

        call_record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "skill_name": skill_name,
            "input": input_data,
            "output": output_data,
            "execution_time": execution_time,
            "status": status
        }

        with open(self.current_session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        session_data["calls"].append(call_record)

        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        获取所有会话列表

        Returns:
            sessions: 会话列表，按时间倒序排列
        """
        sessions = []

        for log_file in sorted(self.log_dir.glob("*.json"), reverse=True):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                # 从 messages 重新计算实际 model_calls 数和 tokens
                # 过滤掉旧代码产生的占位消息（[未设置]/[在进行中...]），只统计真实消息
                total_model_calls = 0
                total_tokens_calculated = 0
                real_message_count = 0
                for msg in session_data.get("messages", []):
                    # 跳过旧版代码产生的占位消息
                    if msg.get("user_message") == "[未设置]" and msg.get("bot_response") == "[在进行中...]":
                        # 占位消息中的 model_calls 仍需计入统计（是真实调用）
                        for call in msg.get("model_calls", []):
                            total_tokens_calculated += call.get(
                                "tokens_total",
                                call.get("tokens_input", 0) + call.get("tokens_output", 0)
                            )
                            total_model_calls += 1
                        continue
                    real_message_count += 1
                    model_calls = msg.get("model_calls", [])
                    total_model_calls += len(model_calls)
                    for call in model_calls:
                        total_tokens_calculated += call.get(
                            "tokens_total",
                            call.get("tokens_input", 0) + call.get("tokens_output", 0)
                        )

                sessions.append({
                    "session_id": session_data["session_id"],
                    "created_at": session_data["created_at"],
                    "message_count": real_message_count,
                    "call_count": total_model_calls,
                    "total_duration": session_data["statistics"]["total_duration"],
                    "total_tokens": total_tokens_calculated
                })
            except Exception as e:
                print(f"读取日志文件 {log_file} 失败: {e}")

        return sessions

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话的详细信息

        Args:
            session_id: 会话ID

        Returns:
            session_data: 会话数据，如果不存在返回None
        """
        session_file = self.log_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # 重新计算统计数据以确保准确性
            # 过滤掉旧版代码产生的占位消息（[未设置]/[在进行中...]），只计入真实消息
            total_messages = 0
            total_tokens_calculated = 0
            total_model_calls = 0
            total_duration = 0

            for msg in session_data.get("messages", []):
                # 跳过旧版代码产生的占位消息，但其 model_calls 仍计入统计
                is_placeholder = (
                    msg.get("user_message") == "[未设置]"
                    and msg.get("bot_response") == "[在进行中...]"
                )
                if is_placeholder:
                    for call in msg.get("model_calls", []):
                        total_tokens_calculated += call.get(
                            "tokens_total",
                            call.get("tokens_input", 0) + call.get("tokens_output", 0)
                        )
                        total_model_calls += 1
                    continue

                total_messages += 1
                total_duration += msg.get("execution_time", 0)
                model_calls = msg.get("model_calls", [])
                total_model_calls += len(model_calls)
                for call in model_calls:
                    total_tokens_calculated += call.get(
                        "tokens_total",
                        call.get("tokens_input", 0) + call.get("tokens_output", 0)
                    )

            session_data["statistics"] = {
                "total_messages": total_messages,
                "total_tokens_used": total_tokens_calculated,
                "total_duration": total_duration,
                "total_calls": total_model_calls
            }

            return session_data
        except Exception as e:
            print(f"读取会话文件失败: {e}")
            return None

    def export_session(self, session_id: str, export_path: str) -> bool:
        """
        导出会话数据为JSON文件

        Args:
            session_id: 会话ID
            export_path: 导出路径

        Returns:
            success: 是否导出成功
        """
        session_data = self.get_session_details(session_id)

        if not session_data:
            return False

        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"导出会话失败: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            success: 是否删除成功
        """
        session_file = self.log_dir / f"{session_id}.json"

        if not session_file.exists():
            return False

        try:
            session_file.unlink()
            return True
        except Exception as e:
            print(f"删除会话失败: {e}")
            return False


# 全局日志管理器实例
_logger_instance = None


def get_logger() -> SessionLogger:
    """获取全局日志管理器实例"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SessionLogger()
    return _logger_instance

