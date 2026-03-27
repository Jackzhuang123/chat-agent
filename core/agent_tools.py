#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 工具系统 - 为 Qwen 模型提供工具支持
包含: bash, read_file, write_file, edit_file 等工具
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ToolExecutor:
    """工具执行器 - 管理和执行所有可用工具"""

    def __init__(self, work_dir: Optional[str] = None, enable_bash: bool = True):
        """
        初始化工具执行器

        Args:
            work_dir: 工作目录,默认为当前目录
            enable_bash: 是否启用 bash 工具 (对安全性有影响)
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.enable_bash = enable_bash
        self.tool_descriptions = self._build_tool_descriptions()

    def _build_tool_descriptions(self) -> List[Dict[str, Any]]:
        """构建工具描述列表,供模型调用"""
        tools = [
            {
                "name": "read_file",
                "description": "读取文件内容。用于查看代码、配置文件等。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要读取的文件路径(相对于工作目录)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": (
                    "写入文件。\n"
                    "- mode=\"overwrite\"（默认）：创建新文件或完全替换已有内容\n"
                    "- mode=\"append\"：追加内容到文件末尾（文件不存在时自动创建）\n"
                    "【重要】整理文档时请用 append 逐步追加，不要用 overwrite 反复覆盖已有内容。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径(相对于工作目录)"
                        },
                        "content": {
                            "type": "string",
                            "description": "文件内容"
                        },
                        "mode": {
                            "type": "string",
                            "description": "写入模式：overwrite（覆盖，默认）或 append（追加到末尾）"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "edit_file",
                "description": "精确编辑文件内容。用于替换文件中的特定部分。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径(相对于工作目录)"
                        },
                        "old_content": {
                            "type": "string",
                            "description": "要替换的原始内容(必须完全匹配)"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "新内容"
                        }
                    },
                    "required": ["path", "old_content", "new_content"]
                }
            },
            {
                "name": "list_dir",
                "description": "列出目录中的文件和子目录。用于探索文件结构。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "目录路径(相对于工作目录),默认为当前目录"
                        }
                    },
                    "required": []
                }
            }
        ]

        # 条件添加 bash 工具
        if self.enable_bash:
            tools.append({
                "name": "bash",
                "description": "运行 shell 命令。用于执行任何系统命令: ls, git, python, npm 等。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的 shell 命令"
                        }
                    },
                    "required": ["command"]
                }
            })

        # todo_write：虚拟工具，由 TodoContextMiddleware 拦截处理（不走本执行器）
        tools.append({
            "name": "todo_write",
            "description": (
                "更新任务计划列表（TODO）中某个步骤的状态。\n"
                "支持操作：\n"
                '  - 标记完成: {"action": "update", "id": 1, "status": "completed", "result_preview": "简短结果"}\n'
                '  - 标记失败: {"action": "update", "id": 1, "status": "failed", "error": "失败原因"}\n'
                '  - 新增步骤: {"action": "add", "task": "步骤描述", "tool": "none"}'
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型: update | add"
                    },
                    "id": {
                        "type": "integer",
                        "description": "要更新的步骤 ID（action=update 时必填）"
                    },
                    "status": {
                        "type": "string",
                        "description": "新状态: pending | in_progress | completed | failed | cancelled"
                    },
                    "task": {
                        "type": "string",
                        "description": "步骤描述（action=add 时必填）"
                    },
                    "tool": {
                        "type": "string",
                        "description": "关联工具名（action=add 时可选）"
                    },
                    "result_preview": {
                        "type": "string",
                        "description": "步骤结果摘要（可选）"
                    },
                    "error": {
                        "type": "string",
                        "description": "失败原因（action=update, status=failed 时可选）"
                    }
                },
                "required": ["action"]
            }
        })

        return tools

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具描述"""
        return self.tool_descriptions

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        执行指定的工具

        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数

        Returns:
            工具执行结果 (JSON 字符串)
        """
        try:
            if tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "write_file":
                return self._write_file(
                    tool_input["path"],
                    tool_input["content"],
                    mode=tool_input.get("mode", "overwrite"),
                )
            elif tool_name == "edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input["old_content"],
                    tool_input["new_content"]
                )
            elif tool_name == "list_dir":
                return self._list_dir(tool_input.get("path", "."))
            elif tool_name == "bash" and self.enable_bash:
                return self._bash(tool_input["command"])
            elif tool_name == "todo_write":
                # 虚拟工具：正常由 TodoContextMiddleware.after_tool_call 拦截处理
                # 若未配置中间件则返回提示（调用方应优先通过 TodoContextMiddleware 处理）
                return json.dumps({
                    "success": False,
                    "error": "todo_write 需要配置 TodoContextMiddleware 才能生效",
                    "hint": "请在 QwenAgentFramework 中添加 TodoContextMiddleware"
                }, ensure_ascii=False)
            else:
                return json.dumps({"error": f"未知工具: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _fuzzy_find_file(self, filename: str, search_home: bool = True) -> Optional[Path]:
        """
        模糊搜索文件，三级策略:
          1. 在 work_dir 递归搜索（速度快，最优先）
          2. 在用户家目录递归搜索（若 search_home=True）
          3. 返回 None，调用方提示用户

        跳过常见无用目录: .git, __pycache__, node_modules, .venv, venv 等

        Args:
            filename: 目标文件名（可以是纯文件名，也可以是末尾部分路径，如 "ui/x.py"）
            search_home: 是否允许搜索家目录（默认 True）

        Returns:
            找到的 Path，或 None
        """
        SKIP_DIRS = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
            ".idea", ".vscode",
        }

        # 只取最后一段文件名用于匹配
        target_name = Path(filename).name

        def _scan(root: Path, depth_limit: int = 8) -> Optional[Path]:
            """递归扫描，depth_limit 防止无限下探"""
            try:
                for item in sorted(root.iterdir()):
                    if item.name in SKIP_DIRS or item.name.startswith("."):
                        continue
                    if item.is_file() and item.name == target_name:
                        return item
                    if item.is_dir() and depth_limit > 0:
                        found = _scan(item, depth_limit - 1)
                        if found:
                            return found
            except (PermissionError, OSError):
                pass
            return None

        # 第一级: 项目工作目录
        result = _scan(self.work_dir)
        if result:
            return result

        # 第二级: 用户家目录（跳过已搜过的 work_dir）
        if search_home:
            home = Path.home()
            try:
                for item in sorted(home.iterdir()):
                    if item.name in SKIP_DIRS or item.name.startswith("."):
                        continue
                    # 跳过已经搜索过的工作目录
                    if item.resolve() == self.work_dir.resolve():
                        continue
                    if item.is_file() and item.name == target_name:
                        return item
                    if item.is_dir():
                        found = _scan(item, depth_limit=6)
                        if found:
                            return found
            except (PermissionError, OSError):
                pass

        return None

    def _read_file(self, path: str) -> str:
        """读取文件，支持绝对路径 + 三级模糊路径搜索（项目内 → 家目录 → 提示）"""
        p = Path(path)

        # ---- 优先：绝对路径直接读取（不做模糊搜索，避免误匹配）----
        if p.is_absolute():
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                    result_abs: Dict[str, Any] = {
                        "success": True,
                        "path": str(p),
                        "content": content
                    }
                    if not content.strip():
                        result_abs["warning"] = "文件存在但内容为空（0字节或全为空白符），不是文件缺失"
                    return json.dumps(result_abs)
                except Exception as e:
                    return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"})
            else:
                return json.dumps({
                    "success": False,
                    "file_not_found": True,
                    "error": f"文件不存在: {path}",
                    "hint": "绝对路径指定的文件不存在，请先用 list_dir 确认路径。"
                })

        # ---- 相对路径：拼接工作目录 ----
        file_path = self.work_dir / path

        # 尝试直接路径
        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                result: Dict[str, Any] = {
                    "success": True,
                    "path": str(file_path.resolve()),
                    "content": content
                }
                if not content.strip():
                    result["warning"] = "文件存在但内容为空（0字节或全为空白符），不是文件缺失"
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"})

        # ---- 直接路径不存在，启动模糊搜索 ----
        filename = Path(path).name
        found = self._fuzzy_find_file(filename, search_home=True)

        if found:
            try:
                with open(found, "r", encoding="utf-8") as f:
                    content = f.read()
                return json.dumps({
                    "success": True,
                    "path": str(found),
                    "fuzzy_match": True,
                    "original_request": path,
                    "note": f"原路径 '{path}' 不存在，自动匹配到: {found}",
                    "content": content
                })
            except Exception as e:
                return json.dumps({"error": f"读取文件失败: {str(e)}"})

        return json.dumps({
            "success": False,
            "file_not_found": True,
            "error": f"文件不存在: {path}",
            "searched_name": filename,
            "hint": (
                f"在工作目录和家目录的全部子目录中均未找到名为 '{filename}' 的文件（不是空文件，而是根本不存在）。"
                f"请先用 list_dir 查看目录结构，确认文件名和路径，再重新调用 read_file。"
            )
        })

    def _write_file(self, path: str, content: str, mode: str = "overwrite") -> str:
        """写入文件，支持 overwrite（覆盖）和 append（追加）两种模式"""
        # 支持绝对路径
        p = Path(path)
        file_path = p if p.is_absolute() else self.work_dir / path

        # 安全性检查：相对路径不得逃逸出工作目录；绝对路径直接放行
        if not p.is_absolute():
            try:
                file_path.resolve().relative_to(self.work_dir.resolve())
            except ValueError:
                return json.dumps({"error": "路径超出工作目录范围"})

        try:
            # 创建父目录
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if mode == "append":
                # 追加模式：在已有内容末尾追加，文件不存在时自动创建
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                total_size = file_path.stat().st_size
                return json.dumps({
                    "success": True,
                    "path": path,
                    "mode": "append",
                    "appended_size": len(content),
                    "total_size": total_size,
                })
            else:
                # 覆盖模式（默认）：完整替换文件内容
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return json.dumps({
                    "success": True,
                    "path": path,
                    "mode": "overwrite",
                    "size": len(content),
                })
        except Exception as e:
            return json.dumps({"error": f"写入文件失败: {str(e)}"})

    def _edit_file(self, path: str, old_content: str, new_content: str) -> str:
        """编辑文件 - 替换指定部分"""
        # 支持绝对路径
        p = Path(path)
        file_path = p if p.is_absolute() else self.work_dir / path

        # 安全性检查：相对路径不得逃逸出工作目录；绝对路径直接放行
        if not p.is_absolute():
            try:
                file_path.resolve().relative_to(self.work_dir.resolve())
            except ValueError:
                return json.dumps({"error": "路径超出工作目录范围"})

        if not file_path.exists():
            return json.dumps({"error": f"文件不存在: {path}"})

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_content not in content:
                return json.dumps({
                    "error": "原始内容未在文件中找到",
                    "hint": "请确保原始内容完全匹配(包括空格和换行)"
                })

            new_file_content = content.replace(old_content, new_content, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_file_content)

            return json.dumps({
                "success": True,
                "path": path,
                "replaced": True
            })
        except Exception as e:
            return json.dumps({"error": f"编辑文件失败: {str(e)}"})

    def _list_dir(self, path: str = ".") -> str:
        """列出目录内容"""
        # 支持绝对路径：若 path 是绝对路径则直接使用，否则相对于 work_dir 拼接
        p = Path(path)
        if p.is_absolute():
            dir_path = p
        else:
            dir_path = self.work_dir / path

        # 安全性检查：相对路径不得逃逸出工作目录；绝对路径直接放行
        if not p.is_absolute():
            try:
                dir_path.resolve().relative_to(self.work_dir.resolve())
            except ValueError:
                return json.dumps({"error": "路径超出工作目录范围"})

        if not dir_path.exists():
            return json.dumps({"error": f"目录不存在: {path}"})

        if not dir_path.is_dir():
            return json.dumps({"error": f"不是目录: {path}"})

        try:
            items = []
            subdirs = []
            for item in sorted(dir_path.iterdir()):
                # 跳过隐藏文件和 __pycache__ 等
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue

                is_dir = item.is_dir()
                items.append({
                    "name": item.name,
                    "type": "dir" if is_dir else "file",
                    "size": item.stat().st_size if not is_dir else None
                })
                if is_dir:
                    subdirs.append(item.name)

            result: Dict[str, Any] = {
                "success": True,
                "path": path,
                "items": items
            }
            if subdirs:
                result["hint"] = (
                    f"以下子目录尚未展开，若目标文件不在当前层，"
                    f"请继续调用 list_dir 探索子目录: {subdirs}"
                )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"列出目录失败: {str(e)}"})

    def _bash(self, command: str) -> str:
        """执行 bash 命令"""
        if not self.enable_bash:
            return json.dumps({"error": "bash 工具已禁用"})

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=30  # 30 秒超时
            )

            return json.dumps({
                "success": True,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "命令执行超时(30秒)"})
        except Exception as e:
            return json.dumps({"error": f"执行命令失败: {str(e)}"})


class ToolParser:
    """工具调用解析器 - 从模型输出中解析工具调用"""

    @staticmethod
    def _parse_input_payload(input_str: str) -> Optional[Dict[str, Any]]:
        """解析工具输入，容忍轻微 JSON 格式问题。

        容错策略（按优先级）：
        1. 直接 json.loads
        2. 补齐缺失右括号后重试
        3. 修复非法 JSON 转义序列（\\s \\; \\( \\w 等正则/shell 转义）后重试
        """
        if not input_str:
            return None

        payload = input_str.strip()
        if not payload:
            return None

        # ── 策略1：直接解析 ───────────────────────────────────────────────
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        # ── 策略2：补齐缺失右括号 ─────────────────────────────────────────
        if payload.startswith("{"):
            missing_right_brace = payload.count("{") - payload.count("}")
            if missing_right_brace > 0:
                repaired = payload + ("}" * missing_right_brace)
                try:
                    data = json.loads(repaired)
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    pass

        # ── 策略3：修复非法 JSON 转义序列 ──────────────────────────────────
        # 模型有时输出 shell/regex 转义（\s \; \( \w \d 等），这些不是合法的 JSON
        # 将 \x（x 不在合法 JSON 转义字符集）替换为 \\x（双重转义，即字面量反斜杠）
        # 合法 JSON 转义: \" \\ \/ \b \f \n \r \t \uXXXX
        _VALID_JSON_ESCAPES = set('"\\//bfnrtu')
        if payload.startswith("{"):
            try:
                def fix_invalid_escapes(s: str) -> str:
                    result = []
                    i = 0
                    while i < len(s):
                        if s[i] == '\\' and i + 1 < len(s):
                            next_ch = s[i + 1]
                            if next_ch in _VALID_JSON_ESCAPES:
                                result.append(s[i])   # 保留合法转义
                            else:
                                result.append('\\\\')  # 非法转义 → 双反斜杠
                            i += 1
                        else:
                            result.append(s[i])
                        i += 1
                    return ''.join(result)

                fixed = fix_invalid_escapes(payload)
                data = json.loads(fixed)
                return data if isinstance(data, dict) else None
            except (json.JSONDecodeError, Exception):
                pass

        return None

    @staticmethod
    def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        从模型输出中解析工具调用

        支持的格式:
        1. JSON 数组格式: [{"tool": "bash", "input": {"command": "ls"}}]
        2. JSON 单对象格式: {"tool": "bash", "input": {"command": "ls"}}
        3. function_call 格式: {"name": "bash", "arguments": {"command": "ls"}}
        4. markdown JSON 代码块: ```json\n{"tool": "list_dir", "input": {...}}\n```
        5. 工具名 + markdown JSON 代码块: list_dir\n```json\n{...}\n```
        6. XML 标记格式: <tool>bash</tool><input>{"command": "ls"}</input>
        7. 裸格式: bash\n{"command": "ls"}

        Returns:
            List[(tool_name, tool_input), ...]
        """
        calls: List[Tuple[str, Dict[str, Any]]] = []
        stripped = text.strip()
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}

        # ---- 格式1/2/3：全文是 JSON（数组或单对象）----
        try:
            if stripped.startswith("["):
                data = json.loads(stripped)
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    # {"tool": ..., "input": ...} 格式
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        calls.append((item["tool"], item["input"]))
                    # {"name": ..., "arguments": ...} 格式（OpenAI function_call 风格）
                    elif "name" in item and "arguments" in item:
                        args = item["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            calls.append((item["name"], args))
                if calls:
                    return calls
            elif stripped.startswith("{"):
                item = json.loads(stripped)
                if isinstance(item, dict):
                    # {"tool": ..., "input": ...}
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        return [(item["tool"], item["input"])]
                    # {"name": ..., "arguments": ...}
                    elif "name" in item and "arguments" in item:
                        args = item["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            return [(item["name"], args)]
                    # {"name": ..., "params": ...} 格式（GLM ReAct 格式）
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return [(item["name"], item["params"])]
                    # {"api": "tool_name", "param": "value", ...} 格式（GLM 常见错误格式）
                    # 工具名在 "api" 字段，其余键值对直接作为工具参数
                    elif "api" in item and isinstance(item["api"], str) and item["api"] in known_tools:
                        tool_name = item["api"]
                        tool_args = {k: v for k, v in item.items() if k != "api"}
                        return [(tool_name, tool_args)]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # ---- 格式1b：markdown 代码块内包含 {"api": ...} 格式 ----
        # 例：```json\n{"api": "read_file", "path": "xxx.py"}\n```
        _md_api_pattern = re.compile(r'```(?:json)?\s*\n\s*(\{[\s\S]*?\})\s*\n?\s*```')
        for _mam in _md_api_pattern.finditer(stripped):
            try:
                _obj = json.loads(_mam.group(1))
                if isinstance(_obj, dict) and "api" in _obj and isinstance(_obj["api"], str) and _obj["api"] in known_tools:
                    _tname = _obj["api"]
                    _targs = {k: v for k, v in _obj.items() if k != "api"}
                    return [(_tname, _targs)]
            except (json.JSONDecodeError, TypeError):
                pass

        # ---- 格式4：整段文本是 markdown JSON 代码块 ----
        # 例：```json\n{"tool": "list_dir", "input": {"path": "core"}}\n```
        md_block_match = re.match(r'^```(?:json|python)?\s*\n([\s\S]*?)\s*```\s*$', stripped)
        if md_block_match:
            inner = md_block_match.group(1).strip()
            try:
                item = json.loads(inner)
                if isinstance(item, dict):
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        return [(item["tool"], item["input"])]
                    elif "name" in item and "arguments" in item:
                        args = item["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            return [(item["name"], args)]
                    # 新增：支持 {"name": "tool", "params": {...}} 格式
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return [(item["name"], item["params"])]
            except (json.JSONDecodeError, TypeError):
                pass

        # ---- 格式4b：自然语言 + ```json {参数} ``` 代码块，从参数键名推断工具名 ----
        # 例：读取文件内容的操作如下：\n```json\n{"path": "/xxx/yyy.py"}\n```
        # 此时 JSON 内没有 "tool"/"name" 字段，只有参数字段，需要根据参数键名推断工具
        #   - 只含 "path" → 结合上文若含"读"/"查看"/"read" 推断 read_file；含"list"/"目录" 推断 list_dir
        #   - 含 "path" + "content" → write_file
        #   - 含 "path" + "old_content"/"new_content" → edit_file
        #   - 含 "command" → bash
        # 注意：末尾用 \s*``` 而不是 \n```，以兼容多行 JSON 中 } 直接紧接 ``` 的情况
        # 例如：```json\n{"path":"x","content":"very long...\n..."}\n``` 或 }```
        _inline_block_pattern = re.compile(
            r'```(?:json|plaintext|python|bash|shell|text)?\s*\n([\s\S]*?)\s*```',
            re.MULTILINE
        )
        for _m in _inline_block_pattern.finditer(stripped):
            _inner = _m.group(1).strip()
            try:
                _item = json.loads(_inner)
            except (json.JSONDecodeError, TypeError):
                _item = None
            if not isinstance(_item, dict):
                continue

            # 检查是否有工具名字段
            if "tool" in _item and "input" in _item and isinstance(_item["input"], dict):
                return [(_item["tool"], _item["input"])]
            elif "name" in _item and "arguments" in _item:
                args = _item["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if isinstance(args, dict):
                    return [(_item["name"], args)]
            elif "name" in _item and "params" in _item and isinstance(_item["params"], dict):
                return [(_item["name"], _item["params"])]

            # 如果已有明确的工具字段，不继续推断
            if "tool" in _item or "name" in _item:
                continue
            _keys = set(_item.keys())
            # 从参数键名推断工具名
            _inferred_tool = None
            if "command" in _keys:
                _inferred_tool = "bash"
            elif {"path", "old_content", "new_content"}.issubset(_keys):
                _inferred_tool = "edit_file"
            elif {"path", "content"}.issubset(_keys):
                _inferred_tool = "write_file"
            elif _keys == {"path"}:
                # 从代码块紧邻的前 150 字符推断是 read_file 还是 list_dir
                # 注意：只取紧邻窗口避免受到全文中 "list_dir" 等词的干扰
                _window_start = max(0, _m.start() - 150)
                _prefix = stripped[_window_start: _m.start()].lower()
                # list_dir 明确信号：出现"列出"/"列目录"/"list_dir"/"listdir"
                # 不使用单独的 "list" 或 "dir"，因为 "list_dir" 作为工具名本身会干扰
                _list_hints_strict = ("列出", "列目录", "list_dir", "listdir", "目录结构", "查看目录", "浏览目录")
                # read_file 信号：出现"读取"/"读文件"/"read_file"/"阅读"/"查看文件"/"打开文件"
                _read_hints = ("读取", "读文件", "read_file", "阅读", "查看文件", "打开文件", "文件内容", "read")
                if any(h in _prefix for h in _list_hints_strict):
                    _inferred_tool = "list_dir"
                else:
                    # 默认推断为 read_file（read 意图更常见，且是更安全的降级选择）
                    _inferred_tool = "read_file"
            if _inferred_tool and _inferred_tool in known_tools:
                calls.append((_inferred_tool, _item))
        if calls:
            return calls

        # ---- 格式5：工具名 + markdown JSON 代码块 ----
        # 例：list_dir\n```json\n{"path": "core"}\n```
        for tool in known_tools:
            pattern = re.compile(
                r'(?:^|(?<=[\s。.，,、！!？?；;：:\n]))' + re.escape(tool) +
                r'\s*\n\s*```(?:json)?\s*\n([\s\S]*?)\s*```',
                re.MULTILINE
            )
            for m in pattern.finditer(stripped):
                inner = m.group(1).strip()
                data = ToolParser._parse_input_payload(inner)
                if data is not None:
                    calls.append((tool, data))
            if calls:
                return calls

        # 尝试严格标记格式
        tool_pattern = r"<tool>(\w+)</tool>"
        input_pattern = r"<input>(.*?)</input>"

        tools = re.findall(tool_pattern, text)
        inputs = re.findall(input_pattern, text, re.DOTALL)

        if tools and inputs and len(tools) == len(inputs):
            for tool, input_str in zip(tools, inputs):
                input_data = ToolParser._parse_input_payload(input_str)
                if input_data is not None:
                    calls.append((tool, input_data))

            if calls:
                return calls

        # 容错: 缺失 </input> 时，读取到下一工具标签或文本末尾
        tool_matches = list(re.finditer(tool_pattern, text))
        if not tool_matches:
            # 兼容 GLM 裸格式: "tool_name\n{...}" / "tool_name\n<input>{...}</input>" / "tool_name\n```json\n{...}\n```"
            bare_calls = ToolParser._parse_bare_format(stripped)
            if bare_calls:
                return bare_calls
            return calls

        tolerant_calls: List[Tuple[str, Dict[str, Any]]] = []
        for idx, tool_match in enumerate(tool_matches):
            tool_name = tool_match.group(1)
            segment_start = tool_match.end()
            segment_end = tool_matches[idx + 1].start() if idx + 1 < len(tool_matches) else len(text)
            segment = text[segment_start:segment_end]

            input_start_match = re.search(r"<input>", segment)
            if not input_start_match:
                continue

            payload_start = input_start_match.end()
            payload_segment = segment[payload_start:]
            payload_segment = re.sub(r"</input>\s*$", "", payload_segment, flags=re.DOTALL).strip()

            input_data = ToolParser._parse_input_payload(payload_segment)
            if input_data is not None:
                tolerant_calls.append((tool_name, input_data))

        if tolerant_calls:
            return tolerant_calls

        # ===== 格式8：最宽松裸格式（兜底解析）=====
        # 适用于模型输出不规范，工具名和JSON参数混在一起的情况
        for tool in known_tools:
            if tool in stripped:
                tool_idx = stripped.find(tool)
                json_start = stripped.find('{', tool_idx)
                if json_start != -1:
                    # 从JSON起始位置向后尝试解析
                    for json_end in range(len(stripped), json_start, -1):
                        try:
                            candidate = stripped[json_start:json_end]
                            args = ToolParser._parse_input_payload(candidate)
                            if args and isinstance(args, dict):
                                return [(tool, args)]
                        except Exception:
                            continue
        # ===== 格式8结束 =====

        # 如果所有格式都解析失败，记录日志
        if not calls:
            print(f"⚠️ 工具解析失败 - 模型输出（前200字符）: {stripped[:200]}")

        return calls

    @staticmethod
    def _parse_bare_format(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        解析 GLM 等模型输出的裸格式工具调用，支持以下变体:

        变体1 - 纯 JSON（工具名独占一行）:
          read_file
          {"path": "README.md"}

        变体2 - GLM 混合格式（有 <input> 标签但没有 <tool> 标签）:
          read_file
          <input>
          {"path": "README.md"}
          </input>

        变体3 - markdown 代码块:
          read_file
          ```json
          {"path": "README.md"}
          ```

        变体4 - 单行混合:
          read_file\n<input>{"path": "README.md"}</input>

        变体5 - 工具名跟在自然语言末尾（GLM plan_mode 常见输出）:
          ...将开始执行计划中的第一步。read_file
          <input>{"path": "README.md"}</input>
        """
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
        calls: List[Tuple[str, Dict[str, Any]]] = []

        # ----- 优先尝试: 工具名 + <input>...</input> 混合格式（含变体5）-----
        # 修复：去掉 ^ 锚点，允许工具名前有任意文字（如"...第一步。read_file" / "执行计划：read_file"）
        # 使用 lookbehind 匹配工具名前是行首/空白/中文标点（含全角冒号、全角标点区 \uff00-\uffef）
        mixed_pattern = re.compile(
            r"(?:^|(?<=[\s。.，,、！!？?；;：:\uff00-\uffef\n]))("
            + "|".join(re.escape(t) for t in known_tools)
            + r")\s*\n\s*<input>\s*(.*?)\s*</input>",
            re.DOTALL | re.MULTILINE,
        )
        for m in mixed_pattern.finditer(text):
            tool_name = m.group(1).strip()
            payload = m.group(2).strip()
            # 去掉可能的 ```json 包裹
            payload = re.sub(r"^```(?:json)?\s*", "", payload)
            payload = re.sub(r"\s*```$", "", payload)
            input_data = ToolParser._parse_input_payload(payload)
            if input_data is not None:
                calls.append((tool_name, input_data))
        if calls:
            return calls

        # ----- 退回: 逐行扫描，收集 JSON 块 -----
        lines = text.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 精确整行匹配（原有逻辑）
            tool_name_candidate = line if line in known_tools else None

            # 变体5：工具名紧跟在行末（前面是自然语言，如 "...第一步。read_file" / "执行计划：read_file"）
            if tool_name_candidate is None:
                for t in known_tools:
                    # 工具名在行末，且前一个字符是标点/空白/中文字符（不是字母数字）
                    # 全角标点：。，、！？；：""''（）【】《》——……
                    if line.endswith(t):
                        prefix = line[: len(line) - len(t)]
                        if not prefix or re.search(
                            r'[\s。.，,、！!？?；;：:\uff00-\uffef\u4e00-\u9fff]$',
                            prefix
                        ):
                            tool_name_candidate = t
                            break

            if tool_name_candidate:
                tool_name = tool_name_candidate
                # 收集后续内容（跳过 <input>/<input> 标签、空行、``` 标记）
                json_lines = []
                cmd_lines = []   # bash 裸命令行（非 JSON 格式）
                j = i + 1
                while j < len(lines):
                    l = lines[j].strip()
                    # 跳过 <input> / </input> / ``` 标记行
                    if l in ("<input>", "</input>") or l.startswith("```"):
                        j += 1
                        continue
                    if l == "" and not json_lines and not cmd_lines:
                        j += 1
                        continue
                    if l.startswith("{") or json_lines:
                        # 标准 JSON 格式
                        json_lines.append(lines[j])
                        combined = "\n".join(json_lines)
                        if combined.count("{") > 0 and combined.count("{") == combined.count("}"):
                            break
                    elif tool_name == "bash" and not json_lines:
                        # bash 裸命令格式：模型直接输出命令行而非 JSON
                        # 例：bash\nfind core -name "*.py" | ...
                        # 收集到空行或下一个工具名为止
                        if l == "" or l in known_tools:
                            break
                        cmd_lines.append(lines[j])
                    else:
                        break
                    j += 1

                if json_lines:
                    payload = "\n".join(json_lines).strip()
                    input_data = ToolParser._parse_input_payload(payload)
                    if input_data is not None:
                        calls.append((tool_name, input_data))
                        i = j + 1
                        continue
                elif cmd_lines and tool_name == "bash":
                    # 将裸命令行包装为标准 {"command": "..."} 格式
                    command = "\n".join(cmd_lines).strip()
                    calls.append(("bash", {"command": command}))
                    i = j
                    continue
            i += 1

        return calls


# ============================================================================
# 可扩展工具注册表 - 借鉴 DeerFlow tools/tools.py 的插件化设计
# ============================================================================

class ToolRegistry:
    """
    工具注册表：支持动态注册自定义工具。

    设计哲学（借鉴 DeerFlow community/ 模块的工具扩展架构）：
    ──────────────────────────────────────────────────────────────────
    1. 内置工具由 ToolExecutor 管理（read_file, write_file 等）
    2. 扩展工具通过 ToolRegistry 动态注册（search_web, image_search 等）
    3. 注册的工具描述自动合并到 ToolExecutor.get_tools() 中
    4. 每个扩展工具需提供：描述（供 LLM 使用）+ 执行函数

    典型用法：
        registry = ToolRegistry()
        registry.register(
            name="search_web",
            description="搜索互联网",
            input_schema={...},
            handler=lambda input: {"results": [...]}
        )
        executor.merge_registry(registry)
    ──────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler,
        enabled: bool = True,
    ) -> None:
        """
        注册一个自定义工具。

        Args:
            name: 工具名称（唯一标识，不能与内置工具重名）
            description: 工具描述（供 LLM 理解何时调用）
            input_schema: 工具输入的 JSON Schema
            handler: 执行函数，签名 fn(tool_input: dict) -> str（返回 JSON 字符串）
            enabled: 是否启用该工具
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler": handler,
            "enabled": enabled,
        }

    def disable(self, name: str) -> None:
        """禁用指定工具（不删除，仅标记为 disabled）。"""
        if name in self._tools:
            self._tools[name]["enabled"] = False

    def enable(self, name: str) -> None:
        """重新启用指定工具。"""
        if name in self._tools:
            self._tools[name]["enabled"] = True

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """返回所有已启用工具的描述列表（供 ToolExecutor 合并）。"""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self._tools.values()
            if t.get("enabled", True)
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
        """
        执行注册的工具。

        Returns:
            工具执行结果（JSON 字符串），若工具不存在返回 None
        """
        tool = self._tools.get(tool_name)
        if not tool or not tool.get("enabled", True):
            return None
        try:
            result = tool["handler"](tool_input)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    def list_tools(self) -> List[str]:
        """返回所有已注册工具的名称列表。"""
        return list(self._tools.keys())


def create_web_search_tool_placeholder() -> Dict[str, Any]:
    """
    创建 web_search 工具的描述占位（借鉴 DeerFlow community/tavily 工具）。

    使用前需提供真实的搜索 API：
      - Tavily Search（https://tavily.com）
      - DuckDuckGo（无 API，需安装 duckduckgo-search）
      - 自定义搜索服务

    集成示例：
        from duckduckgo_search import DDGS
        def ddg_search(input):
            results = list(DDGS().text(input["query"], max_results=5))
            return json.dumps({"results": results})

        registry = ToolRegistry()
        registry.register(
            name="search_web",
            description=create_web_search_tool_placeholder()["description"],
            input_schema=create_web_search_tool_placeholder()["input_schema"],
            handler=ddg_search,
        )
    """
    return {
        "name": "search_web",
        "description": (
            "搜索互联网获取最新信息。用于查找当前事件、技术文档、产品信息等网络内容。"
            "注意：此工具为占位，需配置真实搜索 API 才能使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量（默认 5）",
                    "default": 5,
                }
            },
            "required": ["query"]
        },
    }


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("📋 Agent 工具系统自检")
    print("=" * 60)

    # 1. ToolExecutor - 列出所有内置工具
    executor = ToolExecutor(enable_bash=False)
    print(f"\n🔧 内置工具（共 {len(executor.get_tools())} 个）:")
    for tool in executor.get_tools():
        print(f"  ✓ {tool['name']}: {tool['description'][:50]}...")

    # 2. ToolParser - 验证工具调用解析
    print("\n🔍 ToolParser 解析测试:")
    test_cases = [
        ('<tool>read_file</tool><input>{"path": "README.md"}</input>', "标准 XML 格式"),
        ('[{"tool": "bash", "input": {"command": "ls -la"}}]', "JSON 数组格式"),
        ('read_file\n{"path": "test.py"}', "裸格式（GLM 兼容）"),
    ]
    for text, label in test_cases:
        calls = ToolParser.parse_tool_calls(text)
        status = f"✓ 解析到 {len(calls)} 个调用" if calls else "✗ 未解析到调用"
        print(f"  [{label}] {status}")

    # 3. ToolExecutor - 实际文件工具调用
    print("\n📂 文件工具功能测试:")
    with tempfile.TemporaryDirectory() as tmpdir:
        exec2 = ToolExecutor(work_dir=tmpdir, enable_bash=False)
        write_result = json.loads(exec2.execute_tool("write_file", {"path": "hello.txt", "content": "Hello, Agent!"}))
        read_result  = json.loads(exec2.execute_tool("read_file",  {"path": "hello.txt"}))
        print(f"  write_file: success={write_result.get('success')}")
        print(f"  read_file:  content={read_result.get('content', '')!r}")

    # 4. ToolRegistry - 注册和调用扩展工具
    print("\n📦 ToolRegistry 扩展工具演示:")
    registry = ToolRegistry()
    placeholder = create_web_search_tool_placeholder()
    registry.register(
        name=placeholder["name"],
        description=placeholder["description"],
        input_schema=placeholder["input_schema"],
        handler=lambda inp: json.dumps({"note": "占位工具，未配置真实搜索 API", "query": inp.get("query")}),
    )
    result = registry.execute(placeholder["name"], {"query": "Python Agent 框架"})
    print(f"  已注册: {registry.list_tools()}")
    print(f"  执行结果: {result}")

    print("\n✅ 工具系统自检通过")

