#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 工具系统 - 为 Qwen 模型提供工具支持"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ToolExecutor:
    def __init__(self, work_dir: Optional[str] = None, enable_bash: bool = True):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.enable_bash = enable_bash

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if "error" in tool_input and "path" not in tool_input:
                return json.dumps({"success": False, "error": tool_input["error"]}, ensure_ascii=False)
            if tool_name == "read_file":
                if "path" not in tool_input:
                    return json.dumps({"success": False, "error": "缺少必需参数 'path'"}, ensure_ascii=False)
                return self._read_file(tool_input["path"])
            elif tool_name == "write_file":
                if "path" not in tool_input or "content" not in tool_input:
                    return json.dumps({"success": False, "error": "缺少必需参数 'path' 或 'content'"}, ensure_ascii=False)
                return self._write_file(tool_input["path"], tool_input["content"], mode=tool_input.get("mode", "overwrite"))
            elif tool_name == "edit_file":
                if "path" not in tool_input:
                    return json.dumps({"success": False, "error": "缺少必需参数 'path'"}, ensure_ascii=False)
                return self._edit_file(tool_input["path"], tool_input.get("old_content", ""), tool_input.get("new_content", ""))
            elif tool_name == "list_dir":
                return self._list_dir(tool_input.get("path", "."))
            elif tool_name == "bash" and self.enable_bash:
                if "command" not in tool_input:
                    return json.dumps({"success": False, "error": "缺少必需参数 'command'"}, ensure_ascii=False)
                return self._bash(tool_input["command"])
            else:
                return json.dumps({"error": f"未知工具: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _fuzzy_find_file(self, filename: str, search_home: bool = True) -> Optional[Path]:
        """模糊搜索文件。

        安全约束：
          - 只在 work_dir 内部（最大深度8）或 home 目录的直接子目录内搜索
          - 不跟随符号链接跳出沙箱
          - 解析后的真实路径必须以允许根目录开头，防止路径遍历攻击
        """
        SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                     ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
                     ".idea", ".vscode"}
        # 只取文件名，丢弃任何目录组件（防止 '../../../etc/passwd' 这类输入）
        target_name = Path(filename).name
        if not target_name or target_name in ("..", "."):
            return None

        work_dir_real = self.work_dir.resolve()

        def _is_safe(path: Path, allowed_root: Path) -> bool:
            """确认 path 的真实路径在 allowed_root 内部。"""
            try:
                path.resolve().relative_to(allowed_root)
                return True
            except ValueError:
                return False

        def _scan(root: Path, depth_limit: int = 8) -> Optional[Path]:
            if depth_limit < 0:
                return None
            try:
                for item in sorted(root.iterdir()):
                    if item.name in SKIP_DIRS or item.name.startswith("."):
                        continue
                    # 不跟随符号链接（follow_symlinks=False 需用 lstat）
                    if item.is_symlink():
                        continue
                    if item.is_file() and item.name == target_name:
                        # 双重安全：真实路径必须在 work_dir 内
                        if _is_safe(item, work_dir_real):
                            return item
                    elif item.is_dir() and depth_limit > 0:
                        found = _scan(item, depth_limit - 1)
                        if found:
                            return found
            except (PermissionError, OSError):
                pass
            return None

        result = _scan(self.work_dir)
        if result:
            return result

        # 在 home 目录的直接子目录中搜索（限制为非系统目录）
        if search_home:
            home = Path.home()
            home_real = home.resolve()
            try:
                for item in sorted(home.iterdir()):
                    if item.name in SKIP_DIRS or item.name.startswith("."):
                        continue
                    if item.is_symlink():
                        continue
                    item_real = item.resolve()
                    # 跳过 work_dir 本身（已搜索过）
                    if item_real == work_dir_real:
                        continue
                    if item.is_file() and item.name == target_name:
                        if _is_safe(item, home_real):
                            return item
                    if item.is_dir():
                        # home 内搜索只在该直接子目录（及其子树）中，限制深度
                        sub_real = item_real
                        def _scan_sub(root2: Path, depth: int = 6) -> Optional[Path]:
                            if depth < 0:
                                return None
                            try:
                                for sub in sorted(root2.iterdir()):
                                    if sub.name in SKIP_DIRS or sub.name.startswith("."):
                                        continue
                                    if sub.is_symlink():
                                        continue
                                    if sub.is_file() and sub.name == target_name:
                                        if _is_safe(sub, home_real):
                                            return sub
                                    elif sub.is_dir() and depth > 0:
                                        found2 = _scan_sub(sub, depth - 1)
                                        if found2:
                                            return found2
                            except (PermissionError, OSError):
                                pass
                            return None
                        found = _scan_sub(item, 6)
                        if found:
                            return found
            except (PermissionError, OSError):
                pass
        return None

    def _read_file(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                    result = {"success": True, "path": str(p), "content": content}
                    if not content.strip():
                        result["warning"] = "文件存在但内容为空"
                    return json.dumps(result)
                except Exception as e:
                    return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"})
            else:
                return json.dumps({"success": False, "error": f"文件不存在: {path}", "hint": "绝对路径指定的文件不存在，请先用 list_dir 确认路径。"})
        file_path = self.work_dir / path
        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                result = {"success": True, "path": str(file_path.resolve()), "content": content}
                if not content.strip():
                    result["warning"] = "文件存在但内容为空"
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"})
        filename = Path(path).name
        found = self._fuzzy_find_file(filename, search_home=True)
        if found:
            try:
                with open(found, "r", encoding="utf-8") as f:
                    content = f.read()
                return json.dumps({"success": True, "path": str(found), "fuzzy_match": True, "original_request": path, "note": f"原路径 '{path}' 不存在，自动匹配到: {found}", "content": content})
            except Exception as e:
                return json.dumps({"error": f"读取文件失败: {str(e)}"})
        return json.dumps({"success": False, "error": f"文件不存在: {path}", "searched_name": filename, "hint": f"在工作目录和家目录的全部子目录中均未找到名为 '{filename}' 的文件。请先用 list_dir 查看目录结构，确认文件名和路径。"})

    def _write_file(self, path: str, content: str, mode: str = "overwrite") -> str:
        p = Path(path)
        file_path = p if p.is_absolute() else self.work_dir / path
        if not p.is_absolute():
            try:
                file_path.resolve().relative_to(self.work_dir.resolve())
            except ValueError:
                return json.dumps({"error": "路径超出工作目录范围"})
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                total_size = file_path.stat().st_size
                return json.dumps({"success": True, "path": path, "mode": "append", "appended_size": len(content), "total_size": total_size})
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return json.dumps({"success": True, "path": path, "mode": "overwrite", "size": len(content)})
        except Exception as e:
            return json.dumps({"error": f"写入文件失败: {str(e)}"})

    def _edit_file(self, path: str, old_content: str, new_content: str) -> str:
        p = Path(path)
        file_path = p if p.is_absolute() else self.work_dir / path
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
                return json.dumps({"error": "原始内容未在文件中找到", "hint": "请确保原始内容完全匹配(包括空格和换行)"})
            new_file_content = content.replace(old_content, new_content, 1)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_file_content)
            return json.dumps({"success": True, "path": path, "replaced": True})
        except Exception as e:
            return json.dumps({"error": f"编辑文件失败: {str(e)}"})

    def _list_dir(self, path: str = ".") -> str:
        p = Path(path)
        dir_path = p if p.is_absolute() else self.work_dir / path
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
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue
                is_dir = item.is_dir()
                items.append({"name": item.name, "type": "dir" if is_dir else "file", "size": item.stat().st_size if not is_dir else None})
                if is_dir:
                    subdirs.append(item.name)
            result = {"success": True, "path": path, "items": items}
            if subdirs:
                result["hint"] = f"以下子目录尚未展开，若目标文件不在当前层，请继续调用 list_dir 探索子目录: {subdirs}"
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"列出目录失败: {str(e)}"})

    # 允许执行的命令前缀白名单（防止高危命令注入）
    _BASH_BLOCKED_PATTERNS = [
        r"\brm\s+-rf\s+/",           # rm -rf /
        r"\bmkfs\b",                  # 格式化
        r"\bdd\s+if=.+of=/dev/",      # dd 写磁盘
        r":\s*\(\s*\)\s*\{.*fork\s+bomb",  # fork bomb
        r"chmod\s+-R\s+[0-7]*7[0-7]*\s+/",  # 全局改权限
        r"\bcurl\b.+\|\s*bash",       # 管道执行远程脚本
        r"\bwget\b.+\|\s*bash",
    ]

    def _bash(self, command: str) -> str:
        """执行 Shell 命令。

        安全改进：
          - 使用 shlex.split + shell=False 避免 Shell 注入（当命令为纯字符串时）
          - 对已知高危命令模式进行阻断
          - cwd 限定在 work_dir，防止 cd / 后操作系统文件
        """
        import re as _re
        import shlex

        if not self.enable_bash:
            return json.dumps({"error": "bash 工具已禁用"})

        # 高危模式拦截
        for pat in self._BASH_BLOCKED_PATTERNS:
            if _re.search(pat, command, _re.I):
                return json.dumps({"error": f"命令被安全策略拒绝（匹配高危模式）: {command[:80]}"})

        try:
            # 尝试以 shell=False 方式执行，更安全
            try:
                args = shlex.split(command)
            except ValueError:
                # 包含引号不匹配等情况，回退到 shell=True（仍在 work_dir 下）
                args = None

            if args:
                result = subprocess.run(
                    args,
                    shell=False,
                    cwd=str(self.work_dir),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                # 包含管道/重定向等 shell 特性，必须用 shell=True
                result = subprocess.run(
                    command,
                    shell=True,  # noqa: S603  # 仅在 shlex 失败时退化
                    cwd=str(self.work_dir),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            return json.dumps({
                "success": result.returncode == 0,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "命令执行超时(30秒)"})
        except FileNotFoundError as e:
            return json.dumps({"error": f"命令不存在: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"执行命令失败: {str(e)}"})


class ToolParser:
    @staticmethod
    def _parse_input_payload(input_str: str) -> Optional[Dict[str, Any]]:
        if not input_str:
            return None
        payload = input_str.strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        if payload.startswith("{"):
            missing_right_brace = payload.count("{") - payload.count("}")
            if missing_right_brace > 0:
                repaired = payload + ("}" * missing_right_brace)
                try:
                    data = json.loads(repaired)
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    pass
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
                                result.append(s[i])
                            else:
                                result.append('\\\\')
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
        calls = []
        stripped = text.strip()
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}

        try:
            if stripped.startswith("["):
                data = json.loads(stripped)
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        calls.append((item["tool"], item["input"]))
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
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return [(item["name"], item["params"])]
                    elif "api" in item and isinstance(item["api"], str) and item["api"] in known_tools:
                        tool_name = item["api"]
                        tool_args = {k: v for k, v in item.items() if k != "api"}
                        return [(tool_name, tool_args)]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

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
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return [(item["name"], item["params"])]
            except (json.JSONDecodeError, TypeError):
                pass

        _inline_block_pattern = re.compile(r'```(?:json|plaintext|python|bash|shell|text)?\s*\n([\s\S]*?)\s*```', re.MULTILINE)
        for _m in _inline_block_pattern.finditer(stripped):
            _inner = _m.group(1).strip()
            try:
                _item = json.loads(_inner)
            except (json.JSONDecodeError, TypeError):
                _item = None
            if not isinstance(_item, dict):
                continue
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
            if "tool" in _item or "name" in _item:
                continue
            _keys = set(_item.keys())
            _inferred_tool = None
            if "command" in _keys:
                _inferred_tool = "bash"
            elif {"path", "old_content", "new_content"}.issubset(_keys):
                _inferred_tool = "edit_file"
            elif {"path", "content"}.issubset(_keys):
                _inferred_tool = "write_file"
            elif _keys == {"path"}:
                _window_start = max(0, _m.start() - 150)
                _prefix = stripped[_window_start: _m.start()].lower()
                _list_hints_strict = ("列出", "列目录", "list_dir", "listdir", "目录结构", "查看目录", "浏览目录")
                _read_hints = ("读取", "读文件", "read_file", "阅读", "查看文件", "打开文件", "文件内容", "read")
                if any(h in _prefix for h in _list_hints_strict):
                    _inferred_tool = "list_dir"
                else:
                    _inferred_tool = "read_file"
            if _inferred_tool and _inferred_tool in known_tools:
                calls.append((_inferred_tool, _item))
        if calls:
            return calls

        for tool in known_tools:
            pattern = re.compile(r'(?:^|(?<=[\s。.，,、！!？?；;：:\n]))' + re.escape(tool) + r'\s*\n\s*```(?:json)?\s*\n([\s\S]*?)\s*```', re.MULTILINE)
            for m in pattern.finditer(stripped):
                inner = m.group(1).strip()
                data = ToolParser._parse_input_payload(inner)
                if data is not None:
                    calls.append((tool, data))
            if calls:
                return calls

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

        tool_matches = list(re.finditer(tool_pattern, text))
        if not tool_matches:
            bare_calls = ToolParser._parse_bare_format(stripped)
            if bare_calls:
                return bare_calls
            return calls
        tolerant_calls = []
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

        for tool in known_tools:
            if tool in stripped:
                tool_idx = stripped.find(tool)
                json_start = stripped.find('{', tool_idx)
                if json_start != -1:
                    for json_end in range(len(stripped), json_start, -1):
                        try:
                            candidate = stripped[json_start:json_end]
                            args = ToolParser._parse_input_payload(candidate)
                            if args and isinstance(args, dict):
                                return [(tool, args)]
                        except Exception:
                            continue
        if not calls:
            print(f"⚠️ 工具解析失败 - 模型输出（前200字符）: {stripped[:200]}")
        return calls

    @staticmethod
    def _parse_bare_format(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write"}
        calls = []
        mixed_pattern = re.compile(r"(?:^|(?<=[\s。.，,、！!？?；;：:\uff00-\uffef\n]))(" + "|".join(re.escape(t) for t in known_tools) + r")\s*\n\s*<input>\s*(.*?)\s*</input>", re.DOTALL | re.MULTILINE)
        for m in mixed_pattern.finditer(text):
            tool_name = m.group(1).strip()
            payload = m.group(2).strip()
            payload = re.sub(r"^```(?:json)?\s*", "", payload)
            payload = re.sub(r"\s*```$", "", payload)
            input_data = ToolParser._parse_input_payload(payload)
            if input_data is not None:
                calls.append((tool_name, input_data))
        if calls:
            return calls
        # 预处理：将「JSON}工具名」粘连在同一行的情况拆分为多行
        # 例如 '{"path": "a.py"}bash' → '{"path": "a.py"}' + 'bash'
        _tool_suffix_re = re.compile(
            r'(\{[^{}]*\})\s*(' + '|'.join(re.escape(t) for t in sorted(known_tools, key=len, reverse=True)) + r')\s*$'
        )
        raw_lines = text.strip().splitlines()
        lines: list = []
        for _ln in raw_lines:
            _m = _tool_suffix_re.search(_ln)
            if _m and _ln.strip().startswith('{'):
                # JSON 块后面紧跟工具名，拆开
                lines.append(_m.group(1))
                lines.append(_m.group(2))
            else:
                lines.append(_ln)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            tool_name_candidate = line if line in known_tools else None
            if tool_name_candidate is None:
                for t in known_tools:
                    if line.endswith(t):
                        prefix = line[: len(line) - len(t)]
                        if not prefix or re.search(r'[\s。.，,、！!？?；;：:\uff00-\uffef\u4e00-\u9fff]$', prefix):
                            tool_name_candidate = t
                            break
            if tool_name_candidate:
                tool_name = tool_name_candidate
                json_lines = []
                cmd_lines = []
                j = i + 1
                while j < len(lines):
                    l = lines[j].strip()
                    if l in ("<input>", "</input>") or l.startswith("```"):
                        j += 1
                        continue
                    if l == "" and not json_lines and not cmd_lines:
                        j += 1
                        continue
                    if l.startswith("{") or json_lines:
                        json_lines.append(lines[j])
                        combined = "\n".join(json_lines)
                        if combined.count("{") > 0 and combined.count("{") == combined.count("}"):
                            break
                    elif tool_name == "bash" and not json_lines:
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
                    command = "\n".join(cmd_lines).strip()
                    calls.append(("bash", {"command": command}))
                    i = j
                    continue
            i += 1
        return calls


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, input_schema: Dict[str, Any], handler, enabled: bool = True):
        self._tools[name] = {"name": name, "description": description, "input_schema": input_schema, "handler": handler, "enabled": enabled}

    def disable(self, name: str):
        if name in self._tools:
            self._tools[name]["enabled"] = False

    def enable(self, name: str):
        if name in self._tools:
            self._tools[name]["enabled"] = True

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in self._tools.values() if t.get("enabled", True)]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
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
        return list(self._tools.keys())


def create_web_search_tool_placeholder() -> Dict[str, Any]:
    return {
        "name": "search_web",
        "description": "搜索互联网获取最新信息。用于查找当前事件、技术文档、产品信息等网络内容。注意：此工具为占位，需配置真实搜索 API 才能使用。",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询词"}, "max_results": {"type": "integer", "description": "返回结果数量（默认 5）", "default": 5}}, "required": ["query"]},
    }