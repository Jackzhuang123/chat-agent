#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 工具系统 - 为 Qwen 模型提供工具支持"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time as _time_module
from core.monitor_logger import get_monitor_logger

class ToolExecutor:
    def __init__(self, work_dir: Optional[str] = None, enable_bash: bool = True):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.enable_bash = enable_bash
        self.monitor = get_monitor_logger()

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
                bash_timeout = int(tool_input.get("timeout", 30))
                return self._bash(tool_input["command"], timeout=bash_timeout)
            elif tool_name == "execute_python":
                if "code" not in tool_input:
                    return json.dumps({"success": False, "error": "缺少必需参数 'code'"}, ensure_ascii=False)
                timeout = int(tool_input.get("timeout", 30))
                return self.execute_python(tool_input["code"], timeout=timeout)
            else:
                return json.dumps({"error": f"未知工具: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _fuzzy_find_file(self, filename: str, search_home: bool = True) -> Optional[Path]:
        """模糊搜索文件。"""
        SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                     ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
                     ".idea", ".vscode", "site-packages", "dist-info",
                     "lib", "bin"}
        target_name = Path(filename).name
        if not target_name or target_name in ("..", "."):
            return None

        work_dir_real = self.work_dir.resolve()

        def _is_safe(path: Path, allowed_root: Path) -> bool:
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
                    if item.is_symlink():
                        continue
                    if item.is_file() and item.name == target_name:
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
                    if item_real == work_dir_real:
                        continue
                    if item.is_file() and item.name == target_name:
                        if _is_safe(item, home_real):
                            return item
                    if item.is_dir():
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

    _BLOCKED_PATH_PATTERNS = (
        "/.venv/", "/venv/", "/site-packages/", "/.dist-info/",
        "/__pycache__/", "/.git/", "/node_modules/",
        ".venv/", "venv/", "site-packages/", ".dist-info/",
        "__pycache__/", ".git/", "node_modules/",
    )

    def _read_file(self, path: str) -> str:
        # 路径拦截 (保持原有逻辑不变)
        _norm = path.replace("\\", "/")
        for _blocked in self._BLOCKED_PATH_PATTERNS:
            if _blocked in _norm:
                return json.dumps({
                    "success": False,
                    "error": f"⛔ 路径被拦截：'{path}' 位于受限目录（{_blocked.strip('/')}），与当前任务无关。",
                    "hint": "请只读取项目源码文件（core/、ui/、skills/ 等目录），不要读取依赖库或系统文件。"
                }, ensure_ascii=False)

        p = Path(path)

        # ----- 绝对路径分支 -----
        if p.is_absolute():
            # 若文件存在且不是目录，正常读取
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                    result = {"success": True, "path": str(p), "content": content}
                    if not content.strip():
                        result["warning"] = "文件存在但内容为空"
                    return json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)
            else:
                # ⭐ 新增：绝对路径不存在时，尝试模糊搜索
                filename = p.name
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
                            "note": f"原绝对路径 '{path}' 不存在，自动匹配到: {found}",
                            "content": content
                        }, ensure_ascii=False)
                    except Exception as e:
                        return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)
                # 模糊搜索也未找到
                return json.dumps({
                    "success": False,
                    "error": f"文件不存在: {path}",
                    "hint": "绝对路径指定的文件不存在，且项目内未找到同名文件。请仅提供文件名（如 'session_analyzer.py'），系统会自动搜索。"
                }, ensure_ascii=False)

        # ----- 相对路径分支 (原有逻辑保持不变) -----
        file_path = self.work_dir / path
        if file_path.exists() and file_path.is_dir():
            return json.dumps({
                "success": False,
                "error": f"目标是目录而非文件: {path}",
                "hint": "请改用 list_dir 查看目录内容。",
                "suggested_tool": "list_dir",
                "suggested_input": {"path": path}
            }, ensure_ascii=False)
        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                result = {"success": True, "path": str(file_path.resolve()), "content": content}
                if not content.strip():
                    result["warning"] = "文件存在但内容为空"
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)

        # 相对路径不存在时的模糊搜索 (原有逻辑)
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
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)

        return json.dumps({
            "success": False,
            "error": f"文件不存在: {path}",
            "searched_name": filename,
            "hint": (
                f"在工作目录和全局搜索中均未找到 '{filename}'。"
                f"⚠️ 请立即用 bash find 全局搜索，不要放弃：\n"
                f"  bash {{\"command\": \"find . -name '{filename}' 2>/dev/null\"}}\n"
                f"若文件名可能不同，可模糊搜索：\n"
                f"  bash {{\"command\": \"find . -name '*{Path(filename).stem}*' 2>/dev/null\"}}"
            ),
            "next_action": f"bash {{\"command\": \"find . -name '{filename}' 2>/dev/null\"}}"
        }, ensure_ascii=False)

    def _write_file(self, path: str, content: str, mode: str = "overwrite") -> str:
        p = Path(path)
        file_path = p if p.is_absolute() else self.work_dir / path
        if not p.is_absolute():
            try:
                file_path.resolve().relative_to(self.work_dir.resolve())
            except ValueError:
                return json.dumps({"error": "路径超出工作目录范围"})

        if mode != "append" and content == "" and file_path.exists():
            existing_size = file_path.stat().st_size
            if existing_size > 0:
                return json.dumps({
                    "success": False,
                    "blocked": True,
                    "error": (
                        f"⛔ 拦截危险操作：write_file 试图用空内容覆盖已有内容的文件 {path}"
                        f"（当前大小 {existing_size} 字节）。"
                        "文件内容可能已由 bash 重定向写入，禁止再次用 write_file 空覆盖。"
                        f" 若要验证内容，请用 read_file 或 bash head 命令查看。"
                    ),
                    "fix_hint": "不要再对此文件调用 write_file 空覆盖，内容已正确写入磁盘。直接进行下一步任务。",
                }, ensure_ascii=False)

        _PLACEHOLDER_PATTERNS = [
            r'<完整复制[^>]*>',
            r'<复制[^>]*>',
            r'<粘贴[^>]*>',
            r'<填入[^>]*>',
            r'<插入[^>]*>',
            r'<参见[^>]*>',
            r'<见上方[^>]*>',
            r'\[此处填写[^\]]*\]',
            r'\[TODO[^\]]*\]',
            r'<TODO[^>]*>',
        ]
        for _pat in _PLACEHOLDER_PATTERNS:
            if re.search(_pat, content):
                return json.dumps({
                    "success": False,
                    "blocked": True,
                    "error": (
                        f"⛔ 拦截占位符写入：write_file 的 content 参数包含占位符文本 "
                        f"（匹配模式: {_pat}），而非实际内容。"
                        "请将实际内容直接填入 content 参数，不要写占位符说明。"
                    ),
                    "fix_hint": (
                        "正确做法：先 read_file 读取源数据，然后在 write_file 的 content "
                        "参数中填入从工具结果中提取的实际内容，而非写'<完整复制上方输出>'。"
                        "或者用 execute_python 读取数据并格式化后写入文件。"
                    ),
                }, ensure_ascii=False)

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

    def execute_python(self, code: str, timeout: int = 30) -> str:
        import sys as _sys
        import re as _re
        import subprocess
        import json
        import os

        _has_write = bool(_re.search(
            r'open\s*\([^)]*[\'\"].*[\'\"],\s*[\'\"]w[\'\"]'
            r'|\.write\s*\(|write_file|to_csv|to_json|to_excel'
            r'|shutil\.copy|os\.rename',
            code
        ))
        _has_print = bool(_re.search(r'\bprint\s*\(', code))
        if _has_write and not _has_print:
            code = code + '\nprint("✅ 文件写入操作已完成")\n'

        _PRELUDE = (
            # 保持原有 prelude 不变
            "import sys, traceback, json, math, re, os, os.path as _osp\n"
            "import datetime, collections, itertools, functools, random, statistics\n"
            "import subprocess, shutil, pathlib\n"
            "from io import StringIO\n"
            "from collections import defaultdict, Counter, OrderedDict, deque\n"
            "from itertools import chain, product, combinations, permutations\n"
            "from datetime import date, timedelta\n"
            "from pathlib import Path\n"
            "path = _osp\n"
        )
        wrapper = (
                _PRELUDE
                + "_out = StringIO()\n"
                  "_err = StringIO()\n"
                  "sys.stdout = _out\n"
                  "sys.stderr = _err\n"
                  "_result = {'success': False, 'stdout': '', 'stderr': '', 'error': ''}\n"
                  "try:\n"
                + "\n".join("    " + line for line in code.splitlines()) + "\n"
                + "    _result['success'] = True\n"
                  "except Exception as _e:\n"
                  "    _result['error'] = traceback.format_exc()\n"
                  "finally:\n"
                  "    sys.stdout = sys.__stdout__\n"
                  "    sys.stderr = sys.__stderr__\n"
                  "    _result['stdout'] = _out.getvalue()\n"
                  "    _result['stderr'] = _err.getvalue()\n"
                  "    print(json.dumps(_result, ensure_ascii=False))\n"
        )

        try:
            python_exe = _sys.executable or "python3"
            proc = subprocess.run(
                [python_exe, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_dir),
            )
            output_lines = proc.stdout.strip().splitlines()
            if output_lines:
                try:
                    inner = json.loads(output_lines[-1])
                    prefix_lines = output_lines[:-1]
                    if prefix_lines:
                        inner["stdout"] = "\n".join(prefix_lines) + "\n" + inner.get("stdout", "")
                    stdout_raw = inner.get("stdout", "").strip()
                    _STDOUT_LIMIT = 8000
                    if len(stdout_raw) > _STDOUT_LIMIT:
                        stdout_raw = stdout_raw[
                                         :_STDOUT_LIMIT] + f"\n…[输出过长，已截断，原始长度 {len(stdout_raw)} 字符]"
                    error_raw = inner.get("error", "").strip()[:3000]
                    stderr_raw = (inner.get("stderr", "") or proc.stderr or "").strip()[:2000]
                    success = inner.get("success", False)

                    # ⭐ 新增：写文件但产出可能为空的检测
                    if success and _has_write:
                        # 尝试检测目标文件是否真的写入了内容
                        write_paths = _re.findall(r'open\(["\']([^"\']+)["\']\s*,\s*["\']w["\']', code)
                        is_content_written = False
                        for wpath in write_paths:
                            target = Path(wpath)
                            if not target.is_absolute():
                                target = self.work_dir / target
                            if target.exists() and target.stat().st_size > 0:
                                is_content_written = True
                                break
                        if not is_content_written and not stdout_raw:
                            # 既没有控制台输出，文件也为空 -> 认为实际失败
                            success = False
                            inner["error"] = (
                                "代码执行未报错，但目标文件为空且无控制台输出。"
                                "可能原因：正则匹配无结果、写入逻辑未执行。"
                            )
                            inner["fix_hint"] = (
                                "❌ 任务实际未完成。建议改用 bash 命令完成相同操作，例如：\n"
                                "bash {\"command\": \"grep -E '^class |^def ' core/*.py > API.md\"}"
                            )

                    resp = {
                        "success": success,
                        "stdout": stdout_raw,
                        "stderr": stderr_raw,
                        "error": error_raw,
                        "returncode": proc.returncode,
                    }
                    if not success and "fix_hint" not in resp and inner.get("fix_hint"):
                        resp["fix_hint"] = inner["fix_hint"]
                    elif not success:
                        resp["fix_hint"] = (
                            "❌ 代码执行失败。请仔细阅读上方 error/stderr 中的错误信息，"
                            "修正代码后再次调用 execute_python 重新执行。"
                            "常见修复方向：检查变量名/缩进/语法、确认导入的模块已安装、"
                            "数据类型是否符合预期。"
                        )
                    return json.dumps(resp, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            # 兜底处理 (保持原有)
            stdout_fb = proc.stdout.strip()
            stderr_fb = proc.stderr.strip()
            if len(stdout_fb) > 8000:
                stdout_fb = stdout_fb[:8000] + f"\n…[输出过长，已截断，原始长度 {len(stdout_fb)} 字符]"
            success_fb = proc.returncode == 0
            resp_fb = {
                "success": success_fb,
                "stdout": stdout_fb,
                "stderr": stderr_fb[:2000],
                "returncode": proc.returncode,
            }
            if not success_fb:
                resp_fb["fix_hint"] = (
                    "❌ 代码执行失败（子进程异常退出）。请检查 stderr 中的错误信息，"
                    "修正代码后再次调用 execute_python 重新执行。"
                )
            return json.dumps(resp_fb, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            self.monitor.warning(f"execute_python 超时 (>{timeout}s)")
            return json.dumps({
                "success": False,
                "error": f"代码执行超时（>{timeout}s），请优化算法、减少数据量，或传入更大的 timeout 参数（如 timeout=120）。",
                "fix_hint": "⏱️ 执行超时。可尝试：1) 优化算法复杂度；2) 减少测试数据量；3) 再次调用时传入 timeout=120。",
                "returncode": -1,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"execute_python 内部错误: {str(e)}",
                "fix_hint": "❌ 工具自身出错（非代码问题）。请检查代码格式是否正确（如 JSON 中的换行转义），再次调用 execute_python。",
                "returncode": -1,
            }, ensure_ascii=False)

    _BASH_BLOCKED_PATTERNS = [
        r"\brm\s+-rf\s+/",
        r"\bmkfs\b",
        r"\bdd\s+if=.+of=/dev/",
        r":\s*\(\s*\)\s*\{.*fork\s+bomb",
        r"chmod\s+-R\s+[0-7]*7[0-7]*\s+/",
        r"\bcurl\b.+\|\s*bash",
        r"\bwget\b.+\|\s*bash",
    ]

    _BASH_STDOUT_LIMIT: int = 12000

    def _bash(self, command: str, timeout: int = 30) -> str:
        import re as _re
        import shlex

        if not self.enable_bash:
            return json.dumps({"error": "bash 工具已禁用"})

        for pat in self._BASH_BLOCKED_PATTERNS:
            if _re.search(pat, command, _re.I):
                self.monitor.warning(f"拦截高危 bash 命令: {command[:80]}")
                return json.dumps({"error": f"命令被安全策略拒绝（匹配高危模式）: {command[:80]}"})

        try:
            _SHELL_CHARS = ('|', '>', '<', '&&', '||', ';', '$', '`', '*', '?', '~', '{', '}', '!')
            _needs_shell = any(c in command for c in _SHELL_CHARS)

            if not _needs_shell:
                try:
                    args = shlex.split(command)
                except ValueError:
                    _needs_shell = True
                    args = None
            else:
                args = None

            start_time = _time_module.time()

            if args and not _needs_shell:
                result = subprocess.run(
                    args,
                    shell=False,
                    cwd=str(self.work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False
                )

            elapsed = _time_module.time() - start_time
            if elapsed > 5:
                self.monitor.info(f"bash 命令执行耗时 {elapsed:.2f}s: {command[:100]}")

            stdout = result.stdout
            stderr = result.stderr
            truncated = False
            if len(stdout) > self._BASH_STDOUT_LIMIT:
                stdout = (stdout[:self._BASH_STDOUT_LIMIT]
                          + f"\n…[stdout 过长已截断，原始长度 {len(result.stdout)} 字符，"
                          f"建议加 head -N 或缩小搜索范围]")
                truncated = True
            if len(stderr) > 3000:
                stderr = stderr[:3000] + "\n…[stderr 已截断]"

            resp = {
                "success": result.returncode == 0,
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
            }
            if truncated:
                resp["warning"] = "输出超出限制已截断，建议使用 grep -m N 或 head -N 限制行数"
            return json.dumps(resp)
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"命令执行超时({timeout}秒)"})
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

        if payload.startswith("{"):
            try:
                import re as _re_fix
                def _fix_nested_quotes(s: str) -> str:
                    _field_pattern = _re_fix.compile(r'"(code|command|content|script)"\s*:\s*"((?:[^"\\]|\\.)*)"', _re_fix.DOTALL)
                    def _escape_single_quotes_in_value(m):
                        field_name = m.group(1)
                        value = m.group(2)
                        fixed_value = value.replace("'", "\\'")
                        return f'"{field_name}": "{fixed_value}"'
                    repaired = _field_pattern.sub(_escape_single_quotes_in_value, s)
                    return repaired
                fixed2 = _fix_nested_quotes(payload)
                if fixed2 != payload:
                    data = json.loads(fixed2)
                    return data if isinstance(data, dict) else None
            except (json.JSONDecodeError, Exception):
                pass

        return None

    @staticmethod
    def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not isinstance(text, str):
            print(f"⚠️ 工具解析失败：输入类型错误 (期望 str，实际 {type(text).__name__})")
            return []

        calls = []
        stripped = text.strip()
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write", "execute_python"}

        # ---------- 1. 标准 JSON 格式 ----------
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
                    return ToolParser._normalize_args(calls)
            elif stripped.startswith("{"):
                item = json.loads(stripped)
                if isinstance(item, dict):
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        return ToolParser._normalize_args([(item["tool"], item["input"])])
                    elif "name" in item and "arguments" in item:
                        args = item["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            return ToolParser._normalize_args([(item["name"], args)])
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return ToolParser._normalize_args([(item["name"], item["params"])])
                    elif "api" in item and isinstance(item["api"], str) and item["api"] in known_tools:
                        tool_name = item["api"]
                        tool_args = {k: v for k, v in item.items() if k != "api"}
                        return ToolParser._normalize_args([(tool_name, tool_args)])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # ---------- 2. Markdown 代码块中的 JSON API 格式 ----------
        _md_api_pattern = re.compile(r'```(?:json)?\s*\n\s*(\{[\s\S]*?\})\s*\n?\s*```')
        for _mam in _md_api_pattern.finditer(stripped):
            try:
                _obj = json.loads(_mam.group(1))
                if isinstance(_obj, dict) and "api" in _obj and isinstance(_obj["api"], str) and _obj[
                    "api"] in known_tools:
                    _tname = _obj["api"]
                    _targs = {k: v for k, v in _obj.items() if k != "api"}
                    return ToolParser._normalize_args([(_tname, _targs)])
            except (json.JSONDecodeError, TypeError):
                pass

        # ---------- 3. Markdown 代码块包裹的 JSON 工具调用 ----------
        md_block_match = re.match(r'^```(?:json|python)?\s*\n([\s\S]*?)\s*```\s*$', stripped)
        if md_block_match:
            inner = md_block_match.group(1).strip()
            try:
                item = json.loads(inner)
                if isinstance(item, dict):
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        return ToolParser._normalize_args([(item["tool"], item["input"])])
                    elif "name" in item and "arguments" in item:
                        args = item["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            return ToolParser._normalize_args([(item["name"], args)])
                    elif "name" in item and "params" in item and isinstance(item["params"], dict):
                        return ToolParser._normalize_args([(item["name"], item["params"])])
            except (json.JSONDecodeError, TypeError):
                pass

        # ---------- 4. 内联代码块中的 JSON ----------
        _inline_block_pattern = re.compile(r'```(?:json|plaintext|python|bash|shell|text)?\s*\n([\s\S]*?)\s*```',
                                           re.MULTILINE)
        for _m in _inline_block_pattern.finditer(stripped):
            _inner = _m.group(1).strip()
            try:
                _item = json.loads(_inner)
            except (json.JSONDecodeError, TypeError):
                _item = None
            if not isinstance(_item, dict):
                continue
            if "tool" in _item and "input" in _item and isinstance(_item["input"], dict):
                return ToolParser._normalize_args([(_item["tool"], _item["input"])])
            elif "name" in _item and "arguments" in _item:
                args = _item["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if isinstance(args, dict):
                    return ToolParser._normalize_args([(_item["name"], args)])
            elif "name" in _item and "params" in _item and isinstance(_item["params"], dict):
                return ToolParser._normalize_args([(_item["name"], _item["params"])])
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
                if any(h in _prefix for h in _list_hints_strict):
                    _inferred_tool = "list_dir"
                else:
                    _inferred_tool = "read_file"
            if _inferred_tool and _inferred_tool in known_tools:
                calls.append((_inferred_tool, _item))
        if calls:
            return ToolParser._normalize_args(calls)

        # ---------- 5. 工具名 + 换行 + JSON 代码块 ----------
        for tool in known_tools:
            pattern = re.compile(
                r'(?:^|(?<=[\s。.，,、！!？?；;：:\n]))' + re.escape(tool) + r'\s*\n\s*```(?:json)?\s*\n([\s\S]*?)\s*```',
                re.MULTILINE)
            for m in pattern.finditer(stripped):
                inner = m.group(1).strip()
                data = ToolParser._parse_input_payload(inner)
                if data is not None:
                    calls.append((tool, data))
            if calls:
                return ToolParser._normalize_args(calls)

        # ---------- 6. XML 标签格式 ----------
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
                return ToolParser._normalize_args(calls)

        # ---------- 7. 宽松 XML 格式 ----------
        tool_matches = list(re.finditer(tool_pattern, text))
        if tool_matches:
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
                return ToolParser._normalize_args(tolerant_calls)

        # ---------- 8. 裸格式 <input> 标签 ----------
        bare_calls = ToolParser._parse_bare_format(stripped)
        if bare_calls:
            return ToolParser._normalize_args(bare_calls)

        # ---------- 9. 工具名后跟 JSON 对象（同一行或下一行） ----------
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
                                return ToolParser._normalize_args([(tool, args)])
                        except Exception:
                            continue

        # ========== 10. 非标准格式解析（增强重点） ==========
        if not calls:
            # 10.1 函数调用风格：tool_name("arg") 或 tool_name -c "code"
            func_patterns = [
                (r'execute_python\s+(?:-c|--code)\s+["\'](.*?)["\']', 'execute_python', 'code'),
                (r'read_file\s*\(\s*["\']([^"\']+)["\']\s*\)', 'read_file', 'path'),
                (r'write_file\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']*)["\'](?:\s*,\s*["\'](overwrite|append)["\'])?\s*\)',
                 'write_file', 'path', 'content', 'mode'),
                (r'list_dir\s*\(\s*["\']([^"\']*)["\']?\s*\)', 'list_dir', 'path'),
                (r'bash\s*\(\s*["\']([^"\']+)["\']\s*\)', 'bash', 'command'),
                (r'edit_file\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']\s*\)',
                 'edit_file', 'path', 'old_content', 'new_content'),
            ]
            for pattern, tool_name, *arg_keys in func_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    groups = match.groups()
                    args = {}
                    for i, key in enumerate(arg_keys):
                        if i < len(groups) and groups[i] is not None:
                            if key == 'code':
                                code_str = groups[i]
                                try:
                                    code_str = code_str.encode().decode('unicode_escape')
                                except Exception:
                                    pass
                                args[key] = code_str
                            elif key == 'mode':
                                args[key] = groups[i] if groups[i] else 'overwrite'
                            else:
                                args[key] = groups[i]
                    if tool_name == 'write_file' and 'mode' not in args:
                        args['mode'] = 'overwrite'
                    if tool_name == 'list_dir' and 'path' not in args:
                        args['path'] = '.'
                    calls.append((tool_name, args))
                    return ToolParser._normalize_args(calls)

            # 10.2 裸工具名 + 多行字符串块（作为 code/path 内容）
            for tool in known_tools:
                if tool == 'bash':
                    continue
                pattern = re.compile(
                    rf'(?:^|\n)\s*{re.escape(tool)}\s*\n(?!\s*{{)(.*?)(?=\n\s*(?:{"|".join(re.escape(t) for t in known_tools)})\s*\n|$)',
                    re.DOTALL | re.IGNORECASE
                )
                match = pattern.search(text)
                if match:
                    content = match.group(1).strip()
                    if tool == 'execute_python':
                        calls.append((tool, {"code": content}))
                    elif tool == 'write_file':
                        lines = content.split('\n', 1)
                        if len(lines) >= 2:
                            path = lines[0].strip().strip('"\'')
                            file_content = lines[1].strip()
                            calls.append((tool, {"path": path, "content": file_content}))
                    elif tool == 'read_file':
                        path = content.split('\n')[0].strip().strip('"\'')
                        calls.append((tool, {"path": path}))
                    if calls:
                        return ToolParser._normalize_args(calls)

            # 10.3 类似 {"tool": "read_file", "path": "..."} 但缺少外层结构
            json_like = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*[,}][^{}]*\}', text)
            if json_like:
                try:
                    obj = json.loads(json_like.group())
                    if 'tool' in obj and obj['tool'] in known_tools:
                        tool = obj.pop('tool')
                        if 'input' in obj and isinstance(obj['input'], dict):
                            args = obj['input']
                        else:
                            args = {k: v for k, v in obj.items() if k != 'tool'}
                        calls.append((tool, args))
                        return ToolParser._normalize_args(calls)
                except Exception:
                    pass

            # 10.4 工具名 + 冒号 + JSON 片段
            colon_match = re.search(rf'({"|".join(known_tools)})\s*:\s*(\{{.*?\}})', text, re.DOTALL)
            if colon_match:
                tool = colon_match.group(1)
                try:
                    args = json.loads(colon_match.group(2))
                    calls.append((tool, args))
                    return ToolParser._normalize_args(calls)
                except Exception:
                    pass

        if not calls and any(t in text for t in known_tools):
            print(f"⚠️ [ToolParser] 响应包含工具名但解析失败，前200字符: {text[:200]}")

        return ToolParser._normalize_args(calls)

    @staticmethod
    def _normalize_args(calls: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        """将工具参数中的非标准键名标准化（例如 'param' -> 'path'/'code'）"""
        normalized = []
        for tool_name, args in calls:
            if not isinstance(args, dict):
                normalized.append((tool_name, args))
                continue

            new_args = dict(args)
            # 处理通用 "param" 键（模型常犯错误）
            if "param" in new_args and tool_name in ("read_file", "write_file", "edit_file", "list_dir"):
                if "path" not in new_args:
                    new_args["path"] = new_args.pop("param")
            if "param" in new_args and tool_name == "execute_python":
                if "code" not in new_args:
                    new_args["code"] = new_args.pop("param")
            if "param" in new_args and tool_name == "bash":
                if "command" not in new_args:
                    new_args["command"] = new_args.pop("param")

            # 处理其他常见别名
            if tool_name == "execute_python" and "script" in new_args and "code" not in new_args:
                new_args["code"] = new_args.pop("script")
            if tool_name == "bash" and "cmd" in new_args and "command" not in new_args:
                new_args["command"] = new_args.pop("cmd")

            normalized.append((tool_name, new_args))
        return normalized

    @staticmethod
    def _parse_bare_format(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write", "execute_python"}
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

        _tool_suffix_re = re.compile(
            r'(\{[^{}]*\})\s*(' + '|'.join(re.escape(t) for t in sorted(known_tools, key=len, reverse=True)) + r')\s*$'
        )
        raw_lines = text.strip().splitlines()
        lines: list = []
        for _ln in raw_lines:
            _m = _tool_suffix_re.search(_ln)
            if _m and _ln.strip().startswith('{'):
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