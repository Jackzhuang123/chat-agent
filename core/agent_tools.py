#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 工具系统 - 为 Qwen 模型提供工具支持"""

import json
import re
import subprocess
import time as _time_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.monitor_logger import get_monitor_logger, log_function_call


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

    def _fuzzy_find_file(self, filename: str, search_home: bool = True, depth_limit: int = 8) -> Optional[Path]:
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

        def _scan(root: Path, depth: int) -> Optional[Path]:
            if depth < 0:
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
                    elif item.is_dir() and depth > 0:
                        found = _scan(item, depth - 1)
                        if found:
                            return found
            except (PermissionError, OSError):
                pass
            return None

        result = _scan(self.work_dir, depth_limit)
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
                        def _scan_sub(root2: Path, depth: int) -> Optional[Path]:
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

                        found = _scan_sub(item, min(depth_limit, 6))
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

    @staticmethod
    def _chunk_text_lines(content: str, chunk_size: int = 80) -> List[Dict[str, Any]]:
        lines = content.splitlines()
        chunks = []
        for start in range(0, len(lines), chunk_size):
            end = min(len(lines), start + chunk_size)
            chunks.append({
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(lines[start:end]),
            })
        return chunks or [{"start_line": 1, "end_line": 1, "text": content}]

    @staticmethod
    def _build_file_facts(path: Path, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        classes = []
        functions = []
        imports = []
        for line in lines:
            stripped = line.strip()
            class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
            func_match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
            if class_match:
                classes.append(class_match.group(1))
            if func_match:
                functions.append(func_match.group(1))
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped[:120])

        chunk_summaries = []
        for chunk in ToolExecutor._chunk_text_lines(content)[:8]:
            chunk_lines = chunk["text"].splitlines()
            non_empty = [line.strip() for line in chunk_lines if line.strip()]
            summary = {
                "line_range": f"{chunk['start_line']}-{chunk['end_line']}",
                "classes": [],
                "functions": [],
                "signals": non_empty[:3],
            }
            for line in chunk_lines:
                stripped = line.strip()
                class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
                func_match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
                if class_match:
                    summary["classes"].append(class_match.group(1))
                if func_match:
                    summary["functions"].append(func_match.group(1))
            chunk_summaries.append(summary)

        return {
            "path": str(path),
            "line_count": len(lines),
            "classes": classes[:12],
            "functions": functions[:20],
            "imports": imports[:12],
            "summary": (
                f"共 {len(lines)} 行；"
                f"类 {', '.join(classes[:4]) if classes else '无'}；"
                f"函数 {', '.join(functions[:6]) if functions else '无'}"
            ),
            "chunk_summaries": chunk_summaries,
        }

    def _build_read_result(self, path: Path, content: str, extra: Optional[Dict[str, Any]] = None) -> str:
        result = {
            "success": True,
            "path": str(path),
            "content": content,
            "file_facts": self._build_file_facts(path, content),
        }
        if extra:
            result.update(extra)
        if not content.strip():
            result["warning"] = "文件存在但内容为空"
        return json.dumps(result, ensure_ascii=False)

    @log_function_call()
    def _read_file(self, path: str) -> str:
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
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                    return self._build_read_result(p, content)
                except Exception as e:
                    return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)
            else:
                filename = p.name
                found = self._fuzzy_find_file(filename, search_home=True, depth_limit=5)
                if found:
                    try:
                        with open(found, "r", encoding="utf-8") as f:
                            content = f.read()
                        correction_prefix = (
                            f"⚠️ 系统自动纠正：你请求的路径 '{path}' 不存在。\n"
                            f"✅ 实际读取的文件为：{found}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        )
                        return self._build_read_result(found, correction_prefix + content, extra={
                            "fuzzy_match": True,
                            "original_request": path,
                            "note": f"原绝对路径 '{path}' 不存在，自动匹配到: {found}",
                        })
                    except Exception as e:
                        return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)
                return json.dumps({
                    "success": False,
                    "error": f"文件不存在: {path}",
                    "hint": "绝对路径指定的文件不存在，且项目内未找到同名文件。请仅提供文件名（如 'session_analyzer.py'），系统会自动搜索。"
                }, ensure_ascii=False)

        # ----- 相对路径分支 -----
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
                return self._build_read_result(file_path.resolve(), content)
            except Exception as e:
                return json.dumps({"success": False, "error": f"读取文件失败: {str(e)}"}, ensure_ascii=False)

        # 相对路径不存在时的模糊搜索
        filename = Path(path).name
        found = self._fuzzy_find_file(filename, search_home=True, depth_limit=5)
        if found:
            try:
                with open(found, "r", encoding="utf-8") as f:
                    content = f.read()
                correction_prefix = (
                    f"⚠️ 系统自动纠正：你请求的路径 '{path}' 不存在。\n"
                    f"✅ 实际读取的文件为：{found}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                )
                return self._build_read_result(found, correction_prefix + content, extra={
                    "fuzzy_match": True,
                    "original_request": path,
                    "note": f"原路径 '{path}' 不存在，自动匹配到: {found}",
                })
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

    @log_function_call()
    def execute_python(self, code: str, timeout: int = 30) -> str:
        import sys as _sys
        import re as _re
        import subprocess
        import json

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

                    if success:
                        _has_write_inner = bool(_re.search(
                            r'open\s*\([^)]*[\'\"].*[\'\"],\s*[\'\"]w[\'\"]|\.write\s*\(|write_file|to_csv|to_json|to_excel|shutil\.copy|os\.rename',
                            code
                        ))
                        _is_scan_task = bool(_re.search(r'os\.listdir|glob\.glob|re\.findall|re\.finditer|\.findall\(', code))
                        if _is_scan_task and not stdout_raw and not _has_write_inner:
                            success = False
                            inner["error"] = "代码执行成功但未提取到任何有效结果（stdout为空且未写入文件）。"
                            inner["fix_hint"] = "请检查正则表达式是否正确，或改用 bash 命令重试。"
                            self.monitor.warning(f"execute_python 空结果: {code[:200]}")

                    resp = {
                        "success": success,
                        "stdout": stdout_raw,
                        "stderr": stderr_raw,
                        "error": error_raw,
                        "returncode": proc.returncode,
                    }
                    if not success and "fix_hint" not in resp:
                        resp["fix_hint"] = (
                            "❌ 代码执行失败。请仔细阅读上方 error/stderr 中的错误信息，"
                            "修正代码后再次调用 execute_python 重新执行。"
                        )
                    return json.dumps(resp, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            # 兜底处理
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
    def _fix_bash_command_json(json_str: str) -> str:
        """修复 bash 命令 JSON 中的转义问题"""
        import re
        # 匹配 "command": "..." 字段
        pattern = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
        match = pattern.search(json_str)
        if not match:
            return json_str
        cmd = match.group(1)
        # 将反斜杠加倍（因为 JSON 中需要 \\ 表示单个 \）
        # 但要保护已经转义的双反斜杠
        def escape_backslashes(s: str) -> str:
            # 临时替换 \\ 为特殊标记
            s = s.replace('\\\\', '\x00')
            s = s.replace('\\', '\\\\')
            s = s.replace('\x00', '\\\\')
            return s
        safe_cmd = escape_backslashes(cmd)
        # 重建 JSON
        return json_str[:match.start(1)] + safe_cmd + json_str[match.end(1):]

    # ... 原有的其他方法 ...

    @staticmethod
    def _parse_input_payload(input_str: str) -> Optional[Dict[str, Any]]:
        if not input_str:
            return None
        payload = input_str.strip()
        if not payload:
            return None

        # 修复bash命令
        if '"command"' in payload:
            fixed = ToolParser._fix_bash_command_json(payload)
            try:
                data = json.loads(fixed)
                if isinstance(data, dict):
                    return data
            except:
                pass

        # ---------- 新增：转义 JSON 字符串值内的控制字符 ----------
        def _escape_control_chars_in_strings(s: str) -> str:
            """将 JSON 字符串值内的未转义控制字符（\n, \r, \t）替换为标准转义序列。"""
            result = []
            in_string = False
            escaped = False
            for ch in s:
                if in_string:
                    if escaped:
                        result.append(ch)
                        escaped = False
                    elif ch == '\\':
                        result.append(ch)
                        escaped = True
                    elif ch == '"':
                        result.append(ch)
                        in_string = False
                    elif ch == '\n':
                        result.append('\\n')
                    elif ch == '\r':
                        result.append('\\r')
                    elif ch == '\t':
                        result.append('\\t')
                    else:
                        result.append(ch)
                else:
                    if ch == '"':
                        in_string = True
                    result.append(ch)
            return ''.join(result)

        payload = _escape_control_chars_in_strings(payload)
        # ---------------------------------------------------------

        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        # 原有的补全括号、修复无效转义、修复嵌套引号等逻辑保持不变
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
                    _field_pattern = _re_fix.compile(r'"(code|command|content|script)"\s*:\s*"((?:[^"\\]|\\.)*)"',
                                                     _re_fix.DOTALL)

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
    def _extract_balanced_json_object(text: str, start_idx: int) -> Optional[str]:
        """提取从 start_idx 开始的首个平衡 JSON 对象，正确处理字符串内换行和花括号。"""
        if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]
                if depth < 0:
                    return None
        return None

    @staticmethod
    @log_function_call()
    def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not isinstance(text, str):
            print(f"⚠️ 工具解析失败：输入类型错误 (期望 str，实际 {type(text).__name__})")
            return []

        monitor = get_monitor_logger()
        calls = []
        stripped = text.strip()
        known_tools = {"read_file", "write_file", "edit_file", "list_dir", "bash", "todo_write", "execute_python"}

        # 快速跳过自然语言长文本
        if len(stripped) > 200 and not re.search(r'\n\s*\{', stripped):
            monitor.debug("跳过自然语言长文本解析")
            return []

        # ---------- 0. 专为 execute_python 设计的增强解析 ----------
        if "execute_python" in stripped:
            ep_pattern = re.compile(
                r'(?:^|\n)\s*execute_python\s*\n\s*(\{[\s\S]*?\})\s*(?:$|\n)',
                re.MULTILINE
            )
            ep_match = ep_pattern.search(stripped)
            if ep_match:
                json_str = ep_match.group(1)
                args = ToolParser._parse_input_payload(json_str)
                if args and isinstance(args, dict) and "code" in args:
                    monitor.debug("通过 execute_python 专用解析器成功解析")
                    return ToolParser._normalize_args([("execute_python", args)])
            ep_pattern2 = re.compile(
                r'execute_python\s*(\{[\s\S]*?\})',
                re.MULTILINE
            )
            ep_match2 = ep_pattern2.search(stripped)
            if ep_match2:
                json_str = ep_match2.group(1)
                args = ToolParser._parse_input_payload(json_str)
                if args and isinstance(args, dict) and "code" in args:
                    monitor.debug("通过 execute_python 专用解析器（模式2）成功解析")
                    return ToolParser._normalize_args([("execute_python", args)])

        # ---------- 新增：处理 markdown 代码块内的工具调用（包括 plaintext） ----------
        # 匹配 ```python / ```plaintext / ```json / ``` 等代码块
        code_block_pattern = re.compile(
            r'```(?:python|plaintext|json|text)?\s*\n\s*([\s\S]*?)\s*```',
            re.MULTILINE
        )
        for block_match in code_block_pattern.finditer(stripped):
            inner = block_match.group(1).strip()
            # 检查内部是否包含工具名 + JSON 的格式
            for tool in known_tools:
                tool_pattern = re.compile(
                    rf'{re.escape(tool)}\s*\n\s*(\{{[\s\S]*?\}})',
                    re.MULTILINE
                )
                tool_match = tool_pattern.search(inner)
                if tool_match:
                    json_str = tool_match.group(1)
                    args = ToolParser._parse_input_payload(json_str)
                    if args and isinstance(args, dict):
                        monitor.debug(f"从代码块解析到工具调用: {tool}")
                        return ToolParser._normalize_args([(tool, args)])
            # 尝试直接解析为 JSON 对象
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    if "tool" in obj and "input" in obj:
                        return ToolParser._normalize_args([(obj["tool"], obj["input"])])
                    elif "name" in obj and "arguments" in obj:
                        args = obj["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        return ToolParser._normalize_args([(obj["name"], args)])
            except:
                pass

        # ---------- 原有 JSON 格式解析 ----------
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
                    monitor.debug(f"解析到 JSON 数组格式工具调用: {len(calls)} 个")
                    return ToolParser._normalize_args(calls)
            elif stripped.startswith("{"):
                item = json.loads(stripped)
                if isinstance(item, dict):
                    if "tool" in item and "input" in item and isinstance(item["input"], dict):
                        monitor.debug("解析到 JSON 对象格式工具调用")
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

        # ---------- 原有工具名+换行+JSON解析 ----------
        for tool in known_tools:
            pattern = re.compile(
                rf'(?:^|\n)\s*{re.escape(tool)}\s*\n\s*(\{{)',
                re.MULTILINE | re.DOTALL
            )
            for m in pattern.finditer(stripped):
                json_start = m.start(1)
                candidate = ToolParser._extract_balanced_json_object(stripped, json_start)
                if candidate is None:
                    continue
                args = ToolParser._parse_input_payload(candidate)
                if args and isinstance(args, dict):
                    monitor.debug(f"解析工具调用成功: {tool} (工具名+JSON格式)")
                    return ToolParser._normalize_args([(tool, args)])

        if not calls and any(t in text for t in known_tools):
            monitor.warning(f"响应包含工具名但解析失败，前200字符: {text[:200]}")

        return ToolParser._normalize_args(calls)

    @staticmethod
    def _normalize_args(calls: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        normalized = []
        for tool_name, args in calls:
            if not isinstance(args, dict):
                normalized.append((tool_name, args))
                continue

            new_args = dict(args)
            if "param" in new_args and tool_name in ("read_file", "write_file", "edit_file", "list_dir"):
                if "path" not in new_args:
                    new_args["path"] = new_args.pop("param")
            if "param" in new_args and tool_name == "execute_python":
                if "code" not in new_args:
                    new_args["code"] = new_args.pop("param")
            if "param" in new_args and tool_name == "bash":
                if "command" not in new_args:
                    new_args["command"] = new_args.pop("param")

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
