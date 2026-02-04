#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 工具系统 - 为 Qwen 模型提供工具支持
包含: bash, read_file, write_file, edit_file 等工具
"""

import json
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
                "description": "创建或覆盖文件。用于创建新文件或完全替换文件内容。",
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
                return self._write_file(tool_input["path"], tool_input["content"])
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
            else:
                return json.dumps({"error": f"未知工具: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _read_file(self, path: str) -> str:
        """读取文件"""
        file_path = self.work_dir / path

        # 安全性检查: 确保路径在工作目录内
        try:
            file_path.resolve().relative_to(self.work_dir.resolve())
        except ValueError:
            return json.dumps({"error": "路径超出工作目录范围"})

        if not file_path.exists():
            return json.dumps({"error": f"文件不存在: {path}"})

        if not file_path.is_file():
            return json.dumps({"error": f"不是文件: {path}"})

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({
                "success": True,
                "path": path,
                "content": content
            })
        except Exception as e:
            return json.dumps({"error": f"读取文件失败: {str(e)}"})

    def _write_file(self, path: str, content: str) -> str:
        """写入文件"""
        file_path = self.work_dir / path

        # 安全性检查
        try:
            file_path.resolve().relative_to(self.work_dir.resolve())
        except ValueError:
            return json.dumps({"error": "路径超出工作目录范围"})

        try:
            # 创建父目录
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return json.dumps({
                "success": True,
                "path": path,
                "size": len(content)
            })
        except Exception as e:
            return json.dumps({"error": f"写入文件失败: {str(e)}"})

    def _edit_file(self, path: str, old_content: str, new_content: str) -> str:
        """编辑文件 - 替换指定部分"""
        file_path = self.work_dir / path

        # 安全性检查
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
        dir_path = self.work_dir / path

        # 安全性检查
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
            for item in sorted(dir_path.iterdir()):
                # 跳过隐藏文件和 __pycache__ 等
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue

                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })

            return json.dumps({
                "success": True,
                "path": path,
                "items": items
            })
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
    def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        从模型输出中解析工具调用

        支持的格式:
        1. JSON 格式: [{"tool": "bash", "input": {"command": "ls"}}]
        2. 标记格式: <tool>bash</tool><input>{"command": "ls"}</input>
        3. 文本格式: Tool: bash, Input: {"command": "ls"}

        Returns:
            List[(tool_name, tool_input), ...]
        """
        calls = []

        # 尝试 JSON 格式
        try:
            if text.strip().startswith("["):
                data = json.loads(text)
                for item in data:
                    if isinstance(item, dict) and "tool" in item and "input" in item:
                        calls.append((item["tool"], item["input"]))
                if calls:
                    return calls
        except (json.JSONDecodeError, KeyError):
            pass

        # 尝试标记格式
        import re
        tool_pattern = r"<tool>(\w+)</tool>"
        input_pattern = r"<input>(.*?)</input>"

        tools = re.findall(tool_pattern, text)
        inputs = re.findall(input_pattern, text, re.DOTALL)

        if tools and inputs and len(tools) == len(inputs):
            for tool, input_str in zip(tools, inputs):
                try:
                    input_data = json.loads(input_str)
                    calls.append((tool, input_data))
                except json.JSONDecodeError:
                    pass

            if calls:
                return calls

        return calls


if __name__ == "__main__":
    # 测试工具系统
    executor = ToolExecutor()

    print("📋 可用工具:")
    for tool in executor.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n✅ 工具系统初始化完成")

