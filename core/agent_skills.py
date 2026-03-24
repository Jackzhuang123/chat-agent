#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Skills 系统 - 知识外置化
==========================================================

核心哲学: "知识外置化"

与工具不同:
- 工具 = 模型能做什么 (bash, read_file, etc.)
- 技能 = 模型知道怎么做 (PDF处理, MCP开发, 代码审查等)

技能允许按需加载领域专业知识,不需要重新训练模型。
这就像热插拔 LoRA 适配器,成本从 $10K-$1M 降低到 $0!

关键特性:
1. 渐进式披露: 元数据 -> 详细说明 -> 资源
2. 上下文高效: 技能作为工具结果注入,保留缓存
3. 可编辑知识: 任何人都可以编写 SKILL.md 文件
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class SkillManager:
    """技能管理器 - 发现、加载和管理技能"""

    def __init__(self, skills_dir: Optional[str] = None):
        """
        初始化技能管理器

        Args:
            skills_dir: 技能目录路径，默认为项目根目录下的 skills/
        """
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            # 使用绝对路径，避免从不同工作目录启动时相对路径失效
            self.skills_dir = Path(__file__).parent.parent / "skills"
        self.skills_metadata: Dict[str, Dict[str, Any]] = {}
        self.skills_cache: Dict[str, str] = {}

        # 发现所有可用技能
        self._discover_skills()

    def _discover_skills(self):
        """扫描技能目录,发现所有可用技能"""
        if not self.skills_dir.exists():
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue

            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                self._load_skill_metadata(skill_dir, skill_file)

    def _load_skill_metadata(self, skill_dir: Path, skill_file: Path):
        """
        加载技能的元数据 (第一层: 始终加载)

        SKILL.md 格式（兼容 DeerFlow SKILL.md 规范）:
        ```
        ---
        name: PDF 处理
        description: 处理 PDF 文件的技能
        tags: [pdf, document, parsing]
        license: MIT
        resources:
          - references/pdf_spec.md
          - scripts/extract.py
        ---
        # PDF 处理技能

        详细说明...
        ```

        解析策略（借鉴 DeerFlow skills/parser.py 的健壮解析）：
        - 支持 tags 多种格式：[a, b]、- a\n- b、逗号分隔字符串
        - 必填字段缺失时跳过技能（name + description）
        - 支持 license 字段（DeerFlow 新增的合规字段）
        - 宽容解析：单行/多值字段均兼容
        """
        import re as _re
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取 YAML frontmatter（借鉴 DeerFlow parser.py 的正则策略）
            front_matter_match = _re.match(r"^---\s*\n(.*?)\n---\s*\n", content, _re.DOTALL)
            if not front_matter_match:
                # 尝试宽松格式：--- 开头，下一个 --- 结束
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter_raw = parts[1].strip()
                        body = parts[2].strip()
                    else:
                        return
                else:
                    return
            else:
                frontmatter_raw = front_matter_match.group(1)
                body_start = front_matter_match.end()
                body = content[body_start:].strip()

            # 初始化元数据（默认值）
            metadata: Dict[str, Any] = {
                "name": skill_dir.name,
                "description": "",
                "tags": [],
                "license": None,
                "resources": [],
            }

            # 逐行解析 frontmatter
            # 支持多行列表字段（如 tags: 后跟 - item 格式）
            current_key: Optional[str] = None
            list_buffer: List[str] = []

            for line in frontmatter_raw.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue

                # 检测缩进列表项（YAML 列表格式：- value）
                if stripped.startswith('- ') and current_key in ('tags', 'resources'):
                    list_buffer.append(stripped[2:].strip())
                    continue

                # 键值对行
                if ':' in line:
                    # 保存上一个列表键的缓冲
                    if current_key in ('tags', 'resources') and list_buffer:
                        metadata[current_key] = list_buffer[:]
                        list_buffer = []

                    key, _, value = line.partition(':')
                    key = key.strip().lower()
                    value = value.strip()

                    if key == 'name':
                        if value:
                            metadata['name'] = value
                        current_key = 'name'
                    elif key == 'description':
                        if value:
                            metadata['description'] = value
                        current_key = 'description'
                    elif key == 'license':
                        metadata['license'] = value or None
                        current_key = 'license'
                    elif key == 'tags':
                        current_key = 'tags'
                        if value:
                            # 行内列表：tags: [a, b, c] 或 tags: a, b, c
                            if value.startswith('['):
                                tags_str = value.strip('[]')
                                metadata['tags'] = [t.strip().strip('"\'') for t in tags_str.split(',') if t.strip()]
                            else:
                                metadata['tags'] = [t.strip() for t in value.split(',') if t.strip()]
                    elif key == 'resources':
                        current_key = 'resources'
                        if value:
                            metadata['resources'] = [value]

            # 处理结尾的列表缓冲
            if current_key in ('tags', 'resources') and list_buffer:
                metadata[current_key] = list_buffer[:]

            # 过滤空标签
            metadata['tags'] = [t for t in metadata.get('tags', []) if t]

            # DeerFlow 规范：name 和 description 必须存在
            if not metadata.get('name') or not metadata.get('description'):
                print(f"警告: 技能 {skill_dir.name} 缺少 name 或 description，已跳过")
                return

            metadata["path"] = str(skill_dir)
            metadata["full_content"] = content  # 缓存完整内容
            metadata["body_preview"] = body[:200] + "..." if len(body) > 200 else body
            metadata["enabled"] = True  # 默认启用（DeerFlow 兼容字段）

            self.skills_metadata[skill_dir.name] = metadata
        except Exception as e:
            print(f"警告: 加载技能 {skill_dir.name} 失败: {e}")

    def get_skills_list(self) -> List[Dict[str, Any]]:
        """
        获取所有技能的元数据列表 (第 1 层: 始终加载)

        Returns:
            技能列表,每项包含名称、描述、标签等
        """
        skills = []
        for skill_id, metadata in self.skills_metadata.items():
            skills.append({
                "id": skill_id,
                "name": metadata.get("name", skill_id),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "preview": metadata.get("body_preview", ""),
            })
        return skills

    def get_skill_detail(self, skill_id: str) -> Optional[str]:
        """
        获取技能的详细内容 (第 2 层: 按需加载)

        Args:
            skill_id: 技能 ID

        Returns:
            技能的完整内容 (SKILL.md)
        """
        if skill_id not in self.skills_metadata:
            return None

        # 检查缓存
        if skill_id in self.skills_cache:
            return self.skills_cache[skill_id]

        # 加载完整内容
        content = self.skills_metadata[skill_id].get("full_content", "")
        self.skills_cache[skill_id] = content
        return content

    def get_skill_resources(self, skill_id: str) -> Dict[str, str]:
        """
        获取技能的资源文件 (第 3 层: 需要时加载)

        Args:
            skill_id: 技能 ID

        Returns:
            资源文件内容字典 {filename: content}
        """
        if skill_id not in self.skills_metadata:
            return {}

        skill_path = Path(self.skills_metadata[skill_id]["path"])
        resources = {}

        # 扫描 scripts/ 和 references/ 目录
        for subdir in ["scripts", "references"]:
            subdir_path = skill_path / subdir
            if subdir_path.exists():
                for file in subdir_path.glob("*"):
                    if file.is_file() and not file.name.startswith("."):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                resources[f"{subdir}/{file.name}"] = f.read()
                        except Exception as e:
                            print(f"警告: 读取资源文件 {file} 失败: {e}")

        return resources

    def find_skills_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        基于任务描述查找相关技能 (简单的关键词匹配)

        Args:
            task_description: 任务描述

        Returns:
            相关技能列表
        """
        task_keywords = task_description.lower().split()
        relevant_skills = []

        for skill_id, metadata in self.skills_metadata.items():
            # 检查标签匹配
            tags = [tag.lower() for tag in metadata.get("tags", [])]
            name = metadata.get("name", "").lower()
            desc = metadata.get("description", "").lower()

            match_score = 0
            for keyword in task_keywords:
                if keyword in tags:
                    match_score += 3
                elif keyword in name:
                    match_score += 2
                elif keyword in desc:
                    match_score += 1

            if match_score > 0:
                relevant_skills.append({
                    "id": skill_id,
                    "name": metadata.get("name", skill_id),
                    "score": match_score,
                    "description": metadata.get("description", ""),
                })

        # 按匹配分数排序
        relevant_skills.sort(key=lambda x: x["score"], reverse=True)
        return relevant_skills


class SkillInjector:
    """技能注入器 - 将技能知识注入到模型上下文中"""

    def __init__(self, skill_manager: SkillManager):
        """
        初始化技能注入器

        Args:
            skill_manager: SkillManager 实例
        """
        self.skill_manager = skill_manager

    def inject_skills_to_context(
        self,
        messages: List[Dict[str, str]],
        relevant_skills: List[str],
        include_full_content: bool = False,
    ) -> List[Dict[str, str]]:
        """
        将技能知识注入到消息上下文中

        关键洞察: 技能作为工具结果 (用户消息) 注入,
        而不是系统提示词。这保留了提示词缓存!

        错误: 编辑系统提示词 (缓存失效)
        正确: 添加工具结果 (缓存命中) ✅

        Args:
            messages: 消息列表
            relevant_skills: 要注入的技能 ID 列表
            include_full_content: 是否包含完整内容

        Returns:
            更新后的消息列表
        """
        if not relevant_skills:
            return messages

        # 收集技能信息
        skills_info = []

        for skill_id in relevant_skills:
            if include_full_content:
                # 第 2 层: 加载完整内容
                content = self.skill_manager.get_skill_detail(skill_id)
                if content:
                    skills_info.append(content)
            else:
                # 第 1 层: 仅元数据
                metadata = self.skill_manager.skills_metadata.get(skill_id)
                if metadata:
                    info = f"**技能: {metadata.get('name', skill_id)}**\n"
                    info += f"{metadata.get('description', '')}\n"
                    if metadata.get("tags"):
                        info += f"标签: {', '.join(metadata.get('tags', []))}"
                    skills_info.append(info)

        if not skills_info:
            return messages

        # 构建技能上下文消息
        skills_context = "\n\n".join(skills_info)

        # 作为工具结果注入 (不修改系统提示词,保留缓存!)
        skill_injection_msg = {
            "role": "user",
            "content": f"[可用技能]\n\n{skills_context}\n\n请利用这些技能完成任务。"
        }

        # 将技能信息插入到最后一个用户消息之前
        # (这样模型可以先看到技能,再处理实际任务)
        updated_messages = messages.copy()
        updated_messages.insert(-1, skill_injection_msg)

        return updated_messages

    def format_skills_for_display(self, skills: List[Dict[str, Any]]) -> str:
        """格式化技能列表用于显示"""
        if not skills:
            return "暂无可用技能"

        lines = ["📚 可用技能:"]
        for skill in skills:
            lines.append(f"  • **{skill.get('name', skill.get('id'))}**")
            if skill.get("description"):
                lines.append(f"    {skill['description']}")
            if skill.get("tags"):
                lines.append(f"    标签: {', '.join(skill['tags'])}")

        return "\n".join(lines)


# ============================================================================
# 示例技能文件创建函数 (用于演示)
# ============================================================================

def create_example_skills(skills_dir: str = None):
    """创建示例技能文件，用于演示。若文件已存在则跳过，不覆盖用户修改。"""
    # 默认使用项目根目录下的 skills，避免相对路径问题
    if skills_dir is None:
        skills_dir = Path(__file__).parent.parent / "skills"
    skills_dir = Path(skills_dir)
    skills_dir.mkdir(exist_ok=True)

    # 示例技能 1: PDF 处理
    pdf_skill_dir = skills_dir / "pdf"
    pdf_skill_dir.mkdir(exist_ok=True)

    pdf_skill_content = """---
name: PDF 处理
description: 使用 pdftotext 或 PyMuPDF 处理 PDF 文件
tags: [pdf, document, parsing, extraction]
---

# PDF 处理技能

## 概述

处理 PDF 文件的系统性方法,包括文本提取、页面分析和内容处理。

## 最佳实践

1. **文本提取工具选择**:
   - pdftotext: 简单、快速、适合大文件
   - PyMuPDF: 功能强大、精确、支持复杂操作

2. **常见问题**:
   - 编码问题: 始终使用 UTF-8
   - 页面旋转: 检查 PDF 页面方向
   - 嵌入式字体: 某些 PDF 可能无法提取文本

3. **处理流程**:
   ```
   加载 PDF → 验证有效性 → 提取内容 → 清理数据 → 输出
   ```

## 命令示例

```bash
# 使用 pdftotext
pdftotext input.pdf output.txt

# 使用 PyMuPDF
python3 -c "import fitz; doc=fitz.open('input.pdf'); print(doc[0].get_text())"
```

## 注意事项

- 尊重版权和许可证
- 验证输出的数据完整性
- 处理加密 PDF 时需要密码
"""

    _pdf_skill_file = pdf_skill_dir / "SKILL.md"
    if not _pdf_skill_file.exists():
        with open(_pdf_skill_file, "w", encoding="utf-8") as f:
            f.write(pdf_skill_content)

    # 示例技能 2: 代码审查
    code_review_skill_dir = skills_dir / "code-review"
    code_review_skill_dir.mkdir(exist_ok=True)

    code_review_skill_content = """---
name: 代码审查
description: 系统化的代码审查方法论和检查清单
tags: [code-review, quality, best-practices, python]
---

# 代码审查技能

## 系统化审查流程

### 1. 代码结构 (10%)
- [ ] 函数长度合理 (< 50 行)
- [ ] 类职责单一
- [ ] 模块化设计

### 2. 命名和可读性 (20%)
- [ ] 变量名清晰
- [ ] 函数名描述性强
- [ ] 代码注释完整

### 3. 错误处理 (15%)
- [ ] 异常捕获适当
- [ ] 错误消息有用
- [ ] 边界条件处理

### 4. 性能 (15%)
- [ ] 算法复杂度合理
- [ ] 无明显性能问题
- [ ] 资源释放正确

### 5. 安全性 (20%)
- [ ] 输入验证
- [ ] SQL 注入防护
- [ ] 无硬编码密钥

### 6. 测试 (20%)
- [ ] 单元测试覆盖
- [ ] 边界测试
- [ ] 错误情况测试

## Python 特定检查

- 遵循 PEP 8 风格
- 类型提示完整
- 文档字符串存在

## 审查评语模板

```
总体评价: [好/待改进/需要重审]
关键问题: [列表]
改进建议: [列表]
建议批准: [是/否]
```
"""

    _code_review_skill_file = code_review_skill_dir / "SKILL.md"
    if not _code_review_skill_file.exists():
        with open(_code_review_skill_file, "w", encoding="utf-8") as f:
            f.write(code_review_skill_content)

    # 示例技能 3: Python 开发
    python_skill_dir = skills_dir / "python-dev"
    python_skill_dir.mkdir(exist_ok=True)

    python_skill_content = """---
name: Python 开发
description: Python 开发的最佳实践和常见模式
tags: [python, development, best-practices, patterns]
---

# Python 开发技能

## 环境管理

### 虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate
```

### 依赖管理
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

## 常见设计模式

### 1. 单例模式
用于全局资源,如配置、日志等。

### 2. 工厂模式
用于对象创建逻辑复杂的场景。

### 3. 策略模式
用于可互换的算法实现。

## 性能优化

1. **列表 vs 生成器**:
   - 一次性使用: 生成器更高效
   - 多次访问: 列表更快

2. **字典 vs 集合**:
   - 需要查找: 集合更快
   - 需要值: 字典

3. **字符串操作**:
   - 连接: 使用 ''.join([])
   - 避免: 循环中 += 字符串

## 调试技巧

```python
# 快速打印调试
import pdb; pdb.set_trace()

# 性能分析
import cProfile
cProfile.run('main()')

# 内存分析
from memory_profiler import profile
```

## 测试框架

- **unittest**: 标准库
- **pytest**: 更好的语法
- **mock**: 模拟对象
"""

    _python_skill_file = python_skill_dir / "SKILL.md"
    if not _python_skill_file.exists():
        with open(_python_skill_file, "w", encoding="utf-8") as f:
            f.write(python_skill_content)

    print(f"✅ 示例技能已就绪（已存在的文件不覆盖）：{skills_dir}")
    print(f"   - {pdf_skill_dir}")
    print(f"   - {code_review_skill_dir}")
    print(f"   - {python_skill_dir}")


if __name__ == "__main__":
    # 创建示例技能
    create_example_skills()

    # 演示技能系统
    print("\n" + "=" * 60)
    print("🧪 技能系统演示")
    print("=" * 60)

    skill_manager = SkillManager()

    print("\n📚 可用技能:")
    for skill in skill_manager.get_skills_list():
        print(f"  ✓ {skill['name']}: {skill['description']}")
        if skill.get("tags"):
            print(f"    标签: {', '.join(skill['tags'])}")

    # 任务技能匹配
    print("\n🔍 任务-技能匹配演示:")
    test_tasks = [
        "我想处理一个 PDF 文件",
        "我需要审查代码质量",
        "帮我写一个 Python 脚本",
    ]

    for task in test_tasks:
        print(f"\n  任务: {task}")
        matched = skill_manager.find_skills_for_task(task)
        for skill in matched:
            print(f"    ✓ {skill['name']} (匹配度: {skill['score']})")

    # 技能注入演示
    print("\n💉 技能注入演示:")
    injector = SkillInjector(skill_manager)

    messages = [
        {"role": "user", "content": "帮我审查这个代码"}
    ]

    updated_messages = injector.inject_skills_to_context(
        messages,
        ["code-review"],
        include_full_content=False
    )

    print(f"  原始消息数: {len(messages)}")
    print(f"  注入后消息数: {len(updated_messages)}")
    print("  ✅ 技能已注入到上下文")

    print("\n✅ 技能系统演示完成!")

