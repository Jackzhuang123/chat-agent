# 代码审查技能库

## 概述
提供代码审查、质量分析、安全检测等功能,集成 Qwen Agent 进行自动化代码分析。

## 核心功能

### 1. 代码风格检查
使用 Pylint 和 Flake8 进行代码质量检查。

```python
import subprocess
import json

def check_code_style(file_path, style_type="flake8"):
    """
    检查代码风格

    Args:
        file_path (str): Python 文件路径
        style_type (str): 检查工具 ("flake8" 或 "pylint")

    Returns:
        dict: 检查结果
    """
    issues = []

    if style_type == "flake8":
        result = subprocess.run(
            ["flake8", file_path, "--format=json"],
            capture_output=True,
            text=True
        )
        issues = json.loads(result.stdout) if result.stdout else []

    elif style_type == "pylint":
        result = subprocess.run(
            ["pylint", file_path, "--output-format=json"],
            capture_output=True,
            text=True
        )
        issues = json.loads(result.stdout) if result.stdout else []

    return {
        "file": file_path,
        "tool": style_type,
        "issues": issues,
        "total_issues": len(issues)
    }

# 使用示例
result = check_code_style("example.py", "flake8")
print(f"发现 {result['total_issues']} 个问题")
for issue in result['issues']:
    print(f"  - {issue['code']}: {issue['text']}")
```

### 2. 复杂度分析
使用 Radon 计算圈复杂度和代码指标。

```python
from radon.complexity import cc_visit
from radon.metrics import mi_visit

def analyze_complexity(file_path):
    """
    分析代码复杂度

    Args:
        file_path (str): Python 文件路径

    Returns:
        dict: 复杂度分析结果
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 圈复杂度
    complexity = cc_visit(code)

    # 可维护性指数
    maintainability_index = mi_visit(code, multi=True)

    analysis = {
        "file": file_path,
        "complexity": [],
        "maintainability_index": maintainability_index
    }

    for item in complexity:
        analysis["complexity"].append({
            "name": item.name,
            "complexity": item.complexity,
            "rank": item.complexity_grade,
            "lineno": item.lineno
        })

    return analysis

# 使用示例
result = analyze_complexity("example.py")
print(f"可维护性指数: {result['maintainability_index']}")
for func in result['complexity']:
    print(f"  {func['name']}: 复杂度 {func['complexity']} ({func['rank']})")
```

### 3. 安全漏洞检测
使用 Bandit 检测安全问题。

```python
from bandit.main import Bandit
from bandit.config import BanditConfig
import tempfile

def detect_security_issues(file_path):
    """
    检测代码中的安全问题

    Args:
        file_path (str): Python 文件路径

    Returns:
        dict: 安全问题列表
    """
    import subprocess
    import json

    result = subprocess.run(
        ["bandit", file_path, "-f", "json"],
        capture_output=True,
        text=True
    )

    issues = json.loads(result.stdout) if result.stdout else {}

    return {
        "file": file_path,
        "total_issues": len(issues.get("results", [])),
        "severity_counts": issues.get("metrics", {}).get(file_path, {}).get("_totals", {}),
        "issues": [
            {
                "type": issue["test_id"],
                "severity": issue["severity"],
                "line": issue["line_number"],
                "description": issue["issue_text"]
            }
            for issue in issues.get("results", [])
        ]
    }

# 使用示例
result = detect_security_issues("example.py")
print(f"发现 {result['total_issues']} 个安全问题")
for issue in result['issues']:
    print(f"  [{issue['severity']}] 第 {issue['line']} 行: {issue['type']}")
```

### 4. 代码覆盖率分析
使用 Coverage.py 分析代码覆盖率。

```python
import coverage
import os

def analyze_coverage(test_dir, source_dir):
    """
    分析代码覆盖率

    Args:
        test_dir (str): 测试文件目录
        source_dir (str): 源代码目录

    Returns:
        dict: 覆盖率报告
    """
    cov = coverage.Coverage(source=[source_dir])
    cov.start()

    # 运行测试 (需要自定义)
    import subprocess
    subprocess.run(["pytest", test_dir], capture_output=True)

    cov.stop()
    cov.save()

    # 生成报告
    report = {}
    for module, coverage_data in cov.get_data().items():
        report[module] = {
            "lines": len(coverage_data),
            "coverage_percent": cov.get_statistics()
        }

    return report

# 使用示例
coverage_report = analyze_coverage("tests/", "src/")
```

### 5. 代码重复检查
使用 Radon 检测代码重复。

```python
from radon.complexity import cc_visit
import difflib

def find_duplicate_code(files):
    """
    查找代码重复片段

    Args:
        files (list): Python 文件列表

    Returns:
        dict: 重复代码块
    """
    code_blocks = {}
    duplicates = []

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 提取代码块 (简化版本)
        for i in range(len(lines) - 5):
            block = ''.join(lines[i:i+5])
            if block not in code_blocks:
                code_blocks[block] = []
            code_blocks[block].append((file_path, i))

    # 查找重复
    for block, locations in code_blocks.items():
        if len(locations) > 1:
            duplicates.append({
                "code": block.strip(),
                "locations": locations,
                "count": len(locations)
            })

    return {
        "total_duplicates": len(duplicates),
        "duplicates": duplicates
    }

# 使用示例
files = ["module1.py", "module2.py"]
duplicates = find_duplicate_code(files)
print(f"发现 {duplicates['total_duplicates']} 个重复代码块")
```

### 6. 自动化代码审查报告生成

```python
import json
from datetime import datetime

def generate_review_report(file_path, output_path="review_report.json"):
    """
    生成完整的代码审查报告

    Args:
        file_path (str): 待审查文件
        output_path (str): 报告输出路径

    Returns:
        dict: 审查报告
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "file": file_path,
        "sections": {
            "style": check_code_style(file_path),
            "complexity": analyze_complexity(file_path),
            "security": detect_security_issues(file_path)
        }
    }

    # 生成总体评分
    style_issues = report["sections"]["style"]["total_issues"]
    complexity_score = report["sections"]["complexity"]["maintainability_index"]
    security_issues = report["sections"]["security"]["total_issues"]

    total_score = max(0, 100 - style_issues * 2 - security_issues * 5)
    report["overall_score"] = total_score
    report["grade"] = "A" if total_score >= 90 else "B" if total_score >= 70 else "C"

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

# 使用示例
report = generate_review_report("example.py")
print(f"代码质量评分: {report['overall_score']}/100 (等级: {report['grade']})")
```

## 集成到 Agent

```python
class CodeReviewAgent:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def review_code(self, code_snippet, aspects=["style", "security", "performance"]):
        """
        进行 AI 驱动的代码审查

        Args:
            code_snippet (str): 代码片段
            aspects (list): 审查方面

        Returns:
            str: 审查意见
        """
        prompt = f"""请对以下代码进行专业审查,重点关注 {', '.join(aspects)}:

```python
{code_snippet}
```

审查意见:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)
        review = self.tokenizer.decode(outputs[0])

        return review

# 使用示例
agent = CodeReviewAgent("Qwen/Qwen2.5-0.5B-Instruct")
code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if i % 2 == 0:
            result.append(data[i] * 2)
    return result
"""
review = agent.review_code(code)
print(review)
```

## 依赖库

```bash
pip install flake8 pylint radon bandit coverage pytest
```

## 最佳实践

1. **自动化集成**: 在 CI/CD 流程中集成代码审查工具。
2. **阈值设置**: 为不同指标设置可接受的阈值。
3. **渐进改进**: 定期运行审查,追踪指标改进趋势。
4. **团队标准**: 制定团队的代码规范和审查标准。
5. **文档化**: 记录审查发现和改进建议。

## 相关资源

- [Flake8 文档](https://flake8.pycqa.org/)
- [Pylint 文档](https://www.pylint.org/)
- [Radon 文档](https://radon.readthedocs.io/)
- [Bandit 文档](https://bandit.readthedocs.io/)
- [Coverage.py 文档](https://coverage.readthedocs.io/)

