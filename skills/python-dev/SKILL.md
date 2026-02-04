# Python 开发技能库

## 概述
提供 Python 开发的最佳实践、常用工具库集成、调试技巧等,帮助 Agent 成为一个全能的 Python 开发助手。

## 核心功能

### 1. 代码生成与修复
根据需求生成或修复 Python 代码。

```python
class PythonCodeGenerator:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def generate_code(self, requirement):
        """
        根据需求生成 Python 代码

        Args:
            requirement (str): 功能需求描述

        Returns:
            str: 生成的 Python 代码
        """
        prompt = f"""根据以下需求生成 Python 代码:

需求: {requirement}

约束条件:
- 代码应该是可执行的
- 包含必要的注释
- 遵循 PEP 8 风格指南
- 包含错误处理

代码:
```python"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=1000)
        code = self.tokenizer.decode(outputs[0])

        return code

    def fix_code(self, broken_code, error_message):
        """
        修复有问题的 Python 代码

        Args:
            broken_code (str): 有问题的代码
            error_message (str): 错误信息

        Returns:
            str: 修复后的代码
        """
        prompt = f"""请修复以下 Python 代码:

错误信息:
{error_message}

原始代码:
```python
{broken_code}
```

修复后的代码:
```python"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=1000)
        fixed_code = self.tokenizer.decode(outputs[0])

        return fixed_code

# 使用示例
generator = PythonCodeGenerator("Qwen/Qwen2.5-0.5B-Instruct")

# 生成代码
requirement = "编写一个函数,输入 CSV 文件路径,返回数据统计信息"
code = generator.generate_code(requirement)
print(code)

# 修复代码
broken = """
def read_csv(file):
    data = open(file)
    lines = data.readlines()
    return lines
"""
error = "TypeError: unsupported operand type(s)"
fixed = generator.fix_code(broken, error)
print(fixed)
```

### 2. 依赖管理工具
管理项目依赖和虚拟环境。

```python
import subprocess
import json
import os

class DependencyManager:
    @staticmethod
    def create_venv(venv_name="venv"):
        """
        创建虚拟环境

        Args:
            venv_name (str): 虚拟环境名称

        Returns:
            bool: 是否成功
        """
        result = subprocess.run(
            ["python", "-m", "venv", venv_name],
            capture_output=True
        )
        return result.returncode == 0

    @staticmethod
    def install_packages(packages, venv_path=None):
        """
        安装 Python 包

        Args:
            packages (list): 包名列表
            venv_path (str): 虚拟环境路径

        Returns:
            dict: 安装结果
        """
        pip_cmd = [
            os.path.join(venv_path, "bin", "pip") if venv_path else "pip",
            "install"
        ] + packages

        result = subprocess.run(pip_cmd, capture_output=True, text=True)

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr
        }

    @staticmethod
    def generate_requirements(output_file="requirements.txt"):
        """
        生成 requirements.txt 文件

        Args:
            output_file (str): 输出文件路径
        """
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True
        )

        with open(output_file, 'w') as f:
            f.write(result.stdout)

        return result.stdout

    @staticmethod
    def check_outdated_packages():
        """
        检查过时的包

        Returns:
            list: 过时包列表
        """
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True
        )

        return json.loads(result.stdout) if result.stdout else []

# 使用示例
manager = DependencyManager()

# 创建虚拟环境
manager.create_venv("myenv")

# 安装包
result = manager.install_packages(["numpy", "pandas"], "myenv")
print(f"安装成功: {result['success']}")

# 检查过时包
outdated = manager.check_outdated_packages()
for pkg in outdated:
    print(f"{pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
```

### 3. 性能分析与优化
使用 cProfile 和 memory_profiler 分析性能。

```python
import cProfile
import pstats
import io
from memory_profiler import profile

class PerformanceAnalyzer:
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """
        分析函数性能

        Args:
            func: 要分析的函数
            *args, **kwargs: 函数参数

        Returns:
            dict: 性能分析结果
        """
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # 获取统计信息
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # 打印前 10 个函数

        return {
            "result": result,
            "profile": s.getvalue()
        }

    @staticmethod
    def analyze_memory_usage(func, *args, **kwargs):
        """
        分析内存使用

        Args:
            func: 要分析的函数
            *args, **kwargs: 函数参数

        Returns:
            dict: 内存使用统计
        """
        import tracemalloc

        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "current_memory_mb": current / 1024 / 1024,
            "peak_memory_mb": peak / 1024 / 1024,
            "result": result
        }

# 使用示例
analyzer = PerformanceAnalyzer()

def slow_function(n):
    total = 0
    for i in range(n):
        total += sum(range(i))
    return total

# 性能分析
perf = analyzer.profile_function(slow_function, 1000)
print(perf["profile"])

# 内存分析
mem = analyzer.analyze_memory_usage(slow_function, 1000)
print(f"内存使用: {mem['peak_memory_mb']:.2f} MB")
```

### 4. 测试框架集成
集成 pytest 和 unittest 进行自动化测试。

```python
import pytest
import unittest
from io import StringIO

class TestRunner:
    @staticmethod
    def run_pytest(test_dir="."):
        """
        运行 pytest 测试

        Args:
            test_dir (str): 测试目录

        Returns:
            dict: 测试结果
        """
        import subprocess

        result = subprocess.run(
            ["pytest", test_dir, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr
        }

    @staticmethod
    def generate_test_template(function_name, input_types, expected_output):
        """
        生成测试模板

        Args:
            function_name (str): 函数名
            input_types (dict): 输入参数类型
            expected_output: 期望输出

        Returns:
            str: 测试代码
        """
        test_code = f"""import pytest

class Test{function_name.capitalize()}:

    def setup_method(self):
        \"\"\"测试前准备\"\"\"
        pass

    def teardown_method(self):
        \"\"\"测试后清理\"\"\"
        pass

    def test_{function_name}_basic(self):
        \"\"\"基本功能测试\"\"\"
        result = {function_name}({', '.join(input_types.keys())})
        assert result == {expected_output}

    def test_{function_name}_edge_cases(self):
        \"\"\"边界情况测试\"\"\"
        # TODO: 补充边界情况
        pass

    @pytest.mark.parametrize("input,expected", [
        # TODO: 添加参数化测试用例
    ])
    def test_{function_name}_parametrized(self, input, expected):
        \"\"\"参数化测试\"\"\"
        result = {function_name}(input)
        assert result == expected
"""
        return test_code

# 使用示例
runner = TestRunner()

# 运行测试
result = runner.run_pytest("tests/")
print(result["output"])

# 生成测试模板
template = runner.generate_test_template(
    "calculate_sum",
    {"a": "int", "b": "int"},
    5
)
print(template)
```

### 5. 调试工具集
提供交互式调试和断点功能。

```python
import sys
import traceback
import pdb

class DebugHelper:
    @staticmethod
    def print_stack_trace(exc):
        """
        打印栈轨迹

        Args:
            exc: 异常对象
        """
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    @staticmethod
    def debug_breakpoint(locals_dict):
        """
        设置断点进行调试

        Args:
            locals_dict (dict): 本地变量字典
        """
        debugger = pdb.Pdb()
        debugger.reset()
        sys.set_trace()

    @staticmethod
    def log_function_calls(func):
        """
        装饰器: 记录函数调用

        Args:
            func: 要记录的函数

        Returns:
            function: 包装后的函数
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f">> 调用 {func.__name__}()")
            print(f"   参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                print(f"   返回: {result}")
                return result
            except Exception as e:
                print(f"   异常: {type(e).__name__}: {e}")
                raise

        return wrapper

    @staticmethod
    def assert_debug(condition, message="断言失败"):
        """
        条件断言 (调试时使用)

        Args:
            condition (bool): 条件
            message (str): 失败信息
        """
        if not condition:
            print(f"⚠️  {message}")
            import pdb; pdb.set_trace()

# 使用示例
@DebugHelper.log_function_calls
def add(a, b):
    return a + b

# 测试
add(1, 2)

# 异常调试
try:
    result = 1 / 0
except Exception as e:
    DebugHelper.print_stack_trace(e)
```

### 6. 文档生成
自动生成 API 文档。

```python
import inspect
import ast

class DocGenerator:
    @staticmethod
    def extract_docstring(func):
        """
        提取函数文档字符串

        Args:
            func: Python 函数

        Returns:
            dict: 文档信息
        """
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)

        return {
            "name": func.__name__,
            "signature": str(signature),
            "docstring": docstring,
            "source_file": inspect.getfile(func),
            "line_number": inspect.getsourcelines(func)[1]
        }

    @staticmethod
    def generate_module_docs(module):
        """
        生成模块的完整文档

        Args:
            module: Python 模块

        Returns:
            str: Markdown 格式文档
        """
        doc = f"# {module.__name__}\n\n"
        doc += f"模块文档: {module.__doc__ or '(无文档)'}\n\n"

        # 提取所有函数和类
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if name.startswith('_'):
                    continue

                doc += f"## {name}\n\n"
                doc += f"```python\n{inspect.signature(obj)}\n```\n\n"

                if obj.__doc__:
                    doc += f"{inspect.getdoc(obj)}\n\n"

        return doc

# 使用示例
def example_function(x, y):
    """
    计算两个数的和

    Args:
        x (int): 第一个数
        y (int): 第二个数

    Returns:
        int: 两数之和
    """
    return x + y

info = DocGenerator.extract_docstring(example_function)
print(f"函数名: {info['name']}")
print(f"签名: {info['signature']}")
print(f"文档: {info['docstring']}")
```

## 集成到 Agent

```python
class PythonDevAgent:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.generator = PythonCodeGenerator(model_path)

    def solve_problem(self, problem_description):
        """
        解决 Python 开发问题

        Args:
            problem_description (str): 问题描述

        Returns:
            str: 解决方案
        """
        prompt = f"""你是一个 Python 开发专家。请根据以下问题提供详细的解决方案:

问题: {problem_description}

请包含:
1. 问题分析
2. 解决方案代码
3. 关键要点解释
4. 性能考虑

解决方案:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=1500)
        solution = self.tokenizer.decode(outputs[0])

        return solution

# 使用示例
agent = PythonDevAgent("Qwen/Qwen2.5-0.5B-Instruct")

problem = "如何高效处理大型 CSV 文件?"
solution = agent.solve_problem(problem)
print(solution)
```

## 依赖库

```bash
pip install pytest coverage memory-profiler pandas numpy requests
```

## 最佳实践

1. **代码规范**: 遵循 PEP 8 风格指南。
2. **类型提示**: 使用类型注解提升代码可读性。
3. **错误处理**: 合理使用 try-except 和自定义异常。
4. **单元测试**: 为关键功能编写测试。
5. **文档完善**: 使用文档字符串和类型注解。
6. **性能优化**: 定期进行性能分析和优化。
7. **版本管理**: 使用虚拟环境隔离项目依赖。

## 相关资源

- [PEP 8 风格指南](https://www.python.org/dev/peps/pep-0008/)
- [Pytest 文档](https://docs.pytest.org/)
- [Python 官方文档](https://docs.python.org/3/)
- [Real Python](https://realpython.com/)

