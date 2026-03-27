---
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
