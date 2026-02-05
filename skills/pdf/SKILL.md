---
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
