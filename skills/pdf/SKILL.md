# PDF 处理技能库

## 概述
提供 PDF 文件的解析、提取、转换等功能,可集成到 Qwen Agent 中进行文档智能分析。

## 核心功能

### 1. PDF 文本提取
从 PDF 文件中提取文本内容。

```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件提取文本

    Args:
        pdf_path (str): PDF 文件路径

    Returns:
        str: 提取的文本内容
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# 使用示例
text = extract_text_from_pdf("document.pdf")
print(text)
```

### 2. 页面分割
将 PDF 分解为单个页面。

```python
from PyPDF2 import PdfReader, PdfWriter

def split_pdf_by_page(pdf_path, output_dir="output"):
    """
    将 PDF 按页分割,生成单页 PDF 文件

    Args:
        pdf_path (str): 源 PDF 路径
        output_dir (str): 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        output_path = f"{output_dir}/page_{page_num + 1}.pdf"
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

    return len(reader.pages)

# 使用示例
num_pages = split_pdf_by_page("document.pdf")
print(f"已分割为 {num_pages} 页")
```

### 3. 元数据提取
提取 PDF 的元数据(作者、标题、创建日期等)。

```python
from PyPDF2 import PdfReader
from datetime import datetime

def extract_metadata(pdf_path):
    """
    提取 PDF 元数据

    Args:
        pdf_path (str): PDF 文件路径

    Returns:
        dict: 元数据信息
    """
    reader = PdfReader(pdf_path)
    metadata = reader.metadata

    info = {
        "title": metadata.get("/Title", "Unknown"),
        "author": metadata.get("/Author", "Unknown"),
        "subject": metadata.get("/Subject", "Unknown"),
        "creator": metadata.get("/Creator", "Unknown"),
        "pages": len(reader.pages),
        "creation_date": metadata.get("/CreationDate", "Unknown"),
    }

    return info

# 使用示例
meta = extract_metadata("document.pdf")
for key, value in meta.items():
    print(f"{key}: {value}")
```

### 4. OCR 识别 (图片型 PDF)
使用 Tesseract 对 PDF 中的图片进行 OCR 识别。

```python
import pdf2image
import pytesseract
from PIL import Image
import os

def ocr_pdf(pdf_path, output_text_path="output.txt"):
    """
    对 PDF 进行 OCR 识别,提取图片中的文字

    需要安装: pip install pdf2image pytesseract
    需要系统安装: Tesseract-OCR

    Args:
        pdf_path (str): PDF 文件路径
        output_text_path (str): 输出文本文件路径
    """
    # 将 PDF 转换为图片
    images = pdf2image.convert_from_path(pdf_path)

    full_text = ""
    for page_num, image in enumerate(images):
        # 使用 Tesseract 进行 OCR
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        full_text += f"--- 第 {page_num + 1} 页 ---\n{text}\n"

    # 保存输出
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return full_text

# 使用示例
text = ocr_pdf("scanned_document.pdf")
```

### 5. PDF 合并
合并多个 PDF 文件。

```python
from PyPDF2 import PdfMerger

def merge_pdfs(pdf_list, output_path="merged.pdf"):
    """
    合并多个 PDF 文件

    Args:
        pdf_list (list): PDF 文件路径列表
        output_path (str): 输出文件路径
    """
    merger = PdfMerger()

    for pdf in pdf_list:
        merger.append(pdf)

    merger.write(output_path)
    merger.close()

    return output_path

# 使用示例
pdfs = ["file1.pdf", "file2.pdf", "file3.pdf"]
merge_pdfs(pdfs, "combined.pdf")
```

### 6. PDF 水印
为 PDF 添加文本或图片水印。

```python
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def add_watermark(input_pdf, output_pdf, watermark_text="CONFIDENTIAL"):
    """
    为 PDF 添加文本水印

    Args:
        input_pdf (str): 输入 PDF 路径
        output_pdf (str): 输出 PDF 路径
        watermark_text (str): 水印文本
    """
    # 创建水印
    packet = BytesIO()
    watermark_canvas = canvas.Canvas(packet, pagesize=letter)
    watermark_canvas.setFont("Helvetica", 60)
    watermark_canvas.setFillAlpha(0.3)
    watermark_canvas.drawString(100, 400, watermark_text)
    watermark_canvas.save()

    # 读取源 PDF
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # 添加水印到每一页
    from PyPDF2 import PdfReader as Reader
    watermark = Reader(packet)
    watermark_page = watermark.pages[0]

    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)

    # 保存输出
    with open(output_pdf, 'wb') as output_file:
        writer.write(output_file)

# 使用示例
add_watermark("document.pdf", "watermarked.pdf", "机密文件")
```

## 依赖库

```bash
pip install PyPDF2 pdf2image pytesseract reportlab pillow
```

## 集成到 Agent

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf_skill import extract_text_from_pdf

class PDFAgent:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def analyze_pdf(self, pdf_path, query):
        """
        分析 PDF 文件并回答相关问题

        Args:
            pdf_path (str): PDF 文件路径
            query (str): 用户查询

        Returns:
            str: 模型的回答
        """
        # 提取 PDF 文本
        pdf_text = extract_text_from_pdf(pdf_path)

        # 构建 prompt
        prompt = f"""请分析以下 PDF 文档内容并回答问题:

文档内容:
{pdf_text[:2000]}...

问题: {query}

回答:"""

        # 调用模型
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)
        answer = self.tokenizer.decode(outputs[0])

        return answer

# 使用示例
agent = PDFAgent("Qwen/Qwen2.5-0.5B-Instruct")
result = agent.analyze_pdf("contract.pdf", "这份合同的主要条款是什么?")
print(result)
```

## 最佳实践

1. **大文件处理**: 对于大型 PDF,分页处理以节省内存。
2. **编码问题**: 确保正确处理中文和特殊字符。
3. **错误处理**: 添加异常捕获以处理损坏的 PDF。
4. **性能优化**: 使用缓存存储已提取的文本。
5. **隐私保护**: 处理敏感 PDF 时,及时删除临时文件。

## 相关资源

- [PyPDF2 文档](https://github.com/py-pdf/PyPDF2)
- [pdf2image 文档](https://github.com/Belval/pdf2image)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

