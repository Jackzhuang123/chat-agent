#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载脚本
自动从 Hugging Face 下载 Qwen2.5-0.5B-Instruct 模型
"""

import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ 缺少 huggingface_hub 库")
    print("请先安装: pip install huggingface_hub")
    sys.exit(1)


def download_model():
    """下载 Qwen2.5-0.5B-Instruct 模型"""

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = "./model/qwen2.5-0.5b"

    print("=" * 60)
    print("📥 Qwen2.5-0.5B-Instruct 模型下载器")
    print("=" * 60)
    print(f"模型 ID: {model_id}")
    print(f"保存路径: {local_dir}")
    print(f"模型大小: 约 2GB")
    print("=" * 60)

    # 检查模型是否已存在
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print("⚠️  检测到模型文件已存在")
        user_input = input("是否重新下载? (y/N): ").strip().lower()
        if user_input != 'y':
            print("✅ 使用现有模型")
            return

    print("\n开始下载,请耐心等待...")
    print("(如果速度慢,可以使用镜像站点)")
    print("-" * 60)

    try:
        # 下载模型
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接
            resume_download=True,          # 支持断点续传
        )

        print("-" * 60)
        print("✅ 模型下载完成!")
        print(f"📁 模型位置: {os.path.abspath(local_dir)}")
        print("\n现在可以运行: python web_agent_advanced.py")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 解决方案:")
        print("1. 检查网络连接")
        print("2. 使用国内镜像:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("   python download_model.py")
        print("3. 手动下载:")
        print(f"   访问 https://huggingface.co/{model_id}")
        sys.exit(1)


if __name__ == "__main__":
    # 使用镜像站点 (可选,提高国内下载速度)
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    download_model()

