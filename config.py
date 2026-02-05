#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置模块"""

import os
import torch

# 模型配置
MODEL_NAME = os.getenv("DEBATE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 生成参数
GENERATION_CONFIG = {
    "max_new_tokens": int(os.getenv("DEBATE_MAX_TOKENS", "1024")),
    "temperature": float(os.getenv("DEBATE_TEMPERATURE", "0.7")),
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# 辩论参数
MAX_DEBATE_ROUNDS = int(os.getenv("DEBATE_MAX_ROUNDS", "5"))
CONSENSUS_THRESHOLD = 0.8

# RAG配置
RAG_TOP_K = 3


def get_device() -> torch.device:
    """自动选择最佳计算设备"""
    if torch.cuda.is_available():
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✓ 使用 Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    else:
        print("✓ 使用 CPU (速度较慢)")
        return torch.device("cpu")
