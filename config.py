#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置模块
============

集中管理智能辩论系统的所有配置参数，包括：
- 模型配置：指定使用的大语言模型
- 设备配置：自动选择最佳计算设备
- 生成参数：控制模型输出的质量和多样性
- 辩论参数：控制辩论的轮数和共识判定
- 工具配置：RAG检索和搜索相关参数

环境变量支持
------------
可通过环境变量覆盖默认配置：
- DEBATE_MODEL_NAME: 模型名称
- DEBATE_MAX_ROUNDS: 最大辩论轮数
- DEBATE_TEMPERATURE: 生成温度

使用示例
--------
>>> from config import MODEL_NAME, MAX_DEBATE_ROUNDS
>>> print(f"模型: {MODEL_NAME}")
>>> print(f"最大轮数: {MAX_DEBATE_ROUNDS}")
"""

import os
from typing import Dict, Any
import torch

# =============================================================================
# 模型配置
# =============================================================================
# 指定使用的大语言模型，支持 HuggingFace 模型ID或本地路径
# 可通过环境变量 DEBATE_MODEL_NAME 覆盖
MODEL_NAME = os.getenv("DEBATE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


def get_device() -> torch.device:
    """
    自动选择最佳计算设备
    
    优先级: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Returns
    -------
    torch.device
        选定的计算设备
        
    Notes
    -----
    - CUDA: 推荐 16GB+ 显存，使用 float16 加速
    - MPS: 适用于 M1/M2/M3 Mac，使用 float32 保证数值稳定
    - CPU: 作为兜底方案，速度较慢但兼容性最好
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ 使用 Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("✓ 使用 CPU (速度较慢)")
    return device


# =============================================================================
# 模型生成参数
# =============================================================================
# 控制大语言模型生成文本的行为
# 可通过环境变量覆盖部分参数
GENERATION_CONFIG: Dict[str, Any] = {
    "max_new_tokens": int(os.getenv("DEBATE_MAX_TOKENS", "1024")),
    "temperature": float(os.getenv("DEBATE_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("DEBATE_TOP_P", "0.9")),
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# =============================================================================
# 辩论系统参数
# =============================================================================
MAX_DEBATE_ROUNDS = int(os.getenv("DEBATE_MAX_ROUNDS", "5"))
CONSENSUS_THRESHOLD = float(os.getenv("DEBATE_CONSENSUS_THRESHOLD", "0.8"))

# =============================================================================
# RAG (检索增强生成) 配置
# =============================================================================
RAG_TOP_K = 3               # 每次检索返回的文档数量
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型

# =============================================================================
# 搜索工具配置
# =============================================================================
SEARCH_NUM_RESULTS = 5      # Google搜索返回的最大结果数
