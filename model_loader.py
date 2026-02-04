#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载器 - Qwen 模型封装 (单例模式)
=====================================

本模块提供大语言模型的统一封装，确保模型只加载一次，
多个智能体共享同一实例，有效节省显存/内存。

核心特性
--------
- 单例模式: 全局只有一个模型实例
- 设备自适应: 自动选择 CUDA/MPS/CPU
- 对话模板: 自动应用模型的聊天模板
- 错误处理: 优雅的加载失败处理

使用示例
--------
>>> from model_loader import QwenModel
>>> model = QwenModel()  # 首次调用会加载模型
>>> model2 = QwenModel()  # 返回同一实例，不会重复加载
>>> assert model is model2  # True

>>> response = model.generate([
...     {"role": "system", "content": "你是一个助手"},
...     {"role": "user", "content": "你好"}
... ])
>>> print(response)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional

from config import MODEL_NAME, get_device, GENERATION_CONFIG
from exceptions import ModelLoadError


class QwenModel:
    """
    Qwen 大语言模型封装类 (单例模式)
    
    采用单例模式确保模型只加载一次，所有智能体共享同一实例，
    有效节省显存/内存资源。
    
    Attributes
    ----------
    _instance : QwenModel
        单例实例
    _model : AutoModelForCausalLM
        加载的语言模型
    _tokenizer : AutoTokenizer
        分词器
    _device : torch.device
        计算设备
    
    使用示例
    --------
    >>> model = QwenModel()
    >>> response = model.generate([
    ...     {"role": "system", "content": "你是一个助手"},
    ...     {"role": "user", "content": "你好"}
    ... ])
    >>> print(response)
    """
    
    # =========================================================================
    # 单例模式的类变量
    # =========================================================================
    _instance = None    # 单例实例
    _model = None       # 模型对象
    _tokenizer = None   # 分词器
    _device = None      # 计算设备
    
    def __new__(cls):
        """
        单例模式实现
        
        确保全局只创建一个 QwenModel 实例，后续调用返回同一实例。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化方法
        
        仅在首次创建实例时加载模型，后续调用不会重复加载。
        """
        if QwenModel._model is None:
            self._load_model()
    
    def _load_model(self):
        """
        加载模型和分词器
        
        根据可用设备自动选择最佳配置：
        - CUDA: 使用 float16 精度 + 自动显存分配
        - MPS (Apple Silicon): 使用 float32 避免数值不稳定
        - CPU: 使用 float32，作为兜底方案
        
        Raises
        ------
        ModelLoadError
            模型加载失败时抛出
        """
        print(f"正在加载模型: {MODEL_NAME}")
        
        try:
            # 选择计算设备
            QwenModel._device = get_device()
            
            # 加载分词器
            QwenModel._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            # 根据设备选择数据类型 (MPS用float32避免数值问题)
            dtype = torch.float16 if QwenModel._device.type == "cuda" else torch.float32
            
            # 加载模型
            QwenModel._model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto" if QwenModel._device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # 非CUDA设备需要手动移动模型到设备
            if QwenModel._device.type != "cuda":
                QwenModel._model = QwenModel._model.to(QwenModel._device)
            
            QwenModel._model.eval()  # 设置为评估模式，禁用 dropout
            print("✓ 模型加载成功!")
            
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}") from e
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        生成模型回复
        
        Parameters
        ----------
        messages : List[Dict[str, str]]
            对话消息列表，每条消息包含 role 和 content
            role 可选值: "system", "user", "assistant"
        **kwargs
            额外生成参数，会覆盖默认配置
            可用参数: max_new_tokens, temperature, top_p, do_sample 等
        
        Returns
        -------
        str
            模型生成的回复文本
            
        Examples
        --------
        >>> messages = [
        ...     {"role": "system", "content": "你是辩论专家"},
        ...     {"role": "user", "content": "请阐述你的观点"}
        ... ]
        >>> response = model.generate(messages, temperature=0.8)
        """
        # 合并默认配置和自定义参数
        gen_config = {**GENERATION_CONFIG, **kwargs}
        
        # 应用对话模板
        text = QwenModel._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 分词
        model_inputs = QwenModel._tokenizer(
            [text],
            return_tensors="pt"
        ).to(QwenModel._device)
        
        # 生成回复
        with torch.no_grad():
            if QwenModel._device.type == "mps":
                # MPS设备使用特殊参数避免数值不稳定
                generated_ids = QwenModel._model.generate(
                    **model_inputs,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=max(gen_config["temperature"], 0.1),
                    top_p=gen_config["top_p"],
                    top_k=50,
                    do_sample=gen_config["do_sample"],
                    repetition_penalty=gen_config["repetition_penalty"],
                    pad_token_id=QwenModel._tokenizer.eos_token_id,
                    use_cache=True
                )
            else:
                generated_ids = QwenModel._model.generate(
                    **model_inputs,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_p=gen_config["top_p"],
                    do_sample=gen_config["do_sample"],
                    repetition_penalty=gen_config["repetition_penalty"],
                    pad_token_id=QwenModel._tokenizer.eos_token_id
                )
        
        # 只解码新生成的token
        new_tokens = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = QwenModel._tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
    @property
    def device(self) -> torch.device:
        """获取当前计算设备"""
        return QwenModel._device
    
    @property
    def tokenizer(self):
        """获取分词器"""
        return QwenModel._tokenizer
