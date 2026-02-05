"""Qwen模型加载器 - 单例模式"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from config import MODEL_NAME, get_device, GENERATION_CONFIG
from exceptions import ModelLoadError


class QwenModel:
    """Qwen大语言模型封装类（单例模式）"""
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if QwenModel._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        print(f"正在加载模型: {MODEL_NAME}")
        try:
            QwenModel._device = get_device()
            QwenModel._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            dtype = torch.float16 if QwenModel._device.type == "cuda" else torch.float32
            
            QwenModel._model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto" if QwenModel._device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            if QwenModel._device.type != "cuda":
                QwenModel._model = QwenModel._model.to(QwenModel._device)
            
            QwenModel._model.eval()
            print("✓ 模型加载成功!")
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}") from e
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """生成模型回复"""
        gen_config = {**GENERATION_CONFIG, **kwargs}
        
        text = QwenModel._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = QwenModel._tokenizer([text], return_tensors="pt").to(QwenModel._device)
        
        with torch.no_grad():
            gen_params = {
                "max_new_tokens": gen_config["max_new_tokens"],
                "temperature": max(gen_config["temperature"], 0.1) if QwenModel._device.type == "mps" else gen_config["temperature"],
                "top_p": gen_config["top_p"],
                "do_sample": gen_config["do_sample"],
                "repetition_penalty": gen_config["repetition_penalty"],
                "pad_token_id": QwenModel._tokenizer.eos_token_id,
            }
            if QwenModel._device.type == "mps":
                gen_params["top_k"] = 50
                gen_params["use_cache"] = True
            
            generated_ids = QwenModel._model.generate(**model_inputs, **gen_params)
            
            if QwenModel._device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        new_tokens = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return QwenModel._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    
    @property
    def device(self) -> torch.device:
        return QwenModel._device
    
    @property
    def tokenizer(self):
        return QwenModel._tokenizer
