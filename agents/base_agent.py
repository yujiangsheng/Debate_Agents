#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础智能体类
============

定义所有智能体的抽象基类，提供通用功能：
- 语言模型调用
- 搜索工具集成
- RAG 知识库检索
- 对话历史管理

所有具体智能体 (辩论者、裁判) 都继承此类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict

from model_loader import QwenModel
from tools import SearchTool, RAGTool


class BaseAgent(ABC):
    """
    智能体抽象基类
    
    提供智能体的通用功能，子类需实现 `_build_system_prompt` 方法
    来定义各自的角色和行为。
    
    Attributes
    ----------
    name : str
        智能体名称，如 "智能体A"、"裁判C"
    role : str
        角色描述，用于日志和显示
    model : QwenModel
        语言模型实例 (单例，所有智能体共享)
    history : List[Dict[str, str]]
        对话历史，用于多轮对话上下文
    search_tool : SearchTool or None
        网络搜索工具
    rag_tool : RAGTool or None
        RAG 知识库检索工具
    system_prompt : str
        系统提示词，定义智能体的角色和行为
    """
    
    def __init__(self, name: str, role: str, 
                 use_search: bool = True, use_rag: bool = True):
        """
        初始化智能体
        
        Parameters
        ----------
        name : str
            智能体名称，如 "智能体A"
        role : str
            角色描述，如 "正方辩论者"
        use_search : bool, optional
            是否启用网络搜索 (默认: True)
        use_rag : bool, optional
            是否启用 RAG 知识库 (默认: True)
        """
        self.name = name
        self.role = role
        self.model = QwenModel()  # 单例模式，所有智能体共享同一模型
        self.history: List[Dict[str, str]] = []
        
        # 初始化工具 (根据参数决定是否启用)
        self.search_tool = SearchTool() if use_search else None
        self.rag_tool = RAGTool() if use_rag else None
        
        # 构建系统提示词 (由子类实现)
        self.system_prompt = self._build_system_prompt()
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词 (抽象方法)
        
        子类必须实现此方法，返回定义智能体角色和行为的提示词。
        
        Returns
        -------
        str
            系统提示词
        """
        pass
    
    def search(self, query: str) -> str:
        """
        使用搜索工具获取网络信息
        
        Parameters
        ----------
        query : str
            搜索关键词
            
        Returns
        -------
        str
            格式化的搜索结果，未启用搜索则返回空字符串
        """
        if self.search_tool is None:
            return ""
        results = self.search_tool.search(query)
        return self.search_tool.format_results(results)
    
    def retrieve(self, query: str) -> str:
        """
        使用 RAG 检索知识库中的相关文档
        
        Parameters
        ----------
        query : str
            检索查询
            
        Returns
        -------
        str
            格式化的检索结果，未启用 RAG 则返回空字符串
        """
        if self.rag_tool is None:
            return ""
        results = self.rag_tool.retrieve(query)
        return self.rag_tool.format_context(results)
    
    def add_knowledge(self, documents: List[str]):
        """
        向 RAG 知识库添加文档
        
        Parameters
        ----------
        documents : List[str]
            文档字符串列表
        """
        if self.rag_tool:
            self.rag_tool.add_documents(documents)
    
    def generate(self, prompt: str, context: str = "") -> str:
        """
        调用语言模型生成回复
        
        Parameters
        ----------
        prompt : str
            输入提示
        context : str, optional
            额外上下文信息 (来自搜索或 RAG)
            
        Returns
        -------
        str
            模型生成的回复文本
        """
        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)  # 添加历史对话
        
        # 如果有上下文，将其添加到提示中
        user_msg = f"参考信息:\n{context}\n\n{prompt}" if context else prompt
        messages.append({"role": "user", "content": user_msg})
        
        # 调用模型生成回复
        response = self.model.generate(messages)
        
        # 更新对话历史
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_history(self):
        """
        重置对话历史
        
        在开始新一场辩论前调用，清空之前的对话记录。
        """
        self.history = []
