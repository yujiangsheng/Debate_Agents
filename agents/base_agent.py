"""智能体基类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from model_loader import QwenModel
from tools import SearchTool, RAGTool


class BaseAgent(ABC):
    """智能体抽象基类"""
    
    def __init__(self, name: str, role: str, use_search: bool = True, use_rag: bool = True):
        self.name = name
        self.role = role
        self.model = QwenModel()
        self.history: List[Dict[str, str]] = []
        self.search_tool: Optional[SearchTool] = SearchTool() if use_search else None
        self.rag_tool: Optional[RAGTool] = RAGTool() if use_rag else None
        self.system_prompt = self._build_system_prompt()
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """构建系统提示词（子类实现）"""
        pass
    
    def search(self, query: str) -> str:
        """搜索网络信息"""
        if self.search_tool is None:
            return ""
        results = self.search_tool.search(query)
        return self.search_tool.format_results(results)
    
    def retrieve(self, query: str) -> str:
        """检索知识库"""
        if self.rag_tool is None:
            return ""
        results = self.rag_tool.retrieve(query)
        return self.rag_tool.format_context(results)
    
    def add_knowledge(self, documents: List[str]):
        """向知识库添加文档"""
        if self.rag_tool:
            self.rag_tool.add_documents(documents)
    
    def generate(self, prompt: str, context: str = "") -> str:
        """生成回复"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        user_msg = f"参考信息:\n{context}\n\n{prompt}" if context else prompt
        messages.append({"role": "user", "content": user_msg})
        
        response = self.model.generate(messages)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def reset_history(self):
        """重置对话历史"""
        self.history = []
