#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 工具 - 检索增强生成
=======================

提供基于向量检索的知识库功能，用于从本地文档中检索相关信息。

技术栈
------
- sentence-transformers: 文本嵌入 (向量化)
- FAISS: 高效向量相似度搜索

工作原理
--------
1. 文档向量化: 使用嵌入模型将文档转换为向量
2. 构建索引: 使用 FAISS 构建向量索引
3. 相似度检索: 查询时计算向量相似度，返回最相关的文档

使用示例
--------
>>> from tools import RAGTool
>>> rag = RAGTool()
>>> 
>>> # 添加文档到知识库
>>> rag.add_documents([
...     "人工智能是计算机科学的一个分支...",
...     "机器学习是AI的核心技术之一..."
... ])
>>> 
>>> # 检索相关文档
>>> results = rag.retrieve("什么是人工智能")
>>> print(rag.format_context(results))

依赖要求
--------
- sentence-transformers: 文本嵌入模型
- faiss-cpu: 向量索引和检索

如果未安装依赖，工具会优雅降级，返回空结果。
"""

from typing import List, Dict, Optional
import numpy as np

# 尝试导入依赖 (可选)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from config import RAG_TOP_K, EMBEDDING_MODEL


class RAGTool:
    """
    检索增强生成 (RAG) 工具
    
    使用向量检索从知识库中查找与查询最相关的文档，
    为智能体提供额外的知识支撑。
    
    Attributes
    ----------
    documents : List[str]
        存储的文档列表
    embeddings : np.ndarray or None
        文档的向量表示
    index : faiss.Index or None
        FAISS 向量索引
    model : SentenceTransformer or None
        文本嵌入模型
        
    使用示例
    --------
    >>> rag = RAGTool()
    >>> rag.add_documents(["文档1", "文档2"])
    >>> results = rag.retrieve("查询")
    >>> print(rag.format_context(results))
    """
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """
        初始化 RAG 工具
        
        Parameters
        ----------
        embedding_model : str, optional
            嵌入模型名称或路径 (默认: config.EMBEDDING_MODEL)
        """
        self.documents: List[str] = []          # 文档存储
        self.embeddings: Optional[np.ndarray] = None  # 向量存储
        self.index = None                        # FAISS 索引
        self.model = None                        # 嵌入模型
        
        # 如果依赖可用，加载嵌入模型
        if RAG_AVAILABLE:
            print(f"正在加载嵌入模型: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            print("✓ 嵌入模型加载成功!")
    
    def add_documents(self, documents: List[str]):
        """
        添加文档到知识库
        
        将文档向量化后加入索引，支持增量添加。
        
        Parameters
        ----------
        documents : List[str]
            文档字符串列表
        """
        if not RAG_AVAILABLE:
            return
        
        # 将文档添加到列表
        self.documents.extend(documents)
        
        # 计算新文档的向量表示
        new_embeddings = self.model.encode(documents)
        
        # 合并向量
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # 重建索引
        self._build_index()
    
    def _build_index(self):
        """
        构建 FAISS 向量索引
        
        使用 L2 距离 (欧氏距离) 进行相似度计算。
        """
        if self.embeddings is None:
            return
        dim = self.embeddings.shape[1]  # 向量维度
        self.index = faiss.IndexFlatL2(dim)  # 创建 L2 索引
        self.index.add(self.embeddings.astype('float32'))  # 添加向量
    
    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """
        检索相关文档
        
        Parameters
        ----------
        query : str
            查询字符串
        top_k : int, optional
            返回的文档数量 (默认: config.RAG_TOP_K)
            
        Returns
        -------
        List[Dict]
            相关文档列表，每项包含:
            - document: 文档内容
            - score: 相似度分数 (L2 距离，越小越相似)
            - index: 文档在知识库中的索引
        """
        if not RAG_AVAILABLE or self.index is None:
            return []
        
        # 将查询向量化
        query_vec = self.model.encode([query])
        
        # 执行相似度搜索
        dists, idxs = self.index.search(
            query_vec.astype('float32'), 
            min(top_k, len(self.documents))  # 不超过文档总数
        )
        
        # 构建结果列表
        return [
            {"document": self.documents[i], "score": float(d), "index": int(i)}
            for d, i in zip(dists[0], idxs[0]) if i < len(self.documents)
        ]
    
    def format_context(self, results: List[Dict]) -> str:
        """
        格式化检索结果
        
        Parameters
        ----------
        results : List[Dict]
            retrieve() 返回的结果列表
            
        Returns
        -------
        str
            格式化的上下文字符串，可直接用于提示词
        """
        if not results:
            return ""
        
        text = "【相关知识】\n"
        for i, r in enumerate(results, 1):
            text += f"[{i}] {r['document'][:200]}...\n"
        return text
    
    def clear(self):
        """
        清空知识库
        
        删除所有文档、向量和索引。
        """
        self.documents = []
        self.embeddings = None
        self.index = None
