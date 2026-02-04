"""
工具模块 (tools)
================

本模块提供智能体可用的外部工具：

类列表
------
SearchTool
    网络搜索工具，使用 Google 搜索获取实时信息
RAGTool
    检索增强生成工具，从本地知识库检索相关文档

使用示例
--------
>>> from tools import SearchTool, RAGTool
>>> 
>>> # 网络搜索
>>> search = SearchTool()
>>> results = search.search("人工智能最新进展")
>>> print(search.format_results(results))
>>> 
>>> # RAG 检索
>>> rag = RAGTool()
>>> rag.add_documents(["文档1内容", "文档2内容"])
>>> results = rag.retrieve("关键词")
>>> print(rag.format_context(results))

依赖说明
--------
- SearchTool 依赖: googlesearch-python, requests, beautifulsoup4
- RAGTool 依赖: sentence-transformers, faiss-cpu

如果未安装相关依赖，工具会优雅降级，返回空结果。
"""

from .search_tool import SearchTool
from .rag_tool import RAGTool

__all__ = ['SearchTool', 'RAGTool']
