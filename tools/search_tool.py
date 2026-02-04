#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索工具 - 网络搜索
==================

提供 Google 网络搜索功能，用于获取实时信息支撑辩论论据。

功能特点
--------
- Google 搜索: 根据关键词搜索网页
- 内容抓取: 自动抓取网页标题和摘要
- 优雅降级: 未安装依赖时返回空结果

依赖要求
--------
- googlesearch-python: Google 搜索
- requests: HTTP 请求
- beautifulsoup4: HTML 解析

使用示例
--------
>>> from tools import SearchTool
>>> search = SearchTool(num_results=5)
>>> results = search.search("人工智能最新进展 2024")
>>> print(search.format_results(results))
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time

# 尝试导入 Google 搜索库 (可选依赖)
try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False


class SearchTool:
    """
    Google 搜索工具
    
    提供网页搜索和内容抓取功能。如果未安装 googlesearch-python，
    搜索功能会优雅降级，返回空结果。
    
    Attributes
    ----------
    num_results : int
        每次搜索返回的最大结果数
    headers : dict
        HTTP 请求头，用于模拟浏览器访问
        
    使用示例
    --------
    >>> search = SearchTool(num_results=5)
    >>> results = search.search("人工智能最新进展")
    >>> print(search.format_results(results))
    """
    
    def __init__(self, num_results: int = 5):
        """
        初始化搜索工具
        
        Parameters
        ----------
        num_results : int, optional
            返回的最大搜索结果数 (默认: 5)
        """
        self.num_results = num_results
        # 设置请求头，模拟浏览器访问
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        执行 Google 搜索
        
        Parameters
        ----------
        query : str
            搜索关键词
            
        Returns
        -------
        List[Dict[str, str]]
            搜索结果列表，每项包含:
            - url: 网页 URL
            - title: 网页标题
            - snippet: 内容摘要
        """
        results = []
        
        # 如果未安装搜索库，返回空结果
        if not GOOGLE_SEARCH_AVAILABLE:
            return results
        
        try:
            # 执行 Google 搜索
            urls = list(google_search(query, num_results=self.num_results))
            for url in urls:
                try:
                    # 抓取网页内容
                    content = self._fetch_content(url)
                    if content:
                        results.append({
                            "url": url,
                            "title": content.get("title", ""),
                            "snippet": content.get("snippet", "")
                        })
                except Exception:
                    pass  # 忽略单个网页抓取失败
                time.sleep(0.5)  # 防止请求过快被封
        except Exception as e:
            print(f"搜索出错: {e}")
        
        return results
    
    def _fetch_content(self, url: str, max_len: int = 500) -> Dict[str, str]:
        """
        抓取网页标题和摘要
        
        Parameters
        ----------
        url : str
            网页 URL
        max_len : int, optional
            摘要最大长度 (默认: 500)
            
        Returns
        -------
        Dict[str, str]
            包含 title 和 snippet 的字典
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 移除脚本和样式
            for tag in soup(["script", "style"]):
                tag.decompose()
            
            title = soup.title.string if soup.title else ""
            text = soup.get_text(separator=' ', strip=True)
            snippet = text[:max_len] + "..." if len(text) > max_len else text
            
            return {"title": title, "snippet": snippet}
        except Exception:
            return {"title": "", "snippet": ""}
    
    def format_results(self, results: List[Dict[str, str]]) -> str:
        """格式化搜索结果"""
        if not results:
            return ""
        
        text = "【搜索结果】\n"
        for i, r in enumerate(results, 1):
            text += f"[{i}] {r.get('title', '')[:50]}\n    {r.get('snippet', '')[:150]}...\n"
        return text
