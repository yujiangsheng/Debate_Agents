"""搜索工具 - 网络搜索"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time

try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False


class SearchTool:
    """Google搜索工具"""
    
    def __init__(self, num_results: int = 5):
        self.num_results = num_results
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """执行Google搜索"""
        results = []
        if not GOOGLE_SEARCH_AVAILABLE:
            return results
        
        try:
            urls = list(google_search(query, num_results=self.num_results))
            for url in urls:
                try:
                    content = self._fetch_content(url)
                    if content:
                        results.append({"url": url, "title": content.get("title", ""), "snippet": content.get("snippet", "")})
                except Exception:
                    pass
                time.sleep(0.5)
        except Exception as e:
            print(f"搜索出错: {e}")
        return results
    
    def _fetch_content(self, url: str, max_len: int = 500) -> Dict[str, str]:
        """抓取网页标题和摘要"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
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
