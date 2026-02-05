"""RAG工具 - 检索增强生成"""
from typing import List, Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from config import RAG_TOP_K, EMBEDDING_MODEL


class RAGTool:
    """检索增强生成(RAG)工具"""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.model = None
        
        if RAG_AVAILABLE:
            print(f"正在加载嵌入模型: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            print("✓ 嵌入模型加载成功!")
    
    def add_documents(self, documents: List[str]):
        """添加文档到知识库"""
        if not RAG_AVAILABLE:
            return
        self.documents.extend(documents)
        new_embeddings = self.model.encode(documents)
        self.embeddings = new_embeddings if self.embeddings is None else np.vstack([self.embeddings, new_embeddings])
        self._build_index()
    
    def _build_index(self):
        """构建FAISS向量索引"""
        if self.embeddings is None:
            return
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """检索相关文档"""
        if not RAG_AVAILABLE or self.index is None:
            return []
        query_vec = self.model.encode([query])
        dists, idxs = self.index.search(query_vec.astype('float32'), min(top_k, len(self.documents)))
        return [
            {"document": self.documents[i], "score": float(d), "index": int(i)}
            for d, i in zip(dists[0], idxs[0]) if i < len(self.documents)
        ]
    
    def format_context(self, results: List[Dict]) -> str:
        """格式化检索结果"""
        if not results:
            return ""
        text = "【相关知识】\n"
        for i, r in enumerate(results, 1):
            text += f"[{i}] {r['document'][:200]}...\n"
        return text
