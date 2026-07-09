"""
AI 简历生成器 - 核心基础设施
============================
纯基础设施层：LLM 配置、Embedding、文档索引、混合检索。
不包含任何业务逻辑——业务逻辑在 resume_engine.py 中。
"""

import os, re, warnings
from typing import List

# 抑制依赖库的噪音警告（都不是项目代码的问题）
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import jieba
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv(find_dotenv())

# ============================================================
# LLM 配置
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)

# ============================================================
# Embedding 模型（懒加载 + 单例模式）
# ============================================================

_embeddings = None

def get_embeddings():
    """获取 embedding 模型实例（单例模式，首次调用时下载）。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

# ============================================================
# 全局向量库状态
# ============================================================

_vectorstore = None
_bm25_index = None
_chunks_text: List[str] = []
_chunks_metadata: List[dict] = []

def set_vectorstore(vs, bm25=None, chunks=None):
    """设置全局向量库实例，同时保存 BM25 索引和 chunk 元数据。"""
    global _vectorstore, _bm25_index, _chunks_text, _chunks_metadata
    _vectorstore = vs
    _bm25_index = bm25
    if chunks:
        _chunks_text = [c.page_content for c in chunks]
        _chunks_metadata = [c.metadata for c in chunks]

# ============================================================
# 文档处理
# ============================================================

def load_and_index_documents(file_paths: dict) -> tuple:
    """加载多类型文档，切分，建立 FAISS + BM25 双索引。

    参数：
        file_paths: {"doc_type": ["path1", "path2"], ...}
        例如：{"sample_resume": ["samples/xxx.txt"], "jd": ["jd_python.txt"]}

    返回：
        (vectorstore, bm25_index, chunks)
    """
    all_docs = []
    for doc_type, paths in file_paths.items():
        for path in paths:
            docs = TextLoader(path, encoding="utf-8").load()
            for doc in docs:
                doc.metadata["doc_type"] = doc_type
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 建立 BM25 关键词索引（jieba 中文分词）
    bm25 = BM25Okapi([" ".join(jieba.lcut(c.page_content)) for c in chunks])

    return vectorstore, bm25, chunks

# ============================================================
# 工具定义
# ============================================================

@tool
def search_documents(query: str, k: int = 4) -> str:
    """混合搜索文档数据库（FAISS 语义检索 + BM25 关键词匹配）。
    用于查找用户经历、JD 要求、样本风格等已索引的文档内容。
    query：中文或英文的自然语言搜索词。"""
    if _vectorstore is None:
        return "尚未加载任何文档。"

    # FAISS 语义检索（MMR：相关性 + 多样性，避免返回重复内容）
    faiss_docs = _vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=20
    )

    # BM25 关键词检索（jieba 中文分词）
    bm25_scores = _bm25_index.get_scores(jieba.lcut(query))
    if len(bm25_scores) > 0:
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]
    else:
        bm25_top_indices = []

    # 轮换合并 + 去重
    seen = set()
    merged = []
    for i in range(max(len(faiss_docs), len(bm25_top_indices))):
        if i < len(faiss_docs):
            content = faiss_docs[i].page_content
            if content not in seen:
                doc_type = faiss_docs[i].metadata.get("doc_type", "unknown")
                merged.append(f"[{doc_type}] {content}")
                seen.add(content)
        if i < len(bm25_top_indices):
            idx = bm25_top_indices[i]
            content = _chunks_text[idx]
            if content not in seen:
                doc_type = _chunks_metadata[idx].get("doc_type", "unknown")
                merged.append(f"[{doc_type}] {content}")
                seen.add(content)

    return "\n\n---\n\n".join(merged)

# ============================================================
# 通用工具函数
# ============================================================

def parse_experience(level_text: str) -> dict:
    """从水平描述中提取结构化信息。

    支持的格式：
      - "5年" / "5年以上" / "5年+" → {"years": 5, "label": "5年以上"}
      - "2-3年" → {"years": 3, "label": "2-3年"}（取上限）
      - "精通（5年）" → {"years": 5, "label": "精通（5年）"}
      - "精通" / "熟悉" / "了解" → {"years": None, "label": "精通"}

    返回 dict: {"years": int|None, "label": str}
    """
    text = level_text.strip()

    # 1. 匹配 "X-Y年" 或 "X至Y年"（取上限 Y）
    m = re.search(r'(\d+)\s*[-–—至到]\s*(\d+)\s*年?', text)
    if m:
        return {"years": int(m.group(2)), "label": text}

    # 2. 匹配单独数字
    m = re.search(r'(\d+)\s*年?', text)
    if m:
        return {"years": int(m.group(1)), "label": text}

    # 3. 纯文字描述
    return {"years": None, "label": text}

# ============================================================
# Round 3：兼容性垫片已删除（app.py 不再使用）
# ============================================================
