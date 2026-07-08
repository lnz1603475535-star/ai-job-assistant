"""
第 8 课：RAG 基础建设 —— 文档加载、切分、向量化

核心概念：
  - Document Loader：从各种格式（txt、pdf、csv）读取文档
  - RecursiveCharacterTextSplitter：把长文档切成小块（chunk）
  - Embeddings：把文字变成高维向量（语义相似 = 向量距离近）
  - FAISS：高性能向量数据库，存储和搜索向量
  - Similarity Search：用自然语言搜文档，不是关键词匹配

什么是 RAG（Retrieval-Augmented Generation）？
  检索增强生成 = 先从知识库里"检索"相关文档，再让 AI "生成"回答。
  解决的问题：AI 训练数据是旧的、不知道你的私密文档、容易编造事实。

学完这课你就理解了：RAG 的第一步是"把文档变成可搜索的知识库"——
加载 → 切分 → 向量化 → 存库，这四步是所有 RAG 系统的地基。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

load_dotenv(find_dotenv())

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ============================================================
# 实验①：Document Loader —— 从文件读取文档
# ============================================================
print("=" * 60)
print("实验①：Document Loader —— 加载不同格式的文档")
print("=" * 60)

# TextLoader：加载纯文本文件
resume_path = os.path.join(DATA_DIR, "resume_zhangsan.txt")
jd_path = os.path.join(DATA_DIR, "jd_python_senior.txt")

loader = TextLoader(resume_path, encoding="utf-8")
resume_docs = loader.load()
print(f"加载简历：{resume_path}")
print(f"  文档数量：{len(resume_docs)} 个")
print(f"  文档类型：{type(resume_docs[0]).__name__}")
print(f"  内容长度：{len(resume_docs[0].page_content)} 字符")
print(f"  元数据：{resume_docs[0].metadata}")
print(f"  内容预览：{resume_docs[0].page_content[:150]}...\n")

loader = TextLoader(jd_path, encoding="utf-8")
jd_docs = loader.load()
print(f"加载 JD：{jd_path}")
print(f"  内容预览：{jd_docs[0].page_content[:150]}...")

print("\n>>> Document Loader 统一了不同格式的文件读取，输出都是 Document 对象。")


# ============================================================
# 实验②：RecursiveCharacterTextSplitter —— 文本切分
# ============================================================
print("=" * 60)
print("实验②：RecursiveCharacterTextSplitter —— 把长文档切成小块")
print("=" * 60)

# 为什么要切分？
#   1. LLM 有上下文长度限制（token 上限）
#   2. 小块更容易精确匹配到相关段落
#   3. 太多不相关内容会"稀释"AI 的注意力

# chunk_size：每块最多多少字符
# chunk_overlap：相邻两块之间重叠多少字符（避免关键信息被切断）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 优先在段落/句子边界断开
)

# 把简历和 JD 合并后一起切分
all_docs = resume_docs + jd_docs
chunks = splitter.split_documents(all_docs)

print(f"切分前：{len(all_docs)} 个文档")
print(f"切分后：{len(chunks)} 个 chunk")
print(f"chunk_size=200, chunk_overlap=50\n")

print("切分后的 chunks：")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} (长度: {len(chunk.page_content)} 字符, 来源: {chunk.metadata.get('source', '?')[-30:]}) ---")
    print(chunk.page_content[:120].replace("\n", "↵"))
    print()

print(">>> chunk_overlap 确保关键信息跨 chunk 时不会丢失。")


# ============================================================
# 实验③：Embeddings —— 把文字变成向量
# ============================================================
print("=" * 60)
print("实验③：Embeddings —— 语义相近的文字，向量也相近")
print("=" * 60)

# 用 sentence-transformers 本地模型（免费、无需 API Key）
# 模型首次运行会自动下载，约 100MB
print("加载 embedding 模型（首次运行会下载，请稍候）...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",  # 中文语义模型
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("模型加载完成！\n")

# 演示：把文字变成向量
texts = [
    "Python 后端开发工程师",
    "Django 框架开发",
    "今天天气不错适合出去玩",
]

print("看看语义相近的向量是否真的更接近：")
for i, t in enumerate(texts):
    vec = embeddings.embed_query(t)
    print(f"  [{i}] \"{t}\" → 向量 (维度={len(vec)}, 前5维=[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}...])")

# 计算相似度（向量点积越接近1越相似）
import numpy as np
v0 = embeddings.embed_query(texts[0])
v1 = embeddings.embed_query(texts[1])
v2 = embeddings.embed_query(texts[2])

# 余弦相似度 = 向量点积（因为已经 normalize 了）
sim_01 = np.dot(v0, v1)  # "Python后端" vs "Django开发" —— 应该高
sim_02 = np.dot(v0, v2)  # "Python后端" vs "天气" —— 应该低

print(f"\n语义相似度：")
print(f"  \"{texts[0]}\" vs \"{texts[1]}\" → {sim_01:.3f} {'✓ 高（都是编程）' if sim_01 > 0.5 else ''}")
print(f"  \"{texts[0]}\" vs \"{texts[2]}\" → {sim_02:.3f} {'✓ 低（无关）' if sim_02 < 0.5 else ''}")

print("\n>>> Embeddings 的核心思想：向量距离 = 语义距离。")


# ============================================================
# 实验④：FAISS 向量库 —— 存储和搜索
# ============================================================
print("=" * 60)
print("实验④：FAISS —— 存储向量 + 相似度搜索")
print("=" * 60)

# 把切分好的文档 chunks 向量化后存入 FAISS
# from_documents 会自动调用 embeddings 把文档变成向量
vectorstore = FAISS.from_documents(chunks, embeddings)

print(f"FAISS 索引已创建")
print(f"  存储的文档数：{vectorstore.index.ntotal}")
print(f"  向量维度：{vectorstore.index.d}")

# similarity_search：用自然语言搜索最相关的文档
query = "这个人的 Python 开发经验如何？"
print(f"\n搜索：「{query}」\n")

results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    similarity = ""  # FAISS similarity_search 默认不返回分数
    print(f"结果 {i+1}（来源: {doc.metadata.get('source', '?')[-30:]}）:")
    print(f"  {doc.page_content[:150]}...")
    print()

# 带分数的搜索
print("--- 带相似度分数的搜索 ---\n")
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(results_with_scores):
    # FAISS 返回的是 L2 距离，越小越相似
    print(f"结果 {i+1}: L2距离={score:.2f} | {doc.page_content[:80]}...")

print("\n>>> 你刚才搜索的不是关键词，而是自然语言。这就是向量搜索的威力。")


# ============================================================
# 实验⑤：对比 —— 关键词搜索 vs 语义搜索
# ============================================================
print("=" * 60)
print("实验⑤：关键词搜索 vs 语义搜索 —— 决定性的对比")
print("=" * 60)

# 构造一个"意思相同但措辞完全不同"的查询
query = "这位候选人的技术水平如何？"
keyword = "Python"

print(f"查询：「{query}」")
print(f"如果用关键词 \"{keyword}\" 去搜...")

# 关键词方式的结果（模拟 Ctrl+F）
keyword_results = [doc for doc in chunks if keyword.lower() in doc.page_content.lower()]
print(f"  关键词匹配：{len(keyword_results)} 个 chunk，但不一定是最相关的\n")

# 语义搜索结果
semantic_results = vectorstore.similarity_search(query, k=2)
print("语义搜索结果：")
for i, doc in enumerate(semantic_results):
    source = doc.metadata.get('source', '?').split('/')[-1].split('\\')[-1]
    print(f"  {i+1}. [{source}] {doc.page_content[:100]}...")

print("\n>>> 「技术水平」和「Python」「Django」「架构」语义相关，")
print("    即使文本中没有出现「技术水平」这个词，向量搜索也能找到！")


# ============================================================
# 实验⑥：MMR 搜索 —— 兼顾相关性和多样性
# ============================================================
print("=" * 60)
print("实验⑥：MMR 搜索 —— 不只相关，还要多样")
print("=" * 60)

# 普通相似度搜索可能会返回内容很接近的 3 个 chunk
print("普通相似度搜索结果：")
for i, doc in enumerate(vectorstore.similarity_search("技术经验", k=3)):
    print(f"  {i+1}. {doc.page_content[:80]}...")

# MMR 搜索会挑选：既相关又彼此不太重复的 chunk
print("\nMMR 搜索（相关性+多样性）结果：")
for i, doc in enumerate(vectorstore.max_marginal_relevance_search("技术经验", k=3, fetch_k=6)):
    print(f"  {i+1}. {doc.page_content[:80]}...")

print("\n>>> MMR 避免搜出 3 个内容几乎一样的 chunk，让 AI 看到更全面的信息。")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 8 课 总结 —— RAG 四大基础设施建设")
print("=" * 60)
print("""
任何 RAG 系统都依赖这四步：

1. Document Loader（加载）
     TextLoader / PyPDFLoader / CSVLoader ...
     作用：把各种格式的文件变成 Document 对象

2. Text Splitter（切分）
     RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
     作用：长文档 → 小 chunks，适配 LLM context window

3. Embeddings（向量化）
     HuggingFaceEmbeddings(model_name="中文模型")
     作用：文字 → 高维向量，语义相似 = 向量距离近

4. Vector Store（存储+搜索）
     FAISS.from_documents(chunks, embeddings)
     作用：存向量 + 快速搜索最相关的文档

为什么用 HuggingFace 而不用 OpenAI Embeddings？
  - DeepSeek 不提供 embedding API
  - HuggingFace 本地运行，免费，无需 API Key
  - shibing624/text2vec-base-chinese 是专门的中文语义模型

和后面课程的关联：
  第 5 课 RAG Chain:   把这四步和 LLM 串起来，实现完整问答
  第 6 课 Advanced RAG: 更高级的检索策略
  第 12 课 项目:        用真实简历和 JD 构建求职助手知识库
""")

# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 改 chunk_size=500, chunk_overlap=0，看看搜索结果有什么变化？
   什么时候 chunk 大一点好？什么时候小一点好？
2. 在 data/ 下放一份你自己的简历或任何文本文档，加载、切分、搜索
3. 试一下 PyPDFLoader 加载 PDF（如果 data/ 下有 PDF 的话）
4. 试一下 MMR 搜索和普通搜索，在什么场景下 MMR 明显更好？
""")
