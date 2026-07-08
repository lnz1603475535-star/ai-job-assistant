"""
第 6 课：Advanced RAG —— 高级检索策略

核心概念：
  - MultiQueryRetriever：一个查询 → 多个改写版本 → 合并去重 → 更多召回
  - ContextualCompressionRetriever：检索后用 LLM 压缩，只留相关的
  - ParentDocumentRetriever：用小 chunk 搜索，用大 chunk 喂 LLM

第 5 课用的是最基础的 retriever——一句话搜一次。这课教你三种
更聪明的检索方式，每一种解决一个不同的问题。

学完这课你就理解了：RAG 的质量瓶颈通常不在 LLM，而在"检索到的文档好不好"。
改 retriever 往往比改 prompt 更有效。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.stores import InMemoryStore
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


# ============================================================
# 准备：重建向量库
# ============================================================
print("=" * 60)
print("准备：重建向量库")
print("=" * 60)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "08-rag-foundation", "data")
docs = TextLoader(os.path.join(DATA_DIR, "resume_zhangsan.txt"), encoding="utf-8").load()
docs += TextLoader(os.path.join(DATA_DIR, "jd_python_senior.txt"), encoding="utf-8").load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
).split_documents(docs)

print("加载 embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"向量库就绪：{len(chunks)} 个 chunk\n")

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================
# 实验①：基础 Retriever —— 基线对照
# ============================================================
print("=" * 60)
print("实验①：基础 Retriever（基线对照）")
print("=" * 60)

query = "这个候选人的技术能力"

print(f"查询：「{query}」\n")
results = base_retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"结果 {i+1}：{doc.page_content[:100]}...")
    print()

baseline_docs = results

print(">>> 这是第 5 课的基础检索方式，作为后续实验的对照。\n")


# ============================================================
# 实验②：MultiQueryRetriever —— 多角度查询
# ============================================================
print("=" * 60)
print("实验②：MultiQueryRetriever —— 多角度查询")
print("=" * 60)

# MultiQueryRetriever 用 LLM 把一个问题改写成多个版本，
# 每个版本单独检索，最后合并去重。
# 适合：用户问得太模糊，一个角度搜不全。
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

results = multi_retriever.invoke(query)
print(f"基础检索返回了 {len(baseline_docs)} 个 chunk")
print(f"MultiQuery 返回了 {len(results)} 个 chunk（通常会更多）\n")
for i, doc in enumerate(results):
    print(f"结果 {i+1}：{doc.page_content[:100]}...")

print("\n>>> MultiQuery 把一个问题改写成多个版本，")
print("    从不同角度检索，然后把结果合并去重。")


# ============================================================
# 实验③：ContextualCompressionRetriever —— 上下文压缩
# ============================================================
print("=" * 60)
print("实验③：ContextualCompressionRetriever —— 上下文压缩")
print("=" * 60)

# 先检索出 chunk，再用 LLM 压缩——只保留和问题相关的部分。
# 适合：chunk 很长，但只有一小段真的和问题有关。
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

query2 = "候选人有没有 Docker 和 Kubernetes 经验？"
print(f"查询：「{query2}」\n")

raw_results = base_retriever.invoke(query2)
comp_results = compression_retriever.invoke(query2)

print("压缩前后对比：")
print(f"  原始 chunk 长度：{[len(d.page_content) for d in raw_results]} 字符")
print(f"  压缩后 chunk 长度：{[len(d.page_content) for d in comp_results]} 字符\n")

for i, doc in enumerate(comp_results):
    print(f"压缩结果 {i+1}：{doc.page_content[:150]}...")
    print()

print(">>> 压缩后的 chunk 更短，内容更聚焦于问题本身。")


# ============================================================
# 实验④：ParentDocumentRetriever —— 父子文档
# ============================================================
print("=" * 60)
print("实验④：ParentDocumentRetriever —— 父子文档")
print("=" * 60)

# 用小 chunk 搜索（精准匹配），返回大 chunk（完整上下文）。
# 适合：小 chunk 搜得准但信息太碎，需要更大上下文给 LLM。
child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

store = InMemoryStore()
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2},
)
parent_retriever.add_documents(docs)

print("文档已按父子结构存储：")
print("  子 chunk（150 字符）：用于向量搜索匹配")
print("  父 chunk（500 字符）：返回给 LLM 的完整上下文\n")

results = parent_retriever.invoke("学历背景")
print(f"查询：「学历背景」")
print(f"返回了 {len(results)} 个结果：")
for i, doc in enumerate(results):
    print(f"  结果 {i+1}（{len(doc.page_content)} 字符）：{doc.page_content[:150]}...")

print("\n>>> 注意返回的 chunk 比基础检索的大很多，")
print("    给 LLM 提供了更完整的上下文信息。")


# ============================================================
# 实验⑤：三种策略横向对比
# ============================================================
print("=" * 60)
print("实验⑤：三种策略横向对比")
print("=" * 60)

test_query = "候选人的工作经验和薪资期望是什么？"

print(f"查询：「{test_query}」\n")

print("--- 基础 Retriever ---")
for i, doc in enumerate(base_retriever.invoke(test_query)):
    print(f"  [{i}] {doc.page_content[:100]}...")

print("\n--- MultiQuery Retriever ---")
for i, doc in enumerate(multi_retriever.invoke(test_query)):
    print(f"  [{i}] {doc.page_content[:100]}...")

print("\n--- 压缩 Retriever ---")
for i, doc in enumerate(compression_retriever.invoke(test_query)):
    print(f"  [{i}]（{len(doc.page_content)}字符）{doc.page_content[:100]}...")

print("\n>>> 同一个问题，三种策略返回的结果各不一样。")
print("    没有最好的策略，只有最适合场景的策略。")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第 6 课 总结 —— 高级检索策略选型指南")
print("=" * 60)
print("""
               解决的问题               适用场景
================================================================
基础检索         基础语义搜索              问题清晰、查询简单

MultiQuery      检索不够全面              用户问题模糊或太短
                从多个角度改写问题          需要从多个角度找信息

上下文压缩       检索结果冗余太多           文档很长
                把无关内容压缩掉            只有小部分内容跟问题相关

父子文档         小 chunk 太碎片化          需要准确的搜索 +
                大 chunk 能提供完整上下文    完整的上下文

关键认知：
  所有高级 Retriever 都是"包装"基础 Retriever 的。
  它们改变的是"怎么搜"和"返回什么"，底层向量搜索本身不变。
  甚至可以把它们组合起来使用——比如 MultiQuery + Compression。

和后面课程的关联：
  第 9 课 Agent+RAG：retriever 选型影响 agent 的效率和准确度
  第 12 课 项目：    根据实际场景选择最合适的检索策略
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 试试组合 MultiQuery + Compression：
   compression_retriever = ContextualCompressionRetriever(
       base_compressor=compressor,
       base_retriever=multi_retriever,  # 对 MultiQuery 的结果再做压缩
   )

2. 往 data/ 里多加几种文档，看看 MultiQuery 如何处理多文档场景

3. 思考题：如果你的知识库有 1000+ 份文档，你会选哪种 retriever？为什么？
""")
