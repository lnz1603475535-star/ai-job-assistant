"""
第 9 课：RAG 检索与生成 —— 串联成完整链路

核心概念：
  - Retriever：检索器的抽象接口，向量库、关键词搜索都是 Retriever
  - create_stuff_documents_chain()：把检索到的文档"塞入" prompt
  - create_retrieval_chain()：检索 + 生成，一步到位
  - 对比实验：直接问 AI vs RAG 后问 AI

什么是 RAG Chain？
  把第 4 课的四步（加载→切分→向量化→存库）和第 2 课的 LCEL 串起来：
  question → retriever.invoke() → documents → prompt.format() → llm.invoke() → answer

学完这课你就理解了：RAG 不是魔法，就是把"搜索"和"生成"用 LCEL 串起来。
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
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============================================================
# 准备：重建向量库（第 4 课的内容）
# ============================================================
print("=" * 60)
print("准备：重建向量库")
print("=" * 60)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "08-rag-foundation", "data")
resume_path = os.path.join(DATA_DIR, "resume_zhangsan.txt")
jd_path = os.path.join(DATA_DIR, "jd_python_senior.txt")

docs = TextLoader(resume_path, encoding="utf-8").load()
docs += TextLoader(jd_path, encoding="utf-8").load()

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


# ============================================================
# 实验①：Retriever —— 检索器抽象
# ============================================================
print("=" * 60)
print("实验①：VectorStore → Retriever")
print("=" * 60)

# as_retriever() 把向量库包装成 Retriever 接口
# Retriever 只有一个方法：invoke(query) → List[Document]
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 每次检索返回前 3 个最相关的 chunk
)

# Retriever 也是 Runnable！所以能用 LCEL！
query = "这个候选人有几年经验？会微服务吗？"
docs = retriever.invoke(query)

print(f"检索：「{query}」\n")
print("Retriever 返回的结果：")
for i, doc in enumerate(docs):
    source = doc.metadata['source'].split('\\')[-1]
    print(f"  [{i}] 来自 {source}：")
    print(f"      {doc.page_content[:120]}...")
    print()

print(">>> Retriever 是 Runnable！可以像 model、parser 一样用 | 连接。")


# ============================================================
# 实验②：create_stuff_documents_chain —— 把文档"塞入" prompt
# ============================================================
print("=" * 60)
print("实验②：create_stuff_documents_chain —— 文档入 prompt")
print("=" * 60)

# 这个 prompt 模板有一个特殊的 {context} 变量
# create_stuff_documents_chain 会自动把检索到的文档填入 {context}
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的招聘顾问。根据以下文档内容回答用户的问题。

如果文档中有相关信息，请基于文档回答，并引用出处。
如果文档中没有相关信息，请明确说"文档中未提及"。

文档内容：
{context}"""),
    ("human", "{input}"),
])

# create_stuff_documents_chain: 接收 {context: [Document], input: str}
# → 把 Document 列表格式化为文本塞进 {context}
# → 调用 llm.invoke()
# → 返回答案
stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    output_parser=StrOutputParser(),
)

# context 是 Document 列表（来自 retriever），input 是用户问题
result = stuff_chain.invoke({
    "context": docs,
    "input": query,
})

print(f"问题：{query}")
print(f"\n回答：\n{result}")

print("\n>>> create_stuff_documents_chain 会自动把 Document 列表转成文本填入 prompt。")


# ============================================================
# 实验③：create_retrieval_chain —— 检索 + 生成一步到位
# ============================================================
print("=" * 60)
print("实验③：create_retrieval_chain —— 检索 + 生成一步到位")
print("=" * 60)

# 这才是完整的 RAG Chain！
# create_retrieval_chain 把 retriever 和 stuff_chain 串起来
# 数据流：input → retriever.invoke() → stuff_chain.invoke(context=result)
rag_chain = create_retrieval_chain(retriever, stuff_chain)

# 现在只需要传用户问题！检索 → 生成 全自动
result = rag_chain.invoke({"input": "这个候选人在微服务方面有什么经验？"})

print(f"用户问题：{result['input']}")
print(f"检索到的上下文 chunk 数：{len(result.get('context', []))}")
print(f"\nAI 回答：\n{result['answer']}")

print("\n>>> rag_chain.invoke() 内部自动完成了：检索 → 拼 prompt → 生成。")


# ============================================================
# 实验④：对比 —— 直接问 AI vs RAG 后问 AI
# ============================================================
print("=" * 60)
print("实验④：决定性对比 —— 不开卷 vs 开卷考试")
print("=" * 60)

question = "张三的工作经验是否符合高级 Python 工程师 JD 的要求？请具体分析。"

# --- 实验 A: 直接问 AI（不开卷）---
print(f"用户：{question}\n")

print("--- 实验 A：直接问 AI（不开卷考试）---")
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个招聘顾问。回答用户的问题。"),
    ("human", "{input}"),
])
direct_chain = direct_prompt | llm | StrOutputParser()
direct_answer = direct_chain.invoke({"input": question})

print(f"AI 回答（无文档）：\n{direct_answer[:300]}...\n")

# --- 实验 B: RAG 后问 AI（开卷）---
print("--- 实验 B：RAG + AI（开卷考试）---")
rag_answer = rag_chain.invoke({"input": question})

print(f"AI 回答（有文档）：\n{rag_answer['answer'][:400]}...")

print("\n>>> 不开卷：AI 不知道张三是谁，只能编造或回答不知道")
print(">>> 开卷：AI 基于真实简历和 JD 分析，每个判断都有文档依据")


# ============================================================
# 实验⑤：用 LCEL 手写 RAG Chain
# ============================================================
print("=" * 60)
print("实验⑤：用 LCEL 手写 RAG Chain —— 透明化")
print("=" * 60)

# create_retrieval_chain 很方便，但黑盒。如果我们要自定义流程呢？
# 用 LCEL 从头手写，完全透明

def format_docs(docs):
    """把 Document 列表格式化成纯文本"""
    return "\n\n".join(
        f"[来源：{d.metadata.get('source', '?').split(chr(92))[-1]}]\n{d.page_content}"
        for d in docs
    )

custom_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """基于以下文档回答问题。如果文档中没有，请明确说明。

{context}"""),
    ("human", "{question}"),
])

# 手动组装的 RAG Chain（完全透明）
custom_rag_chain = (
    {
        "context": retriever | format_docs,  # 检索 → 格式化文本
        "question": RunnablePassthrough(),    # 问题原样透传
    }
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# 这就是 LCEL 的威力！
print("RAG Chain 的数据流：")
print("  input → { context: retriever→format, question: passthrough }")
print("        → prompt.format()")
print("        → llm.invoke()")
print("        → parser.parse()")
print("        → 最终答案\n")

result = custom_rag_chain.invoke("候选人是什么学历？")
print(f"查询：候选人是什么学历？")
print(f"回答：\n{result}")

print("\n>>> 手写的 RAG Chain 和 create_retrieval_chain 效果一样，")
print("    但你可以完全控制每一步的行为（比如改 format_docs 里的格式）。")


# ============================================================
# 实验⑥：Context window 管理
# ============================================================
print("=" * 60)
print("实验⑥：检索结果太多怎么办？")
print("=" * 60)

# 如果检索返回很多 chunk，全塞进 prompt 可能超出 token 限制
# 解决方案：控制检索数量 + 在 prompt 中提示 AI

# 只取前 1 个
retriever_k1 = vectorstore.as_retriever(search_kwargs={"k": 1})
result = create_retrieval_chain(retriever_k1, stuff_chain).invoke(
    {"input": "候选人有什么技能？"}
)
print(f"k=1 时，检索到 {len(result['context'])} 个 chunk")
print(f"回答：{result['answer'][:200]}...")

# 取前 5 个（假设更多）
retriever_k5 = vectorstore.as_retriever(search_kwargs={"k": 5})
result = create_retrieval_chain(retriever_k5, stuff_chain).invoke(
    {"input": "候选人有什么技能？"}
)
print(f"\nk=5 时，检索到 {len(result['context'])} 个 chunk")
print(f"回答：{result['answer'][:200]}...")

print("\n>>> k 太小 → 信息不足。k 太大 → context 溢出/稀释。")
print("    根据模型 token 上限和文档 chunk 大小选择 k 值。")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 9 课 总结 —— RAG 完整链路")
print("=" * 60)
print("""
RAG 的两个层次：

1. create_stuff_documents_chain(llm, prompt)
       把 Document 列表"塞"进 prompt 的 {context} 占位符
       只负责"生成"，不管检索

2. create_retrieval_chain(retriever, stuff_chain)
       检索 + 生成一步到位
       input → retriever.invoke() → stuff_chain.invoke()
       对使用者来说：传问题进去，拿答案出来

等价的 LCEL 写法：
   chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough()}
       | prompt | llm | parser
   )

和后面课程的关联：
  第 6 课 Advanced RAG: 换 retriever（MultiQuery / Compression），chain 结构不变
  第 9 课 Agent+RAG:    retriever 作为 agent 的一个 tool
  第 12 课 项目:        完整的求职助手 RAG pipeline
""")

# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 修改 prompt 模板，让 AI 在引用文档时注明"chunk 编号"——验证它是否真的读了文档
2. 自己写一个 format_docs，让每个文档带编号：doc[1]: xxx
3. 如果检索结果完全不相关（比如问天气，但知识库里只有简历），AI 应该怎么回复？
   修改 prompt 来优雅处理这种情况
""")
