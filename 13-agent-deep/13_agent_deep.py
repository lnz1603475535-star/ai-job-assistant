"""
第 9 课：Agent 深入 —— Middleware 与 Agent + RAG

核心概念：
  - Agent middleware：在模型调用前后插入自定义逻辑
  - 自定义 middleware：限流、日志、结果缓存
  - Agent + RAG tool：把检索器包装成 tool，让 Agent 能"查资料"
  - 多工具协作：Agent 自动决定工具调用顺序

第 4 课你用 create_agent() 创建了第一个 agent，但没深入了解内部机制。
这节课教你 agent 的"插件系统"（middleware）和如何把 RAG 集成到 agent 中。

学完这课你就理解了：middleware 是 agent 的"中间件栈"——
每个请求都要穿过这些中间件，你可以在任何环节插入自定义逻辑。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
import os, time

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 准备：构建向量库（供 RAG tool 使用）
# ============================================================
print("=" * 60)
print("准备：为 RAG tool 构建向量库")
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


# ============================================================
# 实验①：回顾 create_agent 内部原理
# ============================================================
print("=" * 60)
print("实验①：create_agent 内部机制回顾")
print("=" * 60)

# create_agent() 组合了以下组件：
#   1. model + tools + system_prompt → 一个自主循环
#   2. 内部流程：模型决策 → 调用工具 → 获取结果 → 再决策 → ... → 最终答案
#   3. 底层其实是一个 LangGraph 状态图！

@tool
def skill_score(skill: str, years: float) -> str:
    """根据工作年限给技能打分，1-10 分。
    skill：技能名称，years：该技能的工作年数。"""
    if years >= 5:
        return f"{skill}：9/10（专家级，{years} 年经验）"
    elif years >= 3:
        return f"{skill}：7/10（熟练，{years} 年经验）"
    elif years >= 1:
        return f"{skill}：4/10（中级，{years} 年经验）"
    else:
        return f"{skill}：2/10（入门，{years} 年经验）"

agent = create_agent(
    model=llm,
    tools=[skill_score],
    system_prompt="你是一个职业顾问。使用 skill_score 工具给技能打分。",
)

print("Agent 已创建，包含 1 个工具：skill_score")
print("Agent 类型：", type(agent).__name__)
print("（底层：create_agent 构建了一个 LangGraph StateGraph）\n")

result = agent.invoke({
    "messages": [HumanMessage(content="评估我的技能：Python 3年，Docker 1年")]
})

# 展示决策流程
for msg in result["messages"]:
    msg_type = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  [{msg_type}] 调用：{tc['name']}({tc['args']})")
    elif msg_type == "ToolMessage":
        print(f"  [{msg_type}] 返回：{msg.content}")
    elif hasattr(msg, "content") and msg.content:
        print(f"  [{msg_type}] 回答：{msg.content[:120]}...")

print("\n>>> Agent 循环：AI 决策 → 调用工具 → 获取结果 → AI 回答。")


# ============================================================
# 实验②：自定义 Middleware —— 计时
# ============================================================
print("=" * 60)
print("实验②：自定义 Middleware —— 给每次模型调用计时")
print("=" * 60)

@wrap_model_call
def timing_and_log(request, handler):
    """包裹模型调用：计时 + 记录日志。"""
    start = time.time()
    response = handler(request)          # 实际执行模型调用
    elapsed = time.time() - start

    # 从 response 取模型刚生成的消息
    last_ai_msg = response.result[0] if response.result else None
    if last_ai_msg:
        content_preview = str(last_ai_msg.content)[:80] if last_ai_msg.content else "(工具调用)"
        tool_calls = len(last_ai_msg.tool_calls) if last_ai_msg.tool_calls else 0
        print(f"  [中间件] 模型调用：{elapsed:.2f}s | 工具调用数：{tool_calls} | {content_preview}...")
    return response

agent_with_middleware = create_agent(
    model=llm,
    tools=[skill_score],
    system_prompt="你是一个职业顾问。简洁打分。",
    middleware=[timing_and_log],
)

print("带计时中间件的 Agent：\n")

result = agent_with_middleware.invoke({
    "messages": [HumanMessage(content="评估：Python 3年。一次只评一个技能。")]
})

print(f"\n最终回答：{result['messages'][-1].content[:200]}...")

print("\n>>> Middleware 挂载在每次模型调用上。可用于日志、限流、缓存等。")


# ============================================================
# 实验③：Agent + RAG Tool
# ============================================================
print("=" * 60)
print("实验③：Agent + RAG Tool —— 让 Agent 能查资料")
print("=" * 60)

# 把向量库检索器包装成一个 tool
@tool
def search_resume(query: str) -> str:
    """搜索候选人数据库中的相关信息。
    用它来查找简历细节、岗位要求或技能需求。
    query：自然语言搜索词。"""
    docs = vectorstore.similarity_search(query, k=3)
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "?").split("\\")[-1]
        results.append(f"[{source}]：{doc.page_content}")
    return "\n\n".join(results)

rag_agent = create_agent(
    model=llm,
    tools=[search_resume, skill_score],
    system_prompt="""你是一个专业的招聘顾问。
    使用 search_resume 查找候选人和 JD 信息。
    使用 skill_score 评估具体技能。
    所有分析必须基于找到的文档内容。""",
)

print("Agent 同时拥有 RAG 检索工具 + 技能分析工具：\n")

questions = [
    "候选人叫什么名字？什么学历？",
    "候选人的 Docker 和 Kubernetes 经验是否满足 JD 要求？",
    "评估候选人的 Python 技能并告诉我整体匹配度。",
]

for q in questions:
    print(f"用户：{q}")
    result = rag_agent.invoke({"messages": [HumanMessage(content=q)]})
    last_msg = result["messages"][-1]
    print(f"AI：{last_msg.content[:200]}...\n")

print(">>> Agent 现在可以先查文档再回答——RAG + Agent 合体！")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 9 课 总结 —— Agent 深入")
print("=" * 60)
print("""
create_agent() 回顾：
  agent = create_agent(model, tools, system_prompt)
  - 底层：一个 LangGraph StateGraph
  - 循环：模型 → 工具调用 → 工具执行 → 模型 → ... → 最终答案
  - 返回的 agent 有 .invoke() 和 .stream()

Middleware 系统：
  @before_model：在每次模型调用之前执行
      用途：输入校验、限流、缓存检查

  @after_model：在每次模型调用之后执行
      用途：日志记录、输出校验、token 统计

Agent + RAG 模式：
  1. 从文档构建向量库
  2. 把 retriever 包装为 @tool
  3. 加入 agent 的 tools 列表
  4. Agent 在需要时自动检索

  这比纯 RAG 更强大，因为：
  - Agent 自行决定"要不要查"（不是每个问题都需要 RAG）
  - Agent 自行决定"查什么"（自动改写查询）
  - Agent 可以查多次（迭代优化查询）

和后面课程的关联：
  第 10-11 课 LangGraph：create_agent 底层就是 LangGraph 图
  第 12 课 项目：完整的求职助手 agent + RAG tools
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 写一个 middleware，统计 agent 调用工具的总次数
2. 给 agent 加第三个 tool（比如计算薪资范围），测试多工具协作
3. 对比：带 RAG tool 的 Agent vs 第 5 课的纯 RAG chain。
   什么场景用哪个更好？
""")
