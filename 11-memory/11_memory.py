"""
第 7 课：Conversation Memory —— 对话记忆

核心概念：
  - ChatMessageHistory：消息的存储容器（内存、数据库、文件都可以）
  - RunnableWithMessageHistory：自动管理对话历史的 wrapper
  - trim_messages()：token 级别的历史裁剪，防止撑爆上下文
  - 三种策略：全量记忆 / 窗口记忆 / 摘要记忆

前 6 课所有的对话都是"一次性"的——AI 不记得你上一句说了什么。
这节课让你构建能记住上下文的多轮对话机器人。

学完这课你就理解了：
  Memory = prompt template 里的 MessagesPlaceholder + 自动管理的历史记录。
  这正是第 1 课 MessagesPlaceholder 的用武之地。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 实验①：没有 Memory 会怎样
# ============================================================
print("=" * 60)
print("实验①：没有 Memory —— AI 每次都是失忆症")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。请简洁回答。"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# 第 1 轮
print("第 1 轮：")
print("  用户：我叫小明，我是一名 Python 开发者。")
result = chain.invoke({"input": "我叫小明，我是一名 Python 开发者。"})
print(f"  AI：{result[:80]}...")

# 第 2 轮——AI 完全不记得！
print("\n第 2 轮：")
print("  用户：我叫什么名字？")
result = chain.invoke({"input": "我叫什么名字？"})
print(f"  AI：{result[:80]}...")

print("\n>>> 没有 Memory，每次 invoke() 都是一次全新的对话。")
print("    AI 在第 2 轮根本不知道你是谁。")


# ============================================================
# 实验②：RunnableWithMessageHistory —— 给 AI 装上记忆
# ============================================================
print("=" * 60)
print("实验②：RunnableWithMessageHistory —— AI 有记忆了")
print("=" * 60)

# 第 1 步：定义带 MessagesPlaceholder 的 prompt
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。请简洁回答。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息会插入到这里！
    ("human", "{input}"),
])

# 第 2 步：构建 chain
chain = memory_prompt | llm | StrOutputParser()

# 第 3 步：用字典存储历史记录（按 session_id 区分）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 第 4 步：用 RunnableWithMessageHistory 包装 chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",       # 输入字典中哪个 key 是用户消息
    history_messages_key="history",   # prompt 中哪个变量接收历史记录
)

# 来一场真正的对话！
session_id = "xiaoming_session"

print("带记忆的对话（session: xiaoming_session）：\n")

questions = [
    "我叫小明，我有 3 年 Python 开发经验。",
    "我叫什么名字？我是做什么的？",
    "我还会 React 和 TypeScript。现在我的完整技术栈是什么？",
]

for q in questions:
    print(f"用户：{q}")
    result = chain_with_history.invoke(
        {"input": q},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"AI：{result[:150]}...\n")

print(">>> 注意第 3 轮：AI 记住了第 1 轮和第 2 轮的内容！")


# ============================================================
# 实验③：多会话 —— 每个用户有独立的记忆
# ============================================================
print("=" * 60)
print("实验③：多会话 —— 不同用户互不干扰")
print("=" * 60)

# 同一个 chain，不同的 session_id，独立的记忆

# 小明的对话
print("--- 小明的 session ---")
result = chain_with_history.invoke(
    {"input": "我叫小明，我喜欢 Python。"},
    config={"configurable": {"session_id": "xiaoming"}},
)
print(f"小明说：我叫小明，我喜欢 Python。")
print(f"AI：{result[:80]}...\n")

# 小红的对话（不同 session，不知道小明是谁）
print("--- 小红的 session ---")
result = chain_with_history.invoke(
    {"input": "我叫小红，我喜欢 JavaScript。"},
    config={"configurable": {"session_id": "xiaohong"}},
)
print(f"小红说：我叫小红，我喜欢 JavaScript。")
print(f"AI：{result[:80]}...\n")

# 回到小明——仍然记得
print("--- 回到小明 ---")
result = chain_with_history.invoke(
    {"input": "我喜欢什么来着？"},
    config={"configurable": {"session_id": "xiaoming"}},
)
print(f"小明问：我喜欢什么来着？")
print(f"AI：{result[:80]}...")

print("\n>>> session_id 是隔离键——不同 session 的记忆互不干扰。")


# ============================================================
# 实验④：trim_messages() —— 防止上下文溢出
# ============================================================
print("=" * 60)
print("实验④：trim_messages() —— 按 token 数裁剪历史")
print("=" * 60)

# 多轮对话后，历史记录会越来越长，最终超出模型的上下文窗口
# trim_messages() 自动删除旧消息，保持在 token 限制内

# 模拟一段长对话
long_history = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！"),
    HumanMessage(content="什么是 Python？"),
    AIMessage(content="Python 是一种编程语言。"),
    HumanMessage(content="什么是变量？"),
    AIMessage(content="变量是存储在内存中的数据。"),
    HumanMessage(content="什么是函数？"),
    AIMessage(content="函数是可复用的代码块。"),
    HumanMessage(content="什么是类？"),
    AIMessage(content="类是对象的蓝图。"),
    HumanMessage(content="什么是模块？"),
    AIMessage(content="模块是包含 Python 代码的文件。"),
]

print(f"完整历史：{len(long_history)} 条消息")

# 使用简单的字符计数器（DeepSeek 不支持 token 计数 API）
def char_counter(messages):
    return sum(len(str(m.content)) for m in messages)

# 裁剪到指定 token 上限以内（中文粗略按 1 字符 ≈ 1 token 估算）
trimmed = trim_messages(
    long_history,
    max_tokens=200,               # 总量控制在这个上限以内
    strategy="last",              # "last" = 保留最新的消息
    token_counter=char_counter,   # 自定义计数器
    include_system=True,          # 始终保留 system message
)

print(f"裁剪后历史：{len(trimmed)} 条消息")
print("保留的消息：")
for msg in trimmed:
    print(f"  [{type(msg).__name__}] {str(msg.content)[:60]}...")

print("\n>>> trim_messages() 确保历史记录永远不会超出模型的 token 限制。")
print("    和窗口裁剪不同，它按实际 token 数计算，而不是消息条数。")


# ============================================================
# 实验⑤：摘要记忆模式
# ============================================================
print("=" * 60)
print("实验⑤：摘要记忆 —— 用 LLM 压缩旧消息")
print("=" * 60)

# 另一种思路：用 LLM 把旧消息压缩成摘要
# 保留了语义内容，但不需要存每一条原始消息

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

summary_chain = summary_prompt | llm | StrOutputParser()

# 自定义 get_session_history，自动摘要
summary_store = {}

class SummaryHistory(InMemoryChatMessageHistory):
    def add_message(self, message):
        super().add_message(message)
        # 每 6 条消息后，把旧的压缩成摘要
        if len(self.messages) >= 6:
            self._summarize()

    def _summarize(self):
        # 保留最近 4 条，把更早的压缩成摘要
        old = self.messages[:-4]
        recent = self.messages[-4:]
        # 跳过已有的摘要 SystemMessage，避免 "[之前的对话摘要]" 层层嵌套
        parts = []
        for m in old:
            if isinstance(m, SystemMessage) and m.content.startswith("[之前的对话摘要]"):
                # 提取内层真实内容，去掉外层前缀
                parts.append(m.content[len("[之前的对话摘要]: "):])
            else:
                parts.append(str(m.content[:100]))
        summary_text = "\n".join(parts)
        summary_msg = SystemMessage(
            content=f"[之前的对话摘要]: {summary_text}"
        )
        self.messages = [summary_msg] + recent

def get_summary_history(session_id: str):
    if session_id not in summary_store:
        summary_store[session_id] = SummaryHistory()
    return summary_store[session_id]

summary_chain_with_history = RunnableWithMessageHistory(
    summary_chain,
    get_summary_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("自动摘要记忆对话：\n")

chat = [
    "我叫小明。",
    "我在一家科技公司工作。",
    "我正在学机器学习。",
    "我每天都在用 Python。",
    "你知道我哪些信息？",
]

for msg in chat:
    print(f"用户：{msg}")
    result = summary_chain_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": "summary_demo"}},
    )
    print(f"AI：{result[:100]}...\n")

# 检查实际存储了多少条消息
stored = summary_store.get("summary_demo")
if stored:
    print(f"存储中的消息数：{len(stored.messages)}（已被摘要压缩，不是完整历史）")
    for m in stored.messages:
        print(f"  [{type(m).__name__}] {str(m.content)[:80]}...")

print("\n>>> 摘要记忆：保留语义含义，不保留每条原始消息。")
print("    适合超长对话，窗口裁剪会丢失早期重要上下文时使用。")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 7 课 总结 —— 记忆策略")
print("=" * 60)
print("""
策略                  原理                        适用场景
==================================================================
无记忆                每次调用独立，不保留历史       无状态工具、一次性问答

全量缓冲              存储所有消息                  短对话
                      简单但快速增长                 演示和原型

窗口缓冲              只保留最近 N 条                 中等长度对话
                      旧消息直接丢弃                  固定的内存预算

摘要记忆              LLM 把旧消息压缩成摘要          长对话
                                                       需要保留早期上下文

trim_messages()       按 token 数量裁剪               生产系统
                                                       严格的 token 预算

核心架构：
  RunnableWithMessageHistory 包装任意 chain：
    - 按 session_id 提取历史记录
    - 通过 MessagesPlaceholder 注入到 prompt 中
    - 把新消息保存回历史记录

和后面课程的关联：
  第 12 课 项目：求职助手中的多轮对话
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 写一个聊天机器人，能记住你的名字、职业和技术栈，跨多轮对话
2. 尝试调整 trim_messages() 的 max_tokens 参数，观察不同值下保留了多少条消息
3. 把 InMemoryChatMessageHistory 替换成文件存储（提示：可以用
   SQLiteChatMessageHistory 或者直接用 pickle 保存 store 字典）
""")
