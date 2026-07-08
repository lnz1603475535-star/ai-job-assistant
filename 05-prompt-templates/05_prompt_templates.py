"""
第 5 课：Prompt Templates —— 告别硬编码字符串

核心概念：
  - ChatPromptTemplate：用变量动态构造 prompt，不用再拼字符串
  - MessagesPlaceholder：为对话历史预留位置（为后续 Memory 课程做铺垫）
  - FewShotChatMessagePromptTemplate：给 AI 看几个示例，它就会照着做

前 4 节课的问题：
  之前所有的 system prompt 和 user message 都是硬编码字符串。每次改内容
  都要改代码。PromptTemplate 让你把 prompt 和数据分离——模板固定，数据可变。

学完这课你就理解了：PromptTemplate 是 LangChain 的"函数签名"——
定义了 AI 接收什么参数、产什么输出。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 实验①：回顾旧方式 —— 硬编码字符串
# ============================================================
print("=" * 60)
print("实验①：旧方式 —— 硬编码字符串（前 4 课的做法）")
print("=" * 60)

resume_text = "张三，3年Python开发，熟悉Django和Vue，参与过电商项目"
jd_text = "招聘高级Python工程师，要求5年经验，熟悉微服务架构"

# 旧方式：每次都要手动 f-string 拼接
old_prompt = f"""你是一个资深HR。请分析以下候选人的简历是否匹配职位要求。

简历：
{resume_text}

职位要求：
{jd_text}

请给出匹配度评分（1-10）和理由。"""

print("旧方式构造的 prompt：")
print(old_prompt[:200] + "...\n")

response = llm.invoke(old_prompt)
print(f"AI 回答：\n{response.content[:300]}...\n")

print(">>> 问题：每次换简历/JD，都要改代码里的字符串。而且 prompt 结构散落在代码中，难以维护。\n")


# ============================================================
# 实验②：ChatPromptTemplate —— 模板与数据分离
# ============================================================
print("=" * 60)
print("实验②：ChatPromptTemplate —— 模板与数据分离")
print("=" * 60)

# 定义模板（结构固定，只写一次）
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深HR。请分析以下候选人的简历是否匹配职位要求。"),
    ("human", """
简历：
{resume}

职位要求：
{jd}

请给出匹配度评分（1-10）和理由。"""),
])

# 用 .format_messages() 填入变量，模板不动，数据可变
messages = template.format_messages(
    resume=resume_text,
    jd=jd_text,
)

# 看看构造出来的消息长什么样
print("构造出的消息列表：")
for msg in messages:
    print(f"  [{type(msg).__name__}] {msg.content[:100]}...")
print()

response = llm.invoke(messages)
print(f"AI 回答：\n{response.content[:300]}...\n")

print(">>> 好处：模板只写一次，换简历/JD 时改变量即可，不用改代码结构。\n")


# ============================================================
# 实验③：MessagesPlaceholder —— 为对话历史预留位置
# ============================================================
print("=" * 60)
print("实验③：MessagesPlaceholder —— 插入对话历史")
print("=" * 60)

# 假设这是之前对话的历史记录（第 7 课 Memory 会系统讲解）
chat_history = [
    HumanMessage(content="我之前在新浪工作了2年，用的是PHP"),
    AIMessage(content="好的，已记录。你有 Python 相关的项目经验吗？"),
]

# 模板中用 MessagesPlaceholder 给历史消息预留位置
template_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个面试记录员，帮候选人梳理经历。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息会插入到这里
    ("human", "根据我的经历，{question}"),
])

# 历史消息 + 新问题一起传入
messages = template_with_history.format_messages(
    history=chat_history,
    question="我适合投什么岗位？",
)

print("带历史记录的消息列表：")
for i, msg in enumerate(messages):
    content_preview = str(msg.content)[:80]
    print(f"  [{i}] {type(msg).__name__}: {content_preview}")
print()

response = llm.invoke(messages)
print(f"AI 回答：\n{response.content[:300]}...\n")

print(">>> MessagesPlaceholder 让 prompt 模板能动态插入任意数量的历史消息。\n")


# ============================================================
# 实验④：Few-Shot Prompting —— 给 AI 看示例，它就会照着做
# ============================================================
print("=" * 60)
print("实验④：Few-Shot —— 用示例教会 AI 特定格式")
print("=" * 60)

# 准备示例
examples = [
    {"input": "今天天气真好", "output": "POSITIVE"},
    {"input": "等了三个小时还没到", "output": "NEGATIVE"},
    {"input": "还行吧，凑合", "output": "NEUTRAL"},
]

# 把示例转成 prompt 格式
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 组装 few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 把 few-shot examples + system prompt + 用户问题拼起来
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析助手。输出只能是 POSITIVE / NEGATIVE / NEUTRAL 三者之一。"),
    few_shot_prompt,                        # 示例插入在这里
    ("human", "{input}"),
])

messages = final_prompt.format_messages(input="这个餐厅环境不错，但上菜太慢了")

print("最终 prompt（含 few-shot 示例）：")
for i, msg in enumerate(messages):
    print(f"  [{i}] {type(msg).__name__}: {msg.content}")
print()

response = llm.invoke(messages)
print(f"AI 回答：{response.content}")

print(">>> Few-Shot 是一种强大的 prompt 技巧——不需要大段描述，给几个例子 AI 就懂了。\n")


# ============================================================
# 实验⑤：Few-Shot 实战 —— 教 AI 按特定 JSON 格式输出
# ============================================================
print("=" * 60)
print("实验⑤：Few-Shot 实战 —— 教 AI 输出特定 JSON 格式")
print("=" * 60)

# 用 few-shot 教 AI 把简历解析成结构化 JSON
# （后面第 3 课会用 Output Parser 更优雅地实现这个功能）
parse_examples = [
    {
        "input": "张三，5年Java开发，擅长Spring Boot和微服务，本科学历",
        "output": '{"name": "张三", "years": 5, "skills": ["Java", "Spring Boot", "微服务"], "degree": "本科"}',
    },
    {
        "input": "李四，2年前端，会React和TypeScript，硕士学历",
        "output": '{"name": "李四", "years": 2, "skills": ["React", "TypeScript"], "degree": "硕士"}',
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=parse_examples,
)

parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简历解析器。把简历描述转成 JSON，只输出 JSON，不要加其他文字。"),
    few_shot,
    ("human", "{input}"),
])

messages = parser_prompt.format_messages(
    input="王五，3年Python后端，熟悉Django和Docker，大专学历"
)

print("few-shot 教 AI 按标准格式输出：")
response = llm.invoke(messages)
print(f"AI 输出：{response.content}")

import json
try:
    parsed = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    print(f"\n成功解析为 Python dict：")
    print(f"  姓名: {parsed.get('name')}")
    print(f"  年限: {parsed.get('years')} 年")
    print(f"  技能: {', '.join(parsed.get('skills', []))}")
    print(f"  学历: {parsed.get('degree')}")
except json.JSONDecodeError:
    print("(JSON 解析失败，但这只是 many-shot 演示，下一课用 Output Parser 解决这个问题)")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第 5 课 总结")
print("=" * 60)
print("""
学到的三个核心工具：

1. ChatPromptTemplate —— 模板和数据分离
      template = ChatPromptTemplate.from_messages([
          ("system", "你是{role}"),
          ("human", "{user_input}"),
      ])
      messages = template.format_messages(role="HR", user_input="帮我分析简历")

2. MessagesPlaceholder —— 为动态消息列表预留位置
      # 后续配合 Memory（第 7 课）实现多轮对话的关键

3. FewShotChatMessagePromptTemplate —— 用示例教 AI 行为
      # 不需要大段 prompt，给几个例子 AI 就能照着做

为什么这些很重要？
  - 前 4 课你一直在用硬编码字符串，实际项目中 prompt 会越来越复杂
  - PromptTemplate 是 LangChain 的"入口协议"——所有链和 Agent 都从它开始
  - 下一课 LCEL 会把 prompt | model | parser 串联起来，模板是链条的第一环

和后面课程的关联：
  第 2 课 LCEL:    prompt | model | parser（模板是链的起点）
  第 3 课 Parser:  结构化输出（比 few-shot JSON 更可靠的方式）
  第 7 课 Memory:  MessagesPlaceholder 是记忆注入的关键
""")

# ============================================================
# 你的练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 把 02_system_prompt.py 改成用 ChatPromptTemplate 实现（不要破坏已有文件，在新目录复制一份改）
2. 用 Few-Shot 做一个"emoji 翻译器"：给 3 个中文→emoji 示例，让 AI 把新中文句子转成 emoji
3. 试一下 ChatPromptTemplate.from_messages() 里的不同角色：
   ("system", ...)、("human", ...)、("ai", ...) 分别是什么意思？
""")
