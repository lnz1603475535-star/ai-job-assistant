"""
第 6 课：LCEL 核心 —— Pipe 语法与 Runnable 接口

核心概念：
  - Runnable 接口：LangChain 所有组件都遵循的统一协议
  - Pipe 语法 (|)：把组件串成链条，像 Linux 管道一样
  - RunnablePassthrough：透传数据，不加工但能让链条跑通
  - RunnableParallel：并行执行多个分支，各自独立
  - RunnableLambda：把普通 Python 函数包装成 Runnable

为什么 LCEL 是 LangChain 的"语法"？
  前 4 课你用的是 LangChain 的"单词"——ChatOpenAI、SystemMessage、@tool。
  这课教你 LangChain 的"语法"——怎么把这些词串成句子。
  所有的链(chain)、Agent、RAG 管道，底层都是 LCEL。

学完这课你就理解了：| 不是装饰语法糖，它是 LangChain 统一所有组件的"胶水"。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
import os, json, time

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 实验①：Runnable 统一接口 —— invoke / batch / stream
# ============================================================
print("=" * 60)
print("实验①：Runnable 接口 —— LangChain 所有组件的统一协议")
print("=" * 60)

# 所有的 LangChain 组件都是 Runnable，都有这三个方法：
#   invoke:   单个输入 → 单个输出（同步、等结果）
#   batch:    多个输入 → 多个输出（批量处理）
#   stream:   单个输入 → 逐步输出（流式，一个一个吐）

# 模型是 Runnable
print("1. invoke —— 一问一答")
result = llm.invoke("用一句话解释什么是 Python 装饰器")
print(f"   回答：{result.content[:80]}...\n")

print("2. batch —— 批量提问（比循环 invoke 快）")
inputs = [
    "什么是 Python 装饰器？",
    "什么是闭包？",
    "装饰器和闭包有什么关系？",
]
results = llm.batch(inputs)
for i, r in enumerate(results):
    print(f"   Q{i+1}: {inputs[i][:30]}...")
    print(f"   A{i+1}: {r.content[:60]}...\n")

print("3. stream —— 逐 token 输出")
print("   ", end="", flush=True)
for chunk in llm.stream("用3个词描述AI"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print("\n")

print(">>> invoke / batch / stream 适用于所有 Runnable，不只是模型。")


# ============================================================
# 实验②：Pipe 语法 —— 用 | 串联组件
# ============================================================
print("=" * 60)
print("实验②：Pipe 语法 —— prompt | model | parser")
print("=" * 60)

# 核心公式： chain = prompt | model | parser
#
# 数据流向：
#   用户输入 → prompt.format() → model.invoke() → parser.parse() → 最终输出
#
# 每个 | 把左边组件的输出喂给右边组件的输入

prompt = ChatPromptTemplate.from_template(
    "把这段文字翻译成{target_lang}，只输出译文：{text}"
)

parser = StrOutputParser()  # 把 AIMessage 对象转成纯字符串

# 这就是 LCEL！三个组件用 | 串成一条链
chain = prompt | llm | parser

# 这条链本身也是 Runnable！所以它也有 invoke / batch / stream
result = chain.invoke({
    "target_lang": "英文",
    "text": "人生苦短，我用Python",
})
print(f"翻译结果：{result}")

print("\n>>> 对比旧方式（前 4 课）：")
print("""
  旧方式：
    messages = prompt.format_messages(target_lang="英文", text="...")
    response = llm.invoke(messages)
    result = response.content               # 还要手动取 .content

  LCEL 方式：
    chain = prompt | llm | parser
    result = chain.invoke({"target_lang": "英文", "text": "..."})

  LCEL 把三步合成一步，而且 chain 本身可以继续参与更长的链条。
""")


# ============================================================
# 实验③：RunnablePassthrough —— 数据透传
# ============================================================
print("=" * 60)
print("实验③：RunnablePassthrough —— 透传与分流")
print("=" * 60)

# RunnablePassthrough 本身不做任何处理，但能帮你把数据"分流"到不同位置
# 最常见的场景：把原始输入同时传给 prompt 和 model

# 场景：给一段代码做 code review，要输出原始代码 + 审查意见
review_prompt = ChatPromptTemplate.from_template("""
请审查以下代码，指出潜在问题：
```
{code}
```
只输出问题和建议。
""")

review_chain = review_prompt | llm | StrOutputParser()

# 用 RunnablePassthrough 把 user_input 同时传给两个地方
final_chain = {
    "original_code": RunnablePassthrough(),   # 原封不动透传
    "review": review_chain,                    # 经过审查链处理后
}

# 注意：这里传给 final_chain 的是普通字符串，但 prompt 中的变量是 {code}
# 所以我们需要调整一下
code = """
def divide(a, b):
    return a / b

def read_file(path):
    f = open(path)
    return f.read()
"""

# 实际上，更常用的 RunnablePassthrough 用法是 .assign()
chain_with_assign = (
    RunnablePassthrough.assign(review=review_chain)
)

result = chain_with_assign.invoke({"code": code})
print("原始代码：")
print(result["code"][:100] + "...")
print(f"\n审查意见：\n{result['review'][:300]}...")

print("\n>>> RunnablePassthrough.assign() 往输入 dict 里追加字段，不修改已有字段。")


# ============================================================
# 实验④：RunnableParallel —— 并行执行
# ============================================================
print("=" * 60)
print("实验④：RunnableParallel —— 并行执行多个分支")
print("=" * 60)

# RunnableParallel 让多个分支同时跑，各自拿到结果后合并
# 每个分支都是独立的 Runnable，互不依赖

translate_prompt = ChatPromptTemplate.from_template(
    "把以下文字翻译成{lang}，只输出译文：{text}"
)
translate_chain = translate_prompt | llm | StrOutputParser()

# 同时翻译成三种语言
parallel_chain = RunnableParallel(
    english=lambda x: translate_chain.invoke({"lang": "英文", "text": x}),
    japanese=lambda x: translate_chain.invoke({"lang": "日文", "text": x}),
    french=lambda x: translate_chain.invoke({"lang": "法文", "text": x}),
)

text = "你好，世界！"
print(f"原文：{text}\n")

start = time.time()
result = parallel_chain.invoke(text)
elapsed = time.time() - start

print("并行翻译结果：")
for lang, translation in result.items():
    print(f"  {lang}: {translation}")
print(f"\n耗时：{elapsed:.1f} 秒（三次翻译并行执行）")

# 对比：如果串行执行
print("\n如果串行执行（先英再日再法），需要 3 倍时间。")
print(">>> RunnableParallel 适合所有分支互不依赖的场景——速度就是分支数倍。")


# ============================================================
# 实验⑤：RunnableLambda —— 把普通函数变成 Runnable
# ============================================================
print("=" * 60)
print("实验⑤：RunnableLambda —— 普通函数也能加入 LCEL 链")
print("=" * 60)

# 你不是非得用 LangChain 的组件——任何 Python 函数都能包装成 Runnable

def word_count(text: str) -> dict:
    """统计文本字数（中文按字符数，英文按空格分词）"""
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    words = len(text.split())
    return {"chinese_chars": chinese_chars, "english_words": words, "raw_length": len(text)}

def format_stats(stats: dict) -> str:
    """把统计数据格式化为可读字符串"""
    return f"中文字符: {stats['chinese_chars']}, 英文单词: {stats['english_words']}, 总字符: {stats['raw_length']}"

# 把普通函数包装成 Runnable
count_runnable = RunnableLambda(word_count)
format_runnable = RunnableLambda(format_stats)

# 然后就能加入 LCEL 链了！
analysis_chain = count_runnable | format_runnable

result = analysis_chain.invoke("LangChain 是一个强大的 LLM 应用框架 framework")
print(f"文本分析结果：{result}")

# 更复杂的例子：在 LLM 链前后插入自定义处理
print("\n--- 复杂例子：翻译 + 字数统计 ---")

# 翻译链的输出是字符串，我们用 RunnableLambda 接住它做后处理
def add_stats(translation: str) -> str:
    stats = word_count(translation)
    stats_line = f"[中文字符:{stats['chinese_chars']}, 英文单词:{stats['english_words']}]"
    return f"{translation}\n{stats_line}"

translate_with_stats = translate_chain | RunnableLambda(add_stats)

result = translate_with_stats.invoke({"lang": "英文", "text": "深度学习正在改变世界"})
print(result)

print("\n>>> RunnableLambda 让你能把任何 Python 函数无缝插入 LCEL 链中。")


# ============================================================
# 实验⑥：串联一切 —— 一个完整的分析管道
# ============================================================
print("=" * 60)
print("实验⑥：串联一切 —— 简历分析管道")
print("=" * 60)

# 把所有组件串起来：Parallel → Prompt → Model → Parser → Lambda

# 准备两条分析 prompt（并行执行）
score_prompt = ChatPromptTemplate.from_template("""
你是一个严格的 HR。给以下简历和 JD 的匹配度打分（只输出数字 1-10）：
简历：{resume}
JD：{jd}
""")

suggestion_prompt = ChatPromptTemplate.from_template("""
你是一个简历优化专家。针对以下 JD，指出简历最大 3 个改进点（每条一句话）：
简历：{resume}
JD：{jd}
""")

# 两条链
score_chain = score_prompt | llm | StrOutputParser()
suggestion_chain = suggestion_prompt | llm | StrOutputParser()

# 并行执行打分和建议
parallel_analysis = RunnableParallel(
    score=score_chain,
    suggestions=suggestion_chain,
)

# 自定义后处理
def format_report(data: dict) -> str:
    return f"""
{'='*40}
简历匹配度报告
{'='*40}
匹配度评分：{data['score'].strip()}/10

改进建议：
{data['suggestions'].strip()}
{'='*40}
"""

final_chain = parallel_analysis | RunnableLambda(format_report)

# 跑起来
resume = "张三，3年Python开发，熟悉Django和Vue，参与过电商项目"
jd = "招聘高级Python工程师，要求5年经验，熟悉微服务架构"

result = final_chain.invoke({"resume": resume, "jd": jd})
print(result)


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 6 课 总结")
print("=" * 60)
print("""
LCEL 的五个核心概念：

1. Pipeline (|) —— 数据从左流到右
      chain = prompt | model | parser
      每个 | 都是一次数据传递：输出 → 输入

2. Runnable.invoke() —— 所有组件的统一调用方式
      模型、prompt、parser、chain、agent 都可以 .invoke()

3. RunnablePassthrough —— 透传数据不加工
      RunnablePassthrough.assign(extra_field=some_chain)
      往 dict 里追加字段，不改变已有字段

4. RunnableParallel —— 无依赖的分支同时跑
      RunnableParallel(branch_a=chain_a, branch_b=chain_b)
      速度 = 单个分支速度（不是所有分支之和）

5. RunnableLambda —— 普通函数加入 LCEL 链
      RunnableLambda(my_func)
      任何 Python 函数都能变成链上的一环

LCEL 是 LangChain 的"统一语言"：
  - 你前面学的 ChatOpenAI 是 Runnable
  - 你今天学的 ChatPromptTemplate 是 Runnable
  - 你后面学的 Retriever、Memory、Agent 都是 Runnable
  - 用 | 串起来，万物皆可链

和后面课程的关联：
  第 3 课 Output Parser:   chain = prompt | model | parser（parser 是最后一环）
  第 5 课 RAG Chain:       chain = retriever | prompt | model | parser
  第 7 课 Memory:          chain = history | prompt | model | parser
  第 10 课 LangGraph:      图的每个 node 内部也是 LCEL 链
""")

# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 用 LCEL 改写 02_system_prompt.py：prompt | llm | StrOutputParser()
2. 写一个 RunnableParallel，同时让 AI 用 3 种不同语气（正式/幽默/极简）回答同一个问题
3. 写一个 RunnableLambda，统计 AI 每次回答的字数，追加到输出后面
4. 试着用 .batch() 批量处理 5 个翻译请求，看比循环 invoke 快多少
""")
