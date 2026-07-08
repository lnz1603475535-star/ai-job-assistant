"""
第 7 课：Output Parsers —— 让 AI 输出结构化数据

核心概念：
  - StrOutputParser：把 AIMessage 转成纯字符串（你已经在第 2 课用过了）
  - CommaSeparatedListOutputParser：让 AI 输出逗号分隔的列表
  - JsonOutputParser：让 AI 输出 JSON
  - PydanticOutputParser：用 Pydantic 模型严格约束输出格式
  - with_structured_output()：让模型原生输出结构化数据（最推荐）

为什么需要 Output Parser？
  前几课你一直用 response.content 取文本。但真实项目里 AI 的输出
  需要被程序处理——存数据库、调 API、做计算。纯文本没法直接用。

学完这课你就理解了：parser 是 LCEL 链条的"最后一环"——raw text in, structured data out。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 实验①：StrOutputParser —— 你已经在用但可能没意识到
# ============================================================
print("=" * 60)
print("实验①：StrOutputParser —— 最基础的 parser")
print("=" * 60)

# 没有 parser 时，invoke 返回的是 AIMessage 对象
prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
chain_without_parser = prompt | llm

result = chain_without_parser.invoke({"concept": "递归"})
print(f"不带 parser 的返回类型：{type(result).__name__}")
print(f"取值方式：result.content\n")

# 有 parser 时，invoke 返回的就是纯字符串
chain_with_parser = prompt | llm | StrOutputParser()
result = chain_with_parser.invoke({"concept": "递归"})
print(f"带 parser 的返回类型：{type(result).__name__}")
print(f"取值方式：直接用 result\n")

print(">>> StrOutputParser 就是帮你自动调用 .content，让链的输出是纯字符串。")


# ============================================================
# 实验②：CommaSeparatedListOutputParser —— 列表型输出
# ============================================================
print("=" * 60)
print("实验②：CommaSeparatedListOutputParser —— 输出列表")
print("=" * 60)

list_parser = CommaSeparatedListOutputParser()

# parser 自带 format_instructions——告诉 AI 应该怎么输出
print(f"Parser 自动生成的格式说明：\n{list_parser.get_format_instructions()}\n")

list_prompt = ChatPromptTemplate.from_messages([
    ("system", "列出用户提到的技术栈关键词。\n{format_instructions}"),
    ("human", "{input}"),
])

# .partial() 是 ChatPromptTemplate 的方法，在 pipe 之前调用
list_prompt_filled = list_prompt.partial(format_instructions=list_parser.get_format_instructions())
list_chain = list_prompt_filled | llm | list_parser

result = list_chain.invoke({"input": "我们后端用 Django + DRF，数据库 MySQL + Redis，部署用 Docker + K8s"})
print(f"AI 提取的技能列表：{result}")
print(f"类型：{type(result).__name__}，可以直接遍历")
for skill in result:
    print(f"  - {skill}")

print("\n>>> CommaSeparatedListOutputParser 让 AI 输出 Python list，直接能用。")


# ============================================================
# 实验③：JsonOutputParser —— JSON 格式输出
# ============================================================
print("=" * 60)
print("实验③：JsonOutputParser —— JSON 格式输出")
print("=" * 60)

json_parser = JsonOutputParser()

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "把用户提供的简历信息转成 JSON。\n{format_instructions}"),
    ("human", "{input}"),
])

json_prompt_filled = json_prompt.partial(format_instructions=json_parser.get_format_instructions())
json_chain = json_prompt_filled | llm | json_parser

result = json_chain.invoke({
    "input": "张三，28岁，3年Python开发经验，熟悉Django和FastAPI，北邮本科毕业"
})

print(f"AI 输出的结构化数据：")
print(f"  类型：{type(result).__name__}")
print(f"  内容：{json.dumps(result, ensure_ascii=False, indent=2)}")
print(f"\n  可以直接取值：result['name'] → {result.get('name', 'N/A')}")

print("\n>>> JsonOutputParser 让 AI 输出 Python dict，比手动 json.loads 可靠得多。")


# ============================================================
# 实验④：PydanticOutputParser —— 严格类型约束
# ============================================================
print("=" * 60)
print("实验④：PydanticOutputParser —— Pydantic 模型约束")
print("=" * 60)

# 定义数据结构（这就是你和后端约定的"协议"）
class Skill(BaseModel):
    name: str = Field(description="技能名称")
    level: str = Field(description="熟练程度：入门/熟练/精通")
    years: Optional[float] = Field(default=None, description="使用年限")

class ResumeAnalysis(BaseModel):
    name: str = Field(description="候选人姓名")
    total_years: int = Field(description="总工作经验年限")
    education: str = Field(description="最高学历")
    skills: List[Skill] = Field(description="技能列表及熟练度")
    match_score: int = Field(description="与 JD 的匹配度评分 1-10")
    summary: str = Field(description="一句话总结")

pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

# 看看 parser 给 AI 的格式要求——自动生成的！
print("PydanticOutputParser 自动生成的格式说明（部分）：")
instructions = pydantic_parser.get_format_instructions()
print(instructions[:400] + "...\n")

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业HR，分析候选人简历与 JD 的匹配度。\n{format_instructions}"),
    ("human", "简历：{resume}\n\nJD：{jd}"),
])

pydantic_prompt_filled = pydantic_prompt.partial(format_instructions=instructions)
pydantic_chain = pydantic_prompt_filled | llm | pydantic_parser

result = pydantic_chain.invoke({
    "resume": "张三，3年Python后端开发，熟悉Django和FastAPI，会Docker，南京大学本科",
    "jd": "招Python高级工程师，5年经验，精通FastAPI和微服务，本科及以上",
})

print(f"输出类型：{type(result).__name__}")
print(f"姓名：{result.name}")
print(f"工作年限：{result.total_years} 年")
print(f"学历：{result.education}")
print(f"匹配度：{result.match_score}/10")
print(f"总结：{result.summary}")
print(f"技能列表：")
for skill in result.skills:
    years_str = f" ({skill.years}年)" if skill.years else ""
    print(f"  - {skill.name}: {skill.level}{years_str}")

print("\n>>> PydanticOutputParser 自动校验类型——AI 漏了字段或类型不对会报错。")


# ============================================================
# 实验⑤：with_structured_output() —— 最推荐的方式
# ============================================================
print("=" * 60)
print("实验⑤：with_structured_output() —— 原生结构化输出")
print("=" * 60)

# with_structured_output() 利用模型原生的 structured output 能力
# 比 PydanticOutputParser 更可靠，因为模型在生成时就遵守 schema

try:
    structured_llm = llm.with_structured_output(ResumeAnalysis)

    # 直接用！不需要 prompt、不需要 parser
    result = structured_llm.invoke(
        "分析此候选人：李四，5年Go开发，精通微服务和K8s，清华硕士。" +
        "JD要求：云原生工程师，3年以上Go经验"
    )

    print(f"输出类型：{type(result).__name__}（直接就是 Pydantic 对象！）")
    print(f"姓名：{result.name}")
    print(f"技能列表：")
    for skill in result.skills:
        print(f"  - {skill.name}: {skill.level}")
    print(f"\n>>> with_structured_output 是最简洁的方式：不需要 prompt template，")
    print("    不需要 parser，模型原生支持，出错概率最低。")
    print("    但注意：需要模型支持 tool calling（DeepSeek 支持）。")

except Exception as e:
    print(f"DeepSeek 可能不完全支持此功能，错误信息：{e}")
    print("如果报错，回到实验④用 PydanticOutputParser，效果类似。")


# ============================================================
# 实验⑥：Parser 对比 + 选型指南
# ============================================================
print("=" * 60)
print("实验⑥：配合 LCEL —— parser 是链条的最后一环")
print("=" * 60)

# 回到我们熟悉的 LCEL 公式：chain = prompt | model | parser
# 不同 parser 让同一条链的输出类型不同

# 列表格式输出（注意这里 format 字段已经包含在 prompt 模板中，不需要再 partial）
prompt = ChatPromptTemplate.from_template(
    "列出{num}个{category}相关的 Python 库名。{format}"
)

list_chain_v2 = prompt | llm | CommaSeparatedListOutputParser()
json_chain_v2 = prompt | llm | JsonOutputParser()

result = list_chain_v2.invoke({
    "num": "5", "category": "机器学习",
    "format": "只输出逗号分隔的库名，不要编号不要解释"
})
print(f"列表格式：{result}\n")

result = json_chain_v2.invoke({
    "num": "3", "category": "Web框架",
    "format": "输出 JSON 数组，每个元素是 {{\"name\": \"库名\", \"description\": \"一句话\"}}"
})
print(f"JSON 格式：{json.dumps(result, ensure_ascii=False, indent=2)}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第 7 课 总结 —— Parser 选型指南")
print("=" * 60)
print("""
场景                          推荐 Parser
─────────────────────────────────────────────
只要纯文本                     StrOutputParser
只要列表                       CommaSeparatedListOutputParser
要 JSON，结构不固定            JsonOutputParser
要 JSON，结构固定且要校验      PydanticOutputParser
要 JSON，结构固定，模型支持    with_structured_output()（最推荐）

实操建议：
  1. 能用 with_structured_output() 就用它（最省事、最可靠）
  2. 不行就用 PydanticOutputParser（兼容性最好）
  3. 简单场景用 JsonOutputParser / CommaSeparatedListOutputParser
  4. 别手动 json.loads(response.content)——那是前 LCEL 时代的做法

和后面课程的关联：
  第 5 课 RAG:      chain = retriever | prompt | model | parser
  第 9 课 Agent:    agent 中也可以用 structured output
  第 12 课 项目:    技能匹配结果用 Pydantic 模型承载
""")

# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 定义一个 Pydantic 模型表示 "面试问题"（含：问题内容、考察点、难度、参考答案），
   让 AI 根据 JD 生成 3 个面试问题
2. 试一下当 AI 输出不满足 Pydantic 模型要求时（比如少了一个必填字段），会发生什么？
3. 用 with_structured_output() 做一个"翻译 + 语言检测"的链：
   输入任意文字，输出 {"detected_lang": "...", "translation": "..."}
""")
