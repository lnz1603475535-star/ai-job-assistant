"""
第 3 课：Tools —— 让 AI 能做事

核心概念：
  - @tool 装饰器：    把普通 Python 函数变成 AI 可以调用的工具
  - bind_tools()：    告诉模型"这些工具你可以用，需要时自己选"
  - 工具调用流程：    用户提问 → AI 决定调哪个工具 → 执行工具 → AI 读结果 → 最终回答

对比：
  - 没工具：AI 凭记忆回答（可能不准、瞎编数字）
  - 有工具：AI 主动调用你的函数获取准确结果

学完这课你就明白了：LangChain 的 @tool 省掉了手写 JSON Schema 的麻烦
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import math
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# Step 1: 定义工具
# ============================================================
# @tool 装饰器做了两件事：
#   1. 把函数标记为"工具"
#   2. 用 docstring 作为工具的描述（AI 靠这个理解工具能干嘛！）

@tool
def huilv(yuan: str, xian: str) -> str:
    '''转换汇率，yuan 是源货币，xian 是目标货币。如果 yuan 是"美元"，xian 是"人民币"，就把 1 美元换算成人民币的汇率返回。'''
    
    return f"{yuan}可以转换成多少{xian}（模拟）"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。city 参数是城市名称，如 北京、上海、深圳。"""
    weather_data = {
        "北京": "晴，25°C，湿度 40%，适合户外活动",
        "上海": "多云，28°C，湿度 65%，有点闷热",
        "深圳": "雷阵雨，30°C，湿度 80%，建议带伞",
        "杭州": "阴，22°C，湿度 55%，凉爽舒适",
    }
    return weather_data.get(city, f"{city}：晴，23°C，湿度 50%（模拟数据）")


@tool
def calculator(expression: str) -> str:
    """执行数学计算。expression 参数是数学表达式，如 3*4+5、sqrt(64)、2**10。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "pi": math.pi, "e": math.e,
        })
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{e}"


tools = [get_weather, calculator, huilv]
llm_with_tools = llm.bind_tools(tools)  # 关键一步：把工具"装"到模型上


# ============================================================
# 实验①：不给工具 —— AI 只能瞎猜
# ============================================================
print("=" * 60)
print("实验①：不给 AI 工具 —— 让它算 12345 × 67890")
print("=" * 60)

response = llm.invoke([HumanMessage(content="请计算 12345 × 67890 等于多少？")])
print(f"\nAI 回答：\n{response.content}\n")


# ============================================================
# 实验②：给工具 —— AI 主动调用计算器
# ============================================================
print("=" * 60)
print("实验②：给 AI 绑定 calculator 工具 —— 同样的问题")
print("=" * 60)

messages = [HumanMessage(content="请计算 12345 × 67890 等于多少？")]
response = llm_with_tools.invoke(messages)

print(f"\n>>> AI 返回的消息类型：{type(response).__name__}")
print(f">>> 有 tool_calls 吗？{bool(response.tool_calls)}")

if response.tool_calls:
    print(f">>> AI 决定调用的工具：{[tc['name'] for tc in response.tool_calls]}")
    print(f">>> 调用参数：{[tc['args'] for tc in response.tool_calls]}")

    # ---------- 手动执行工具（第 4 课 Agent 会帮你自动做这一步）----------
    messages.append(response)
    tool_map = {"get_weather": get_weather, "calculator": calculator}

    for tc in response.tool_calls:
        func = tool_map[tc["name"]]
        result = func.invoke(tc["args"])
        print(f">>> {tc['name']} 返回：{result}")
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # ---------- 第二次调用：AI 读工具结果，给出最终回答 ----------
    final_response = llm_with_tools.invoke(messages)
    print(f"\n>>> AI 最终回答：\n{final_response.content}")


# ============================================================
# 实验③：多个工具混在一起，AI 自己选
# ============================================================
print("\n" + "=" * 60)
print("实验③：同时提供天气 + 计算器 + 翻译工具，看 AI 怎么选")
print("=" * 60)

questions = [
    "北京今天天气怎么样？",
    "帮我算一下 2 的 10 次方是多少？",
    "深圳会不会下雨？顺便帮我算一下 156 × 23 等于多少",
    "1美元等于多少人民币？"
]

for q in questions:
    print(f"\n--- 用户：{q}")
    messages = [HumanMessage(content=q)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        names = [f"{tc['name']}({tc['args']})" for tc in response.tool_calls]
        print(f"   调用了 {len(response.tool_calls)} 个工具：{names}")

        messages.append(response)
        for tc in response.tool_calls:
            func = {"get_weather": get_weather, "calculator": calculator, "huilv": huilv}[tc["name"]]
            result = func.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        final = llm_with_tools.invoke(messages)
        print(f"   最终回答：{final.content[:120]}...")
    else:
        print(f"   没调工具，直接回答：{response.content[:80]}...")


# ============================================================
# 实验④：加上 System Prompt —— 第 2 课 + 第 3 课的组合
# ============================================================
print("\n" + "=" * 60)
print("实验④：System Prompt + Tools 一起用（第 2 课 + 第 3 课）")
print("=" * 60)

system = SystemMessage(content="你是一个旅行助手，回答要带 emoji，语气热情活泼。")

messages = [
    system,
    HumanMessage(content="我周末想去上海玩，那边的天气适合出门吗？")
]
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    messages.append(response)
    for tc in response.tool_calls:
        func = {"get_weather": get_weather, "calculator": calculator, "huilv": huilv}[tc["name"]]
        result = func.invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    final = llm_with_tools.invoke(messages)
    print(f"\nAI 回答：\n{final.content}")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("三个核心概念")
print("=" * 60)
print("""
1. @tool 装饰器
   - 把 Python 函数变成 AI 可调用的工具
   - docstring 很重要！AI 靠它理解这个工具的用途和参数

2. bind_tools()
   - 把工具列表"装"到模型上
   - AI 收到消息后会自己判断：需要调工具吗？调哪个？

3. 工具调用是"两轮对话"
   - 第 1 轮：用户提问 → AI 决定调工具 → 返回 tool_calls（不是最终答案！）
   - 第 2 轮：执行工具 → 告诉 AI 结果 → AI 根据结果组织最终回答

4. 和不用 LangChain 的区别
   - 不用 LangChain：手写 JSON Schema，手动判断 tool_calls，手动拼 ToolMessage
   - 用 LangChain：@tool 装饰器自动生成 Schema，消息类型帮你拼好对话链
""")


# ============================================================
# 你的练习
# ============================================================
print("=" * 60)
print("练习：自己写一个新工具，加进去试试")
print("=" * 60)
print("""
参考模板：

@tool
def translate(text: str, target_lang: str) -> str:
    '''将文本翻译成目标语言。text 是要翻译的内容，target_lang 是目标语言（如 英文、日文）。'''
    # 模拟翻译（真实项目里调翻译 API）
    return f"「{text}」的{target_lang}翻译结果（模拟）"

# 1. 把 translate 加入 tools 列表
# 2. 重新运行，试试问 AI："帮我把'你好世界'翻译成日文"
""")
