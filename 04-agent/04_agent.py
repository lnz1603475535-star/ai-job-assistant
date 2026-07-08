"""
第 4 课：Agent —— 模型 + 工具 + 提示词，三合一

核心概念：
  - create_agent()：  把模型、工具、prompt 组装成一个能"自主思考"的 Agent
  - Agent 循环：           思考 → 调工具 → 看结果 → 再思考 → …… → 最终回答
  - stream()：             让你看到 Agent 每一步在干什么（调试利器）

和第 3 课的对比：
  第 3 课（手动）：你写循环，手动判断 tool_calls，手动执行，手动拼消息
  第 4 课（Agent）：create_agent() 全部自动搞定，而且能多轮反复调用工具

学完这课你就理解了：Agent 是 LangChain 最核心的价值，前面 3 课都是铺垫
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
import math
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# Step 1: 定义工具（和第 3 课一模一样）
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。city 参数是城市名称，如 北京、上海、深圳。"""
    weather_data = {
        "北京": "晴，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "雷阵雨，30°C，湿度 80%",
        "杭州": "阴，22°C，湿度 55%",
    }
    return weather_data.get(city, f"{city}：晴，23°C（模拟数据）")


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

@tool
def translate(text: str, target_lang: str) -> str:
    '''将文本翻译成目标语言。text 是要翻译的内容，target_lang 是目标语言（如 英文、日文）。'''
    # 模拟翻译（真实项目里调翻译 API）
    return f"「{text}」的{target_lang}翻译结果（模拟）"

# ============================================================
# Step 2: 创建 Agent —— 一行代码，模型+工具+提示词合体
# ============================================================
system_prompt = "你是一个旅游助手。查询到的数据要如实汇报，不要编造。"

agent = create_agent(
    model=llm,
    tools=[get_weather, calculator, translate],
    system_prompt=system_prompt,
)

# ============================================================
# 实验①：Agent 自动处理工具调用（对比第 3 课的手动循环）
# ============================================================
print("=" * 60)
print("实验①：Agent 自动查天气 + 算数")
print("=" * 60)
print("你只需要一行 agent.invoke()，不需要手动判断 tool_calls、")
print("不需要手动执行工具、不需要手动拼 ToolMessage。Agent 全包了。\n")

result = agent.invoke({
    "messages": [HumanMessage(content="上海天气怎么样？顺便算一下 88 × 77")]
})

# 打印所有消息，看 Agent 干了什么
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    content = str(msg.content)[:100] if msg.content else "(无内容)"
    tool_info = ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_info = f" [准备调用: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}]"
    print(f"[{i}] {msg_type}: {content}{tool_info}")

print(f"\n>>> 最终回答：\n{result['messages'][-1].content}")


# ============================================================
# 实验②：stream() —— 实时看到 Agent 的每一步思考
# ============================================================
print("\n" + "=" * 60)
print("实验②：用 stream() 观察 Agent 的思考过程")
print("=" * 60)

user_input = "北京天气如何？温度超过 30 度的话帮我算一下 30 的平方"

print(f"用户：{user_input}\n")

for chunk in agent.stream({"messages": [HumanMessage(content=user_input)]}):
    # chunk 是一个 dict，key 是节点名（agent / tools）
    for node_name, node_data in chunk.items():
        messages = node_data.get("messages", [])
        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type == "AIMessage" and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🧠 AI 决定调用：{tc['name']}({tc['args']})")
            elif msg_type == "ToolMessage":
                print(f"  🔧 {msg.name} 返回：{msg.content}")
            elif msg_type == "AIMessage" and msg.content:
                print(f"  💬 AI 回答：{msg.content[:150]}...")


# ============================================================
# 实验③：复杂任务 —— Agent 的多轮推理
# ============================================================
print("\n" + "=" * 60)
print("实验③：多轮推理 —— Agent 反复调用工具直到解决问题")
print("=" * 60)

user_input = "上海和北京哪边更热？温差是多少？把这两个地名翻译成日语。"

print(f"用户：{user_input}\n")

result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

# 只看关键步骤
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    if msg_type == "HumanMessage":
        print(f"[{i}] 用户提问：{msg.content}")
    elif msg_type == "AIMessage" and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"[{i}] AI 调工具：{tc['name']}({tc['args']})")
    elif msg_type == "ToolMessage":
        print(f"[{i}] 工具返回：{msg.content}")
    elif msg_type == "AIMessage" and msg.content:
        print(f"[{i}] AI 回答：\n{msg.content}")


# ============================================================
# 关键对比：第 3 课 vs 第 4 课
# ============================================================
print("\n" + "=" * 60)
print("第 3 课 vs 第 4 课 —— 代码量对比")
print("=" * 60)
print("""
第 3 课（手动，20+ 行胶水代码）：
  response = llm_with_tools.invoke(messages)
  if response.tool_calls:                          # 判断要不要调工具
      messages.append(response)
      for tc in response.tool_calls:               # 遍历每个 tool_call
          func = tool_map[tc["name"]]              # 找到对应函数
          result = func.invoke(tc["args"])          # 执行函数
          messages.append(ToolMessage(...))         # 拼 ToolMessage
      final = llm_with_tools.invoke(messages)       # 再问一次 AI

第 4 课（Agent，1 行）：
  result = agent.invoke({"messages": [HumanMessage(...)]})

区别：
  1. Agent 自动循环——调一次工具不够？它会再调，直到满意
  2. 第 4 课的实验③"比较温度"就需要查两次天气，Agent 自动查了两次
  3. stream() 让你能实时看到 AI 的决策过程，调试非常方便
""")


# ============================================================
# 你的练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 加一个新工具（如翻译、查快递），重新创建 agent，跑通
2. 把 system_prompt 改成不同角色（旅行助手、算命先生），看 Agent 行为怎么变
3. 问一个需要连续调用 3 次工具的问题，观察 Agent 怎么一步步解决
""")
