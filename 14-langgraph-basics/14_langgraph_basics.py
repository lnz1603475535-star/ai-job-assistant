"""
第 10 课：LangGraph 基础 —— 手写 Agent 状态机

核心概念：
  - StateGraph：用状态图定义工作流，nodes = 步骤，edges = 流向
  - State (TypedDict)：定义图中流转的数据结构
  - Conditional edges：根据条件决定下一步走哪个 node
  - 从零手写 Agent loop：call_model → should_use_tool? → call_tool → back to model

create_agent 是"黑盒"——你传工具和 prompt，它帮你跑。LangGraph 是"白盒"——
你定义每一步干什么、下一步往哪走。当业务逻辑复杂到 create_agent 不够用时，
LangGraph 就是你的答案。

学完这课你就理解了：LangGraph 让你精确控制 agent 的每一个决策分支。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    weather = {"北京": "晴天 25°C", "上海": "多云 28°C", "深圳": "小雨 30°C"}
    return weather.get(city, f"{city}：晴 22°C")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {"abs": abs, "sqrt": __import__('math').sqrt}))
    except Exception as e:
        return f"错误：{e}"

tools = [get_weather, calculator]
tools_map = {t.name: t for t in tools}


# ============================================================
# 实验①：最简单的图 —— 线性流程
# ============================================================
print("=" * 60)
print("实验①：最简单的 LangGraph —— 线性流程")
print("=" * 60)

# 第 1 步：定义 State（图中流转的数据）
class SimpleState(TypedDict):
    messages: Annotated[list, operator.add]  # operator.add = 追加新消息
    step_count: int

# 第 2 步：定义 Node（操作 state 的函数）
def node_a(state: SimpleState) -> dict:
    print("  [节点 A] 开始...")
    return {"messages": [AIMessage(content="你好，来自节点 A！")], "step_count": 1}

def node_b(state: SimpleState) -> dict:
    print("  [节点 B] 处理中...")
    return {
        "messages": [AIMessage(content=f"节点 B 在此。已执行步数：{state['step_count']}")],
        "step_count": state["step_count"] + 1,
    }

# 第 3 步：构建图 —— 加节点、加边
graph = StateGraph(SimpleState)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge(START, "a")    # 开始 → 节点 A
graph.add_edge("a", "b")      # 节点 A → 节点 B
graph.add_edge("b", END)      # 节点 B → 结束

app = graph.compile()

print("图编译完成。运行中...\n")
result = app.invoke({"messages": [HumanMessage(content="开始！")]})

print(f"\n最终状态：")
print(f"  消息列表：{[type(m).__name__ for m in result['messages']]}")
print(f"  步数：{result['step_count']}")
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(f"    {type(msg).__name__}：{msg.content[:80]}")

print("\n>>> 每个节点是一个 Python 函数，边定义了流转方向。")


# ============================================================
# 实验②：条件边 —— 分支逻辑
# ============================================================
print("=" * 60)
print("实验②：条件边 —— 分支决策")
print("=" * 60)

class BranchState(TypedDict):
    messages: Annotated[list, operator.add]

# 路由函数：决定下一步去哪个节点
def length_router(state: BranchState) -> Literal["short_answer", "long_answer"]:
    last_msg = state["messages"][-1]
    if len(str(last_msg.content)) > 50:
        print(f"  [路由] 消息很长（{len(str(last_msg.content))} 字符）→ 短回答")
        return "short_answer"
    else:
        print(f"  [路由] 消息很短（{len(str(last_msg.content))} 字符）→ 长回答")
        return "long_answer"

def short_node(state: BranchState) -> dict:
    return {"messages": [AIMessage(content="好的。")]}

def long_node(state: BranchState) -> dict:
    return {"messages": [AIMessage(content="收到你的详细问题，我会给出全面的回答。")]}

branch_graph = StateGraph(BranchState)
branch_graph.add_node("router", lambda s: {"messages": []})  # 透传
branch_graph.add_node("short_answer", short_node)
branch_graph.add_node("long_answer", long_node)

branch_graph.add_edge(START, "router")
branch_graph.add_conditional_edges("router", length_router, {
    "short_answer": "short_answer",
    "long_answer": "long_answer",
})
branch_graph.add_edge("short_answer", END)
branch_graph.add_edge("long_answer", END)

app = branch_graph.compile()

print("\n测试 1：短输入")
app.invoke({"messages": [HumanMessage(content="你好")]})

print("\n测试 2：长输入")
app.invoke({"messages": [HumanMessage(content="请详细告诉我计算机科学从 1900 年到 2000 年的完整历史")]})

print("\n>>> 条件边 = 工作流中的 if/else 分支。")


# ============================================================
# 实验③：从零手写 Agent 循环 —— LangGraph 核心
# ============================================================
print("=" * 60)
print("实验③：从零手写 Agent 循环 —— LangGraph 核心")
print("=" * 60)

# 这就是 create_agent 底层做的事情！

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 给模型绑定工具
llm_with_tools = llm.bind_tools(tools)

# 节点 1：调用模型
def call_model(state: AgentState) -> dict:
    print("  [call_model] 调用 LLM...")
    response = llm_with_tools.invoke(
        [SystemMessage(content="你是一个有帮助的助手。")] + state["messages"]
    )
    return {"messages": [response]}

# 节点 2：执行工具
def call_tools(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_messages = []
    for tc in last_msg.tool_calls:
        tool_func = tools_map[tc["name"]]
        result = tool_func.invoke(tc["args"])
        print(f"  [call_tools] {tc['name']}({tc['args']}) → {result}")
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_messages}

# 路由：继续调工具还是结束？
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("  [路由] → 执行工具")
        return "tools"
    print("  [路由] → 结束")
    return "__end__"

# 构建 Agent 图
agent_graph = StateGraph(AgentState)
agent_graph.add_node("call_model", call_model)
agent_graph.add_node("call_tools", call_tools)

agent_graph.add_edge(START, "call_model")
agent_graph.add_conditional_edges("call_model", should_continue, {
    "tools": "call_tools",
    "__end__": END,
})
agent_graph.add_edge("call_tools", "call_model")  # 工具执行完，回到模型继续！

agent_app = agent_graph.compile()

print("Agent 图编译完成！\n")

# 运行！
print("用户：北京天气怎么样？再算一下 15 * 7\n")
result = agent_app.invoke({
    "messages": [HumanMessage(content="北京天气怎么样？再算一下 15 * 7")]
})

print(f"\n最终回答：")
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"  {msg.content}")


# ============================================================
# 实验④：手写 vs create_agent 对比
# ============================================================
print("=" * 60)
print("实验④：手写 Agent vs create_agent —— 效果一样，控制力不同")
print("=" * 60)

from langchain.agents import create_agent

auto_agent = create_agent(model=llm, tools=tools, system_prompt="你是一个有帮助的助手。")

print("--- create_agent ---")
auto_result = auto_agent.invoke({
    "messages": [HumanMessage(content="上海天气怎么样？简短回答。")]
})
print(f"  {auto_result['messages'][-1].content[:120]}...")

print("\n--- 手写 LangGraph Agent ---")
manual_result = agent_app.invoke({
    "messages": [HumanMessage(content="上海天气怎么样？简短回答。")]
})
for msg in manual_result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"  {msg.content[:120]}...")

print("""
两者都能完成任务。区别在于：
  create_agent  = 快速、简单，适合 80% 的场景
  LangGraph     = 完全控制每一步决策和状态
                  当你需要自定义路由、复杂状态或非标准流程时使用。

>>> LangGraph vs create_agent：效果一样，但 LangGraph 把方向盘交到你手里。""")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 10 课 总结 —— LangGraph 三个核心概念")
print("=" * 60)
print("""
LangGraph 的三大构建块：

1. State（TypedDict）
     定义图中"流动的数据"
     字段使用 Annotated[类型, reducer] 来定义合并逻辑

2. Node（普通 Python 函数）
     定义每个步骤"做什么"
     接收 state，返回部分更新的 dict

3. Edge（边）
     普通边：A → B（始终从 A 到 B）
     条件边：A → 路由函数 → B 或 C（根据 state 决策）

Agent 循环 = 最简单的 LangGraph 模式：
  START → call_model → [检查是否有 tool_calls]
                           ├─ 有 → call_tools → call_model（循环！）
                           └─ 没有 → END

什么时候用 create_agent vs LangGraph：
  create_agent  简单 agent、标准工具、快速原型
  LangGraph     自定义路由、复杂状态、人机协作、
                多 agent、持久化、任何非标准流程

和后面课程的关联：
  第 11 课 LangGraph 进阶：checkpointing、人机协作、子图
  第 12 课 项目：LangGraph 驱动的求职助手
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 给手写 Agent 加一个新工具（比如 get_time），测试它能否正确调用
2. 加一个"反思"节点，在给出最终答案前让 AI 先检查一遍
3. 在白纸上画出 Agent 流程图：START → call_model → should_continue → call_tools/END
   标出每一步的输入和输出，走查一个需要两次工具调用的场景
""")
