"""
第 11 课：LangGraph 进阶 —— 持久化与人机协作

核心概念：
  - Checkpointing：保存 Agent 每一步的状态，支持断点续传
  - Human-in-the-loop：Agent 在关键步骤暂停，等待人类审批
  - interrupt() / Command()：让图暂停和恢复的机制

第 10 课你学会了手写 Agent 状态机。这课教你两个生产级功能：
  1. 持久化——Agent 崩溃了？从断点继续，不会丢状态
  2. 人机协作——Agent 不擅长的决策，交给人类拍板

学完这课你就理解了：
  LangGraph 的真正威力不在"能跑 Agent"，而在"能控制 Agent 怎么跑"。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, Literal
import operator
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============================================================
# 实验①：Checkpointing —— 状态持久化
# ============================================================
print("=" * 60)
print("实验①：Checkpointing —— 保存和恢复状态")
print("=" * 60)

class CheckpointState(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

def increment(state: CheckpointState) -> dict:
    new_count = state.get("count", 0) + 1
    print(f"  [increment] count: {state.get('count', 0)} → {new_count}")
    return {"count": new_count, "messages": [AIMessage(content=f"第 {new_count} 步")]}

graph = StateGraph(CheckpointState)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_edge("increment", END)

# MemorySaver 在内存中存储检查点
# 生产环境用：pip install langgraph-checkpoint-sqlite
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 用 thread_id 区分不同对话
config = {"configurable": {"thread_id": "demo-thread-1"}}
print("Thread ID：demo-thread-1\n")

# 逐步运行
for i in range(3):
    result = app.invoke(
        {"messages": [HumanMessage(content=f"第 {i+1} 次运行")]},
        config=config,
    )
    print(f"  第 {i+1} 次：count={result['count']}")

# 状态跨 invoke 持久化！
print(f"\n最终 count：{result['count']}（跨越 3 次独立调用，状态保持）")
print(">>> 没有 checkpointer，每次 invoke() 都是重新开始。有了它，状态持续存在。")


# ============================================================
# 实验②：Human-in-the-loop —— 人机协作
# ============================================================
print("=" * 60)
print("实验②：Human-in-the-loop —— 暂停等待人类审批")
print("=" * 60)

@tool
def modify_resume(section: str, new_content: str) -> str:
    """修改简历的某个部分。section：部分名称，new_content：更新后的内容。"""
    return f"[已修改] {section} 更新为：{new_content}"

tools = [modify_resume]
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    pending_approval: bool

def call_model(state: ApprovalState) -> dict:
    """第 1 步：AI 决定做什么。"""
    print("  [AI] 思考中...")
    sys_msg = SystemMessage(content="你是简历编辑助手。收到修改请求时，使用 modify_resume 工具。")
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def human_approval(state: ApprovalState) -> dict:
    """第 2 步：人类审批，通过后执行。"""
    last_msg = state["messages"][-1]
    if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
        return {}

    tc = last_msg.tool_calls[0]
    print(f"\n  {'='*50}")
    print(f"  AI 想要执行：{tc['name']}({tc['args']})")
    print(f"  {'='*50}")

    approval = interrupt("批准这个修改吗？(yes/no)")
    print(f"  [人类] 回复：{approval}")

    if approval.lower() != "yes":
        return {"messages": [
            ToolMessage(content="被人类拒绝", tool_call_id=tc["id"]),
            AIMessage(content="修改被拒绝。人类没有批准。"),
        ]}

    # 执行工具
    tool_func = tools_map[tc["name"]]
    result = tool_func.invoke(tc["args"])
    print(f"  [工具] 已执行：{result}")
    return {"messages": [ToolMessage(content=result, tool_call_id=tc["id"])]}



def final_model(state: ApprovalState) -> dict:
    """第 3 步：AI 总结结果。"""
    print("  [AI] 总结中...")
    response = llm.invoke(
        [SystemMessage(content="用一句话总结刚才做了什么。")]
        + state["messages"]
    )
    return {"messages": [response]}

# 构建审批流程图（简化版：3 个节点）
approval_graph = StateGraph(ApprovalState)
approval_graph.add_node("ai_decide", call_model)
approval_graph.add_node("human_review", human_approval)
approval_graph.add_node("ai_respond", final_model)

approval_graph.add_edge(START, "ai_decide")
approval_graph.add_edge("ai_decide", "human_review")
approval_graph.add_edge("human_review", "ai_respond")
approval_graph.add_edge("ai_respond", END)

checkpointer2 = MemorySaver()
app = approval_graph.compile(checkpointer=checkpointer2)

config = {"configurable": {"thread_id": "approval-demo"}}

# 启动对话
print("\n用户：用 modify_resume 把「技能」改成「Python 5年，Django，Docker」\n")

# LangGraph 1.x: interrupt() 不再抛出异常，而是静默暂停图
# 通过 app.get_state(config).interrupts 判断图是否在 interrupt() 处暂停
result = app.invoke(
    {"messages": [HumanMessage(content="用 modify_resume 把「技能」改成「Python 5年，Django，Docker」")]},
    config=config,
)

snapshot = app.get_state(config)
if snapshot.interrupts:  # 非空 → 图在 interrupt() 处暂停了
    print(f"\n⏸️  图已暂停，等待人类审批")
    print(f"   提示: {snapshot.interrupts[0].value}")
    print("在实际应用中，你会：")
    print("  1. 通过 UI 向用户展示待审批的修改")
    print("  2. 等待用户点击（批准/拒绝）")
    print("  3. 用 Command(resume='yes') 或 Command(resume='no') 恢复执行")
else:
    print(f"\n最终结果：{result['messages'][-1].content[:150]}...")


# ============================================================
# 实验③：恢复中断的图
# ============================================================
print("=" * 60)
print("实验③：用人类决定恢复执行")
print("=" * 60)

# 用人类的决定恢复图执行
print("正在以人类批准恢复...\n")

resumed_result = app.invoke(
    Command(resume="yes"),  # 人类说 YES！
    config=config,
)
print(f"\n最终 AI 回复：{resumed_result['messages'][-1].content[:200]}...")

print("\n>>> Command(resume=...) 把人类的决定传回图中。")
print("    图从上次中断的地方继续执行。")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 11 课 总结 —— LangGraph 进阶")
print("=" * 60)
print("""
Checkpointing（检查点机制）：
  checkpointer = MemorySaver()  # 演示用（内存存储）
  # 生产环境：
  # checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

  graph.compile(checkpointer=checkpointer)
  config = {"configurable": {"thread_id": "session-123"}}

  - 每个节点（超步）执行后自动保存状态
  - 相同 thread_id → 状态跨 invoke 持久化
  - 不同 thread_id → 独立对话

Human-in-the-loop（人机协作）：
  interrupt("提示消息")    →  暂停图，显示消息给人类
                            LangGraph 1.x: 不抛异常，用 get_state().next 判断暂停
  Command(resume=...)     →  用人类的输入恢复执行

  适用场景：
  - 审批工作流（本次演示）
  - AI 提议，人类审核后再执行
  - 敏感操作（删除、发邮件、发布）

子图（概念）：
  - 复杂工作流可以拆分成子图
  - 每个子图是独立编译的图
  - add_node("子图名", 子图.compile())
  - 让大图保持可维护性

LangGraph Agent 设计模式：
  - Supervisor（主管模式）：一个 agent 分发给多个专业子 agent
  - Hierarchical（层级模式）：多层 agent，每层有更窄的职责范围
  - Custom routing（自定义路由）：基于任意 state 字段的条件分支

和后面课程的关联：
  第 12 课 项目：求职助手用 LangGraph 编排完整工作流
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 给人机协作流程加一条"拒绝"路径：
   如果人类说"no"，跳过工具执行，让 AI 解释为什么被拒绝了
2. 试试用两个不同的 thread_id，观察状态如何隔离
3. 在工具执行后加第二个 interrupt()（在 AI 回答之前），
   让人类也能审核工具实际执行的结果
""")
