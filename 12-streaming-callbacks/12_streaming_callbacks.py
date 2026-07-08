"""
第 8 课：Streaming & Callbacks —— 流式输出与可观测性

核心概念：
  - stream()：逐 chunk 输出，像 ChatGPT 一样逐字显示
  - astream()：异步流式（适合 async 应用）
  - Custom Callbacks：监听 LLM 调用、工具调用等事件
  - stream_mode：控制流的粒度（messages / updates / custom）

前几课你用过 agent.stream()，但没深入理解。这节课系统学习流式输出
和 callback 机制——这是生产环境必备的能力。

学完这课你就理解了：stream() 不只是"好看"，它决定了用户体验和生产可观测性。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
import os, time

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 实验①：invoke vs stream
# ============================================================
print("=" * 60)
print("实验①：invoke vs stream —— 两种获取结果的方式")
print("=" * 60)

prompt = ChatPromptTemplate.from_template("用三句话讲一个关于{topic}的短故事。")
chain = prompt | llm | StrOutputParser()

# invoke：等全部完成再返回
print("--- invoke（等全部完成）---")
start = time.time()
result = chain.invoke({"topic": "一个机器人学画画"})
elapsed = time.time() - start
print(f"结果：{result[:100]}...")
print(f"耗时：{elapsed:.2f}s（一次性返回）\n")

# stream：逐 token 输出
print("--- stream（逐 token 输出）---")
start = time.time()
print("结果：", end="", flush=True)
first_token_time = None
for chunk in chain.stream({"topic": "一个机器人学画画"}):
    if first_token_time is None:
        first_token_time = time.time() - start
    print(chunk, end="", flush=True)
total_time = time.time() - start
print(f"\n首个 token：{first_token_time:.2f}s | 总耗时：{total_time:.2f}s")
print("\n>>> stream() 立刻显示输出，invoke() 让用户干等。")


# ============================================================
# 实验②：从模型直接流式输出
# ============================================================
print("=" * 60)
print("实验②：模型级别的 token 流式")
print("=" * 60)

# 模型本身就可以流式输出（不需要 chain）
print("直接从模型流式输出：\n")
for chunk in llm.stream("用一句话解释什么是 Python 装饰器。"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print("\n\n>>> 每个 chunk 是一个 AIMessageChunk，.content 是一个或多个 token。")


# ============================================================
# 实验③：自定义 Callback 处理器
# ============================================================
print("=" * 60)
print("实验③：自定义 Callback —— 监听 LLM 调用")
print("=" * 60)

class TimingCallback(BaseCallbackHandler):
    """一个记录 LLM 调用耗时和 token 用量的回调处理器。"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start_time = time.time()
        # 显示发给 LLM 的 prompt 预览
        prompt_preview = str(prompts[0])[:80] if prompts else "(无 prompt)"
        print(f"[回调] LLM 调用开始 | prompt: {prompt_preview}...")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self._start_time
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        print(f"[回调] LLM 调用结束 | 耗时: {elapsed:.2f}s | "
              f"tokens: {token_usage.get('total_tokens', '未知')}")

    def on_llm_error(self, error, **kwargs):
        print(f"[回调] LLM 出错: {error}")

# 创建一个带自定义回调的 LLM
llm_with_callback = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    callbacks=[TimingCallback()],
)

print("使用自定义回调调用 LLM：\n")
result = llm_with_callback.invoke("1+1 等于几？")
print(f"\n回复：{result.content}")

print("\n>>> Callback 让你能挂载到每次 LLM 调用上，做日志、监控、告警。")


# ============================================================
# 实验④：Agent 流式 + Callback 组合
# ============================================================
print("=" * 60)
print("实验④：Agent 流式输出 + 回调")
print("=" * 60)

@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    weather = {"北京": "晴天，25°C", "上海": "多云，28°C"}
    return weather.get(city, f"{city}：晴天，22°C")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个天气助手。简洁回答。",
)

print("Agent 流式输出（显示每个决策步骤）：\n")
print("用户：北京天气怎么样？\n")

for chunk in agent.stream(
    {"messages": [HumanMessage(content="北京天气怎么样？")]},
    stream_mode="updates",  # 显示每个节点的更新
):
    node_name = list(chunk.keys())[0]
    node_data = chunk[node_name]
    messages = node_data.get("messages", [])
    for msg in messages:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [{node_name}] AI 决定调用：{tc['name']}({tc['args']})")
        elif msg_type == "ToolMessage":
            print(f"  [{node_name}] 工具返回：{msg.content}")
        elif hasattr(msg, "content") and msg.content:
            print(f"  [{node_name}] AI 回答：{msg.content[:120]}")

print("\n>>> stream_mode='updates' 按节点显示输出，可以看到每一步的决策过程。")
print("    stream_mode='messages' 则按消息粒度显示。")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("第 8 课 总结 —— 流式输出与可观测性")
print("=" * 60)
print("""
流式模式：

  chain.stream(input)
      返回：chain 最后一个步骤的逐 chunk 输出
      适用：想要最终结果逐字显示

  agent.stream(input, stream_mode="updates")
      返回：{节点名: {messages: [...]}} 按步骤输出
      适用：想看到 agent 每一步的推理过程

  agent.stream(input, stream_mode="messages")
      返回：跨所有节点的逐条消息
      适用：需要聊天式的消息流

Callback 生命周期：
  on_llm_start  → LLM 调用开始（记录 prompt，记录开始时间）
  on_llm_end    → LLM 调用结束（记录 token 用量、延迟）
  on_llm_error  → LLM 调用失败（记录错误、触发告警）
  on_tool_start / on_tool_end / on_tool_error（工具调用相关）
  on_chain_start / on_chain_end（chain 调用相关）

LangSmith（可选）：
  LangChain 的商业可观测平台
  - 在 .env 中添加 LANGCHAIN_TRACING_V2=true
  - 在 smith.langchain.com 注册
  - 自动追踪所有 LLM 调用、工具调用、chain 运行
  - 个人开发者有免费额度

和后面课程的关联：
  第 9 课 Agent+RAG：流式输出 RAG agent
  第 10-11 课 LangGraph：LangGraph 有自己的流式 API
  第 12 课 项目：Streamlit UI 中实现 token 级流式输出
""")


# ============================================================
# 练习
# ============================================================
print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 写一个 TimingCallback，统计多次调用的平均响应时间
2. 试试 agent.stream() 的 stream_mode="messages"，和 "updates" 对比差异
3. 给 callback 加错误处理——LLM 返回错误时会触发什么？
""")
