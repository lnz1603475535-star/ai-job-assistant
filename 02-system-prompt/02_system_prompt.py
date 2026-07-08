"""
第 2 课：System Prompt —— 给 AI 一个身份

核心概念：
  - SystemMessage: 给 AI 设定角色、行为规则（AI 看不到，但会遵循）
  - HumanMessage:  用户说的话（你和 AI 对话的内容）

对比实验：
  ① 不带 system prompt → AI 是"通用助手"
  ② 带 system prompt → AI 变成"资深简历顾问"

自己动手改：把 system prompt 改成另一个角色，看 AI 怎么变。
"""

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============================================================
# 实验 ①：不带 system prompt（通用 AI）
# ============================================================
print("=" * 60)
print("实验①：不带 system prompt（AI 不知道自己的角色）")
print("=" * 60)

messages = [
    HumanMessage(content="帮我看一下这段简历写得怎么样：'我熟练使用 Office 办公软件，工作认真负责'")
]
response = llm.invoke(messages)
print(f"\nAI 回答：\n{response.content}\n")


# ============================================================
# 实验 ②：带 system prompt（角色扮演）
# ============================================================
print("=" * 60)
print("实验②：带 system prompt（告诉 AI 它是资深简历顾问）")
print("=" * 60)

system_prompt = """你是一个严厉的面试官，说话刻薄但一针见血。找出简历里所有吹牛和空洞的地方，毫不留情地指出来。"""

messages_with_system = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="帮我看一下这段简历写得怎么样：'我熟练使用 Office 办公软件，工作认真负责'")
]
response2 = llm.invoke(messages_with_system)
print(f"\nAI 回答：\n{response2.content}\n")


# ============================================================
# 给你的练习
# ============================================================
print("=" * 60)
print("试试看：把上面的 system_prompt 换成下面任意一个角色，重新运行")
print("=" * 60)
print("""
角色 A - 毒舌面试官：
"你是一个严厉的面试官，说话刻薄但一针见血。找出简历里所有吹牛和空洞的地方，毫不留情地指出来。"

角色 B - 幼儿园老师：
"你是一位幼儿园老师，要用最简单的话、打比方的方式给孩子解释什么是好简历。"

角色 C - 你自己的想法：
______（填入你想让 AI 扮演的角色）
""")
