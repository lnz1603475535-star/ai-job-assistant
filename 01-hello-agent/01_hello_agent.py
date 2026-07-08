"""
第一步：Hello Agent —— 用 LangChain 调用 DeepSeek

这个脚本展示最基础的用法：
1. 加载 API Key
2. 创建一个 Chat 模型
3. 发送一条消息，获取回复

跑之前先把 .env 里的 DEEPSEEK_API_KEY 填上。
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# 自动查找并加载 .env 文件
load_dotenv(find_dotenv())

# 创建模型实例
# DeepSeek 兼容 OpenAI 的 API 格式，所以用 ChatOpenAI 就能直接调用
llm = ChatOpenAI(
    model="deepseek-chat",                     # DeepSeek 的模型名
    api_key=os.getenv("DEEPSEEK_API_KEY"),     # 从环境变量读取 key
    base_url=os.getenv("DEEPSEEK_BASE_URL"),   # DeepSeek 的 API 地址
)

# 发送第一条消息
response = llm.invoke("你好！请用一句话介绍一下你自己。")

print("AI 回复:", response.content)
