"""
AI 简历生成器 - 引擎模块
=========================
5 个核心函数，每个都可以独立调用和测试。
从用户输入到生成定制简历的完整链路。
"""

import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from models import UserProfile, StyleProfile, JDRequirements
from prompts import (
    USER_INFO_PARSE_PROMPT,
    STYLE_EXTRACTION_PROMPT,
    JD_REQUIREMENTS_PROMPT,
    BASE_RESUME_PROMPT,
    JD_CUSTOMIZE_SYSTEM_PROMPT,
)
from core import llm, search_documents, load_file_content

logger = logging.getLogger(__name__)

# 缓存最近一次 Agent 调用的 token 用量
_last_agent_token_usage: dict = {}


# ============================================================
# 1. 用户信息提取
# ============================================================

def parse_user_info(text: str) -> UserProfile:
    """从自由文本中提取结构化用户画像。

    技术：LCEL 链（prompt → LLM → PydanticOutputParser），不需要 Agent 或 tool。
    模式：与第 3 课 Output Parsers 中 PydanticOutputParser 的用法一致。

    参数：
        text：用户自由输入的文本（口语描述自己的经历和技能）
    返回：
        UserProfile 结构化对象
    """
    parser = PydanticOutputParser(pydantic_object=UserProfile)
    prompt = USER_INFO_PARSE_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | llm | parser

    try:
        return chain.invoke({"user_text": text})
    except Exception:
        logger.exception("用户信息解析失败")
        return UserProfile(
            name="", contact="", skills=[], experience=[], education=""
        )


# ============================================================
# 2. 风格提取
# ============================================================

def extract_style(sample_path: str) -> StyleProfile:
    """分析样本简历的风格特征。

    技术：将样本简历全文加载到 prompt 中（不分块），通过 LCEL 链提取风格。
    不分块的原因：风格分析需要完整的文档上下文——章节顺序、整体语气、
    格式模式这些在切分后会丢失。

    参数：
        sample_path：样本简历文件路径
    返回：
        StyleProfile 结构化对象
    """
    parser = PydanticOutputParser(pydantic_object=StyleProfile)
    prompt = STYLE_EXTRACTION_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | llm | parser

    try:
        resume_text = load_file_content(sample_path)
        return chain.invoke({"resume_text": resume_text})
    except Exception:
        logger.exception("风格提取失败: %s", sample_path)
        return StyleProfile(
            structure="个人信息 → 技能 → 工作经历 → 教育",
            tone="简洁专业",
            format_patterns="动词开头，每段经历 3-4 条",
        )


# ============================================================
# 3. JD 要求提取
# ============================================================

def extract_jd_requirements(jd_path: str) -> JDRequirements:
    """从 JD 文件中提取结构化要求。

    技术：LCEL 链（prompt → LLM → PydanticOutputParser），模式同 parse_user_info。

    参数：
        jd_path：JD 文件路径
    返回：
        JDRequirements 结构化对象
    """
    parser = PydanticOutputParser(pydantic_object=JDRequirements)
    prompt = JD_REQUIREMENTS_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | llm | parser

    try:
        jd_text = load_file_content(jd_path)
        return chain.invoke({"jd_text": jd_text})
    except Exception:
        logger.exception("JD 要求提取失败: %s", jd_path)
        return JDRequirements(
            title="未知岗位", must_have=[], nice_to_have=[],
            keywords=[], hidden_preferences=""
        )


# ============================================================
# 4. 基础简历生成
# ============================================================

def generate_base_resume(user: UserProfile, style: StyleProfile) -> str:
    """根据用户画像和风格偏好生成基础简历。

    技术：直接 prompt → LLM 调用，不需要 Agent 或 tool。
    生成型任务不需要工具调用——LLM 只需要根据给定的信息生成文本。

    参数：
        user：UserProfile 结构化用户画像
        style：StyleProfile 风格偏好
    返回：
        Markdown 格式的简历文本
    """
    chain = BASE_RESUME_PROMPT | llm
    try:
        result = chain.invoke({
            "user_profile": user.model_dump_json(indent=2, ensure_ascii=False),
            "style_profile": style.model_dump_json(indent=2, ensure_ascii=False),
        })
        return result.content
    except Exception:
        logger.exception("基础简历生成失败")
        return ""


# ============================================================
# 5. JD 定制优化
# ============================================================

def customize_for_jd(base_resume: str, jd_reqs: JDRequirements) -> str:
    """根据 JD 要求定制简历。返回定制后的 Markdown 文本。

    参数：
        base_resume：generate_base_resume 生成的基础简历（Markdown）
        jd_reqs：extract_jd_requirements 提取的 JD 结构化要求
    返回：
        定制后的 Markdown 格式简历
    """
    agent = create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt=JD_CUSTOMIZE_SYSTEM_PROMPT,
    )

    user_message = HumanMessage(content=(
        f"根据以下 JD 要求，优化这份简历：\n\n"
        f"=== JD 要求 ===\n"
        f"岗位：{jd_reqs.title}\n"
        f"必备要求：{', '.join(jd_reqs.must_have)}\n"
        f"加分项：{', '.join(jd_reqs.nice_to_have)}\n"
        f"关键词：{', '.join(jd_reqs.keywords)}\n"
        f"隐性偏好：{jd_reqs.hidden_preferences}\n\n"
        f"=== 简历原文 ===\n{base_resume}"
    ))

    try:
        result = agent.invoke({"messages": [user_message]})

        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]

        # 缓存本次 token 用量
        global _last_agent_token_usage
        _last_agent_token_usage = _extract_agent_token_usage(result["messages"])

        return ai_messages[-1].content if ai_messages else base_resume
    except Exception:
        logger.exception("JD 定制优化失败")
        return base_resume


def _extract_agent_token_usage(messages: list) -> dict:
    """从 Agent 消息中提取 API 原始 token 用量（字段名归一化交给 TokenBudget）。"""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "response_metadata", {}) or {}
        tu = meta.get("token_usage", {}) or meta.get("usage", {})
        if tu:
            usage["prompt_tokens"] += (
                tu["prompt_tokens"] if "prompt_tokens" in tu
                else tu.get("input_tokens", 0)
            )
            usage["completion_tokens"] += (
                tu["completion_tokens"] if "completion_tokens" in tu
                else tu.get("output_tokens", 0)
            )
    return usage


def get_last_token_usage() -> dict:
    """返回最近一次 customize_for_jd 调用的真实 token 用量。"""
    return _last_agent_token_usage
