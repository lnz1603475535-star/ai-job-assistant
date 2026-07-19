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
from core import llm, search_documents, load_file_content, TokenBudget

logger = logging.getLogger(__name__)

# 缓存最近一次 customize_for_jd 成功调用的 token 用量
# 每次进入函数时重置为 {}；全部失败则保持 {}
# 注意：重试时如果第 1 次异常（无 result 对象），其 token 用量无法提取，
# 仅计入第 2 次成功的用量。瞬时网络错误通常未实际扣费，误差可接受。
# TODO: 后端阶段拆除全局变量，token 用量改为通过 workflow state 传递
_last_agent_token_usage: dict = {}

# 标记最近一次 customize_for_jd 是否触发了降级（返回 base_resume）
_last_customize_failed = False


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
            is_fallback=True,
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

def customize_for_jd(
    base_resume: str,
    jd_reqs: JDRequirements,
    notifications: list[str] | None = None,
) -> str:
    """根据 JD 要求定制简历。返回定制后的 Markdown 文本。

    内部最多 2 次尝试：第 1 次正常调用；若抛异常且为网络瞬时错误
    （超时/连接失败/限流/5xx），第 2 次以相同策略重试；若为非瞬时异常
    或空 AIMessage，直接降级返回 base_resume，不再重试。

    参数：
        base_resume：generate_base_resume 生成的基础简历（Markdown）
        jd_reqs：extract_jd_requirements 提取的 JD 结构化要求
        notifications：上游节点的提醒（解析失败/降级等），Agent 会据此调整策略
    返回：
        定制后的 Markdown 格式简历
    """
    global _last_agent_token_usage, _last_customize_failed

    # 重置为本次调用的真实用量——如果重试全部失败，不应保留上次成功的旧数据
    _last_agent_token_usage = {}
    _last_customize_failed = False

    # 拷贝入参，避免修改调用方的列表（operator.add 依赖原列表不变）
    notes = list(notifications or [])
    max_attempts = 2

    agent = create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt=JD_CUSTOMIZE_SYSTEM_PROMPT,
    )

    # 构建系统提醒段落（Agent 会据此调整策略）
    notes_section = ""
    if notes:
        notes_section = "=== 系统提醒（注意以下信息可能不完整，请据此调整优化策略） ===\n"
        notes_section += "\n".join(notes) + "\n\n"

    user_message = HumanMessage(content=(
        f"根据以下 JD 要求，优化这份简历：\n\n"
        f"{notes_section}"
        f"=== JD 要求 ===\n"
        f"岗位：{jd_reqs.title}\n"
        f"必备要求：{', '.join(jd_reqs.must_have)}\n"
        f"加分项：{', '.join(jd_reqs.nice_to_have)}\n"
        f"关键词：{', '.join(jd_reqs.keywords)}\n"
        f"隐性偏好：{jd_reqs.hidden_preferences}\n\n"
        f"=== 简历原文 ===\n{base_resume}"
    ))

    for attempt in range(max_attempts):
        try:
            result = agent.invoke({"messages": [user_message]})

            ai_messages = [
                m for m in result["messages"]
                if isinstance(m, AIMessage) and m.content
            ]

            # 缓存本次 token 用量（记录最后一次成功调用的用量）
            _last_agent_token_usage = _extract_agent_token_usage(result["messages"])

            if ai_messages:
                _last_customize_failed = False
                return ai_messages[-1].content

            # 空 AIMessage → 不重试（系统性问题，重试大概率还是空）
            logger.warning("Agent 未输出有效 AIMessage，降级返回 base_resume")
            _last_customize_failed = True
            return base_resume

        except Exception as e:
            if attempt < max_attempts - 1 and _is_transient_error(e):
                # 网络瞬时异常 → 正常重试，不改变策略
                logger.warning(
                    "网络异常（第 %d/%d 次）：%s，正常重试",
                    attempt + 1, max_attempts, str(e)[:100],
                )
                # continue 进入下一次循环
            else:
                # Agent 执行异常 or 重试次数耗尽 → 直接放弃
                logger.exception(
                    "JD 定制优化失败（第 %d/%d 次），降级返回 base_resume",
                    attempt + 1, max_attempts,
                )
                _last_customize_failed = True
                return base_resume


def _is_transient_error(exc: Exception) -> bool:
    """判断异常是否为网络/服务端瞬时错误（值得原策略重试）。

    网络抖动、超时、限流、5xx 等是外部原因，与 Agent 执行能力无关，
    重试时不需要降级策略。其他异常（如 SDK 内部错误、响应解析失败）
    才需要降级重试。
    """
    error_str = str(exc).lower()
    transient_keywords = [
        "timeout", "timed out",
        "connection",
        "network", "refused", "connection reset",
        "rate limit", "too many requests",
        "server error", "internal server error",
        "service unavailable", "bad gateway", "gateway timeout",
        "503", "502", "504", "429",
    ]
    return any(kw in error_str for kw in transient_keywords)


def _extract_agent_token_usage(messages: list) -> dict:
    """从 Agent 消息中提取 API 原始 token 用量。

    从后往前取最后一条带 token_usage 的 AIMessage（最终响应），
    避免中间 tool-call 消息的 usage 被重复累加。
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "response_metadata", {}) or {}
        tu = meta.get("token_usage") if "token_usage" in meta else meta.get("usage", {})
        if tu:
            input_t, output_t = TokenBudget._parse_usage(tu)
            return {"prompt_tokens": input_t, "completion_tokens": output_t}
    return {"prompt_tokens": 0, "completion_tokens": 0}


def get_last_token_usage() -> dict:
    """返回最近一次 customize_for_jd 调用的真实 token 用量。

    TODO: 后端阶段接入 TokenBudget 后，此函数改为从 TokenBudget 实例读取累计值。
    """
    return _last_agent_token_usage


def is_customize_failed() -> bool:
    """返回最近一次 customize_for_jd 是否触发了降级（异常返回 base_resume）。"""
    return _last_customize_failed
