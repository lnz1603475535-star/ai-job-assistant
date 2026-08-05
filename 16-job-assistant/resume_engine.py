"""
AI 简历生成器 - 引擎模块
=========================
5 个核心函数，每个都可以独立调用和测试。
从用户输入到生成定制简历的完整链路。
"""

import logging
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from core import TokenBudget, llm, load_file_content, search_documents
from models import JDRequirements, StyleProfile, UserProfile
from prompts import (
    BASE_RESUME_PROMPT,
    JD_CUSTOMIZE_SYSTEM_PROMPT,
    JD_REQUIREMENTS_PROMPT,
    STYLE_EXTRACTION_PROMPT,
    USER_INFO_PARSE_PROMPT,
)

logger = logging.getLogger(__name__)

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
        return UserProfile(name="", contact="", skills=[], experience=[], education="")


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
            title="未知岗位",
            must_have=[],
            nice_to_have=[],
            keywords=[],
            hidden_preferences="",
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
        result = chain.invoke(
            {
                "user_profile": user.model_dump_json(indent=2, ensure_ascii=False),
                "style_profile": style.model_dump_json(indent=2, ensure_ascii=False),
            }
        )
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
    user_supplement: str = "",
) -> tuple[str, dict, bool]:
    """根据 JD 要求定制简历。返回 (定制简历, token 用量, 是否降级)。

    内部最多 2 次尝试：第 1 次正常调用；若抛异常且为网络瞬时错误
    （超时/连接失败/限流/5xx），第 2 次以相同策略重试；若为非瞬时异常
    或空 AIMessage，直接降级返回 base_resume，不再重试。

    参数：
        base_resume：generate_base_resume 生成的基础简历（Markdown）
        jd_reqs：extract_jd_requirements 提取的 JD 结构化要求
        notifications：上游节点的提醒（解析失败/降级等），Agent 会据此调整策略
        user_supplement：用户对简历的补充指引（侧重点/弱化项/岗位看法），可选
    返回：
        (定制后的 Markdown 格式简历, token 用量 dict, 是否降级 bool)。
        token 用量只在调用成功时提取（重试失败的尝试无 API 响应对象，返回 {}）。
    """
    max_attempts = 2

    agent = _build_customize_agent()
    user_message = _build_customize_messages(
        base_resume, jd_reqs, notifications, user_supplement
    )

    for attempt in range(max_attempts):
        try:
            result = agent.invoke({"messages": [user_message]})

            ai_messages = [
                m for m in result["messages"] if isinstance(m, AIMessage) and m.content
            ]

            token_usage = _extract_agent_token_usage(result["messages"])

            if ai_messages:
                return ai_messages[-1].content, token_usage, False

            # 空 AIMessage → 不重试（系统性问题，重试大概率还是空）
            logger.warning("Agent 未输出有效 AIMessage，降级返回 base_resume")
            return base_resume, token_usage, True

        except Exception as e:
            if attempt < max_attempts - 1 and _is_transient_error(e):
                # 网络瞬时异常 → 正常重试，不改变策略
                logger.warning(
                    "网络异常（第 %d/%d 次）：%s，正常重试",
                    attempt + 1,
                    max_attempts,
                    str(e)[:100],
                )
                # continue 进入下一次循环
            else:
                # Agent 执行异常 or 重试次数耗尽 → 直接放弃
                logger.exception(
                    "JD 定制优化失败（第 %d/%d 次），降级返回 base_resume",
                    attempt + 1,
                    max_attempts,
                )
                return base_resume, {}, True


def _build_customize_agent():
    """构建 JD 定制 Agent（create_agent v1.0，返回 langgraph 图，支持流式）。

    同步路径（customize_for_jd）与流式路径（stream_customize_for_jd）共用，
    保证两条路径的 Agent 配置一致。
    """
    return create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt=JD_CUSTOMIZE_SYSTEM_PROMPT,
    )


def _build_customize_messages(
    base_resume: str,
    jd_reqs: JDRequirements,
    notifications: list[str] | None = None,
    user_supplement: str = "",
) -> HumanMessage:
    """组装定制请求消息（含系统提醒段落，Agent 会据此调整策略）。"""
    notes = list(notifications or [])
    notes_section = ""
    if notes:
        notes_section = (
            "=== 系统提醒（注意以下信息可能不完整，请据此调整优化策略） ===\n"
        )
        notes_section += "\n".join(notes) + "\n\n"

    return HumanMessage(
        content=(
            f"根据以下 JD 要求，优化这份简历：\n\n"
            f"{notes_section}"
            f"=== JD 要求 ===\n"
            f"岗位：{jd_reqs.title}\n"
            f"必备要求：{', '.join(jd_reqs.must_have)}\n"
            f"加分项：{', '.join(jd_reqs.nice_to_have)}\n"
            f"关键词：{', '.join(jd_reqs.keywords)}\n"
            f"隐性偏好：{jd_reqs.hidden_preferences}\n\n"
            + (
                f"=== 用户偏好 ===\n{user_supplement}\n\n"
                if user_supplement.strip()
                else ""
            )
            + f"=== 简历原文 ===\n{base_resume}"
        )
    )


def stream_customize_for_jd(
    base_resume: str,
    jd_reqs: JDRequirements,
    notifications: list[str] | None = None,
    user_supplement: str = "",
) -> Iterator[tuple[str, object]]:
    """流式 JD 定制：实时产出 token 片段。

    yield 事件：
        ("token", 文本块)  —— LLM 逐步生成的 token（拼起来即定制简历）
        ("done", 结果 dict) —— {"customized_resume", "token_usage", "failed"}

    与 customize_for_jd 的差异：
        - 无重试——流式输出中途重跑体验差，失败时由调用方捕获异常后降级
          （同步端点保留完整重试逻辑，两条路径各司其职）
        - stream_mode=["messages", "values"]：messages 产出 token 块，
          values 在结束时输出完整 state（含最后一条 AIMessage 的 token_usage）
    """
    agent = _build_customize_agent()
    user_message = _build_customize_messages(
        base_resume, jd_reqs, notifications, user_supplement
    )

    final_messages: list = []
    try:
        # 组合模式输出 (mode, data)：mode="messages" 时 data=(chunk, metadata)；
        # mode="values" 时 data=完整 state（节点完成后输出，含 messages 与 token_usage）
        for mode, data in agent.stream(
            {"messages": [user_message]},
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                chunk, _metadata = data
                if isinstance(chunk, AIMessageChunk):
                    content = chunk.content
                    if content:
                        yield ("token", content)
            elif mode == "values":
                final_messages = data.get("messages", [])

        ai_messages = [
            m for m in final_messages if isinstance(m, AIMessage) and m.content
        ]
        yield (
            "done",
            {
                "customized_resume": ai_messages[-1].content
                if ai_messages
                else base_resume,
                "token_usage": _extract_agent_token_usage(final_messages),
                "failed": not bool(ai_messages),
            },
        )
    except Exception:
        # 异常不在此降级（调用方需要区分"异常"与"空结果"），只记录日志
        logger.exception("流式 JD 定制失败")
        raise


def _is_transient_error(exc: Exception) -> bool:
    """判断异常是否为网络/服务端瞬时错误（值得原策略重试）。

    网络抖动、超时、限流、5xx 等是外部原因，与 Agent 执行能力无关，
    重试时不需要降级策略。其他异常（如 SDK 内部错误、响应解析失败）
    才需要降级重试。
    """
    error_str = str(exc).lower()
    transient_keywords = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "refused",
        "connection reset",
        "rate limit",
        "too many requests",
        "server error",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "503",
        "502",
        "504",
        "429",
    ]
    return any(kw in error_str for kw in transient_keywords)


def _extract_agent_token_usage(messages: list) -> dict:
    """从 Agent 消息中提取 API 原始 token 用量。

    从后往前取最后一条带 token 用量的 AIMessage（最终响应），
    避免中间 tool-call 消息的 usage 被重复累加。

    usage 来源优先级：
        1. response_metadata["token_usage"] / ["usage"]（同步 invoke 路径）
        2. message.usage_metadata 属性（流式聚合路径——chunk 合并后
           usage 只写在该属性，response_metadata 不透传，实测验证）
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "response_metadata", {}) or {}
        tu = meta.get("token_usage") if "token_usage" in meta else meta.get("usage")
        if not tu:
            tu = getattr(msg, "usage_metadata", None)
        if tu:
            input_t, output_t = TokenBudget._parse_usage(tu)
            return {"prompt_tokens": input_t, "completion_tokens": output_t}
    return {"prompt_tokens": 0, "completion_tokens": 0}
