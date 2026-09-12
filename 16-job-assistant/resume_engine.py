"""
AI 简历生成器 - 引擎模块
=========================
5 个核心函数，每个都可以独立调用和测试。
从用户输入到生成定制简历的完整链路。
"""

import logging
import re
import threading
import time
from collections.abc import Iterator
from typing import Any, NotRequired, cast

import httpx
import openai
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from core import (
    DEFAULT_TOKEN_BUDGET,
    TokenBudget,
    llm,
    load_file_content,
    search_documents,
)
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
        content = result.content
        return content if isinstance(content, str) else ""
    except Exception:
        logger.exception("基础简历生成失败")
        return ""


# ============================================================
# 5. JD 定制优化
# ============================================================


class TokenBudgetState(AgentState):
    """Agent 图 state 扩展：携带累计 token 用量与已提醒级别。

    预算状态放在图 state 而非中间件实例上——中间件才能被多个请求
    安全共用（Agent 单例缓存），每个请求的用量互不串数据。
    """

    token_input: NotRequired[int]
    token_output: NotRequired[int]
    token_warned_level: NotRequired[int]


class TokenBudgetMiddleware(AgentMiddleware):
    """把 TokenBudget 接进 Agent 循环：每轮模型调用前注入提醒、调用后记账。

    这是"过程控制"而非"事后统计"：提醒作为消息进入对话，模型能看到
    预算将尽并据此收敛输出；用量逐轮累积，工具调用轮次也不会漏算。
    """

    state_schema = TokenBudgetState

    def __init__(
        self,
        max_tokens: int = DEFAULT_TOKEN_BUDGET,
        warning_ratio: float = 0.7,
    ):
        super().__init__()
        self._max_tokens = max_tokens
        self._warning_ratio = warning_ratio

    def _budget_from_state(self, state: Any) -> TokenBudget:
        """从图 state 重建预算对象（含累计用量与已提醒级别）。"""
        budget = TokenBudget(self._max_tokens, self._warning_ratio)
        budget.restore(
            input_tokens=state.get("token_input", 0) or 0,
            output_tokens=state.get("token_output", 0) or 0,
            notified_level=state.get("token_warned_level", 0) or 0,
        )
        return budget

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """每轮模型调用后累积真实用量。"""
        messages = state.get("messages") or []
        if not messages:
            return None
        budget = self._budget_from_state(state)
        budget.record_from_message(messages[-1])
        return {
            "token_input": budget.input_tokens,
            "token_output": budget.output_tokens,
        }

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """每轮模型调用前注入预算提醒（每级只注入一次）。"""
        budget = self._budget_from_state(state)
        warning = budget.get_warning()
        if not warning:
            return None
        logger.info("Token 预算提醒已注入模型：%s", budget.get_usage_report())
        return {
            "messages": [HumanMessage(content=warning)],
            "token_warned_level": budget.notified_level,
        }


# Agent 单例（无状态，预算随图 state 走；每次请求重建纯属浪费）
_customize_agent: Any = None
_agent_lock = threading.Lock()


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
                m.content
                for m in result["messages"]
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content
            ]

            token_usage = _extract_agent_token_usage(result)

            if ai_messages:
                return ai_messages[-1], token_usage, False

            # 空 AIMessage → 不重试（系统性问题，重试大概率还是空）
            logger.warning("Agent 未输出有效 AIMessage，降级返回 base_resume")
            return base_resume, token_usage, True

        except Exception as e:
            if attempt < max_attempts - 1 and _is_transient_error(e):
                # 网络瞬时异常 → 退避后原策略重试。限流/超时立即重试大概率
                # 再撞一次，指数退避给服务端留出恢复窗口
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "网络异常（第 %d/%d 次）：%s，%.1fs 后重试",
                    attempt + 1,
                    max_attempts,
                    str(e)[:100],
                    delay,
                )
                time.sleep(delay)
            else:
                # Agent 执行异常 or 重试次数耗尽 → 直接放弃
                logger.exception(
                    "JD 定制优化失败（第 %d/%d 次），降级返回 base_resume",
                    attempt + 1,
                    max_attempts,
                )
                return base_resume, {}, True

    # 防御性兜底：循环内所有路径均已 return，此处实际不可达（满足静态检查）
    return base_resume, {}, True


def _build_customize_agent():
    """获取 JD 定制 Agent（create_agent v1.0，返回 langgraph 图，支持流式）。

    同步路径（customize_for_jd）与流式路径（stream_customize_for_jd）共用，
    保证两条路径的 Agent 配置一致。

    单例缓存：Agent 本身无状态（Token 预算存在图 state 里，见
    TokenBudgetMiddleware），每次请求重建纯属浪费。
    """
    global _customize_agent
    if _customize_agent is None:
        with _agent_lock:
            if _customize_agent is None:  # 等锁期间可能已被别的线程建好
                _customize_agent = create_agent(
                    model=llm,
                    tools=[search_documents],
                    system_prompt=JD_CUSTOMIZE_SYSTEM_PROMPT,
                    middleware=[TokenBudgetMiddleware()],
                    state_schema=TokenBudgetState,
                )
    return _customize_agent


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
) -> Iterator[tuple[str, Any]]:
    """流式 JD 定制：实时产出 token 片段。

    yield 事件：
        ("token", 文本块)  —— LLM 逐步生成的 token（拼起来即定制简历）
        ("done", 结果 dict) —— {"customized_resume", "token_usage", "failed"}

    与 customize_for_jd 的差异：
        - 无重试——流式输出中途重跑体验差，失败时由调用方捕获异常后降级
          （同步端点保留完整重试逻辑，两条路径各司其职）
        - stream_mode=["messages", "values"]：messages 产出 token 块，
          values 在结束时输出完整 state（含中间件累积的 token 用量）
        - 只推送最终答案的 token：Agent 调工具前可能先输出一段铺垫文字，
          那属于过程性内容、不在最终简历里，流式时直接丢弃，
          否则用户看到的流与最终落盘的简历对不上
    """
    agent = _build_customize_agent()
    user_message = _build_customize_messages(
        base_resume, jd_reqs, notifications, user_supplement
    )

    final_state: dict = {}
    try:
        # 组合模式输出 (mode, data)：mode="messages" 时 data=(chunk, metadata)；
        # mode="values" 时 data=完整 state（节点完成后输出，含 messages 与 token 用量）
        tool_call_msg_ids: set[str] = set()
        for mode, data in agent.stream(
            {"messages": [user_message]},
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                chunk, _metadata = data
                if not isinstance(chunk, AIMessageChunk):
                    continue
                msg_id = chunk.id or ""
                # 带 tool_call 的 chunk 属于工具调用轮次，不是最终答案
                if getattr(chunk, "tool_call_chunks", None):
                    tool_call_msg_ids.add(msg_id)
                    continue
                if msg_id in tool_call_msg_ids:
                    continue
                content = chunk.content
                if content:
                    yield ("token", content)
            elif mode == "values":
                final_state = cast(Any, data) or {}

        messages = final_state.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        yield (
            "done",
            {
                "customized_resume": ai_messages[-1].content
                if ai_messages
                else base_resume,
                "token_usage": _extract_agent_token_usage(final_state),
                "failed": not bool(ai_messages),
            },
        )
    except Exception:
        # 异常不在此降级（调用方需要区分"异常"与"空结果"），只记录日志
        logger.exception("流式 JD 定制失败")
        raise


# 瞬时错误重试的退避基数（秒）——指数退避：delay * 2^attempt
_RETRY_BASE_DELAY = 1.5

# 兜底关键词：异常类型无法识别时（被第三方包装、丢失类型信息）的最后一道判断。
# 数字状态码用词边界匹配，避免误伤错误文本中恰好含 429/503 的无关数字
_TRANSIENT_KEYWORDS = (
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
)
_TRANSIENT_STATUS_RE = re.compile(r"\b(429|500|502|503|504)\b")


def _is_transient_error(exc: Exception) -> bool:
    """判断异常是否为网络/服务端瞬时错误（值得原策略重试）。

    网络抖动、超时、限流、5xx 等是外部原因，与 Agent 执行能力无关，
    重试时不需要降级策略。其他异常（如 SDK 内部错误、响应解析失败）
    才需要降级。

    按可靠性从高到低判断：异常类型 → HTTP 状态码 → 关键词兜底。
    早期版本只做关键词字符串匹配，会把错误文本里恰好出现的 "429"
    等数字误判成限流（来源：_is_transient_error 原始实现）。
    """
    # 1) 类型判断（首选）：openai SDK 的异常层次是稳定的公开契约
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True
    # 2) httpx 网络层异常（SDK 未包装时直接抛出）
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    # 3) 带 HTTP 状态码的异常：5xx / 429 视为瞬时
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status == 429):
        return True
    # 4) 兜底：类型信息丢失时退回关键词匹配
    error_str = str(exc).lower()
    return any(kw in error_str for kw in _TRANSIENT_KEYWORDS) or bool(
        _TRANSIENT_STATUS_RE.search(error_str)
    )


def _extract_agent_token_usage(state: dict) -> dict:
    """从 Agent 最终 state 中取出累计 token 用量。

    由 TokenBudgetMiddleware 在每轮模型调用后累积写入 state，因此包含
    工具调用轮次的消耗（早期版本只取最后一条 AIMessage 的 usage，
    Agent 调过 search_documents 时成本会被系统性低估）。
    """
    return {
        "prompt_tokens": int(state.get("token_input", 0) or 0),
        "completion_tokens": int(state.get("token_output", 0) or 0),
    }
