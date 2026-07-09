"""
AI 简历生成器 - 引擎模块
=========================
5 个核心函数，每个都可以独立调用和测试。
从用户输入到生成定制简历的完整链路。
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from models import UserProfile, StyleProfile, JDRequirements
from prompts import (
    USER_INFO_PARSE_PROMPT,
    STYLE_EXTRACTION_PROMPT,
    JD_REQUIREMENTS_PROMPT,
    BASE_RESUME_PROMPT,
    JD_CUSTOMIZE_PROMPT,
)
from core import llm, search_documents, load_file_content


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
        # Pydantic 解析失败时返回空画像
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
    resume_text = load_file_content(sample_path)

    parser = PydanticOutputParser(pydantic_object=StyleProfile)
    prompt = STYLE_EXTRACTION_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | llm | parser

    try:
        return chain.invoke({"resume_text": resume_text})
    except Exception:
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
    jd_text = load_file_content(jd_path)

    parser = PydanticOutputParser(pydantic_object=JDRequirements)
    prompt = JD_REQUIREMENTS_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | llm | parser

    try:
        return chain.invoke({"jd_text": jd_text})
    except Exception:
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
    result = chain.invoke({
        "user_profile": user.model_dump_json(indent=2, ensure_ascii=False),
        "style_profile": style.model_dump_json(indent=2, ensure_ascii=False),
    })
    return result.content


# ============================================================
# 5. JD 定制优化
# ============================================================

def customize_for_jd(base_resume: str, jd_reqs: JDRequirements,
                     token_warning: str = "") -> str:
    """根据 JD 要求定制简历。

    技术：使用 create_agent + search_documents tool。
    为什么需要 Agent？定制过程中可能需要从已索引的用户信息文档中
    检索更多细节来支持改写。例如 JD 要求"高并发经验"，Agent 可以
    搜索用户文档看看有没有相关经历可以展开描述。

    参数：
        base_resume：generate_base_resume 生成的基础简历（Markdown）
        jd_reqs：extract_jd_requirements 提取的 JD 结构化要求
        token_warning：Token 预算警告文本（空字符串表示预算充足）
    返回：
        定制后的 Markdown 格式简历
    """
    system_prompt = (
        (token_warning + "\n\n") if token_warning else ""
    ) + """你是一个简历优化专家。根据 JD 要求优化简历。

规则：
1. 使用 search_documents 查找用户经历中与 JD 相关的细节
2. 绝对不编造用户没有的经历或技能——严禁修改公司名称、时间、职位等事实信息
3. 将 JD 关键词自然地融入已有经历的描述中，不堆砌
4. 调整措辞使经历听起来更贴合 JD 要求
5. 输出格式与输入简历的 Markdown 结构保持一致
6. 输出优化后的完整简历（Markdown 格式），不要输出解释性文字"""

    agent = create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt=system_prompt,
    )

    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"根据以下 JD 要求，优化这份简历：\n\n"
            f"=== JD 要求 ===\n"
            f"岗位：{jd_reqs.title}\n"
            f"必备要求：{', '.join(jd_reqs.must_have)}\n"
            f"加分项：{', '.join(jd_reqs.nice_to_have)}\n"
            f"关键词：{', '.join(jd_reqs.keywords)}\n"
            f"隐性偏好：{jd_reqs.hidden_preferences}\n\n"
            f"=== 简历原文 ===\n{base_resume}"
        ))]
    })

    # 提取最终 AI 回复
    ai_messages = [
        m for m in result["messages"]
        if isinstance(m, AIMessage) and m.content
    ]
    return ai_messages[-1].content if ai_messages else base_resume
