# WorkflowState 是 TypedDict(total=False)（LangGraph 惯例：节点只写部分键），
# 直取键被误报为"可能不存在"；键存在性由图执行顺序保证。部署前补标注时重开。
# pyright: reportTypedDictNotRequiredAccess=false

"""
AI 简历生成器 — LangGraph 工作流
=================================
7 节点线性工作流：一函数一节点。
支持 MemorySaver 断点恢复。
"""

import functools
import logging
import operator
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

logger = logging.getLogger(__name__)

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core import TokenBudget, search_documents_impl
from models import JDRequirements, StyleProfile, UserProfile
from resume_engine import (
    customize_for_jd,
    extract_jd_requirements,
    extract_style,
    generate_base_resume,
    parse_user_info,
)

# MemorySaver 单例——断点恢复依赖同一实例跨请求保持状态
_checkpointer: MemorySaver | None = None

# 编译后的工作流单例——同一实例跨请求保持 checkpoint 状态
_compiled_graph: CompiledStateGraph | None = None

# 惰性单例的初始化锁（并发首次调用会各建一份图/checkpointer）
_init_lock = threading.Lock()

# 已登记 thread 的 LRU（队首最旧）。MemorySaver 为每个 thread_id 永久保留
# 全部 checkpoint——长跑进程不清理必然 OOM。超出上限时淘汰最旧的 thread
# 并释放其 checkpoint。第 14 课换 PostgresSaver 后由数据库侧管理生命周期。
# 注：多用户部署时需按并发会话数调大，或改为带 TTL 的存储。
_MAX_TRACKED_THREADS = 50
_tracked_threads: OrderedDict[str, None] = OrderedDict()
_threads_lock = threading.Lock()


def get_checkpointer() -> MemorySaver:
    """获取 MemorySaver 单例（首次调用时创建）。"""
    global _checkpointer
    if _checkpointer is None:
        with _init_lock:
            if _checkpointer is None:
                _checkpointer = MemorySaver()
    return _checkpointer


def register_thread(thread_id: str) -> None:
    """登记 thread_id；超出上限时淘汰最旧的 thread 及其 checkpoint。"""
    with _threads_lock:
        _tracked_threads[thread_id] = None
        _tracked_threads.move_to_end(thread_id)
        # thread_id 已在队尾，淘汰队首即可，不会误删本次要用的 thread
        while len(_tracked_threads) > _MAX_TRACKED_THREADS:
            oldest, _ = _tracked_threads.popitem(last=False)
            try:
                get_checkpointer().delete_thread(oldest)
            except Exception:
                logger.warning("淘汰 thread checkpoint 失败：%s", oldest, exc_info=True)
            else:
                logger.info("已淘汰最旧 thread 的 checkpoint：%s", oldest)


# 经验库全文超过该字符数时启用按需检索（小于阈值直接喂全文，检索纯属多余）
_EXPERIENCE_FULL_TEXT_THRESHOLD = 5000
# 按需检索的候选块数（约 500 字符/块 ≈ 5000 字符输入，token 恒定不随库增长）
_EXPERIENCE_RETRIEVE_K = 10
# 经验库头部区（基本信息/技能）的保留上限——按行截断，不切碎字段
_EXPERIENCE_HEAD_LINES = 25
_EXPERIENCE_HEAD_CHARS = 800


# ============================================================
# State
# ============================================================


class WorkflowState(TypedDict, total=False):
    """工作流状态。一函数一节点的中间产物。"""

    user_text: str
    user_supplement: str  # Step 3 可选的用户补充（侧重点/岗位看法等）
    sample_resume_path: str
    jd_path: str
    style_profile: StyleProfile
    user_profile: UserProfile
    jd_requirements: JDRequirements
    base_resume: str
    customized_resume: str
    token_usage: (
        dict  # customize 阶段 token 用量（TokenBudget 在 node_customize 按请求实例化）
    )
    errors: list[
        str
    ]  # 不用 operator.add：每次 node_validate_inputs 返回全新错误列表，覆盖旧值
    notifications: Annotated[
        list[str], operator.add
    ]  # 用 operator.add：多节点各自追加，不覆盖


# ============================================================
# 节点
# ============================================================


def node_validate_inputs(state: WorkflowState) -> dict[str, Any]:
    """节点 1：验证输入。"""
    errors = []

    if not state.get("user_text", "").strip():
        errors.append("缺少用户口述信息（user_text）")
    if not state.get("sample_resume_path", "").strip():
        errors.append("缺少样本简历路径（sample_resume_path）")
    if not state.get("jd_path", "").strip():
        errors.append("缺少 JD 路径（jd_path）")

    for key, label in [("sample_resume_path", "样本简历"), ("jd_path", "JD")]:
        path = state.get(key, "")
        if path and not os.path.exists(path):
            errors.append(f"{label}文件不存在：{path}")

    return {"errors": errors}


def node_extract_style(state: WorkflowState) -> dict[str, Any]:
    """节点 2：提取样本简历风格。"""
    result = extract_style(state["sample_resume_path"])
    return {"style_profile": result}


def retrieve_experience_for_jd(jd: JDRequirements) -> str:
    """按 JD 关键词从经验库检索相关经历段落（复用 search_documents 的 RRF 融合检索）。

    只检索 user_experience 类型，避免串入样本简历/JD 内容。
    返回空字符串表示检索不可用或无结果——调用方回退全文。
    """
    if not jd or not jd.keywords:
        return ""
    query = " ".join(jd.keywords[:10]).strip()
    if not query:
        return ""
    try:
        result = search_documents_impl(
            query, k=_EXPERIENCE_RETRIEVE_K, doc_types=["user_experience"]
        )
    except Exception:
        logger.exception("按需检索经验库失败，回退全文")
        return ""
    # 正常结果以 [doc_type] 开头（代码保证的格式）；哨兵文案（"未找到相关文档。"、
    # "尚未加载任何文档。"、"关键词索引不可用" 等）都不以 [ 开头——
    # 按格式判断而非枚举文案，未来新增哨兵也不会漏
    if not result or not result.startswith("["):
        return ""
    return result


def _experience_head(user_text: str) -> str:
    """截取经验库头部区（姓名/联系方式/学历/技能所在段），按行截断不切碎字段。

    头部区按约定置于经验库开头，与 USER_INFO_PARSE_PROMPT 的
    "姓名从文本开头提取"一致。
    """
    head_lines: list[str] = []
    used = 0
    for line in user_text.splitlines()[:_EXPERIENCE_HEAD_LINES]:
        if used + len(line) > _EXPERIENCE_HEAD_CHARS:
            break
        head_lines.append(line)
        used += len(line) + 1
    return "\n".join(head_lines).strip()


def node_parse_user(state: WorkflowState) -> dict[str, Any]:
    """节点 4：解析用户信息。经验库较大时按 JD 关键词按需检索，只提取相关经历。

    ⚠️ 检索结果只能替换"经历"部分，不能替换全文：检索用 JD 关键词做 query，
    而姓名/邮箱/学历所在的基本信息段与 JD 关键词没有任何关键词交集，检索不到。
    早期版本直接 context = retrieved，经验库一旦超过 5000 字符，UserProfile 的
    name/contact/education 就会静默变空，简历丢掉姓名和学历。
    故这里始终保留头部区（基本信息/技能），只对经历部分做按需检索。
    """
    user_text = state.get("user_text", "")
    jd = state.get("jd_requirements")

    context = user_text
    # 经验库超过阈值 → 按需检索（token 恒定 + 提取更精准）；否则直接用全文
    if jd and len(user_text) > _EXPERIENCE_FULL_TEXT_THRESHOLD:
        retrieved = retrieve_experience_for_jd(jd)
        if retrieved:
            head = _experience_head(user_text)
            context = (
                f"{head}\n\n"
                f"=== 与岗位相关的经历（已按 JD 关键词检索） ===\n{retrieved}"
            )
            logger.info(
                "按需检索：经验库 %d 字符 → 头部区 %d 字符 + 检索上下文 %d 字符",
                len(user_text),
                len(head),
                len(retrieved),
            )

    result = parse_user_info(context)
    return {"user_profile": result}


def node_extract_jd(state: WorkflowState) -> dict[str, Any]:
    """节点 3：提取 JD 要求。"""
    result = extract_jd_requirements(state["jd_path"])
    return {"jd_requirements": result}


def node_generate_base(state: WorkflowState) -> dict[str, Any]:
    """节点 5：生成基础简历。"""
    result = generate_base_resume(
        state["user_profile"],
        state["style_profile"],
    )
    return {"base_resume": result}


def node_check_parsed(state: WorkflowState) -> dict[str, Any]:
    """节点 6：检查解析结果是否为空/默认值，追加提醒但不中断流程。"""
    notifications = []

    style = state.get("style_profile")
    if style and style.is_fallback:
        notifications.append(
            "⚠️ 风格提取失败，已使用默认风格（非样本风格），简历排版可能与预期不同"
        )

    user = state.get("user_profile")
    if user:
        if not user.name:
            notifications.append("⚠️ 未识别到姓名，简历可能不完整")
        if not user.skills:
            notifications.append("⚠️ 未识别到技能，简历可能缺少技术栈")

    jd = state.get("jd_requirements")
    if jd:
        if not jd.title or jd.title == "未知岗位":
            notifications.append("⚠️ JD 解析不完整，定制可能不够精准")
        if not jd.keywords:
            notifications.append("⚠️ 未提取到 JD 关键词，简历定制效果有限")

    base = state.get("base_resume", "")
    if not base.strip():
        notifications.append("⚠️ 基础简历生成为空，请检查输入信息是否完整")

    return {"notifications": notifications}


def node_customize(state: WorkflowState) -> dict[str, Any]:
    """节点 7：JD 定制优化。"""
    # 短路保护：基础简历为空时不调用 LLM（省一次无效调用），直接标记跳过
    base_resume = state.get("base_resume", "")
    if not base_resume.strip():
        logger.warning("基础简历为空，跳过 JD 定制")
        return {
            "customized_resume": "",
            "token_usage": {},
            "notifications": [
                "⚠️ 基础简历为空，已跳过 JD 定制。请检查输入信息后重新生成。"
            ],
        }

    customized, token_usage, failed = customize_for_jd(
        base_resume,
        state["jd_requirements"],
        notifications=state.get("notifications"),
        user_supplement=state.get("user_supplement", ""),
    )

    notifications = []
    if failed:
        notifications.append("⚠️ JD 定制优化失败，已使用基础简历代替")

    # 用量提示给用户看。预算对象由最终用量构造——面向模型的那份提醒
    # 已由 TokenBudgetMiddleware 在生成过程中注入（过程控制在此处完成），
    # 这里只是事后向用户展示同一份数据
    budget = TokenBudget.from_usage(token_usage)
    notice = budget.get_status_notice()
    if notice:
        notifications.append(notice)
    logger.info("customize token 用量：%s", budget.get_usage_report())

    return {
        "customized_resume": customized,
        "token_usage": token_usage,
        "notifications": notifications,
    }


# ============================================================
# 路由
# ============================================================


def router_after_validate(state: WorkflowState) -> str:
    """validate_inputs 后的条件路由：有错误直接结束，否则继续。

    触发频率统计（metrics 采集）：错误路由带 [路由] 标记打 WARNING，
    正常路由打 DEBUG——grep app.log 即可统计两种路由的比例。
    """
    if state.get("errors"):
        logger.warning(
            "[路由] 输入验证失败，错误路由触发：%d 条错误", len(state["errors"])
        )
        return END
    logger.debug("[路由] 输入验证通过，正常路由")
    return "extract_style"


def _timed_node(node_fn: Callable) -> Callable:
    """包装节点函数：记录单节点耗时（metrics 采集，编译前包装覆盖所有调用路径）。

    图内节点名取 add_node 的 key，不受包装影响；日志带 [节点耗时] 标记便于 grep。
    """

    @functools.wraps(node_fn)
    def wrapper(state: WorkflowState) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return node_fn(state)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("[节点耗时] %s = %.1fs", node_fn.__name__, elapsed)

    return wrapper


# ============================================================
# 构建工作流
# ============================================================


def build_workflow() -> CompiledStateGraph:
    """构建并编译 LangGraph 工作流（单例模式）。

    首次调用时构建图并编译；后续调用直接返回已编译的实例。
    编译时启用 interrupt_after=["check_parsed"]，工作流在检查解析结果后暂停，
    用户可以看到基础简历 + 所有提醒（姓名/技能/JD/风格是否异常），再决定继续定制。
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    # 先取 checkpointer（它内部有同一把 _init_lock），再进临界区：
    # threading.Lock 不可重入，嵌套获取会直接死锁（曾导致整个工作流卡死）
    checkpointer = get_checkpointer()
    with _init_lock:
        if _compiled_graph is not None:  # 等锁期间可能已被别的线程编译好
            return _compiled_graph
        _compiled_graph = _compile_workflow(checkpointer)
    return _compiled_graph


def _compile_workflow(checkpointer: MemorySaver) -> CompiledStateGraph:
    """构建并编译工作流图（调用方负责单例与加锁，checkpointer 由外部传入）。"""
    graph = StateGraph(WorkflowState)

    # 添加节点（_timed_node 包装：metrics 采集单节点耗时，图内名称不变）
    graph.add_node("validate_inputs", _timed_node(node_validate_inputs))
    graph.add_node("extract_style", _timed_node(node_extract_style))
    graph.add_node("parse_user", _timed_node(node_parse_user))
    graph.add_node("extract_jd", _timed_node(node_extract_jd))
    graph.add_node("generate_base", _timed_node(node_generate_base))
    graph.add_node("check_parsed", _timed_node(node_check_parsed))
    graph.add_node("customize", _timed_node(node_customize))

    # 连接边
    graph.add_edge(START, "validate_inputs")
    graph.add_conditional_edges(
        "validate_inputs",
        router_after_validate,
        {
            "extract_style": "extract_style",
            END: END,
        },
    )
    # extract_jd 在 parse_user 之前：parse_user 需要 JD 关键词做按需检索
    graph.add_edge("extract_style", "extract_jd")
    graph.add_edge("extract_jd", "parse_user")
    graph.add_edge("parse_user", "generate_base")
    graph.add_edge("generate_base", "check_parsed")
    graph.add_edge("check_parsed", "customize")
    graph.add_edge("customize", END)

    # 编译：单例 checkpointer + check_parsed 后暂停（人工审核断点）
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["check_parsed"],
    )


# ============================================================
# 便利函数
# ============================================================


def run_workflow(
    user_text: str,
    sample_resume_path: str,
    jd_path: str,
    thread_id: str = "default",
    user_supplement: str = "",
) -> dict[str, Any]:
    """运行工作流，在检查解析结果后暂停（interrupt_after=["check_parsed"]）。

    返回的 state 包含 base_resume 但不含 customized_resume。
    用户审核基础简历后，调用 resume_workflow(thread_id) 继续执行 JD 定制。

    参数：
        user_text：用户信息文本（从经验库读取）
        sample_resume_path：样本简历文件路径
        jd_path：JD 文件路径
        thread_id：线程 ID（同一 thread_id 可跨调用保持状态）
        user_supplement：可选的用户补充信息（侧重点/岗位看法等）

    返回：
        暂停时的 WorkflowState 字典（含 base_resume，不含 customized_resume）
        如果 validate_inputs 发现错误，直接返回错误 state（不暂停）
    """
    app = build_workflow()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # 登记 thread 并淘汰超限的旧 checkpoint（MemorySaver 永久保留，需主动回收）
    register_thread(thread_id)

    initial_state = {
        "user_text": user_text,
        "user_supplement": user_supplement,
        "sample_resume_path": sample_resume_path,
        "jd_path": jd_path,
        "errors": [],
        "notifications": [],
        "base_resume": "",
        "customized_resume": "",
    }

    result = app.invoke(initial_state, config)
    # 标记是否暂停在断点
    state = app.get_state(config)
    result["_interrupted"] = bool(state.next) if state else False
    return result


def resume_workflow(thread_id: str = "default") -> dict[str, Any]:
    """从 check_parsed 后的断点恢复执行，继续运行 customize。

    使用同一个 thread_id 以匹配 checkpoint。

    参数：
        thread_id：与 run_workflow() 相同的线程 ID

    返回：
        最终 WorkflowState 字典（包含 customized_resume）

    异常：
        RuntimeError：当前 thread_id 没有已保存的 checkpoint
    """
    app = build_workflow()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 先查有没有 checkpoint，没有就直接报错，不盲调 invoke
    current_state = app.get_state(config)
    if current_state is None or not current_state.values:
        raise RuntimeError(
            f"无法恢复工作流：thread_id={thread_id} 没有已保存的 checkpoint。"
            "请先调用 run_workflow() 启动工作流。"
        )

    # 有 checkpoint → 正常恢复，任何异常直接透传
    return app.invoke(None, config)
