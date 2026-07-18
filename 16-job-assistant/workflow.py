"""
AI 简历生成器 — LangGraph 工作流
=================================
7 节点线性工作流：一函数一节点。
支持 MemorySaver 断点恢复。
"""

import operator
import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from models import UserProfile, StyleProfile, JDRequirements
from resume_engine import (
    parse_user_info,
    extract_style,
    extract_jd_requirements,
    generate_base_resume,
    customize_for_jd,
    get_last_token_usage,
    is_customize_failed,
)


# MemorySaver 单例——断点恢复依赖同一实例跨请求保持状态
_checkpointer: MemorySaver | None = None

# 编译后的工作流单例——同一实例跨请求保持 checkpoint 状态
_compiled_graph: CompiledStateGraph | None = None


def get_checkpointer() -> MemorySaver:
    """获取 MemorySaver 单例（首次调用时创建）。"""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


# ============================================================
# State
# ============================================================

class WorkflowState(TypedDict, total=False):
    """工作流状态。一函数一节点的中间产物。"""

    user_text: str
    sample_resume_path: str
    jd_path: str
    style_profile: StyleProfile
    user_profile: UserProfile
    jd_requirements: JDRequirements
    base_resume: str
    customized_resume: str
    token_usage: dict  # TODO: 后端阶段接入 TokenBudget 进行成本控制
    errors: list[str]  # 不用 operator.add：每次 node_validate_inputs 返回全新错误列表，覆盖旧值
    notifications: Annotated[list[str], operator.add]  # 用 operator.add：多节点各自追加，不覆盖


# ============================================================
# 节点
# ============================================================

def node_validate_inputs(state: WorkflowState) -> dict[str, object]:
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


def node_extract_style(state: WorkflowState) -> dict[str, object]:
    """节点 2：提取样本简历风格。"""
    result = extract_style(state["sample_resume_path"])
    return {"style_profile": result}


def node_parse_user(state: WorkflowState) -> dict[str, object]:
    """节点 3：解析用户信息。"""
    result = parse_user_info(state["user_text"])
    return {"user_profile": result}


def node_extract_jd(state: WorkflowState) -> dict[str, object]:
    """节点 4：提取 JD 要求。"""
    result = extract_jd_requirements(state["jd_path"])
    return {"jd_requirements": result}


def node_generate_base(state: WorkflowState) -> dict[str, object]:
    """节点 5：生成基础简历。"""
    result = generate_base_resume(
        state["user_profile"],
        state["style_profile"],
    )
    return {"base_resume": result}


def node_check_parsed(state: WorkflowState) -> dict[str, object]:
    """节点 6：检查解析结果是否为空/默认值，追加提醒但不中断流程。"""
    notifications = []

    style = state.get("style_profile")
    if style and style.is_fallback:
        notifications.append("⚠️ 风格提取失败，已使用默认风格（非样本风格），简历排版可能与预期不同")

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


def node_customize(state: WorkflowState) -> dict[str, object]:
    """节点 7：JD 定制优化。"""
    result = customize_for_jd(
        state["base_resume"],
        state["jd_requirements"],
        notifications=state.get("notifications"),
    )
    notifications = []
    if is_customize_failed():
        notifications.append("⚠️ JD 定制优化失败，已使用基础简历代替")
    return {
        "customized_resume": result,
        "token_usage": get_last_token_usage(),
        "notifications": notifications,
    }


# ============================================================
# 路由
# ============================================================

def router_after_validate(state: WorkflowState) -> str:
    """validate_inputs 后的条件路由：有错误直接结束，否则继续。"""
    if state.get("errors"):
        return END
    return "extract_style"


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

    graph = StateGraph(WorkflowState)

    # 添加节点
    graph.add_node("validate_inputs", node_validate_inputs)
    graph.add_node("extract_style", node_extract_style)
    graph.add_node("parse_user", node_parse_user)
    graph.add_node("extract_jd", node_extract_jd)
    graph.add_node("generate_base", node_generate_base)
    graph.add_node("check_parsed", node_check_parsed)
    graph.add_node("customize", node_customize)

    # 连接边
    graph.add_edge(START, "validate_inputs")
    graph.add_conditional_edges("validate_inputs", router_after_validate, {
        "extract_style": "extract_style",
        END: END,
    })
    graph.add_edge("extract_style", "parse_user")
    graph.add_edge("parse_user", "extract_jd")
    graph.add_edge("extract_jd", "generate_base")
    graph.add_edge("generate_base", "check_parsed")
    graph.add_edge("check_parsed", "customize")
    graph.add_edge("customize", END)

    # 编译：单例 checkpointer + generate_base 后暂停（人工审核断点）
    _compiled_graph = graph.compile(
        checkpointer=get_checkpointer(),
        interrupt_after=["check_parsed"],
    )
    return _compiled_graph


# ============================================================
# 便利函数
# ============================================================

def run_workflow(
    user_text: str,
    sample_resume_path: str,
    jd_path: str,
    thread_id: str = "default",
) -> dict[str, object]:
    """运行工作流，在检查解析结果后暂停（interrupt_after=["check_parsed"]）。

    返回的 state 包含 base_resume 但不含 customized_resume。
    用户审核基础简历后，调用 resume_workflow(thread_id) 继续执行 JD 定制。

    参数：
        user_text：用户口述文本
        sample_resume_path：样本简历文件路径
        jd_path：JD 文件路径
        thread_id：线程 ID（同一 thread_id 可跨调用保持状态）

    返回：
        暂停时的 WorkflowState 字典（含 base_resume，不含 customized_resume）
        如果 validate_inputs 发现错误，直接返回错误 state（不暂停）
    """
    app = build_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_text": user_text,
        "sample_resume_path": sample_resume_path,
        "jd_path": jd_path,
        "errors": [],
        "notifications": [],
        "base_resume": "",
        "customized_resume": "",
    }

    return app.invoke(initial_state, config)


def resume_workflow(thread_id: str = "default") -> dict[str, object]:
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
    config = {"configurable": {"thread_id": thread_id}}

    # 先查有没有 checkpoint，没有就直接报错，不盲调 invoke
    current_state = app.get_state(config)
    if current_state is None or not current_state.values:
        raise RuntimeError(
            f"无法恢复工作流：thread_id={thread_id} 没有已保存的 checkpoint。"
            "请先调用 run_workflow() 启动工作流。"
        )

    # 有 checkpoint → 正常恢复，任何异常直接透传
    return app.invoke(None, config)
