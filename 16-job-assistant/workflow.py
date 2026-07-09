"""
AI 简历生成器 — LangGraph 工作流
=================================
6 节点线性工作流：一函数一节点。
支持 MemorySaver 断点恢复。
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from models import UserProfile, StyleProfile, JDRequirements
from core import TokenBudget
from resume_engine import (
    parse_user_info,
    extract_style,
    extract_jd_requirements,
    generate_base_resume,
    customize_for_jd,
)


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
    errors: list[str]  # 不用 operator.add：每次 node_validate_inputs 返回全新错误列表，覆盖旧值


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


def node_customize(state: WorkflowState) -> dict[str, object]:
    """节点 6：JD 定制优化（带 Token 预算监控）。"""
    budget = TokenBudget()

    # 计入输入消耗
    base = state.get("base_resume", "")
    jd = state.get("jd_requirements")
    if jd:
        budget.add_usage(jd.model_dump_json(ensure_ascii=False))
    budget.add_usage(base)

    warning = budget.get_warning()
    result = customize_for_jd(
        state["base_resume"],
        state["jd_requirements"],
        token_warning=warning,
    )

    # 记录输出消耗
    budget.add_usage(result)

    return {"customized_resume": result}


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
    """构建并编译 LangGraph 工作流。"""
    graph = StateGraph(WorkflowState)

    # 添加节点
    graph.add_node("validate_inputs", node_validate_inputs)
    graph.add_node("extract_style", node_extract_style)
    graph.add_node("parse_user", node_parse_user)
    graph.add_node("extract_jd", node_extract_jd)
    graph.add_node("generate_base", node_generate_base)
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
    graph.add_edge("generate_base", "customize")
    graph.add_edge("customize", END)

    # 编译（带 checkpointer，支持断点恢复）
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 便利函数
# ============================================================

def run_workflow(
    user_text: str,
    sample_resume_path: str,
    jd_path: str,
    thread_id: str = "default",
) -> dict[str, object]:
    """运行完整工作流，返回最终 state。

    参数：
        user_text：用户口述文本
        sample_resume_path：样本简历文件路径
        jd_path：JD 文件路径
        thread_id：线程 ID（同一 thread_id 可跨调用保持状态）

    返回：
        最终 WorkflowState 字典
    """
    app = build_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_text": user_text,
        "sample_resume_path": sample_resume_path,
        "jd_path": jd_path,
        "errors": [],
        "base_resume": "",
        "customized_resume": "",
    }

    return app.invoke(initial_state, config)
