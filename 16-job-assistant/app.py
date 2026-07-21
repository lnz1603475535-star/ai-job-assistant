"""
AI 简历生成器 — Streamlit UI (Round 4)
=====================================
分步向导 + 简历预览 + 经验库编辑器。

技术栈：LangChain | LangGraph | DeepSeek | FAISS | BM25 | jieba | Streamlit
"""

import logging
import streamlit as st
import sys, os, tempfile, uuid

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from core import (
    load_and_index_documents,
    set_vectorstore,
    llm,
    load_file_content,
    setup_logging,
)
from workflow import run_workflow, resume_workflow
from prompts import EXPERIENCE_EXTRACTION_PROMPT
from models import validate_experience_markdown
from exporters import markdown_to_pdf_bytes, markdown_to_docx_bytes

# ============================================================
# 常量
# ============================================================

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXP_BANK_PATH = os.path.join(DATA_DIR, "experience_bank.md")

AVAILABLE_RESUMES = [
    {"label": "张三 - Python 后端", "path": os.path.join(SAMPLE_DIR, "resume_zhangsan.txt")},
    {"label": "李四 - 前端工程师", "path": os.path.join(SAMPLE_DIR, "resume_lisi.txt")},
]
AVAILABLE_JDS = [
    {"label": "高级 Python 后端工程师", "path": os.path.join(SAMPLE_DIR, "jd_python_senior.txt")},
    {"label": "高级前端工程师", "path": os.path.join(SAMPLE_DIR, "jd_frontend_senior.txt")},
]
WIZARD_STEPS = ["样本简历", "JD 要求", "补充信息", "生成预览", "下载导出"]


# ============================================================
# 辅助函数
# ============================================================

def load_experience_bank() -> str:
    """读取经验库文件内容，首次运行时自动创建。"""
    if not os.path.exists(EXP_BANK_PATH):
        default = "# 经验库\n\n在此粘贴你的项目经历，AI 生成简历时会参考这些内容。\n"
        try:
            os.makedirs(os.path.dirname(EXP_BANK_PATH), exist_ok=True)
            with open(EXP_BANK_PATH, "w", encoding="utf-8") as f:
                f.write(default)
        except OSError:
            st.warning("经验库文件创建失败，修改可能无法保存。")
        return default
    try:
        with open(EXP_BANK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content.strip() else "# 经验库\n\n在此粘贴你的项目经历。\n"
    except (UnicodeDecodeError, OSError, PermissionError) as e:
        st.error(f"经验库文件读取失败：{_sanitize_error(e)}，请手动检查或删除 data/experience_bank.md")
        return "# 经验库\n\n在此粘贴你的项目经历。\n"


def initialize_session_state():
    """初始化所有 session_state 变量。"""
    defaults = {
        "wizard_step": 1,
        "session_id": str(uuid.uuid4()),
        "resume_path": None,
        "resume_name": None,
        "jd_path": None,
        "jd_name": None,
        "docs_indexed": False,
        "user_text": "",
        "workflow_result": None,
        "workflow_paused": False,       # True 表示工作流停在 check_parsed 断点，等待审核
        "paused_result": None,          # 断点暂停时的中间 state（含 base_resume + notifications）
        "paused_lost": False,           # True 表示审核状态意外丢失，需提示用户
        "processing": False,
        "show_ai_extract": False,
        "ai_extract_result": None,
        "index_error": None,
        "index_error_detail": "",
        "export_base_pdf": None,
        "export_base_docx": None,
        "export_customized_pdf": None,
        "export_customized_docx": None,
        "user_photo_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "exp_bank_content" not in st.session_state:
        st.session_state.exp_bank_content = load_experience_bank()


def save_uploaded_file(uploaded_file, prefix: str = "upload") -> tuple[str, str | None]:
    """保存上传文件到临时目录，返回 (路径, 错误消息)。
    成功时路径有效、错误为 None；失败时路径为空、错误为用户可读的提示。
    支持 .txt / .pdf / .docx / .md 格式。
    """
    try:
        content = uploaded_file.getvalue()
    except Exception:
        return "", "无法读取文件内容，请重新上传。"

    if len(content) == 0:
        return "", "文件内容为空，请检查后重新上传。"

    suffix = os.path.splitext(uploaded_file.name)[1]
    if not suffix or suffix == ".":           # 无扩展名或只有点 → 默认 txt
        suffix = ".txt"
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(content)
    except (OSError, PermissionError):
        return "", "文件保存失败，请检查磁盘空间或临时目录权限后重试。"

    return path, None


def _sanitize_error(exc: Exception) -> str:
    """脱敏异常信息：替换用户目录路径，截断到 200 字符。"""
    msg = str(exc)
    home = os.path.expanduser("~")
    if home and home != "~":
        msg = msg.replace(home, "~")
    return msg[:200]


def _render_file_preview(state_key: str, label_prefix: str):
    """通用文件预览组件。state_key: 'resume' 或 'jd'"""
    path_key = f"{state_key}_path"
    name_key = f"{state_key}_name"
    path = st.session_state.get(path_key)
    if not path:
        return
    st.info(f"{label_prefix} 当前选择：**{st.session_state.get(name_key, '')}**")
    with st.expander(f"{label_prefix} 内容预览"):
        try:
            content = load_file_content(path)
            if not content.strip():
                st.warning("文件内容为空，请重新选择。")
                st.session_state[path_key] = None
                st.session_state[name_key] = None
            else:
                st.text(content[:600] + ("..." if len(content) > 600 else ""))
        except ValueError as e:
            st.error(str(e))
            st.session_state[path_key] = None
            st.session_state[name_key] = None
        except FileNotFoundError:
            st.warning("文件已被移动或删除，请重新选择。")
            st.session_state[path_key] = None
            st.session_state[name_key] = None
        except (OSError, PermissionError):
            st.error("无法读取文件，请检查文件权限后重试。")
            st.session_state[path_key] = None
            st.session_state[name_key] = None


def validate_current_step() -> bool:
    """检查当前步骤是否可以前进。"""
    step = st.session_state.wizard_step
    if step == 1:
        return st.session_state.resume_path is not None
    if step == 2:
        return st.session_state.jd_path is not None
    if step == 3:
        return True  # 可选步骤
    if step == 4:
        return st.session_state.workflow_result is not None
    return True


def _reset_workflow_state() -> None:
    """重置工作流相关状态，生成新 session_id 避免 checkpoint 冲突。"""
    st.session_state.workflow_result = None
    st.session_state.workflow_paused = False
    st.session_state.paused_result = None
    st.session_state.processing = False
    st.session_state.session_id = str(uuid.uuid4())
    # 清除导出缓存（重新生成简历后需要重新导出）
    st.session_state.export_base_pdf = None
    st.session_state.export_base_docx = None
    st.session_state.export_customized_pdf = None
    st.session_state.export_customized_docx = None
    # 注意：user_photo_path 不清除——用户不希望重新上传照片


def index_documents_if_needed():
    """如果还没索引，就索引文档。失败时给出分类提示并允许重试。"""
    if st.session_state.docs_indexed:
        return
    if not st.session_state.resume_path or not st.session_state.jd_path:
        return

    with st.spinner("📚 正在索引文档..."):
        try:
            vs, chunks = load_and_index_documents({
                "user_experience": [EXP_BANK_PATH],
                "sample_resume": [st.session_state.resume_path],
                "jd": [st.session_state.jd_path],
            })
            set_vectorstore(vs, chunks)
            st.session_state.docs_indexed = True
            st.session_state.index_error = None
            st.toast(f"✅ 已索引 {len(chunks)} 个文本块")
        except (UnicodeDecodeError, ValueError) as e:
            logger.error("文档索引失败（编码/格式错误）：%s", _sanitize_error(e))
            error_str = str(e).lower()
            if "不存在" in error_str or "无法访问" in error_str or "not found" in error_str:
                st.session_state.index_error = "missing"
            elif "加密" in error_str or "encrypt" in error_str or "密码" in error_str:
                st.session_state.index_error = "pdf_encrypted"
            elif "扫描" in error_str or "scan" in error_str:
                st.session_state.index_error = "pdf_scanned"
            elif "损坏" in error_str or "corrupt" in error_str:
                st.session_state.index_error = "pdf_corrupted"
            elif "编码" in error_str or "encode" in error_str:
                st.session_state.index_error = "encoding"
            else:
                st.session_state.index_error = "unknown"
                st.session_state.index_error_detail = _sanitize_error(e)
        except Exception as e:
            logger.exception("文档索引失败（未知错误）")
            error_str = str(e).lower()
            if "empty" in error_str or "no text" in error_str:
                st.session_state.index_error = "empty"
            elif "connect" in error_str or "timeout" in error_str or "download" in error_str:
                st.session_state.index_error = "network"
            else:
                st.session_state.index_error = "unknown"
                st.session_state.index_error_detail = _sanitize_error(e)


def show_index_error():
    """根据 index_error 类型显示对应的错误提示和重试按钮。"""
    error_type = st.session_state.get("index_error")
    if not error_type:
        return

    messages = {
        "encoding": "文件编码不支持，请确保上传的是 UTF-8 编码的 .txt 文件。",
        "missing": "文件已被移动或删除，请回到前几步重新选择。",
        "empty":   "文件内容为空，请检查后重新选择。",
        "network": "首次使用需下载模型（约 400MB），请检查网络连接后重试。",
        "pdf_encrypted": "PDF 文件已加密，请先解密为普通 PDF 后重新上传。",
        "pdf_scanned": "PDF 可能是扫描件，无法提取文字。请上传含文本的 PDF 或使用 OCR 工具转换。",
        "pdf_corrupted": "PDF 文件已损坏或格式异常，请检查后重新上传。",
        "unknown": f"索引失败：{st.session_state.get('index_error_detail', '未知错误')}",
    }
    st.error(messages.get(error_type, messages["unknown"]))

    # 只有 network 类错误值得重试，其他需要用户修复文件
    if error_type == "network":
        if st.button("🔄 重试索引", use_container_width=True):
            st.session_state.index_error = None
            st.session_state.docs_indexed = False
            st.rerun()
    else:
        if st.button("↩ 返回重新选择文件", use_container_width=True):
            st.session_state.index_error = None
            st.session_state.docs_indexed = False
            st.session_state.wizard_step = 1
            st.rerun()


# ============================================================
# UI 组件
# ============================================================

def render_step_indicator(current_step: int):
    """顶部分步进度条。"""
    cols = st.columns(5)
    icons = ["📄", "📋", "✏️", "⚙️", "📥"]
    for i, (col, label) in enumerate(zip(cols, WIZARD_STEPS)):
        num = i + 1
        with col:
            if num == current_step:
                st.info(f"**{icons[i]} Step {num}**\n{label}")
            elif num < current_step:
                st.success(f"✅ Step {num}\n~~{label}~~")
            else:
                st.caption(f"{icons[i]} Step {num}\n{label}")
    st.divider()


def render_navigation():
    """底部上一步/下一步按钮。"""
    c1, _, c3 = st.columns([1, 2, 1])

    with c1:
        if st.session_state.wizard_step > 1:
            if st.button("← 上一步", use_container_width=True):
                st.session_state.wizard_step -= 1
                st.rerun()

    with c3:
        step = st.session_state.wizard_step
        if step < 5:
            can_proceed = validate_current_step()
            if st.button(
                "下一步 →" if step < 4 else "去下载 →",
                type="primary",
                use_container_width=True,
                disabled=not can_proceed,
            ):
                st.session_state.wizard_step += 1
                st.rerun()
        elif step == 5:
            if st.button("🔄 重新开始", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


# ============================================================
# Step 1：选择样本简历
# ============================================================

def step_1_resume():
    st.header("① 选择样本简历")
    st.caption("上传一份你喜欢的简历作为风格参考，或从样例中选择。AI 会学习它的结构、措辞和排版风格。")

    source = st.radio(
        "简历来源",
        ["📁 上传文件", "📋 选择样例"],
        horizontal=True,
        key="resume_source",
    )

    if source == "📁 上传文件":
        uploaded = st.file_uploader("上传简历 (.txt, .pdf, .docx, .md)", type=["txt", "pdf", "docx", "md"], key="step1_uploader")
        if uploaded:
            path, error = save_uploaded_file(uploaded, "resume")
            if error:
                st.error(f"❌ {error}")
                st.session_state.resume_path = None
                st.session_state.resume_name = None
            else:
                st.session_state.resume_path = path
                st.session_state.resume_name = uploaded.name
                st.session_state.docs_indexed = False
                st.toast(f"✅ 已上传：{uploaded.name}")
    else:
        labels = ["-- 请选择 --"] + [r["label"] for r in AVAILABLE_RESUMES]
        choice = st.selectbox("选择样例简历", labels, key="step1_sample_select")
        if choice and choice != "-- 请选择 --":
            for r in AVAILABLE_RESUMES:
                if r["label"] == choice:
                    st.session_state.resume_path = r["path"]
                    st.session_state.resume_name = choice
                    st.session_state.docs_indexed = False
                    break
            st.toast(f"✅ 已选择：{choice}")
        elif choice == "-- 请选择 --" and st.session_state.get("resume_path"):
            st.session_state.resume_path = None
            st.session_state.resume_name = None
            st.session_state.docs_indexed = False

    # 当前选择提示
    _render_file_preview("resume", "📄")


# ============================================================
# Step 2：选择 JD
# ============================================================

def step_2_jd():
    st.header("② 选择职位描述 (JD)")
    st.caption("上传文件、粘贴文字、或从样例中选择。AI 会根据 JD 要求定制简历内容。")

    source = st.radio(
        "JD 来源",
        ["📁 上传文件", "📝 粘贴文字", "📋 选择样例"],
        horizontal=True,
        key="jd_source",
    )

    if source == "📁 上传文件":
        uploaded = st.file_uploader("上传 JD (.txt, .pdf, .docx, .md)", type=["txt", "pdf", "docx", "md"], key="step2_uploader")
        if uploaded:
            path, error = save_uploaded_file(uploaded, "jd")
            if error:
                st.error(f"❌ {error}")
                st.session_state.jd_path = None
                st.session_state.jd_name = None
            else:
                st.session_state.jd_path = path
                st.session_state.jd_name = uploaded.name
                st.session_state.docs_indexed = False
                st.toast(f"✅ 已上传：{uploaded.name}")

    elif source == "📝 粘贴文字":
        jd_text = st.text_area(
            "粘贴 JD 文字",
            height=200,
            placeholder="将招聘 JD 的岗位职责和任职要求粘贴到这里",
            key="step2_paste_input",
            label_visibility="collapsed",
        )
        jd_label = st.text_input(
            "给这段 JD 起个名字（可选）",
            placeholder="例如：OPPO AI产品实习生",
            key="step2_paste_label",
        )
        if st.button("✅ 确认内容", key="step2_paste_confirm"):
            stripped = jd_text.strip()
            if not stripped:
                st.warning("请先粘贴 JD 文字。")
            elif len(stripped) < 20:
                st.warning("粘贴的文字太短，请至少包含完整的岗位职责。")
            else:
                path = os.path.join(tempfile.gettempdir(), f"jd_paste_{uuid.uuid4().hex[:8]}.txt")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(stripped)
                except OSError:
                    st.error("文件保存失败，请检查磁盘空间后重试。")
                else:
                    name = jd_label.strip() or "手动粘贴 JD"
                    st.session_state.jd_path = path
                    st.session_state.jd_name = name
                    st.session_state.docs_indexed = False
                    st.toast(f"✅ 已保存：{name}（{len(stripped)} 字符）")

    else:
        labels = ["-- 请选择 --"] + [j["label"] for j in AVAILABLE_JDS]
        choice = st.selectbox("选择样例 JD", labels, key="step2_sample_select")
        if choice and choice != "-- 请选择 --":
            for j in AVAILABLE_JDS:
                if j["label"] == choice:
                    st.session_state.jd_path = j["path"]
                    st.session_state.jd_name = choice
                    st.session_state.docs_indexed = False
                    break
            st.toast(f"✅ 已选择：{choice}")
        elif choice == "-- 请选择 --" and st.session_state.get("jd_path"):
            st.session_state.jd_path = None
            st.session_state.jd_name = None
            st.session_state.docs_indexed = False

    # 当前选择提示
    _render_file_preview("jd", "📋")


# ============================================================
# Step 3：补充信息（可选）
# ============================================================

def step_3_user_info():
    st.header("③ 补充信息（可选）")
    st.caption("你的工作经历已从经验库自动读取。这里可以补充简历侧重点、岗位理解、想强调或弱化的内容。")

    supplement = st.text_area(
        "补充指引",
        value=st.session_state.user_text,
        height=150,
        placeholder="例如：突出高并发优化经验，弱化前端部分；希望简历体现团队管理能力；这个岗位偏架构方向，侧重系统设计经历",
        key="user_text_input",
    )
    st.session_state.user_text = supplement

    st.divider()

    # 照片上传（可选，嵌入 PDF 简历首页右上角）
    st.caption("📷 **简历照片**（可选，仅用于 PDF 导出）")
    uploaded_photo = st.file_uploader(
        "上传照片",
        type=["jpg", "jpeg", "png"],
        key="step3_photo_uploader",
        label_visibility="collapsed",
    )
    if uploaded_photo:
        path, error = save_uploaded_file(uploaded_photo, "photo")
        if error:
            st.error(f"❌ {error}")
            st.session_state.user_photo_path = None
        else:
            st.session_state.user_photo_path = path
            st.toast(f"✅ 已上传照片：{uploaded_photo.name}")
            # 预览
            st.image(uploaded_photo.getvalue(), width=80, caption=uploaded_photo.name)
    elif st.session_state.user_photo_path:
        # 之前上传过，显示预览
        try:
            st.image(st.session_state.user_photo_path, width=80, caption="当前照片")
        except Exception:
            logger.warning("照片文件加载失败，已清除：%s", st.session_state.user_photo_path, exc_info=True)
            st.session_state.user_photo_path = None
            st.warning("照片文件已失效，请重新上传。")
        if st.button("🗑 移除照片", key="remove_photo"):
            st.session_state.user_photo_path = None
            st.rerun()


# ============================================================
# Step 4：生成预览
# ============================================================

def step_4_generate_preview():
    st.header("④ 生成预览")

    # ================================================================
    # 状态 1：已完成（workflow_result 非空）
    # ================================================================
    if st.session_state.workflow_result is not None:
        result = st.session_state.workflow_result
        base = result.get("base_resume", "")
        customized = result.get("customized_resume", "")

        if result.get("_interrupted") is False:
            st.info("简历已自动生成，未经过人工审核步骤。")

        # 展示提醒（解析失败/降级等）
        notifications = result.get("notifications", [])
        if notifications:
            for note in notifications:
                st.warning(note)

        # 中间结果（调试用）
        with st.expander("🔍 中间分析结果"):
            user = result.get("user_profile")
            style = result.get("style_profile")
            jd_reqs = result.get("jd_requirements")

            if user:
                st.write(f"**解析用户**：{user.name}，{len(user.skills)} 项技能，{len(user.experience)} 段经历")
            if style:
                st.write(f"**风格**：{style.structure[:60]}...")
            if jd_reqs:
                st.write(f"**JD**：{jd_reqs.title}，{len(jd_reqs.keywords)} 个关键词")

        tab1, tab2 = st.tabs(["🎯 JD 定制简历", "📄 基础简历"])
        with tab1:
            st.markdown(customized)
        with tab2:
            st.markdown(base)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重新生成", use_container_width=True):
                _reset_workflow_state()
                st.rerun()
        with col2:
            if st.button("ℹ️ 调整信息", use_container_width=True):
                _reset_workflow_state()
                st.session_state.wizard_step = 3
                st.rerun()
        return

    # ================================================================
    # 状态 2：断点暂停（workflow_paused 为 True，审核基础简历 + 提醒后继续）
    # ================================================================
    if st.session_state.workflow_paused:
        paused = st.session_state.paused_result
        if paused is None:
            # 异常情况：标记为暂停但没有数据，回退到初始状态
            logger.warning("workflow_paused=True 但 paused_result 为空，回退到初始状态")
            st.session_state.workflow_paused = False
            st.session_state.paused_lost = True
            st.rerun()

        base_resume = paused.get("base_resume", "")
        user = paused.get("user_profile")
        style = paused.get("style_profile")
        jd_reqs = paused.get("jd_requirements")
        notifications = paused.get("notifications", [])

        # 展示提醒（check_parsed 在暂停前已执行）
        if notifications:
            for note in notifications:
                st.warning(note)

        st.success("✅ 基础简历已生成，请审核后再继续 JD 定制。")

        # 解析摘要
        with st.expander("🔍 解析摘要（点击展开）"):
            if user:
                st.write(f"**姓名**：{user.name or '（未识别）'}")
                st.write(f"**技能**：{', '.join(user.skills) if user.skills else '（未识别）'}")
                st.write(f"**经历**：{len(user.experience)} 段")
            if style:
                fallback_tag = " ⚠️ 默认风格" if style.is_fallback else ""
                st.write(f"**风格**：{style.structure[:80]}...{fallback_tag}")
            if jd_reqs:
                st.write(f"**目标岗位**：{jd_reqs.title}")
                st.write(f"**关键词**：{', '.join(jd_reqs.keywords) if jd_reqs.keywords else '（未提取到）'}")

        # 基础简历预览
        st.subheader("📄 基础简历预览")
        with st.container(border=True):
            if base_resume.strip():
                st.markdown(base_resume)
            else:
                st.warning("基础简历生成为空，建议在侧边栏补充经验库内容后重新生成。")

        # 操作按钮
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 审核通过，继续 JD 定制", type="primary", use_container_width=True, disabled=st.session_state.processing):
                st.session_state.processing = True
                with st.spinner("🤖 AI 正在根据 JD 定制简历... 这可能需要 20-40 秒"):
                    try:
                        result = resume_workflow(thread_id=st.session_state.session_id)
                        st.session_state.workflow_result = result
                        st.session_state.workflow_paused = False
                        st.session_state.paused_result = None
                        st.rerun()
                    except RuntimeError:
                        logger.warning("resume_workflow 失败：checkpoint 不存在", exc_info=True)
                        st.error(
                            "工作流状态丢失，无法继续。请点击下方「重新生成」按钮重新开始。"
                            "这通常是因为服务重启导致缓存被清空。"
                        )
                        st.session_state.workflow_paused = False
                        st.session_state.paused_result = None
                        st.session_state.processing = False
                    except Exception as e:
                        logger.exception("resume_workflow 执行失败")
                        st.session_state.processing = False
                        error_str = str(e).lower()
                        if "timeout" in error_str or "timed out" in error_str:
                            st.error("请求超时，请检查网络后重试。您可以再次点击「审核通过」按钮继续。")
                        elif "rate limit" in error_str or "too many" in error_str:
                            st.warning("请求过于频繁，请稍等片刻后重试。")
                        elif "unauthorized" in error_str or "auth" in error_str or "api key" in error_str or "apikey" in error_str:
                            st.error("API 认证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否正确。")
                        elif "connect" in error_str or "network" in error_str or "refused" in error_str:
                            st.error("无法连接到 AI 服务，请检查网络连接后重试。")
                        else:
                            st.error(f"JD 定制失败：{_sanitize_error(e)}")
        with c2:
            if st.button("🔄 放弃并重新生成", use_container_width=True):
                _reset_workflow_state()
                st.rerun()

        return

    # ================================================================
    # 状态 3：未开始
    # ================================================================
    if st.session_state.paused_lost:
        st.warning("上一次的审核状态已丢失（通常由服务重启导致），请重新生成。")
        st.session_state.paused_lost = False

    st.info("请确认以下信息无误后，点击生成按钮。")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"📄 **样本简历**：{st.session_state.resume_name}")
    with c2:
        st.write(f"📋 **目标 JD**：{st.session_state.jd_name}")

    if st.session_state.user_text.strip():
        with st.expander("📝 补充信息（点击展开）"):
            st.text(st.session_state.user_text[:800] + ("..." if len(st.session_state.user_text) > 800 else ""))

    if st.button("🚀 开始生成简历", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.session_state.workflow_result = None
        st.session_state.workflow_paused = False
        st.session_state.paused_result = None

        # 先索引文档
        index_documents_if_needed()

        if not st.session_state.docs_indexed:
            show_index_error()
            st.session_state.processing = False
            return

        # 运行工作流（会在 check_parsed 后暂停）
        with st.spinner("🤖 AI 正在生成基础简历... 这可能需要 20-40 秒"):
            try:
                result = run_workflow(
                    user_text=load_experience_bank(),
                    sample_resume_path=st.session_state.resume_path,
                    jd_path=st.session_state.jd_path,
                    thread_id=st.session_state.session_id,
                    user_supplement=st.session_state.user_text,
                )

                errors = result.get("errors", [])
                if errors:
                    st.session_state.processing = False
                    for err in errors:
                        st.error(f"❌ {err}")
                    return

                # 判断是否暂停在断点（由 LangGraph checkpoint 状态决定）
                if result["_interrupted"]:
                    # 预期情况：暂停在 check_parsed 之后
                    st.session_state.paused_result = result
                    st.session_state.workflow_paused = True
                else:
                    # 意外情况：工作流没有暂停直接完成了
                    logger.warning("工作流未在断点暂停，直接完成了（interrupt_after 可能未生效）")
                    st.session_state.workflow_result = result
                st.session_state.processing = False
                st.rerun()
            except Exception as e:
                logger.exception("工作流执行失败")
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    st.error("请求超时，请检查网络后点击【重新生成】重试。")
                elif "rate limit" in error_str or "too many" in error_str:
                    st.warning("请求过于频繁，请稍等片刻后重试。")
                elif "unauthorized" in error_str or "auth" in error_str or "api key" in error_str or "apikey" in error_str:
                    st.error("API 认证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否正确。")
                elif "connect" in error_str or "network" in error_str or "refused" in error_str:
                    st.error("无法连接到 AI 服务，请检查网络连接后重试。")
                else:
                    st.error(f"生成失败：{_sanitize_error(e)}")
                st.session_state.processing = False


# ============================================================
# Step 5：下载导出
# ============================================================

def _ensure_export_cache(result: dict):
    """确保导出 bytes 已缓存到 session_state，避免每次 rerun 重新生成。"""
    base = result.get("base_resume", "")
    customized = result.get("customized_resume", "")

    photo = st.session_state.get("user_photo_path")
    jd_reqs = result.get("jd_requirements")
    job_target = jd_reqs.title if jd_reqs else ""

    if st.session_state.export_customized_pdf is None and customized:
        pdf_bytes, pdf_err = markdown_to_pdf_bytes(customized, photo_path=photo, job_target=job_target)
        if pdf_err:
            logger.error("定制简历 PDF 导出失败：%s", pdf_err)
            st.session_state.export_customized_pdf = ("error", pdf_err)
        else:
            st.session_state.export_customized_pdf = ("ok", pdf_bytes)

    if st.session_state.export_customized_docx is None and customized:
        docx_bytes, docx_err = markdown_to_docx_bytes(customized, job_target=job_target)
        if docx_err:
            logger.error("定制简历 Word 导出失败：%s", docx_err)
            st.session_state.export_customized_docx = ("error", docx_err)
        else:
            st.session_state.export_customized_docx = ("ok", docx_bytes)

    if st.session_state.export_base_pdf is None and base:
        pdf_bytes, pdf_err = markdown_to_pdf_bytes(base, photo_path=photo, job_target=job_target)
        if pdf_err:
            logger.error("基础简历 PDF 导出失败：%s", pdf_err)
            st.session_state.export_base_pdf = ("error", pdf_err)
        else:
            st.session_state.export_base_pdf = ("ok", pdf_bytes)

    if st.session_state.export_base_docx is None and base:
        docx_bytes, docx_err = markdown_to_docx_bytes(base, job_target=job_target)
        if docx_err:
            logger.error("基础简历 Word 导出失败：%s", docx_err)
            st.session_state.export_base_docx = ("error", docx_err)
        else:
            st.session_state.export_base_docx = ("ok", docx_bytes)


def _render_download_buttons(label_prefix: str, md_text: str, file_prefix: str,
                              pdf_key: str, docx_key: str):
    """渲染一组三列下载按钮（Markdown / PDF / Word）。

    Args:
        label_prefix: 按钮标签前缀，如 "📥 定制简历"
        md_text: Markdown 原文
        file_prefix: 文件名前缀，如 "resume_python_customized"
        pdf_key: session_state key for cached PDF bytes
        docx_key: session_state key for cached Word bytes
    """
    col_md, col_pdf, col_docx = st.columns(3)

    with col_md:
        st.download_button(
            label=f"{label_prefix} (.md)",
            data=md_text.encode("utf-8"),
            file_name=f"{file_prefix}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_pdf:
        cached = st.session_state.get(pdf_key)
        if cached and cached[0] == "ok":
            st.download_button(
                label=f"{label_prefix} (.pdf)",
                data=cached[1],
                file_name=f"{file_prefix}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        elif cached and cached[0] == "error":
            st.button(
                label=f"{label_prefix} (.pdf) ⚠️",
                disabled=True,
                use_container_width=True,
            )
            st.caption(f"❌ {cached[1][:80]}")
        else:
            st.button(
                label=f"{label_prefix} (.pdf) ⏳",
                disabled=True,
                use_container_width=True,
            )

    with col_docx:
        cached = st.session_state.get(docx_key)
        if cached and cached[0] == "ok":
            st.download_button(
                label=f"{label_prefix} (.docx)",
                data=cached[1],
                file_name=f"{file_prefix}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        elif cached and cached[0] == "error":
            st.button(
                label=f"{label_prefix} (.docx) ⚠️",
                disabled=True,
                use_container_width=True,
            )
            st.caption(f"❌ {cached[1][:80]}")
        else:
            st.button(
                label=f"{label_prefix} (.docx) ⏳",
                disabled=True,
                use_container_width=True,
            )


def step_5_download():
    st.header("⑤ 下载导出")

    try:
        if st.session_state.workflow_result is None:
            st.warning("请先在 Step 4 生成简历。")
            if st.button("← 返回生成", use_container_width=True):
                st.session_state.wizard_step = 4
                st.rerun()
            return

        result = st.session_state.workflow_result
        base = result.get("base_resume", "")
        customized = result.get("customized_resume", "")
        jd_reqs = result.get("jd_requirements")
        jd_title = str(jd_reqs.title) if jd_reqs else "custom"
        safe_title = "".join(c for c in jd_title if c.isalnum() or c in " _-")[:30]

        # 确保导出缓存已生成
        _ensure_export_cache(result)

        # ── 定制简历 ──
        st.subheader("🎯 JD 定制简历")
        st.markdown(customized)
        _render_download_buttons(
            label_prefix="📥 定制简历",
            md_text=customized,
            file_prefix=f"resume_{safe_title}_customized",
            pdf_key="export_customized_pdf",
            docx_key="export_customized_docx",
        )

        st.divider()

        # ── 基础简历 ──
        st.subheader("📄 基础简历")
        st.markdown(base)
        _render_download_buttons(
            label_prefix="📥 基础简历",
            md_text=base,
            file_prefix=f"resume_{safe_title}_base",
            pdf_key="export_base_pdf",
            docx_key="export_base_docx",
        )

        # 操作按钮
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 重新生成", use_container_width=True):
                _reset_workflow_state()
                st.rerun()
        with c2:
            if st.button("ℹ️ 调整信息", use_container_width=True):
                _reset_workflow_state()
                st.session_state.wizard_step = 3
                st.rerun()
    except Exception:
        logger.exception("Step 5 下载导出页面异常")
        st.error("下载页面加载失败，请返回上一步重新生成简历。")
        if st.button("← 返回 Step 4", use_container_width=True):
            st.session_state.wizard_step = 4
            st.rerun()


# ============================================================
# 侧边栏：经验库管理
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.header("📚 经验库管理")
        st.caption("维护你的项目经历，AI 生成简历时会搜索这些内容。")

        # 编辑器
        edited = st.text_area(
            "Markdown 编辑",
            value=st.session_state.exp_bank_content,
            height=180,
            key="exp_bank_editor",
            label_visibility="collapsed",
        )

        # 预览（用 st.text 避免 markdown 渲染溢出侧边栏）
        with st.expander("👁️ 预览（纯文本）"):
            preview = edited if edited else st.session_state.exp_bank_content
            st.text(preview[:1200] + ("..." if len(preview) > 1200 else ""))

        if st.button("💾 保存修改", use_container_width=True):
            try:
                with open(EXP_BANK_PATH, "w", encoding="utf-8") as f:
                    f.write(edited)
                st.session_state.exp_bank_content = edited
                st.session_state.docs_indexed = False
                st.toast("✅ 经验库已保存")
            except PermissionError:
                st.error("保存失败：文件被占用或没有写入权限，请关闭其他程序后重试。")
            except OSError:
                st.error("保存失败：磁盘空间不足或文件系统错误，请检查后重试。")
            except Exception as e:
                st.error(f"保存失败：{_sanitize_error(e)}")

        st.divider()

        # AI 整理
        st.subheader("🤖 AI 帮我整理")
        st.caption("粘贴一段经历，AI 自动格式化为经验库条目。")

        raw = st.text_area(
            "经历描述",
            height=80,
            placeholder="例如：我在某公司做后端开发，负责订单系统改造...",
            key="ai_extract_input",
            label_visibility="collapsed",
        )

        if st.button("🔍 提取并预览", use_container_width=True):
            if len(raw.strip()) < 20:
                st.warning("请至少输入 20 个字符")
            else:
                with st.spinner("AI 正在整理..."):
                    try:
                        chain = EXPERIENCE_EXTRACTION_PROMPT | llm
                        result = chain.invoke({"raw_text": raw})
                        st.session_state.ai_extract_result = result.content
                        # 格式校验
                        is_valid, validation_msg = validate_experience_markdown(
                            st.session_state.ai_extract_result
                        )
                        if is_valid:
                            st.toast("✅ " + validation_msg)
                        else:
                            st.warning("⚠️ " + validation_msg)
                    except Exception as e:
                        error_str = str(e).lower()
                        if "timeout" in error_str or "timed out" in error_str:
                            st.error("AI 提取超时，请检查网络后重试。")
                        elif "rate limit" in error_str or "too many" in error_str:
                            st.warning("请求过于频繁，请稍等片刻后重试。")
                        elif "connect" in error_str or "network" in error_str:
                            st.error("无法连接到 AI 服务，请检查网络连接。")
                        else:
                            st.error(f"提取失败：{_sanitize_error(e)}")

        # 确认追加（用 st.text 避免溢出）
        if st.session_state.ai_extract_result:
            st.markdown("---")
            st.caption("### 提取结果预览")
            st.info("请确认以下内容准确无误：")
            st.text(st.session_state.ai_extract_result[:800] +
                    ("..." if len(st.session_state.ai_extract_result) > 800 else ""))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 确认", type="primary", use_container_width=True):
                    try:
                        with open(EXP_BANK_PATH, "a", encoding="utf-8") as f:
                            f.write(f"\n\n{st.session_state.ai_extract_result}")
                        st.session_state.exp_bank_content = load_experience_bank()
                        st.session_state.docs_indexed = False
                        st.session_state.ai_extract_result = None
                        st.toast("✅ 已追加到经验库！")
                        st.rerun()
                    except PermissionError:
                        st.error("写入失败：文件被占用或没有写入权限，请关闭其他程序后重试。")
                    except OSError:
                        st.error("写入失败：磁盘空间不足或文件系统错误，请检查后重试。")
                    except Exception as e:
                        st.error(f"写入失败：{_sanitize_error(e)}")
            with c2:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.ai_extract_result = None
                    st.rerun()

        st.divider()

        # 状态
        st.caption("### 📊 当前状态")
        resume_ok = st.session_state.resume_name is not None
        jd_ok = st.session_state.jd_name is not None
        st.caption(f"{'✅' if resume_ok else '❌'} 简历：{st.session_state.resume_name or '未选择'}")
        st.caption(f"{'✅' if jd_ok else '❌'} JD：{st.session_state.jd_name or '未选择'}")
        st.caption(f"{'✅' if st.session_state.docs_indexed else '⏳'} 文档索引")
        if st.session_state.workflow_result:
            cr = st.session_state.workflow_result.get("customized_resume", "")
            st.caption(f"✅ 简历已生成 ({len(cr)} 字符)")


# ============================================================
# 主函数
# ============================================================

def main():
    setup_logging()

    st.set_page_config(
        page_title="AI 简历生成器",
        page_icon="📝",
        layout="wide",
    )

    initialize_session_state()
    render_sidebar()

    st.title("📝 AI 简历生成器")
    st.caption("上传样本简历 + JD + 你的经历 → AI 生成定制简历")

    render_step_indicator(st.session_state.wizard_step)

    # 分发当前步骤
    step = st.session_state.wizard_step
    if step == 1:
        step_1_resume()
    elif step == 2:
        step_2_jd()
    elif step == 3:
        step_3_user_info()
    elif step == 4:
        step_4_generate_preview()
    elif step == 5:
        step_5_download()

    render_navigation()

    st.divider()
    st.caption(
        "技术栈：LangChain | LangGraph | DeepSeek | FAISS | BM25 | jieba | Streamlit"
    )


if __name__ == "__main__":
    main()
