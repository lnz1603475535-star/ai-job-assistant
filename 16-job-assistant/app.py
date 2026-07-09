"""
AI 简历生成器 — Streamlit UI (Round 3)
=====================================
分步向导 + 简历预览 + 经验库编辑器。

技术栈：LangChain | LangGraph | DeepSeek | FAISS | Streamlit
"""

import streamlit as st
import sys, os, tempfile, uuid

sys.path.insert(0, os.path.dirname(__file__))

from core import (
    load_and_index_documents,
    set_vectorstore,
    llm,
)
from workflow import run_workflow
from resume_engine import parse_user_info

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
WIZARD_STEPS = ["样本简历", "JD 要求", "个人信息", "生成预览", "下载导出"]

USER_TEXT_EXAMPLE = """我叫李思，邮箱 lisi@email.com，电话 13800002222。
技能包括：Python、Django、FastAPI、Docker、MySQL、Redis、Linux。
工作经历：
2022年6月到2025年3月，在某互联网公司做 Python 后端开发，负责订单系统从单体拆分为微服务，
用 FastAPI 重写了核心 API，性能提升了 3 倍，日均处理 200 万订单。
还搭建了 CI/CD 流水线，用 Docker 容器化部署。
2020年7月到2022年5月，在某创业公司做全栈开发，用 Django 写后端，Vue 写前端，
独立负责用户系统和支付模块。
学历：浙江大学 软件工程 本科 2016-2020"""


# ============================================================
# 辅助函数
# ============================================================

def load_experience_bank() -> str:
    """读取经验库文件内容。"""
    try:
        if os.path.exists(EXP_BANK_PATH):
            with open(EXP_BANK_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return "# 经验库\n\n在此粘贴你的项目经历，AI 生成简历时会参考这些内容。\n"


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
        "user_parsed": None,
        "workflow_result": None,
        "processing": False,
        "show_ai_extract": False,
        "ai_extract_result": None,
        "index_error": None,        # 索引失败类型：encoding/missing/empty/network/unknown
        "index_error_detail": "",   # unknown 类型时的原始错误信息
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "exp_bank_content" not in st.session_state:
        st.session_state.exp_bank_content = load_experience_bank()


def save_uploaded_file(uploaded_file) -> tuple[str, str | None]:
    """保存上传文件到临时目录，返回 (路径, 错误消息)。
    成功时路径有效、错误为 None；失败时路径为空、错误为用户可读的提示。
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return "", "文件编码不支持，请保存为 UTF-8 编码的 .txt 文件后重新上传。"

    if not content.strip():
        return "", "文件内容为空，请检查后重新上传。"

    suffix = os.path.splitext(uploaded_file.name)[1] or ".txt"
    path = os.path.join(tempfile.gettempdir(), f"resume_{uuid.uuid4().hex[:8]}{suffix}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except (OSError, PermissionError):
        return "", "文件保存失败，请检查磁盘空间或临时目录权限后重试。"

    return path, None


def validate_current_step() -> bool:
    """检查当前步骤是否可以前进。"""
    step = st.session_state.wizard_step
    if step == 1:
        return st.session_state.resume_path is not None
    if step == 2:
        return st.session_state.jd_path is not None
    if step == 3:
        return len(st.session_state.user_text.strip()) >= 20
    if step == 4:
        return st.session_state.workflow_result is not None
    return True


def index_documents_if_needed():
    """如果还没索引，就索引文档。失败时给出分类提示并允许重试。"""
    if st.session_state.docs_indexed:
        return
    if not st.session_state.resume_path or not st.session_state.jd_path:
        return

    with st.spinner("📚 正在索引文档..."):
        try:
            vs, bm25, chunks = load_and_index_documents({
                "user_experience": [EXP_BANK_PATH],
                "sample_resume": [st.session_state.resume_path],
                "jd": [st.session_state.jd_path],
            })
            set_vectorstore(vs, bm25, chunks)
            st.session_state.docs_indexed = True
            st.session_state.chunk_count = len(chunks)
            st.session_state.index_error = None
            st.toast(f"✅ 已索引 {len(chunks)} 个文本块")
        except UnicodeDecodeError:
            st.session_state.index_error = "encoding"
        except FileNotFoundError:
            st.session_state.index_error = "missing"
        except Exception as e:
            error_str = str(e).lower()
            if "empty" in error_str or "no text" in error_str:
                st.session_state.index_error = "empty"
            elif "connect" in error_str or "timeout" in error_str or "download" in error_str:
                st.session_state.index_error = "network"
            else:
                st.session_state.index_error = "unknown"
                st.session_state.index_error_detail = str(e)[:200]


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
    c1, c2, c3 = st.columns([1, 2, 1])

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
        uploaded = st.file_uploader("上传简历 (.txt)", type=["txt"], key="step1_uploader")
        if uploaded:
            path, error = save_uploaded_file(uploaded)
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
        labels = [r["label"] for r in AVAILABLE_RESUMES]
        choice = st.selectbox("选择样例简历", labels, key="step1_sample_select")
        if choice:
            for r in AVAILABLE_RESUMES:
                if r["label"] == choice:
                    st.session_state.resume_path = r["path"]
                    st.session_state.resume_name = choice
                    st.session_state.docs_indexed = False
                    break
            st.toast(f"✅ 已选择：{choice}")

    # 当前选择提示
    if st.session_state.resume_path:
        st.info(f"📄 当前选择：**{st.session_state.resume_name}**")
        with st.expander("📄 内容预览"):
            try:
                with open(st.session_state.resume_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    st.warning("文件内容为空，请重新选择。")
                    st.session_state.resume_path = None
                    st.session_state.resume_name = None
                else:
                    st.text(content[:600] + ("..." if len(content) > 600 else ""))
            except UnicodeDecodeError:
                st.error("文件编码不支持，请保存为 UTF-8 格式后重新上传。")
                st.session_state.resume_path = None
                st.session_state.resume_name = None
            except FileNotFoundError:
                st.warning("文件已被移动或删除，请重新选择。")
                st.session_state.resume_path = None
                st.session_state.resume_name = None
            except (OSError, PermissionError):
                st.error("无法读取文件，请检查文件权限后重试。")


# ============================================================
# Step 2：选择 JD
# ============================================================

def step_2_jd():
    st.header("② 选择职位描述 (JD)")
    st.caption("上传目标岗位的 JD，或从样例中选择。AI 会根据 JD 要求定制简历内容。")

    source = st.radio(
        "JD 来源",
        ["📁 上传文件", "📋 选择样例"],
        horizontal=True,
        key="jd_source",
    )

    if source == "📁 上传文件":
        uploaded = st.file_uploader("上传 JD (.txt)", type=["txt"], key="step2_uploader")
        if uploaded:
            path, error = save_uploaded_file(uploaded)
            if error:
                st.error(f"❌ {error}")
                st.session_state.jd_path = None
                st.session_state.jd_name = None
            else:
                st.session_state.jd_path = path
                st.session_state.jd_name = uploaded.name
                st.session_state.docs_indexed = False
                st.toast(f"✅ 已上传：{uploaded.name}")
    else:
        labels = [j["label"] for j in AVAILABLE_JDS]
        choice = st.selectbox("选择样例 JD", labels, key="step2_sample_select")
        if choice:
            for j in AVAILABLE_JDS:
                if j["label"] == choice:
                    st.session_state.jd_path = j["path"]
                    st.session_state.jd_name = choice
                    st.session_state.docs_indexed = False
                    break
            st.toast(f"✅ 已选择：{choice}")

    # 当前选择提示
    if st.session_state.jd_path:
        st.info(f"📋 当前选择：**{st.session_state.jd_name}**")
        with st.expander("📋 内容预览"):
            try:
                with open(st.session_state.jd_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    st.warning("文件内容为空，请重新选择。")
                    st.session_state.jd_path = None
                    st.session_state.jd_name = None
                else:
                    st.text(content[:600] + ("..." if len(content) > 600 else ""))
            except UnicodeDecodeError:
                st.error("文件编码不支持，请保存为 UTF-8 格式后重新上传。")
                st.session_state.jd_path = None
                st.session_state.jd_name = None
            except FileNotFoundError:
                st.warning("文件已被移动或删除，请重新选择。")
                st.session_state.jd_path = None
                st.session_state.jd_name = None
            except (OSError, PermissionError):
                st.error("无法读取文件，请检查文件权限后重试。")


# ============================================================
# Step 3：输入个人信息
# ============================================================

def step_3_user_info():
    st.header("③ 填写个人信息")
    st.caption("用口语描述你的教育背景、技能和工作经历。AI 会自动提取结构化信息。")

    user_text = st.text_area(
        "自由描述",
        value=st.session_state.user_text,
        height=250,
        placeholder=USER_TEXT_EXAMPLE,
        key="user_text_input",
    )
    st.session_state.user_text = user_text

    with st.expander("💡 参考写法（点击展开）"):
        st.code(USER_TEXT_EXAMPLE, language=None)

    # 输入量提示
    char_count = len(user_text.strip())
    if char_count == 0:
        pass  # 刚进入页面，不打扰
    elif char_count < 20:
        st.caption(f"📝 已输入 {char_count} 字，还差 {20 - char_count} 字即可预览解析结果")

    # 解析预览
    if char_count >= 20:
        if st.button("🔍 预览解析结果"):
            with st.spinner("正在解析..."):
                try:
                    profile = parse_user_info(user_text)
                    st.session_state.user_parsed = profile
                    if not profile.name and not profile.skills:
                        st.warning("未识别到姓名和技能，建议补充更多描述后再试。")
                except Exception:
                    st.session_state.user_parsed = None
                    st.warning("解析失败，请检查输入内容后重试。这不影响后续生成，但建议至少包含姓名和技能信息。")

        if st.session_state.user_parsed:
            profile = st.session_state.user_parsed
            st.subheader("解析结果")
            st.write(f"**姓名**：{profile.name or '（未识别）'}")
            st.write(f"**联系方式**：{profile.contact or '（未识别）'}")
            if profile.skills:
                st.write(f"**技能**：{', '.join(profile.skills)}")
            if profile.experience:
                st.write(f"**工作经历**（{len(profile.experience)} 段）：")
                for exp in profile.experience:
                    st.write(f"- {exp.title} @ {exp.company} ({exp.duration})")
            st.write(f"**教育背景**：{profile.education or '（未识别）'}")


# ============================================================
# Step 4：生成预览
# ============================================================

def step_4_generate_preview():
    st.header("④ 生成预览")

    # 未生成时显示确认信息
    if st.session_state.workflow_result is None:
        st.info("请确认以下信息无误后，点击生成按钮。")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"📄 **样本简历**：{st.session_state.resume_name}")
        with c2:
            st.write(f"📋 **目标 JD**：{st.session_state.jd_name}")

        with st.expander("📝 个人信息（点击展开）"):
            st.text(st.session_state.user_text[:800] + ("..." if len(st.session_state.user_text) > 800 else ""))

        if st.button("🚀 开始生成简历", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            st.session_state.workflow_result = None

            # 先索引文档
            index_documents_if_needed()

            if not st.session_state.docs_indexed:
                show_index_error()
                st.session_state.processing = False
                return

            # 运行工作流
            with st.spinner("🤖 AI 正在生成简历... 这可能需要 30-60 秒"):
                try:
                    result = run_workflow(
                        user_text=st.session_state.user_text,
                        sample_resume_path=st.session_state.resume_path,
                        jd_path=st.session_state.jd_path,
                        thread_id=st.session_state.session_id,
                    )

                    errors = result.get("errors", [])
                    if errors:
                        for err in errors:
                            st.error(f"❌ {err}")
                    else:
                        st.session_state.workflow_result = result
                        st.session_state.processing = False
                        st.rerun()
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str or "timed out" in error_str:
                        st.error("请求超时，请检查网络后点击【重新生成】重试。")
                    elif "rate limit" in error_str or "too many" in error_str:
                        st.warning("请求过于频繁，请稍等片刻后重试。")
                    elif "unauthorized" in error_str or "auth" in error_str or "key" in error_str:
                        st.error("API 认证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否正确。")
                    elif "connect" in error_str or "network" in error_str or "refused" in error_str:
                        st.error("无法连接到 AI 服务，请检查网络连接后重试。")
                    else:
                        st.error(f"生成失败：{str(e)[:200]}")
                    st.session_state.processing = False

    # 已生成时显示预览
    else:
        result = st.session_state.workflow_result
        base = result.get("base_resume", "")
        customized = result.get("customized_resume", "")

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
                st.session_state.workflow_result = None
                st.rerun()
        with col2:
            if st.button("ℹ️ 调整信息", use_container_width=True):
                st.session_state.workflow_result = None
                st.session_state.wizard_step = 3
                st.rerun()


# ============================================================
# Step 5：下载导出
# ============================================================

def step_5_download():
    st.header("⑤ 下载导出")

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
    jd_title = jd_reqs.title if jd_reqs else "custom"
    safe_title = "".join(c for c in jd_title if c.isalnum() or c in " _-")[:30]

    st.subheader("🎯 JD 定制简历")
    st.markdown(customized)
    st.download_button(
        label="📥 下载定制简历 (.md)",
        data=customized.encode("utf-8"),
        file_name=f"resume_{safe_title}_customized.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()

    st.subheader("📄 基础简历")
    st.markdown(base)
    st.download_button(
        label="📥 下载基础简历 (.md)",
        data=base.encode("utf-8"),
        file_name=f"resume_{safe_title}_base.md",
        mime="text/markdown",
        use_container_width=True,
    )


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
            except Exception as e:
                st.error(f"保存失败：{e}")

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
                        from langchain_core.messages import HumanMessage
                        extraction_prompt = """你是一个经验库整理专家。从用户的口语描述中提取工作经历，格式化为结构化的 Markdown。

提取规则：
1. 找出每段工作经历的：公司/项目名称、时间段、角色、技术栈、具体成就
2. 每段经历一个 ## 标题，格式如下：

## 项目：{项目名称}
- 时间：{时间段}
- 角色：{角色}
- 技术栈：{技术栈}
- 详情：
   {每个成就或职责用独立段落}

3. 不要编造任何用户没有提到的信息。用中文输出。"""
                        response = llm.invoke([
                            HumanMessage(content=f"{extraction_prompt}\n\n用户描述：\n{raw}")
                        ])
                        st.session_state.ai_extract_result = response.content
                    except Exception as e:
                        st.error(f"提取失败：{e}")

        # 确认追加（用 st.text 避免溢出）
        if st.session_state.ai_extract_result:
            st.markdown("---")
            st.caption("### 提取结果预览")
            st.info("请确认以下内容准确无误：")
            st.text(st.session_state.ai_extract_result[:800] +
                    ("..." if len(st.session_state.ai_extract_result) > 800 else ""))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 确认追加", type="primary", use_container_width=True):
                    try:
                        with open(EXP_BANK_PATH, "a", encoding="utf-8") as f:
                            f.write(f"\n\n{st.session_state.ai_extract_result}")
                        st.session_state.exp_bank_content = load_experience_bank()
                        st.session_state.docs_indexed = False
                        st.session_state.ai_extract_result = None
                        st.toast("✅ 已追加到经验库！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"写入失败：{e}")
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
        "技术栈：LangChain · LangGraph · DeepSeek · FAISS · BM25 · jieba · Streamlit | "
        "Round 3 — 分步向导 + 经验库管理"
    )


if __name__ == "__main__":
    main()
