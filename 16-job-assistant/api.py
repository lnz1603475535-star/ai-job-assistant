"""
AI 简历生成器 — FastAPI 后端（第 13 课）
========================================
5 个核心端点：简历生成 / JD 定制 / 导出下载 / 用户管理 / 面试问答。

运行：uvicorn api:app --reload --port 8000
文档：http://127.0.0.1:8000/docs（Swagger UI）

设计约定：
- 端点是薄封装，业务逻辑全部复用 core / resume_engine / workflow / exporters
- 阻塞调用（索引、LLM、导出）用 asyncio.to_thread 包住，不阻塞事件循环
- 存储均为进程内（内存 dict + 临时文件），重启即清——第 14 课迁移 PostgreSQL
- 检索服务复用 core 的模块级注册表（RetrievalService）：API 进程内单实例，
  每次 generate 重建索引并替换。多租户 per-request 注入是后续并发改造点。
"""

import asyncio
import json
import logging
import os
import queue
import tempfile
import threading
import uuid
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from core import llm, load_and_index_documents, setup_logging
from exporters import markdown_to_docx_bytes, markdown_to_pdf_bytes, sanitize_error
from models import UserProfile, WorkExperience
from resume_engine import stream_customize_for_jd
from workflow import build_workflow, resume_workflow, run_workflow

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 简历生成器 API",
    description="样本简历(风格) + 用户信息(内容) + JD(方向) → 定制简历",
    version="0.1.0",
)

# 上传文件保存目录（进程内临时目录，重启即清；第 14 课迁 PostgreSQL 时改文件表）
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai-job-assistant-uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 经验库路径（与 app.py 的 EXP_BANK_PATH 保持一致）
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXPERIENCE_BANK_PATH = os.path.join(DATA_DIR, "experience_bank.md")


# ============================================================
# Pydantic 请求模型
# ============================================================


class UserUpdate(BaseModel):
    """用户信息部分更新（全部可选，未传字段保持不变）。"""

    name: str | None = None
    contact: str | None = None
    skills: list[str] | None = None
    experience: list[WorkExperience] | None = None
    education: str | None = None


class CustomizeRequest(BaseModel):
    """JD 定制请求：run_workflow 返回的 thread_id。"""

    thread_id: str


class ExportRequest(BaseModel):
    """导出请求：简历文本 + 目标格式。"""

    resume_text: str
    format: Literal["pdf", "docx"]
    job_target: str = ""


class InterviewChatRequest(BaseModel):
    """面试问答请求：简历作为面试官上下文，thread_id 保持多轮对话。"""

    resume_text: str
    question: str
    thread_id: str = ""


# ============================================================
# 用户管理（内存存储 + 锁保护；第 14 课迁移 PostgreSQL）
# ============================================================

_users: dict[str, UserProfile] = {}
_users_lock = threading.Lock()


@app.post("/api/users", status_code=201)
async def create_user(user: UserProfile) -> dict:
    """创建用户信息（姓名/联系方式/技能/经历/教育）。"""
    user_id = uuid.uuid4().hex[:12]
    with _users_lock:
        _users[user_id] = user.model_copy()
    logger.info("创建用户 %s：%s", user_id, user.name)
    return {"user_id": user_id, **user.model_dump()}


@app.get("/api/users/{user_id}")
async def get_user(user_id: str) -> dict:
    """查询用户信息。"""
    with _users_lock:
        user = _users.get(user_id)
    if user is None:
        raise HTTPException(404, f"用户不存在：{user_id}")
    return user.model_dump()


@app.put("/api/users/{user_id}")
async def update_user(user_id: str, patch: UserUpdate) -> dict:
    """部分更新用户信息（未传字段保持不变）。"""
    with _users_lock:
        user = _users.get(user_id)
        if user is None:
            raise HTTPException(404, f"用户不存在：{user_id}")
        data = user.model_dump()
        data.update(patch.model_dump(exclude_unset=True))
        _users[user_id] = UserProfile(**data)
    logger.info("更新用户 %s", user_id)
    return _users[user_id].model_dump()


@app.delete("/api/users/{user_id}", status_code=204)
async def delete_user(user_id: str) -> None:
    """删除用户信息。"""
    with _users_lock:
        existed = _users.pop(user_id, None) is not None
    if not existed:
        raise HTTPException(404, f"用户不存在：{user_id}")
    logger.info("删除用户 %s", user_id)


# ============================================================
# 简历生成 / 定制 / 导出
# ============================================================


async def _save_upload(upload: UploadFile, prefix: str) -> str:
    """保存上传文件到 UPLOAD_DIR，返回本地路径。"""
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if not ext:
        ext = ".txt"
    path = os.path.join(UPLOAD_DIR, f"{prefix}-{uuid.uuid4().hex[:8]}{ext}")
    content = await upload.read()
    await asyncio.to_thread(_write_upload, path, content)
    return path


def _write_upload(path: str, content: bytes) -> None:
    """同步写文件（经 to_thread 调用，不阻塞事件循环）。"""
    with open(path, "wb") as f:
        f.write(content)


@app.post("/api/resume/generate")
async def generate_resume(
    user_text: Annotated[
        str, Form(description="用户口述信息（从经验库读取或直接输入）")
    ],
    sample_resume: Annotated[UploadFile, File(description="样本简历文件")],
    jd: Annotated[UploadFile, File(description="JD 文件")],
    user_supplement: Annotated[
        str, Form(description="可选的用户补充（侧重点/弱化项/岗位看法）")
    ] = "",
) -> dict:
    """生成基础简历（在 check_parsed 断点暂停）。

    流程：保存上传文件 → 建立 FAISS+BM25 双索引 → 运行 LangGraph 工作流
    至 check_parsed 断点。返回 thread_id 和 base_resume，
    用户审核后调用 POST /api/resume/customize 继续 JD 定制。
    """
    try:
        sample_path = await _save_upload(sample_resume, "sample")
        jd_path = await _save_upload(jd, "jd")

        # 重建索引并注册为当前检索服务（按需检索与 Agent 共用）
        load_and_index_documents(
            {
                "user_experience": [EXPERIENCE_BANK_PATH],
                "sample_resume": [sample_path],
                "jd": [jd_path],
            }
        )

        thread_id = uuid.uuid4().hex[:16]
        result = await asyncio.to_thread(
            run_workflow,
            user_text=user_text,
            sample_resume_path=sample_path,
            jd_path=jd_path,
            thread_id=thread_id,
            user_supplement=user_supplement,
        )
        logger.info(
            "简历生成：thread=%s，interrupted=%s，errors=%d",
            thread_id,
            result.get("_interrupted"),
            len(result.get("errors", [])),
        )
        return {"thread_id": thread_id, **result}
    except ValueError as e:
        # 文件格式/内容问题 → 400（消息已脱敏）
        raise HTTPException(400, sanitize_error(e)) from e
    except Exception as e:
        logger.exception("简历生成失败")
        raise HTTPException(500, sanitize_error(e)) from e


@app.post("/api/resume/customize")
async def customize_resume(req: CustomizeRequest) -> dict:
    """从 check_parsed 断点恢复执行，完成 JD 定制。

    返回定制简历 + token 用量 + 通知（含降级/Token 预算提醒）。
    """
    try:
        result = await asyncio.to_thread(resume_workflow, req.thread_id)
        logger.info("JD 定制完成：thread=%s", req.thread_id)
        return {
            "thread_id": req.thread_id,
            "customized_resume": result.get("customized_resume", ""),
            "token_usage": result.get("token_usage", {}),
            "notifications": result.get("notifications", []),
        }
    except RuntimeError as e:
        # checkpoint 不存在（服务重启/无效 thread_id）→ 404
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        logger.exception("JD 定制失败")
        raise HTTPException(500, sanitize_error(e)) from e


@app.post("/api/resume/export")
async def export_resume(req: ExportRequest) -> Response:
    """导出简历为 PDF 或 Word 文件下载。

    照片嵌入（photo_path）暂不支持 API 路径——Streamlit UI 已有此功能，
    第 13 课 API 版优先保证核心导出闭环。
    """
    if not req.resume_text.strip():
        raise HTTPException(422, "resume_text 不能为空")

    try:
        if req.format == "pdf":
            data, err = await asyncio.to_thread(
                markdown_to_pdf_bytes, req.resume_text, None, req.job_target
            )
            media_type = "application/pdf"
            filename = "resume.pdf"
        else:
            data, err = await asyncio.to_thread(
                markdown_to_docx_bytes, req.resume_text, req.job_target, None
            )
            media_type = (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
            filename = "resume.docx"
    except Exception as e:
        logger.exception("导出失败")
        raise HTTPException(500, sanitize_error(e)) from e

    if err is not None:
        logger.warning("导出失败：%s", err)
        raise HTTPException(500, sanitize_error(err))
    return Response(
        data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# 流式定制（异步提交 + SSE 流式 + 轮询兜底）
# ============================================================


# 任务注册表：task_id → CustomizeTask
# 线程模型：POST /stream 提交后由后台线程执行流式定制，SSE 端点只消费事件队列。
# 前端断开 SSE 不中断任务（任务线程与 HTTP 请求生命周期解耦）——这正是
# Streamlit 运行中点击按钮会取消任务的根治方案（2026-08-04 决策）。
class CustomizeTask:
    """一个流式定制任务：事件队列 + 最终结果。"""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self.result: dict | None = None
        self.error: str | None = None
        self.done = False


_tasks: dict[str, CustomizeTask] = {}
_tasks_lock = threading.Lock()

# token 批次大小：攒够一批再入队，减少队列事件数与 SSE 唤醒次数
_TOKEN_BATCH = 50


def _get_workflow_state(thread_id: str):
    """读取工作流 checkpoint 状态（只读，不执行）。"""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return build_workflow().get_state(config)


def _run_customize_task(task_id: str, task: CustomizeTask) -> None:
    """后台线程：从 checkpoint 读上下文 → 流式定制 → 事件入队。

    LLM 调用失败时降级 base_resume（与同步路径降级语义一致）；
    任务结束（成功或降级）后 put 哨兵 None，SSE 端点据此结束。
    """
    base_resume = ""
    try:
        state = _get_workflow_state(task.thread_id)
        values = state.values if state else {}
        base_resume = values.get("base_resume", "")
        jd_reqs = values.get("jd_requirements")
        if not base_resume or jd_reqs is None:
            task.error = (
                f"checkpoint（thread_id={task.thread_id}）缺少 base_resume 或"
                "jd_requirements，无法定制。请先调用 generate。"
            )
            logger.warning("流式定制任务启动失败：%s", task.error)
            task.events.put(("error", task.error))
            return

        # 攒批次再入队：token 逐块入队会让 SSE 端点频繁唤醒
        buf: list[str] = []

        def flush() -> None:
            if buf:
                task.events.put(("token", "".join(buf)))
                buf.clear()

        for kind, payload in stream_customize_for_jd(
            base_resume,
            jd_reqs,
            notifications=values.get("notifications"),
            user_supplement=values.get("user_supplement", ""),
        ):
            if kind == "token":
                buf.append(payload)
                if len(buf) >= _TOKEN_BATCH:
                    flush()
            elif kind == "done":
                flush()
                task.result = {
                    "thread_id": task.thread_id,
                    **payload,  # customized_resume / token_usage / failed
                }
                task.events.put(("done", task.result))
                return
    except Exception:
        # 流式失败 → 降级 base_resume（结果不空，前端可正常展示）
        logger.exception("流式定制任务失败：task=%s", task_id)
        task.result = {
            "thread_id": task.thread_id,
            "customized_resume": base_resume,
            "token_usage": {},
            "failed": True,
        }
        task.events.put(("done", task.result))
    finally:
        task.done = True
        task.events.put(None)  # 哨兵：SSE 端点据此结束


def _wait_task_event(task: CustomizeTask) -> tuple[str, Any] | None:
    """阻塞等待任务事件；任务已结束且队列空（哨兵）返回 None。"""
    while True:
        try:
            return task.events.get(timeout=1.0)
        except queue.Empty:
            if task.done:
                return None


@app.post("/api/resume/customize/stream")
async def start_customize_stream(req: CustomizeRequest) -> dict:
    """提交异步流式定制任务，立即返回 task_id（不等待生成）。

    流程：校验 checkpoint 可用 → 后台线程开始流式定制 → 返回 task_id。
    前端用 GET /api/resume/customize/stream/{task_id} 订阅 SSE 流；
    连接中断不影响任务执行，完成后可轮询
    GET /api/resume/customize/result/{task_id} 取结果。
    """
    try:
        state = await asyncio.to_thread(_get_workflow_state, req.thread_id)
        if state is None or not state.values.get("base_resume"):
            raise HTTPException(
                404,
                f"无法恢复工作流：thread_id={req.thread_id} 没有可用的 checkpoint。"
                "请先调用 generate。",
            )

        task_id = uuid.uuid4().hex[:12]
        task = CustomizeTask(req.thread_id)
        with _tasks_lock:
            _tasks[task_id] = task
        threading.Thread(
            target=_run_customize_task, args=(task_id, task), daemon=True
        ).start()
        logger.info("启动流式定制任务：task=%s thread=%s", task_id, req.thread_id)
        return {"task_id": task_id, "thread_id": req.thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("启动流式定制失败")
        raise HTTPException(500, sanitize_error(e)) from e


@app.get("/api/resume/customize/stream/{task_id}")
async def stream_customize(task_id: str) -> StreamingResponse:
    """SSE 流式输出：逐 token 推送 messages/partial 事件（对齐 LangGraph 标准）。

    事件序列：
        event: messages/partial
        data: {"messages": [{"type": "AIMessageChunk", "content": "..."}]}

        事件: done
        data: {"thread_id": ..., "customized_resume": "...", "token_usage": {...},
               "failed": false}

    客户端断开连接不影响后台任务；任务完成后可轮询
    GET /api/resume/customize/result/{task_id} 取结果。
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, f"任务不存在：{task_id}")

    def sse_event(event_name: str, payload: Any) -> str:
        return (
            f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        )

    async def event_stream():
        while True:
            event = await asyncio.to_thread(_wait_task_event, task)
            if event is None:
                return
            kind, payload = event
            if kind == "token":
                yield sse_event(
                    "messages/partial",
                    {"messages": [{"type": "AIMessageChunk", "content": payload}]},
                )
            elif kind == "done":
                yield sse_event("done", payload)
                return
            elif kind == "error":
                yield sse_event("error", {"detail": payload})
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/resume/customize/result/{task_id}")
async def get_customize_result(task_id: str) -> dict:
    """查询流式定制任务状态/结果（前端断线后轮询兜底）。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    return {
        "status": "done" if task.done else "running",
        "result": task.result,
        "error": task.error,
    }


# ============================================================
# 面试问答（内存对话历史；第 14 课接 PostgresSaver 持久化）
# ============================================================

# 保留最近 N 轮问答（每条消息一条），防止上下文无限增长
_INTERVIEW_HISTORY_MAX = 12

_interviews: dict[str, list[dict]] = {}
_interviews_lock = threading.Lock()

INTERVIEW_SYSTEM_PROMPT = """你是模拟面试官，正在根据候选人的简历进行技术面试。

职责：
1. 基于简历内容提问：追问技术细节、项目难点、量化结果
2. 点评候选人的回答：肯定亮点、指出不足、给出改进建议
3. 一题一题来，每轮聚焦一个点，不要一次问多个问题

回答风格：像真实面试官，先简短回应候选人的回答，再抛出下一个问题。"""


@app.post("/api/interview/chat")
async def interview_chat(req: InterviewChatRequest) -> dict:
    """模拟面试问答：面试官基于简历 + 对话历史回应。

    thread_id 为空时自动创建（响应中返回）；同一 thread_id 连续调用保持多轮对话。
    """
    if not req.resume_text.strip():
        raise HTTPException(422, "resume_text 不能为空")
    if not req.question.strip():
        raise HTTPException(422, "question 不能为空")

    thread_id = req.thread_id or uuid.uuid4().hex[:16]

    # 锁内只拷贝历史（快照），LLM 调用在锁外——避免持锁阻塞其他请求
    with _interviews_lock:
        history = list(_interviews.get(thread_id, []))

    messages = [
        SystemMessage(INTERVIEW_SYSTEM_PROMPT),
        HumanMessage(content=f"候选人简历：\n{req.resume_text}"),
    ]
    for m in history[-_INTERVIEW_HISTORY_MAX:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=req.question))

    try:
        response = await asyncio.to_thread(llm.invoke, messages)
        answer = response.content
    except Exception as e:
        logger.exception("面试问答失败：thread=%s", thread_id)
        raise HTTPException(500, sanitize_error(e)) from e

    # LLM 调用完成后才写回历史（失败不写，下次重发同问）
    with _interviews_lock:
        history = _interviews.setdefault(thread_id, [])
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > _INTERVIEW_HISTORY_MAX * 2:
            del history[: len(history) - _INTERVIEW_HISTORY_MAX * 2]

    logger.info("面试问答：thread=%s，%d 轮", thread_id, len(history) // 2)
    return {"thread_id": thread_id, "answer": answer}
