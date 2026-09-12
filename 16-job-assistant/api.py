"""
AI 简历生成器 — FastAPI 后端（第 13 课）
========================================
端点分组：用户管理（CRUD）/ 简历生成 / JD 定制（同步 + 流式）/ 导出下载 / 面试问答。
（具体路由见下方 @app 装饰器，或 http://127.0.0.1:8000/docs——不在文档里写死数量，
数字必会随迭代过期。）

运行：uvicorn api:app --reload --port 8000
文档：http://127.0.0.1:8000/docs（Swagger UI）

设计约定：
- 端点是薄封装，业务逻辑全部复用 core / resume_engine / workflow / exporters
- 阻塞调用（索引、LLM、导出）用 asyncio.to_thread 包住，不阻塞事件循环
- 存储均为进程内（内存 dict + 临时文件），重启即清——第 14 课迁移 PostgreSQL
- 检索服务复用 core 的模块级注册表（RetrievalService）：API 进程内单实例，
  每次 generate 重建索引并替换。多租户 per-request 注入是后续并发改造点。
- 流式定制的后台线程向 queue.Queue 投递事件，SSE 端点用 get_nowait + asyncio.sleep
  短轮询消费——不占用 asyncio.to_thread 的默认线程池（该池与 run_workflow /
  llm.invoke 等业务调用共用），也不依赖事件循环的生命周期。见 CustomizeTask。
"""

import asyncio
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from typing import Annotated, Any, Literal, NamedTuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from pydantic import BaseModel

from core import (
    EXPERIENCE_BANK_PATH,
    llm,
    load_and_index_documents,
    sanitize_error,
    setup_logging,
)
from exporters import markdown_to_docx_bytes, markdown_to_pdf_bytes
from models import JDRequirements, UserProfile, WorkExperience
from resume_engine import stream_customize_for_jd
from workflow import build_workflow, resume_workflow, run_workflow

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 简历生成器 API",
    description="样本简历(风格) + 用户信息(内容) + JD(方向) → 定制简历",
    version="0.1.0",
)

# 上传文件保存目录（系统临时目录——**重启并不会自动清空**，
# 由 _cleanup_stale_uploads() 在启动时按时间清理；第 14 课迁 PostgreSQL 时改文件表）
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai-job-assistant-uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 单个上传文件的大小上限（对齐 core 加载文档时的 _MAX_FILE_SIZE）
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
# 上传文件的保留时长（小时）——超时即清，避免随使用无限增长
_UPLOAD_MAX_AGE_HOURS = 24


def _cleanup_stale_uploads() -> None:
    """清理过期的上传文件（进程启动时调用一次）。

    上传文件只在 generate 期间被读取（checkpoint 里存的路径仅作记录），
    不清理会随使用无限增长——%TEMP% 并不会自动回收。
    按**时间**而不是"启动即清空"：将来 `uvicorn --workers N` 多进程启动时，
    清空会误删其他 worker 正在使用的文件。
    """
    cutoff = time.time() - _UPLOAD_MAX_AGE_HOURS * 3600
    removed = 0
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            logger.debug("清理上传文件失败（可忽略）：%s", path)
    if removed:
        logger.info(
            "已清理 %d 个过期上传文件（超过 %d 小时）", removed, _UPLOAD_MAX_AGE_HOURS
        )


_cleanup_stale_uploads()


# ============================================================
# Pydantic 请求模型
# ============================================================


class UserUpdate(BaseModel):
    """用户信息部分更新（全部可选，未传字段保持不变）。

    本类只是 UserProfile 的"全部可选"版本，字段一一对应——
    UserProfile 新增可更新字段时两处需同步（Pydantic 默认忽略未知字段，
    漏同步不会报错，只会让该字段在 PUT 时被静默忽略）。
    注意 _users 里存的是 UserProfile，将来若其增加服务端字段（如 user_id /
    created_at），它们不该出现在这里。
    """

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
        # 在锁内取值返回：锁外再读一次会与并发删除竞态（KeyError → 500）
        updated = _users[user_id].model_dump()
    logger.info("更新用户 %s", user_id)
    return updated


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
    """保存上传文件到 UPLOAD_DIR，返回本地路径。

    大小在**读取之前**用 upload.size 拦截：`await upload.read()` 会把整个文件
    读进内存，`_write_upload` 再落一份盘——等 core 在加载阶段才发现超限，
    内存尖峰和双份磁盘占用都已经付出了。

    Raises:
        ValueError: 文件超过 _MAX_UPLOAD_BYTES（端点映射为 400）
    """
    if upload.size is not None and upload.size > _MAX_UPLOAD_BYTES:
        raise ValueError(
            f"文件过大（{upload.size / 1024 / 1024:.1f}MB），"
            f"上限 {_MAX_UPLOAD_BYTES // 1024 // 1024}MB。请压缩后重试。"
        )
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
        # 精选响应键（不直接 **result）：整个 WorkflowState 还含 user_text
        # （把用户上传的整份经验库原样回显）、sample_resume_path / jd_path
        # （服务器临时路径）等只应留在服务端的字段。
        # 与 customize_resume 的响应风格保持一致。
        return {
            "thread_id": thread_id,
            "base_resume": result.get("base_resume", ""),
            "customized_resume": result.get("customized_resume", ""),
            "user_profile": result.get("user_profile"),
            "style_profile": result.get("style_profile"),
            "jd_requirements": result.get("jd_requirements"),
            "notifications": result.get("notifications", []),
            "errors": result.get("errors", []),
            "_interrupted": result.get("_interrupted", False),
        }
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

    # 关键字传参：两个导出函数的参数顺序相反（pdf 是 photo→job，docx 是 job→photo），
    # 位置传参时极易看错、也极易传反
    try:
        if req.format == "pdf":
            data, err = await asyncio.to_thread(
                markdown_to_pdf_bytes,
                req.resume_text,
                photo_path=None,
                job_target=req.job_target,
            )
            media_type = "application/pdf"
            filename = "resume.pdf"
        else:
            data, err = await asyncio.to_thread(
                markdown_to_docx_bytes,
                req.resume_text,
                job_target=req.job_target,
                photo_path=None,
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
    """一个流式定制任务：事件队列 + 最终结果。

    events 是 queue.Queue（线程安全的普通队列），后台线程直接 put；
    SSE 端点用 get_nowait + asyncio.sleep 短轮询消费。

    为什么不用 asyncio.Queue：它要求生产端拿到"消费端正在跑的那个事件循环"，
    再 call_soon_threadsafe 投递——而事件循环的生命周期不归本模块掌握
    （uvicorn 是进程级长循环，TestClient 则每个请求一个 portal 循环）。
    一旦捕获到的是短命循环，投递抛 RuntimeError、事件被静默丢弃。
    queue.Queue + 短轮询没有这个耦合：生产端只依赖队列本身，与事件循环无关。

    也不同于早期实现（asyncio.to_thread + queue.get(timeout=1.0) 轮询）：
    那条路径长期占用**默认线程池**的槽位，而该池同时服务 run_workflow /
    llm.invoke / 导出——并发的 SSE 连接足以占满池，连带卡住"生成简历"。
    现在消费端只在 asyncio.sleep 上等待，**完全不碰线程**。
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self.result: dict | None = None
        self.done = False

    def emit(self, event: tuple[str, Any] | None) -> None:
        """投递事件（由后台线程调用；queue.Queue.put 本身线程安全）。"""
        self.events.put(event)


_tasks: OrderedDict[str, CustomizeTask] = OrderedDict()
_tasks_lock = threading.Lock()

# 任务注册表上限：完成的任务持有完整简历文本，不回收会随着使用无限增长
_MAX_TRACKED_TASKS = 20

# token 批次大小：攒够一批再入队，减少队列事件数与 SSE 唤醒次数
_TOKEN_BATCH = 50

# SSE 心跳间隔（秒）：生成过程中 LLM 可能长时间不吐 token，期间一个字节都不发，
# 公网部署时会被反向代理判定为空闲连接掐断。空闲超过该间隔就发一行 SSE 注释保活。
_SSE_HEARTBEAT_SECONDS = 15
# SSE 消费端的取事件间隔（秒）：用 asyncio.sleep 挂起，不占线程；
# 50ms 的送达延迟对逐 token 展示无感知，换来与事件循环完全解耦
_SSE_POLL_SECONDS = 0.05


def _sse_event(event_name: str, payload: Any) -> str:
    """按 SSE 规范序列化一个事件（模块级，无需每请求重建）。"""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _register_task(task_id: str, task: CustomizeTask) -> None:
    """登记任务；超出上限时淘汰最旧的已完成任务（在跑的任务永不淘汰）。"""
    with _tasks_lock:
        _tasks[task_id] = task
        _tasks.move_to_end(task_id)
        while len(_tasks) > _MAX_TRACKED_TASKS:
            evicted = False
            for old_id, old_task in _tasks.items():
                if old_id != task_id and old_task.done:
                    del _tasks[old_id]
                    evicted = True
                    break
            if not evicted:  # 全部在跑，无可淘汰，等待下次登记再试
                break


class _CustomizeContext(NamedTuple):
    """定制所需的 checkpoint 上下文：端点读一次、校验一次，交给后台线程开工。"""

    base_resume: str
    jd_reqs: JDRequirements
    notifications: list[str] | None
    user_supplement: str


def _load_customize_context(thread_id: str) -> _CustomizeContext | None:
    """读取并校验定制所需的 checkpoint 上下文；None 表示该 thread 不可定制。

    这是"能否定制"的**唯一判定点**：端点用它决定 404，同一份数据直接交给
    后台线程序列化使用——线程不再自己重读 checkpoint，因此

    - 不存在"端点读完、线程开跑之间 checkpoint 被 LRU 淘汰"的竞态；
    - 不存在两份判据不一致的割裂契约。早期实现端点只查 base_resume、
      线程还要求 jd_requirements：一个"有 base_resume 但无 jd_requirements"
      的 checkpoint 会让 POST 返回 200（任务已启动），随后 SSE 才报错。
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state: StateSnapshot | None = build_workflow().get_state(config)
    values = state.values if state else {}
    base_resume = values.get("base_resume", "")
    jd_reqs = values.get("jd_requirements")
    if not base_resume or jd_reqs is None:
        return None
    return _CustomizeContext(
        base_resume=base_resume,
        jd_reqs=jd_reqs,
        notifications=values.get("notifications"),
        user_supplement=values.get("user_supplement", ""),
    )


def _run_customize_task(
    task_id: str, task: CustomizeTask, context: _CustomizeContext
) -> None:
    """后台线程：流式定制 → 事件入队。

    上下文由端点读取并校验后传入（见 _load_customize_context），线程自身
    不触碰 checkpoint，因此这里没有"状态不可用"的失败分支。
    LLM 调用失败时降级 base_resume（与同步路径降级语义一致）；
    任务结束（成功或降级）后投递哨兵 None，SSE 端点据此结束。
    """
    base_resume = context.base_resume
    try:
        # 攒批次再入队：token 逐块入队会让 SSE 端点频繁唤醒
        buf: list[str] = []

        def flush() -> None:
            if buf:
                task.emit(("token", "".join(buf)))
                buf.clear()

        for kind, payload in stream_customize_for_jd(
            context.base_resume,
            context.jd_reqs,
            notifications=context.notifications,
            user_supplement=context.user_supplement,
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
                task.emit(("done", task.result))
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
        task.emit(("done", task.result))
    finally:
        task.done = True
        task.emit(None)  # 哨兵：SSE 端点据此结束


@app.post("/api/resume/customize/stream")
async def start_customize_stream(req: CustomizeRequest) -> dict:
    """提交异步流式定制任务，立即返回 task_id（不等待生成）。

    流程：校验 checkpoint 可用 → 后台线程开始流式定制 → 返回 task_id。
    前端用 GET /api/resume/customize/stream/{task_id} 订阅 SSE 流；
    连接中断不影响任务执行，完成后可轮询
    GET /api/resume/customize/result/{task_id} 取结果。
    """
    try:
        context = await asyncio.to_thread(_load_customize_context, req.thread_id)
        if context is None:
            raise HTTPException(
                404,
                f"无法恢复工作流：thread_id={req.thread_id} 没有可用的 checkpoint。"
                "请先调用 generate。",
            )

        task_id = uuid.uuid4().hex[:12]
        task = CustomizeTask(req.thread_id)
        _register_task(task_id, task)
        threading.Thread(
            target=_run_customize_task, args=(task_id, task, context), daemon=True
        ).start()
        logger.info("启动流式定制任务：task=%s thread=%s", task_id, req.thread_id)
        return {"task_id": task_id, "thread_id": req.thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("启动流式定制失败")
        raise HTTPException(500, sanitize_error(e)) from e


@app.get("/api/resume/customize/stream/{task_id}")
async def consume_customize_stream(task_id: str) -> StreamingResponse:
    """消费任务的 SSE 事件流：逐 token 推送 messages/partial（对齐 LangGraph 标准）。

    事件序列：
        event: messages/partial
        data: {"messages": [{"type": "AIMessageChunk", "content": "..."}]}

        事件: done
        data: {"thread_id": ..., "customized_resume": "...", "token_usage": {...},
               "failed": false}

    空闲超过 _SSE_HEARTBEAT_SECONDS 未产出 token 时发一行 SSE 注释（": ..."）
    保活——代理会掐断长时间静默的连接，而前端的解析器天然忽略注释行。

    客户端断开连接不影响后台任务；任务完成后可轮询
    GET /api/resume/customize/result/{task_id} 取结果。
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, f"任务不存在：{task_id}")

    async def event_stream():
        last_event_at = time.monotonic()
        while True:
            try:
                event = task.events.get_nowait()
            except queue.Empty:
                if task.done:
                    # 任务已结束且事件已取完（哨兵可能被重复订阅的消费者取走）
                    return
                if time.monotonic() - last_event_at >= _SSE_HEARTBEAT_SECONDS:
                    last_event_at = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(_SSE_POLL_SECONDS)
                continue
            last_event_at = time.monotonic()
            if event is None:
                return
            kind, payload = event
            if kind == "token":
                yield _sse_event(
                    "messages/partial",
                    {"messages": [{"type": "AIMessageChunk", "content": payload}]},
                )
            elif kind == "done":
                yield _sse_event("done", payload)
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
    }


# ============================================================
# 面试问答（内存对话历史；第 14 课接 PostgresSaver 持久化）
# ============================================================

# 单个会话保留的问答轮数上限（1 轮 = 用户提问 + 面试官回答，共 2 条消息）。
# 只此一个常量，存多少就喂多少——早期实现是"存 12 轮、只回喂 6 轮"的
# 双重上限，两处口径不一致，看代码无法判断实际记忆深度。
# 想加深记忆只改这一个数（注意 token 成本随轮数线性增长）。
_INTERVIEW_MAX_TURNS = 6
# 会话（thread）数量上限：每个 thread 的历史永久驻留内存，需按 LRU 回收
_MAX_INTERVIEW_THREADS = 50

_interviews: OrderedDict[str, list[dict]] = OrderedDict()
_interviews_lock = threading.Lock()


def _get_interview_history(thread_id: str) -> list[dict]:
    """取会话历史快照（锁内只拷贝，LLM 调用在锁外——避免持锁阻塞其他请求）。"""
    with _interviews_lock:
        history = list(_interviews.get(thread_id, []))
        if thread_id in _interviews:
            _interviews.move_to_end(thread_id)
    return history


def _append_interview_turns(thread_id: str, question: str, answer: str) -> int:
    """写回一轮问答并返回保留的轮数；超限的最旧会话按 LRU 淘汰。"""
    with _interviews_lock:
        history = _interviews.setdefault(thread_id, [])
        _interviews.move_to_end(thread_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        keep = _INTERVIEW_MAX_TURNS * 2
        if len(history) > keep:
            del history[: len(history) - keep]
        turns = len(history) // 2
        # thread_id 已在队尾，淘汰队首的旧会话即可，不会误删当前会话
        while len(_interviews) > _MAX_INTERVIEW_THREADS:
            _interviews.popitem(last=False)
    return turns

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

    history = _get_interview_history(thread_id)

    messages = [
        SystemMessage(INTERVIEW_SYSTEM_PROMPT),
        HumanMessage(content=f"候选人简历：\n{req.resume_text}"),
    ]
    # 历史在写入时已按 _INTERVIEW_MAX_TURNS 截断，这里全部回喂——
    # 存多少喂多少，不再有第二重隐式截断
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=req.question))

    try:
        response = await asyncio.to_thread(llm.invoke, messages)
        content = response.content
    except Exception as e:
        logger.exception("面试问答失败：thread=%s", thread_id)
        raise HTTPException(500, sanitize_error(e)) from e

    # AIMessage.content 声明为 str | list[内容块]（多模态），本场景只接受纯文本。
    # 不用 str(content) 兜底——那会把内容块列表变成一串 Python repr 混进回答。
    if not isinstance(content, str) or not content.strip():
        logger.warning("面试官返回非文本或空内容：thread=%s", thread_id)
        raise HTTPException(502, "面试官未返回有效回答，请稍后重试。")
    answer = content

    # LLM 调用完成后才写回历史（失败不写，下次重发同问）
    turns = _append_interview_turns(thread_id, req.question, answer)

    logger.info("面试问答：thread=%s，%d 轮", thread_id, turns)
    return {"thread_id": thread_id, "answer": answer}
