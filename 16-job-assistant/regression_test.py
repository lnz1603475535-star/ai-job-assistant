"""
回归测试 — 代码审查缺陷锁定
============================
每个用例对应 2026-09-12 代码审查中发现并修复的一个缺陷。
全部为纯本地断言，**零 LLM 调用**，秒级跑完，可随时回归。

运行：python regression_test.py

与其他测试的分工：
    terminal_test.py   工作流全链路（含 LLM，慢）
    api_test.py        FastAPI 端点全链路（含 LLM，慢）
    regression_test.py 本次修复的缺陷点（无 LLM，快）——改检索/预算/导出先跑这个
"""

import io as _io
import os
import sys
import threading
from typing import Any

# 输出编码加固：Windows 控制台重定向到文件时默认 GBK+strict，
# 遇到 ⚠️ 等非 GBK 字符会直接抛 UnicodeEncodeError 中断测试
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, _io.TextIOWrapper):
        _stream.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from core import setup_logging

setup_logging()

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import app as app_module
import exporters
import workflow
from core import DEFAULT_TOKEN_BUDGET, RetrievalService, TokenBudget, _tokenize
from resume_engine import (
    TokenBudgetMiddleware,
    _extract_agent_token_usage,
    _is_transient_error,
)

_results: list[tuple[str, bool, str]] = []


def check(desc: str, ok: bool, detail: str = "") -> None:
    """记录一条断言结果。"""
    _results.append((desc, bool(ok), detail))


class _StubVectorStore:
    """桩向量库：按固定顺序返回指定块，用于隔离 FAISS 分支。"""

    def __init__(self, chunks: list[Document], order: list[int]):
        self._chunks = chunks
        self._order = order
        self.last_k = 0

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self.last_k = k
        return [self._chunks[i] for i in self._order[:k]]


# ============================================================
# 缺陷 ③：BM25 零分块挤占 RRF 名额
# ============================================================


def test_bm25_zero_score_no_pollution() -> None:
    print("\n" + "─" * 60)
    print("[1/8] 缺陷③：BM25 零分块不得挤占检索名额")
    print("─" * 60)

    chunks = [
        Document(
            page_content=f"无关块{i}与查询毫无关系的内容",
            metadata={"_chunk_idx": i, "doc_type": "user_experience"},
        )
        for i in range(40)
    ]
    # FAISS 判定的语义相关顺序：20 最相关，依次递减
    order = list(range(20, 40))
    svc = RetrievalService(_StubVectorStore(chunks, order), chunks)

    # 查询与语料零 token 重叠 → BM25 全部 0 分
    query = "量子光刻机半导体"
    assert svc._bm25_index is not None  # 构造场景下必有 BM25 索引
    scores = svc._bm25_index.get_scores(_tokenize(query))
    check("BM25 分数确实全为 0（构造前提成立）", all(float(s) == 0 for s in scores))

    result = svc.search(query, k=4, doc_types=["user_experience"])
    returned = [
        int(block.split("无关块")[1].split("与")[0])
        for block in result.split("\n\n---\n\n")
    ]
    check(
        "BM25 全零分时，top-4 严格等于 FAISS 语义 top-4",
        returned == [20, 21, 22, 23],
        f"实际返回 {returned}（修复前为 [20, 0, 21, 1]——一半名额被无关块占用）",
    )


# ============================================================
# 缺陷 ②：按需检索丢掉姓名/学历
# ============================================================


def test_experience_head_keeps_identity() -> None:
    print("\n" + "─" * 60)
    print("[2/8] 缺陷②：按需检索必须保留头部区（姓名/学历/技能）")
    print("─" * 60)

    bank = (
        "# 基本信息\n- 姓名: 李思\n- 邮箱: lisi@email.com\n"
        "- 学历: 浙江大学 计算机工程 本科 2016-2020\n\n"
        "# 技能\n\nPython, Django, FastAPI\n\n"
        + "\n".join(f"## 项目：第{i}个项目\n- 时间：2023.0{i%9}-2024.0{i%9}\n"
                    f"- 角色：后端开发\n- 技术栈：Python\n- 详情：\n  做了很多事情。"
                    for i in range(40))
    )
    head = workflow._experience_head(bank)

    check("头部区含姓名", "李思" in head)
    check("头部区含学历", "浙江大学" in head)
    check("头部区含技能段", "Python, Django, FastAPI" in head)
    check("头部区按行截断，不切碎字段", not head.endswith("做了很多事"))
    check(
        "头部区长度受控（不超过上限）",
        len(head) <= workflow._EXPERIENCE_HEAD_CHARS,
        f"实际 {len(head)} 字符",
    )

    # 模拟按需检索路径：检索结果 + 头部区拼装
    retrieved = "[user_experience] ## 项目：第3个项目\n- 时间：2023.03-2024.03"
    context = f"{head}\n\n=== 与岗位相关的经历 ===\n{retrieved}"
    check("按需检索后的上下文仍含姓名", "李思" in context)
    check("按需检索后的上下文仍含学历", "浙江大学" in context)


# ============================================================
# 缺陷 ③ 附带：分词清洗
# ============================================================


def test_tokenize_cleans_tokens() -> None:
    print("\n" + "─" * 60)
    print("[3/8] 分词清洗：空格/标点不得进入 BM25 打分")
    print("─" * 60)

    tokens = _tokenize("Python，Docker 微服务（高并发）")
    check("过滤掉空格 token", " " not in tokens)
    check("过滤掉标点 token", not any(t in "，（）" for t in tokens))
    check("统一小写", "python" in tokens and "Python" not in tokens)
    check(
        "中文词元原样保留（jieba 切分结果不丢）",
        "服务" in tokens and "并发" in tokens,
        f"实际 {tokens}",
    )


# ============================================================
# 缺陷 ⑫：TokenBudget 必须能真正注入提醒（而非事后统计）
# ============================================================


def test_token_budget_levels() -> None:
    print("\n" + "─" * 60)
    print("[4/8] TokenBudget：分级提醒每级只触发一次")
    print("─" * 60)

    budget = TokenBudget(max_tokens=1000, warning_ratio=0.7)
    check("初始级别为 0", budget.level == 0)
    check("初始无提醒", budget.get_warning() == "")

    budget.record(input_tokens=700, output_tokens=0)
    check("达警戒线后级别为 1", budget.level == 1)
    first = budget.get_warning()
    check("警戒提醒非空", bool(first))
    check("同一级别不重复提醒", budget.get_warning() == "")

    budget.record(input_tokens=250, output_tokens=0)  # 累计 950/1000
    check("接近耗尽后级别为 2", budget.level == 2)
    check("升级时应再次提醒", bool(budget.get_warning()))
    check("升级后同样只提醒一次", budget.get_warning() == "")

    check("展示用提示不消耗级别", bool(budget.get_status_notice()))

    # 中间件依赖的 state 往返
    restored = TokenBudget(max_tokens=1000)
    restored.restore(
        input_tokens=budget.input_tokens,
        output_tokens=budget.output_tokens,
        notified_level=budget.notified_level,
    )
    check(
        "restore 可完整复现累计状态",
        restored.total_tokens == budget.total_tokens
        and restored.notified_level == budget.notified_level,
    )

    usage = {"prompt_tokens": 900, "completion_tokens": 100}
    check(
        "from_usage 可构造展示用预算",
        TokenBudget.from_usage(usage, max_tokens=1000).usage_ratio == 1.0,
    )


def test_agent_token_usage_from_state() -> None:
    print("\n" + "─" * 60)
    print("[5/8] 缺陷13：token 用量须含工具调用轮次")
    print("─" * 60)

    state = {"token_input": 1234, "token_output": 567}
    usage = _extract_agent_token_usage(state)
    check("从中间件累积的 state 读取", usage == {
        "prompt_tokens": 1234, "completion_tokens": 567,
    })
    check("缺字段时安全降级", _extract_agent_token_usage({}) == {
        "prompt_tokens": 0, "completion_tokens": 0,
    })


def test_token_budget_middleware_mechanism() -> None:
    """直接驱动中间件（零 LLM）：验证"记账 + 注入提醒"的完整回路。

    这是缺陷⑫的核心——预算必须真正把提醒送进对话，而不只是事后统计。
    """
    print("\n" + "─" * 60)
    print("[5b/8] Token 预算中间件：逐轮记账 + 超标注入提醒")
    print("─" * 60)

    middleware = TokenBudgetMiddleware()
    check(
        "预算上限与 TokenBudget 默认值一致（避免两处漂移）",
        middleware._max_tokens == TokenBudget().max_tokens
        == DEFAULT_TOKEN_BUDGET,
    )

    # 第 1 轮：一次昂贵的模型调用（模拟含检索的定制轮）
    round1 = AIMessage(
        content="...",
        usage_metadata={
            "input_tokens": 15000, "output_tokens": 2000, "total_tokens": 17000,
        },
    )
    update1 = middleware.after_model({"messages": [round1]}, None) or {}
    check(
        "after_model 按轮累积真实用量",
        update1 == {"token_input": 15000, "token_output": 2000},
        f"实际 {update1}",
    )

    state: dict[str, Any] = {"messages": [round1], **update1}
    check(
        "未达警戒线时不打扰模型",
        middleware.before_model(state, None) is None,
    )

    # 第 2 轮：再来一轮，累计跨过警戒线（20000+3000 = 23000 / 30000 ≈ 77%）
    round2 = AIMessage(
        content="...",
        usage_metadata={
            "input_tokens": 5000, "output_tokens": 1000, "total_tokens": 6000,
        },
    )
    update2 = middleware.after_model({**state, "messages": [round1, round2]}, None) or {}
    check(
        "第二轮用量累加到第一轮之上（不漏算工具调用轮）",
        update2 == {"token_input": 20000, "token_output": 3000},
        f"实际 {update2}",
    )

    state.update(update2)
    injected = middleware.before_model(state, None) or {}
    injected_messages = injected.get("messages") or []
    check(
        "突破警戒线后向对话注入提醒（原实现永远做不到）",
        bool(injected_messages),
    )
    check(
        "注入的是提醒文本",
        bool(injected_messages) and "Token" in injected_messages[0].content,
    )
    check(
        "级别记忆写回 state（供下一轮判断）",
        injected.get("token_warned_level") == 1,
        f"实际 {injected.get('token_warned_level')}",
    )

    state["messages"] = [*state["messages"], *injected_messages]
    state["token_warned_level"] = injected.get("token_warned_level")
    check(
        "同一级别不重复注入（避免刷屏式提醒）",
        middleware.before_model(state, None) is None,
    )


# ============================================================
# 缺陷 ⑪：瞬时错误判断改用异常类型
# ============================================================


def test_transient_error_detection() -> None:
    print("\n" + "─" * 60)
    print("[6/8] 缺陷11：瞬时错误按异常类型判断")
    print("─" * 60)

    import httpx
    import openai

    _req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    check(
        "openai 超时 → 瞬时",
        _is_transient_error(openai.APITimeoutError(request=_req)),
    )
    check("httpx 连接错误 → 瞬时", _is_transient_error(httpx.ConnectError("boom")))
    check("业务异常 → 非瞬时", not _is_transient_error(ValueError("字段缺失")))

    class _FakeStatusError(Exception):
        status_code = 503

    check("带 503 状态码 → 瞬时", _is_transient_error(_FakeStatusError()))
    check(
        "错误文本里恰好含 429 的无关数字不再误判",
        not _is_transient_error(ValueError("token budget 1429 exceeded")),
    )
    check(
        "关键词兜底仍生效（类型信息丢失时）",
        _is_transient_error(ValueError("Connection reset by peer")),
    )


# ============================================================
# 缺陷 ⑦：SSE error 事件必须被消费
# ============================================================


def test_sse_parser_handles_error_event() -> None:
    print("\n" + "─" * 60)
    print("[7/8] 缺陷⑦：SSE 解析须识别 event 名（含 error）")
    print("─" * 60)

    lines = [
        "event: messages/partial",
        'data: {"messages": [{"type": "AIMessageChunk", "content": "你好"}]}',
        "",
        "event: messages/partial",
        'data: {"messages": [{"type": "AIMessageChunk", "content": "世界"}]}',
        "",
        "event: done",
        'data: {"customized_resume": "你好世界", "failed": false}',
        "",
    ]
    events = list(app_module._parse_sse_lines(lines))
    check("解析出全部事件", len(events) == 3, f"实际 {len(events)}")
    check("事件名正确", [name for name, _ in events] == [
        "messages/partial", "messages/partial", "done",
    ])
    check("载荷正确", events[-1][1]["customized_resume"] == "你好世界")

    err_lines = ["event: error", 'data: {"detail": "checkpoint 缺失"}', ""]
    err_events = list(app_module._parse_sse_lines(err_lines))
    check(
        "error 事件被识别（修复前被静默丢弃 → 前端白等 4 分钟）",
        bool(err_events) and err_events[0][0] == "error",
    )

    bad_lines = ["event: done", "data: {不是合法 JSON", "", "event: done",
                 'data: {"ok": true}', ""]
    bad_events = list(app_module._parse_sse_lines(bad_lines))
    check("非法 JSON 跳过且不中断后续事件", len(bad_events) == 1)

    # 服务端心跳（": keep-alive" 注释行）必须被客户端解析器忽略，
    # 否则前端会把保活信号当成事件——这是新增心跳后的跨端契约
    hb_events = list(
        app_module._parse_sse_lines(
            [": keep-alive", "", 'data: {"messages": []}', "", ": keep-alive", ""]
        )
    )
    check(
        "心跳注释行被客户端解析器忽略（不产生伪事件）",
        hb_events == [("", {"messages": []})],
        f"实际 {hb_events}",
    )

    # 字节行（resp.iter_lines 在部分环境下产出 bytes）
    byte_events = list(app_module._parse_sse_lines([b'data: {"ok": 1}', b""]))
    check("兼容 bytes 行", byte_events == [("", {"ok": 1})])


# ============================================================
# 缺陷 ㉗ / ㉛：导出层
# ============================================================


def test_job_target_scan_and_classifier() -> None:
    print("\n" + "─" * 60)
    print("[8/8] 缺陷27/31：求职意向只扫头部 + 两条导出路径共用行解析")
    print("─" * 60)

    head_has = ["# 张三", "求职意向：Python 后端", "邮箱：a@b.com"]
    check("头部写明求职意向 → 抑制顶栏重复", exporters._hides_job_target(head_has))

    body_only = ["# 张三", "", "## 工作经历"]
    body_only += [f"- 第{i}条经历描述" for i in range(20)]
    body_only.append("- 深入了解目标岗位要求，主动对齐业务")
    check(
        "正文深处出现「目标岗位」不抑制顶栏（修复前会误伤）",
        not exporters._hides_job_target(body_only),
    )

    cases = {
        "## 专业技能": ("h2", "专业技能", 0),
        "### ABC 科技 - 后端": ("h3", "ABC 科技 - 后端", 0),
        "- **Python** 是主力": ("list", "**Python** 是主力", 0),
        "  - 嵌套项": ("list", "嵌套项", 2),
        "": ("blank", "", 0),
        "---": ("hr", "", 0),
        "联系方式：a@b.com": ("text", "联系方式：a@b.com", 0),
    }
    for line, expected in cases.items():
        check(f"行分类 {line!r}", exporters._classify_line(line) == expected,
              f"实际 {exporters._classify_line(line)}")

    # 分类器保留行内标记，PDF 侧自行剥离
    _kind, content, _indent = exporters._classify_line("- **加粗**内容")
    check("PDF 路径剥离行内标记", exporters._strip_inline_format(content) == "加粗内容")


# ============================================================
# 缺陷 ⑧：checkpoint 内存回收
# ============================================================


def test_thread_registry_eviction() -> None:
    print("\n" + "─" * 60)
    print("[附] 缺陷⑧：checkpoint 按 LRU 回收，不再无限增长")
    print("─" * 60)

    workflow._tracked_threads.clear()
    for i in range(workflow._MAX_TRACKED_THREADS + 10):
        workflow.register_thread(f"regression-thread-{i}")

    check(
        f"登记数被限制在 {workflow._MAX_TRACKED_THREADS} 以内",
        len(workflow._tracked_threads) == workflow._MAX_TRACKED_THREADS,
        f"实际 {len(workflow._tracked_threads)}",
    )
    check(
        "最旧的 thread 已被淘汰",
        "regression-thread-0" not in workflow._tracked_threads,
    )
    check(
        "最新的 thread 仍在册",
        f"regression-thread-{workflow._MAX_TRACKED_THREADS + 9}"
        in workflow._tracked_threads,
    )
    workflow._tracked_threads.clear()


# ============================================================
# SSE 流式：asyncio.Queue + 跨线程投递 + 心跳
# ============================================================


def test_customize_task_cross_thread_emit() -> None:
    """零 LLM 直接驱动 SSE 生成器：验证跨线程投递、心跳与哨兵收尾。

    这是"方案甲"的核心风险点——后台线程通过 loop.call_soon_threadsafe
    把事件投递给 SSE 端点，投递路径错了会表现为"前端一直空转"。
    """
    print("\n" + "─" * 60)
    print("[附] SSE：跨线程投递 + 心跳 + 哨兵收尾")
    print("─" * 60)

    import asyncio
    from typing import cast

    import api

    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        # 缩短心跳间隔，让测试在秒级内观察到心跳
        original_hb = api._SSE_HEARTBEAT_SECONDS
        api._SSE_HEARTBEAT_SECONDS = 0.2

        def open_stream(task_id: str, task: api.CustomizeTask) -> Any:
            """注册任务并打开它的 SSE 生成器，返回 (task, 取下一块的函数)。"""
            with api._tasks_lock:
                api._tasks[task_id] = task
            resp = asyncio.run_coroutine_threadsafe(
                api.consume_customize_stream(task_id), loop
            ).result(timeout=10)
            stream = cast(Any, resp.body_iterator)

            def pull(timeout: float = 5.0) -> str | None:
                """取下一块；StopAsyncIteration 表示流已结束。"""
                try:
                    return asyncio.run_coroutine_threadsafe(
                        stream.__anext__(), loop
                    ).result(timeout=timeout)
                except StopAsyncIteration:
                    return None

            return pull

        # ── 场景 1：心跳 → token → done → 哨兵 ──
        task = api.CustomizeTask("regression-thread")
        pull = open_stream("regression-sse-task", task)

        first = pull()
        check("空闲时产出 SSE 心跳注释行", first is not None and first.startswith(":"),
              f"实际 {first!r}")

        # 从**测试线程**投递事件（真正的跨线程路径）
        task.emit(("token", "你好"))
        second = pull()
        check(
            "跨线程投递的 token 事件送达 SSE 端点",
            second is not None
            and second.startswith("event: messages/partial")
            and "你好" in second,
            f"实际 {second!r}",
        )

        task.emit(("done", {"customized_resume": "你好世界", "failed": False}))
        third = pull()
        check("done 事件送达", third is not None and third.startswith("event: done"),
              f"实际 {third!r}")

        task.done = True
        task.emit(None)  # 哨兵
        check("哨兵送达后生成器结束（不挂起）", pull() is None)

        # 注：error 事件通道已随"唯一判定点"重构删除——端点校验不过时直接
        # 返回 404，不存在"任务已启动后才报错"的路径，该事件没有生产者。
        # 详见 test_customize_context_single_source_of_truth。

        api._SSE_HEARTBEAT_SECONDS = original_hb
    finally:
        with api._tasks_lock:
            api._tasks.pop("regression-sse-task", None)
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=5)
        loop.close()


def test_customize_context_single_source_of_truth() -> None:
    """定制的"能否进行"只有一处判定、一次读取。

    早期实现端点与后台线程各读一次 checkpoint、判据还不一样（端点只查
    base_resume，线程还要 jd_requirements）——会出现 POST 返回 200（任务已
    启动）随后 SSE 才报错的割裂契约。现由 _load_customize_context 统一读取
    与校验，线程只消费结果，因此也不再需要"状态不可用"的失败分支。
    """
    print("\n" + "─" * 60)
    print("[附] 定制上下文：唯一判定点，一次读取")
    print("─" * 60)

    import api

    check(
        "未知 thread → 返回 None（端点据此返回 404）",
        api._load_customize_context("regression-no-such-thread") is None,
    )
    check(
        "旧的独立读取函数已并入判定点（防止出现第二处读取）",
        not hasattr(api, "_get_workflow_state"),
    )
    check(
        "后台线程不再持有 error 字段（该分支已无触发条件）",
        not hasattr(api.CustomizeTask("regression-x"), "error"),
    )


def test_customize_task_decoupled_from_event_loop() -> None:
    """投递事件不得依赖事件循环（TestClient 场景暴露的缺陷）。

    中途尝试过的实现让后台线程 loop.call_soon_threadsafe 投递，需要"消费端正
    跑着的那个循环"——而 TestClient 每个请求一个 portal 循环，POST 返回后循环
    即结束，投递抛 RuntimeError 被静默吞掉，SSE 一个事件都收不到
    （api_test 抓到：done 事件永远不来）。现在 emit 只依赖 queue.Queue，
    本用例在没有运行中事件循环的普通同步线程里验证这一点。
    """
    print("\n" + "─" * 60)
    print("[附] SSE：投递事件不依赖事件循环")
    print("─" * 60)

    import api

    detail = ""
    task: Any = None
    try:
        # 本线程是普通同步上下文——没有运行中的事件循环
        task = api.CustomizeTask("regression-no-loop")
        task.emit(("token", "片段"))
        task.emit(None)
        ok = True
    except Exception as exc:  # 记录后由断言判定
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    check("无事件循环的线程里可构造并投递事件", ok, detail)
    if ok and task is not None:
        check(
            "事件按序进入队列（普通队列，无循环绑定）",
            task.events.get_nowait() == ("token", "片段")
            and task.events.get_nowait() is None,
        )


def test_classify_index_error() -> None:
    """索引错误分类是纯函数：判据只看文本，不看异常类型。

    早期实现是两条 if/elif 链分处两个 except 块（ValueError 一条、Exception
    一条），等于"用哪个错误码"取决于异常是哪个类。合并后由内容决定，
    并且第一次可以脱离 Streamlit 直接单测。
    """
    print("\n" + "─" * 60)
    print("[附] 索引错误分类：纯函数 + 与文案表同步")
    print("─" * 60)

    cases = [
        (ValueError("文件不存在或无法访问：a.pdf"), "missing"),
        (ValueError("文件已加密，无法读取：a.pdf"), "pdf_encrypted"),
        (ValueError("PDF 可能是扫描件，无法提取文字。"), "pdf_scanned"),
        (ValueError("文件已损坏或格式异常：a.pdf"), "pdf_corrupted"),
        (Exception("no text could be extracted"), "empty"),
        (Exception("Connection error while downloading model"), "network"),
        (ValueError("完全无法归类的错误"), "unknown"),
    ]
    for exc, expect in cases:
        got = app_module._classify_index_error(exc)
        check(
            f"「{str(exc)[:26]}」→ {expect}",
            got == expect,
            f"实际 {got}",
        )

    # 核心性质：同一段文本换异常类，分类结果不变（旧实现会变）
    check(
        "判据与异常类型解耦（同一文本换类结果不变）",
        app_module._classify_index_error(ValueError("Connection reset")) == "network"
        and app_module._classify_index_error(OSError("Connection reset")) == "network",
    )
    # UnicodeDecodeError 的真实现文本形如 "'utf-8' codec can't decode byte..."
    check(
        "真实解码错误文本能归到 encoding（旧关键词漏了 codec/decode）",
        app_module._classify_index_error(
            UnicodeDecodeError("utf-8", b"\xc0", 88, 89, "invalid start byte")
        )
        == "encoding",
    )

    # 两表同步守卫：分类码必须都在文案表里有对应项
    codes = {
        app_module._classify_index_error(exc) for exc, _ in cases
    } | {"encoding"}  # encoding 由上一条覆盖
    missing_keys = codes - set(app_module._INDEX_ERROR_MESSAGES)
    check(
        "分类码集合 ⊆ 文案表键集合（防新增错误码掉进 unknown）",
        not missing_keys,
        f"文案表缺少：{missing_keys}",
    )


def test_upload_guard_and_cleanup() -> None:
    """上传入口的两处有界化：落盘前拦大小、启动时清过期文件。"""
    print("\n" + "─" * 60)
    print("[附] 上传：大小前置拦截 + 过期文件清理")
    print("─" * 60)

    import asyncio
    import time as _time

    # 必须用 FastAPI 的 UploadFile（Starlette 的是其父类，方向不可逆，
    # 传入父类实例 pyright 会报类型不匹配）
    from fastapi import UploadFile as FastAPIUploadFile

    import api

    limit = api._MAX_UPLOAD_BYTES

    # 1) 超限文件必须在 read() 之前被拒（端点把 ValueError 映射成 400）
    oversized = FastAPIUploadFile(
        file=_io.BytesIO(b""), size=limit + 1, filename="too-big.pdf"
    )
    detail = ""
    try:
        asyncio.run(api._save_upload(oversized, "regression"))
        rejected = False
    except ValueError as exc:
        rejected = "过大" in str(exc)
        detail = str(exc)[:80]
    check("超限上传在读取前被拒（ValueError → 400）", rejected, detail)

    # 2) 恰好等于上限应放行（边界不能误伤）
    exact = FastAPIUploadFile(
        file=_io.BytesIO(b"hello"), size=limit, filename="exact.txt"
    )
    path = asyncio.run(api._save_upload(exact, "regression"))
    check("恰好等于上限放行并落盘", os.path.exists(path))
    if os.path.exists(path):
        os.remove(path)

    # 3) size 未知（None）时不拦截——交给加载阶段兜底
    unknown = FastAPIUploadFile(
        file=_io.BytesIO(b"x"), size=None, filename="unknown.txt"
    )
    path2 = asyncio.run(api._save_upload(unknown, "regression"))
    check("size 未知时放行（由 core 加载阶段兜底）", os.path.exists(path2))
    if os.path.exists(path2):
        os.remove(path2)

    # 4) 清理：只删过期文件，保留未过期的（多 worker 启动时不能误删）
    old_file = os.path.join(api.UPLOAD_DIR, "regression-old.txt")
    fresh_file = os.path.join(api.UPLOAD_DIR, "regression-fresh.txt")
    for p in (old_file, fresh_file):
        with open(p, "wb") as f:
            f.write(b"x")
    stale = _time.time() - (api._UPLOAD_MAX_AGE_HOURS + 1) * 3600
    os.utime(old_file, (stale, stale))

    api._cleanup_stale_uploads()
    check("过期上传文件被清理", not os.path.exists(old_file))
    check(
        "未过期文件保留（避免误删其他 worker 正在用的文件）",
        os.path.exists(fresh_file),
    )
    if os.path.exists(fresh_file):
        os.remove(fresh_file)


def test_merge_customize_result_keeps_context() -> None:
    """API 定制成功后，基础简历等上下文不得丢失（两条路径必须同构）。"""
    print("\n" + "─" * 60)
    print("[附] 定制结果合并：API 路径不丢基础简历")
    print("─" * 60)

    paused = {
        "base_resume": "基础简历正文",
        "user_profile": "PROFILE",
        "style_profile": "STYLE",
        "jd_requirements": "JD",
        "_interrupted": True,  # 暂停时是 True，定制完成后不应带进结果
    }
    done = {
        "customized_resume": "定制后的正文",
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "failed": False,
    }

    merged = app_module._merge_customize_result(done, paused)
    check("保留基础简历", merged["base_resume"] == "基础简历正文")
    check("保留解析结果（用户/风格/JD）",
          (merged["user_profile"], merged["style_profile"], merged["jd_requirements"])
          == ("PROFILE", "STYLE", "JD"))
    check("定制简历来自后端结果", merged["customized_resume"] == "定制后的正文")
    check("token 用量透传", merged["token_usage"]["prompt_tokens"] == 100)
    check("不带 _interrupted（定制已完成，不该提示未经审核）",
          "_interrupted" not in merged)

    # UI 消费的键集合。注意：本地路径（resume_workflow 返回的完整 WorkflowState）
    # 是它的**超集**（还含 user_text / sample_resume_path 等只应留在服务端的键），
    # 所以下面的相等断言只约束 API 路径：
    #   不少 → 否则某条路径会显示为空；不多 → 不泄漏服务端字段
    ui_keys = {
        "base_resume", "customized_resume", "user_profile",
        "style_profile", "jd_requirements", "token_usage", "notifications",
    }
    check(
        "API 路径产出恰好等于 UI 所读的键（不少：不显示为空；不多：不泄漏）",
        set(merged) == ui_keys,
        f"多出 {set(merged) - ui_keys}，缺少 {ui_keys - set(merged)}",
    )
    # "本地路径是超集"这个真约束，用真方式表达：UI 读的键必须都在 state 声明里
    state_keys = set(workflow.WorkflowState.__annotations__)
    check(
        "UI 所读的键都在本地路径的 state 声明里（两条路径都满足 UI）",
        ui_keys <= state_keys,
        f"本地 state 缺少：{ui_keys - state_keys}",
    )

    # paused 缺失时不得崩（例如用户刷新后 session_state 被重建）
    fallback = app_module._merge_customize_result(done, None)
    check("paused 缺失时安全降级", fallback["base_resume"] is None)

    failed = app_module._merge_customize_result({**done, "failed": True}, paused)
    check("降级时补上失败通知",
          any("定制优化失败" in n for n in failed["notifications"]))


def test_interview_history_single_limit() -> None:
    """面试记忆只有一重上限：存多少就喂多少（曾为"存 12 轮只喂 6 轮"）。"""
    print("\n" + "─" * 60)
    print("[附] 面试记忆：单一上限，存喂一致")
    print("─" * 60)

    import api

    api._interviews.clear()
    thread_id = "regression-interview"
    for i in range(10):
        api._append_interview_turns(thread_id, f"问题{i}", f"回答{i}")

    history = api._get_interview_history(thread_id)
    expected = api._INTERVIEW_MAX_TURNS * 2
    check(
        f"存储被截断到 {api._INTERVIEW_MAX_TURNS} 轮（{expected} 条消息）",
        len(history) == expected,
        f"实际 {len(history)} 条",
    )
    check(
        "保留的是最新一轮（问题9 + 回答9）",
        history[-2]["content"] == "问题9" and history[-1]["content"] == "回答9",
        f"实际末尾两条：{[m['content'] for m in history[-2:]]}",
    )
    check(
        "读路径返回的就是存储的全部（无第二重隐式截断）",
        len(api._get_interview_history(thread_id)) == len(history),
    )
    check(
        "旧的第二个上限常量已不存在（防止双重口径复现）",
        not hasattr(api, "_INTERVIEW_HISTORY_MAX"),
    )
    api._interviews.clear()


# ============================================================
# 缺陷 35：惰性单例加锁引发的死锁
# ============================================================


def test_build_workflow_no_deadlock() -> None:
    """在独立线程里构建工作流，超时即判定死锁。

    build_workflow 持 _init_lock 后又调 get_checkpointer（同一把不可重入的锁）
    曾导致整个工作流永久卡死——这个用例就是防止它回来。
    """
    print("\n" + "─" * 60)
    print("[附] 惰性单例：build_workflow 不得死锁")
    print("─" * 60)

    workflow._compiled_graph = None  # 强制走一次完整的惰性构建路径
    done = threading.Event()
    holder: dict = {}

    def _build() -> None:
        try:
            holder["graph"] = workflow.build_workflow()
        except Exception as exc:  # 记录后由断言统一判定
            holder["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_build, daemon=True)
    worker.start()
    finished = done.wait(timeout=30)

    check(
        "build_workflow 30 秒内返回（不死锁）",
        finished,
        "超时未返回——锁被嵌套获取或循环等待",
    )
    check("无异常抛出", "error" not in holder, str(holder.get("error", ""))[:120])
    check("返回已编译的图", holder.get("graph") is not None)
    check(
        "二次调用返回同一实例（单例生效）",
        workflow.build_workflow() is holder.get("graph"),
    )


# ============================================================
# 主入口
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  AI 简历生成器 — 代码审查缺陷回归测试")
    print("=" * 60)

    test_bm25_zero_score_no_pollution()
    test_experience_head_keeps_identity()
    test_tokenize_cleans_tokens()
    test_token_budget_levels()
    test_agent_token_usage_from_state()
    test_token_budget_middleware_mechanism()
    test_transient_error_detection()
    test_sse_parser_handles_error_event()
    test_job_target_scan_and_classifier()
    test_thread_registry_eviction()
    test_customize_task_cross_thread_emit()
    test_customize_task_decoupled_from_event_loop()
    test_customize_context_single_source_of_truth()
    test_classify_index_error()
    test_upload_guard_and_cleanup()
    test_merge_customize_result_keeps_context()
    test_interview_history_single_limit()
    # 死锁用例必须放在最后：一旦它真死锁，守护线程会一直持有 _init_lock，
    # 后续任何 get_checkpointer() 调用都会跟着阻塞
    test_build_workflow_no_deadlock()

    print("\n" + "=" * 60)
    print("  验证清单")
    print("=" * 60)
    failed = 0
    for desc, ok, detail in _results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        suffix = f"  ← {detail}" if detail and not ok else ""
        print(f"  [{status}] {desc}{suffix}")

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  全部通过！（{len(_results)} 项断言）")
    else:
        print(f"  {failed}/{len(_results)} 项未通过，请检查上述 FAIL 项。")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
