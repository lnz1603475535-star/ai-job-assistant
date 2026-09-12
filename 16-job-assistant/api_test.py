"""
FastAPI 端点终端测试 — 第 13 课
==============================
与 api.py 走相同的调用路径（TestClient 模拟 HTTP 请求）：
用户管理 CRUD → 导出 → 简历生成（断点暂停）→ 流式定制（SSE）→
同步定制（保底）→ 面试问答。

运行：python api_test.py
"""

import io as _io
import json
import os
import sys

# 输出编码加固：Windows 控制台重定向到文件时默认 GBK+strict，
# 遇到 ⚠️ 等非 GBK 字符会直接抛 UnicodeEncodeError 中断测试
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, _io.TextIOWrapper):
        _stream.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from core import setup_logging

setup_logging()

from fastapi.testclient import TestClient

from api import app

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")

client = TestClient(app)


def test_api():
    """第 13 课 FastAPI 端点测试：5 个端点全链路（与 api.py 一致）。"""

    print("=" * 60)
    print("  AI 简历生成器 — 第 13 课 FastAPI 端点测试")
    print("=" * 60)

    # ── 第 1 步：用户管理 CRUD（纯内存，无 LLM 调用）──
    print("\n" + "─" * 60)
    print("[1/6] 用户管理 CRUD...")
    print("─" * 60)

    r = client.post(
        "/api/users",
        json={
            "name": "李思",
            "contact": "lisi@email.com | 13800002222",
            "skills": ["Python", "FastAPI", "Docker"],
            "experience": [],
            "education": "浙江大学 软件工程 本科 2016-2020",
        },
    )
    assert r.status_code == 201, f"创建用户失败：{r.status_code} {r.text}"
    user_id = r.json()["user_id"]
    print(f"  [OK] 创建用户：{user_id}")

    r = client.get(f"/api/users/{user_id}")
    assert r.status_code == 200 and r.json()["name"] == "李思"
    print("  [OK] 查询用户")

    r = client.put(f"/api/users/{user_id}", json={"skills": ["Python", "FastAPI"]})
    assert r.status_code == 200 and r.json()["skills"] == ["Python", "FastAPI"]
    assert r.json()["name"] == "李思", "部分更新不应覆盖未传字段"
    print("  [OK] 部分更新用户（未传字段保持不变）")

    r = client.delete(f"/api/users/{user_id}")
    assert r.status_code == 204
    r = client.get(f"/api/users/{user_id}")
    assert r.status_code == 404
    print("  [OK] 删除用户 + 404 复查")

    # ── 第 2 步：导出端点（无 LLM 调用）──
    print("\n" + "─" * 60)
    print("[2/6] 导出端点（PDF + Word）...")
    print("─" * 60)

    resume_md = "# 李思\n\n## 技能\n\n- Python\n- FastAPI\n"

    r = client.post(
        "/api/resume/export", json={"resume_text": resume_md, "format": "pdf"}
    )
    assert r.status_code == 200, f"PDF 导出失败：{r.status_code} {r.text}"
    assert r.headers["content-type"].startswith("application/pdf")
    assert len(r.content) > 1000
    print(f"  [OK] PDF 导出：{len(r.content)} bytes")

    r = client.post(
        "/api/resume/export",
        json={
            "resume_text": resume_md,
            "format": "docx",
            "job_target": "Python 后端工程师",
        },
    )
    assert r.status_code == 200
    assert len(r.content) > 1000
    print(f"  [OK] Word 导出（含求职意向）：{len(r.content)} bytes")

    r = client.post("/api/resume/export", json={"resume_text": "", "format": "pdf"})
    assert r.status_code == 422, "空简历文本应返回 422"
    r = client.post(
        "/api/resume/export", json={"resume_text": resume_md, "format": "html"}
    )
    assert r.status_code == 422, "不支持的格式应返回 422"
    print("  [OK] 参数校验：空文本/非法格式 → 422")

    # ── 第 3 步：简历生成（上传文件 + 全链路工作流，在断点暂停）──
    print("\n" + "─" * 60)
    print("[3/6] 简历生成（generate，check_parsed 断点暂停）...")
    print("─" * 60)

    user_text = """我叫李思，邮箱 lisi@email.com，电话 13800002222。
技能包括：Python、Django、FastAPI、Docker、MySQL。
工作经历：2022年6月到2025年3月在某互联网公司做 Python 后端开发，
用 FastAPI 重写核心 API，性能提升 3 倍。
学历：浙江大学 软件工程 本科 2016-2020"""

    with (
        open(os.path.join(SAMPLE_DIR, "resume_zhangsan.txt"), "rb") as f1,
        open(os.path.join(SAMPLE_DIR, "jd_python_senior.txt"), "rb") as f2,
    ):
        r = client.post(
            "/api/resume/generate",
            data={"user_text": user_text, "user_supplement": "测试：突出高并发"},
            files={
                "sample_resume": ("resume_zhangsan.txt", f1, "text/plain"),
                "jd": ("jd_python_senior.txt", f2, "text/plain"),
            },
        )
    assert r.status_code == 200, f"生成失败：{r.status_code} {r.text}"
    data = r.json()
    thread_id = data["thread_id"]
    assert data["errors"] == [], f"工作流输入验证失败：{data['errors']}"
    assert data["_interrupted"], "应暂停在 check_parsed 断点"
    assert len(data["base_resume"]) > 100
    print(
        f"  [OK] 生成暂停：thread={thread_id}，base_resume {len(data['base_resume'])} 字符"
    )
    print(
        f"  [OK] 解析结果：{data['user_profile']['name']}，{len(data['user_profile']['skills'])} 项技能"
    )

    # ── 第 4 步：流式定制（异步提交 + SSE + 轮询兜底）──
    print("\n" + "─" * 60)
    print("[4/6] 流式定制（SSE：异步提交 → 事件流 → 轮询兜底）...")
    print("─" * 60)

    # 提交任务（不等待生成，立即返回 task_id）
    r = client.post("/api/resume/customize/stream", json={"thread_id": thread_id})
    assert r.status_code == 200, f"提交流式任务失败：{r.status_code} {r.text}"
    task_id = r.json()["task_id"]
    print(f"  [OK] 提交任务：{task_id}")

    # 消费 SSE 流：messages/partial 事件逐 token + done 事件带结果
    token_events = 0
    streamed_len = 0
    done_payload = None
    with client.stream("GET", f"/api/resume/customize/stream/{task_id}") as resp:
        assert resp.status_code == 200, f"SSE 连接失败：{resp.status_code}"
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if "messages" in payload:
                token_events += 1
                for m in payload["messages"]:
                    streamed_len += len(m.get("content", ""))
            elif "customized_resume" in payload:
                done_payload = payload
    assert done_payload, "SSE 流未收到 done 事件"
    assert not done_payload["failed"], "流式定制不应降级"
    assert len(done_payload["customized_resume"]) > 100
    assert done_payload["token_usage"]["prompt_tokens"] > 0, "token 用量缺失"
    print(
        f"  [OK] SSE 流：{token_events} 个 messages/partial 事件，"
        f"{streamed_len} 字符，token {done_payload['token_usage']}"
    )
    customized_resume = done_payload["customized_resume"]

    # 任务完成后轮询结果接口（断线兜底路径）
    r = client.get(f"/api/resume/customize/result/{task_id}")
    assert r.status_code == 200 and r.json()["status"] == "done"
    assert r.json()["result"]["customized_resume"] == customized_resume
    print("  [OK] result 轮询：任务已完成，结果与 SSE 一致")

    # 无效 task_id / 无效 thread_id → 404
    r = client.get("/api/resume/customize/result/no-such-task")
    assert r.status_code == 404
    r = client.post(
        "/api/resume/customize/stream", json={"thread_id": "no-such-thread"}
    )
    assert r.status_code == 404
    print("  [OK] 无效 task_id / thread_id → 404")

    # ── 第 5 步：同步定制（恢复断点执行，保底路径）──
    print("\n" + "─" * 60)
    print("[5/6] 同步定制（customize，恢复断点执行）...")
    print("─" * 60)

    r = client.post("/api/resume/customize", json={"thread_id": thread_id})
    assert r.status_code == 200, f"定制失败：{r.status_code} {r.text}"
    data = r.json()
    assert len(data["customized_resume"]) > 100
    assert data["token_usage"], "应返回 token 用量"
    print(
        f"  [OK] 定制完成：{len(data['customized_resume'])} 字符，"
        f"token {data['token_usage']}"
    )

    # 无效 thread_id → 404
    r = client.post("/api/resume/customize", json={"thread_id": "no-such-thread"})
    assert r.status_code == 404
    print("  [OK] 无效 thread_id → 404")

    # ── 第 6 步：面试问答（多轮对话）──
    print("\n" + "─" * 60)
    print("[6/6] 面试问答（interview/chat，多轮）...")
    print("─" * 60)

    r = client.post(
        "/api/interview/chat",
        json={
            "resume_text": data["customized_resume"],
            "question": "请介绍一下你的项目经历",
        },
    )
    assert r.status_code == 200, f"面试问答失败：{r.status_code} {r.text}"
    chat = r.json()
    assert chat["answer"]
    chat_thread = chat["thread_id"]
    print(f"  [OK] 第 1 轮回答：{chat['answer'][:60]}...")

    # 第二轮：用返回的 thread_id 保持对话上下文
    r = client.post(
        "/api/interview/chat",
        json={
            "resume_text": data["customized_resume"],
            "question": "追问：你提到的高并发方案具体怎么实现的？",
            "thread_id": chat_thread,
        },
    )
    assert r.status_code == 200 and r.json()["answer"]
    assert r.json()["thread_id"] == chat_thread
    print(f"  [OK] 第 2 轮回答（同 thread 保持上下文）：{r.json()['answer'][:60]}...")

    r = client.post(
        "/api/interview/chat",
        json={"resume_text": "  ", "question": "你好"},
    )
    assert r.status_code == 422, "空简历应返回 422"
    print("  [OK] 参数校验：空简历 → 422")

    r = client.post(
        "/api/interview/chat",
        json={"resume_text": data["customized_resume"], "question": " "},
    )
    assert r.status_code == 422, "空问题应返回 422"
    print("  [OK] 参数校验：空问题 → 422")

    print("\n" + "=" * 60)
    print("  第 13 课 FastAPI 端点测试全部通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
