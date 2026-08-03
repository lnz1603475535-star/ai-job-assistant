"""
终端测试脚本 — Round 5
======================
验证 LangGraph 工作流全链路：索引文档 → 运行工作流（断点暂停）
→ 审核后恢复执行 → JD 定制 → 导出 PDF/Word。
与 app.py 走相同的调用路径。

运行：python terminal_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core import setup_logging
setup_logging()

import core
from core import load_and_index_documents, set_vectorstore
from workflow import run_workflow, resume_workflow

# 使用项目自带的样例文件
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

USER_TEXT = """我叫李思，邮箱 lisi@email.com，电话 13800002222。
技能包括：Python、Django、FastAPI、Docker、MySQL、Redis、Linux。
工作经历：
2022年6月到2025年3月，在某互联网公司做 Python 后端开发，负责订单系统从单体拆分为微服务，
用 FastAPI 重写了核心 API，性能提升了 3 倍，日均处理 200 万订单。
还搭建了 CI/CD 流水线，用 Docker 容器化部署。
2020年7月到2022年5月，在某创业公司做全栈开发，用 Django 写后端，Vue 写前端，
独立负责用户系统和支付模块。
学历：浙江大学 软件工程 本科 2016-2020"""


def test_workflow():
    """Round 4 终端测试：LangGraph 工作流全链路（与 app.py 一致）。"""

    print("=" * 60)
    print("  AI 简历生成器 — Round 4 终端测试")
    print("=" * 60)

    # ── 第 1 步：索引文档 ──
    print("\n" + "─" * 60)
    print("[1/4] 索引文档（FAISS + BM25 双索引）...")
    print("─" * 60)

    sample_resume_path = os.path.join(SAMPLE_DIR, "resume_zhangsan.txt")
    jd_path = os.path.join(SAMPLE_DIR, "jd_python_senior.txt")

    vs, chunks = load_and_index_documents({
        "user_experience": [os.path.join(DATA_DIR, "experience_bank.md")],
        "sample_resume": [sample_resume_path],
        "jd": [jd_path],
    })
    set_vectorstore(vs, chunks)
    print(f"  已索引 {len(chunks)} 个文本块")
    print(f"  FAISS 索引：就绪 | BM25 索引：就绪")

    # ── 第 2 步：运行工作流（在 generate_base 后暂停）──
    print("\n" + "─" * 60)
    print("[2/4] 运行工作流（validate_inputs → extract_style → extract_jd → parse_user → generate_base）...")
    print("─" * 60)

    result = run_workflow(
        user_text=USER_TEXT,
        sample_resume_path=sample_resume_path,
        jd_path=jd_path,
        thread_id="test-round4",
        user_supplement="测试：突出高并发经验，弱化前端",
    )

    errors = result.get("errors", [])
    if errors:
        print(f"\n  [ERROR] 工作流有错误：")
        for e in errors:
            print(f"     - {e}")
        print("\n" + "=" * 60)
        print("  测试中止：输入验证失败。")
        print("=" * 60)
        return

    user = result.get("user_profile")
    style = result.get("style_profile")
    jd_reqs = result.get("jd_requirements")
    base = result.get("base_resume", "")

    print(f"\n  [OK] 输入验证通过")
    if user:
        print(f"  [OK] parse_user：{user.name}，{len(user.skills)} 项技能，{len(user.experience)} 段经历")
    if style:
        fallback_tag = " ⚠️ 默认风格" if style.is_fallback else ""
        print(f"  [OK] extract_style：{style.structure[:50]}...{fallback_tag}")
    if jd_reqs:
        print(f"  [OK] extract_jd：{jd_reqs.title}，{len(jd_reqs.keywords)} 个关键词")
    print(f"  [OK] generate_base：{len(base)} 字符")
    if base:
        print(base[:300] + "..." if len(base) > 300 else base)

    # ── 第 3 步：审核后恢复执行 ──
    print("\n" + "─" * 60)
    print("[3/4] 恢复工作流（check_parsed → customize）...")
    print("─" * 60)

    final_result = resume_workflow(thread_id="test-round4")
    customized = final_result.get("customized_resume", "")
    notifications = final_result.get("notifications", [])

    print(f"\n  [OK] customize：{len(customized)} 字符")
    if customized:
        print(customized[:300] + "..." if len(customized) > 300 else customized)

    if notifications:
        print(f"\n  [提醒] 工作流通知（{len(notifications)} 条）：")
        for note in notifications:
            print(f"     {note}")

    # ── 第 4 步：验证清单 ──
    print("\n" + "=" * 60)
    print("  验证清单")
    print("=" * 60)

    checks = [
        ("无错误", len(errors) == 0),
        ("parse_user：姓名正确", user is not None and user.name and "李" in user.name),
        ("parse_user：技能非空", user is not None and len(user.skills) > 0),
        ("parse_user：经历已拆分", user is not None and len(user.experience) == 2),
        ("extract_style：结构非空", style is not None and len(style.structure) > 0),
        ("extract_jd：关键词已提取", jd_reqs is not None and len(jd_reqs.keywords) > 0),
        ("extract_jd：必备要求已提取", jd_reqs is not None and len(jd_reqs.must_have) > 0),
        ("generate_base：已生成", len(base) > 100),
        ("customize：已生成", len(customized) > 100),
        ("无 JD 定制失败通知", not any("JD 定制优化失败" in n for n in notifications)),
        ("双索引就绪", vs is not None and core._bm25_index is not None),
    ]

    all_pass = True
    for desc, okay in checks:
        status = "PASS" if okay else "FAIL"
        if not okay:
            all_pass = False
        print(f"  [{status}] {desc}")

    print("\n" + "=" * 60)
    if all_pass:
        print("  全部通过！Round 5 改造完成。")
    else:
        print("  部分检查未通过，请检查上述 FAIL 项。")
    print("=" * 60)

    # ── Round 5 新增：导出验证 ──
    print("\n" + "─" * 60)
    print("[5/5] 导出验证（PDF + Word）...")
    print("─" * 60)

    from exporters import markdown_to_pdf_bytes, markdown_to_docx_bytes

    export_checks = []

    # PDF
    pdf_bytes, pdf_err = markdown_to_pdf_bytes(customized)
    export_checks.append(("PDF 导出", pdf_err is None and pdf_bytes is not None and len(pdf_bytes) > 1000, pdf_err))

    # Word
    docx_bytes, docx_err = markdown_to_docx_bytes(customized)
    export_checks.append(("Word 导出", docx_err is None and docx_bytes is not None and len(docx_bytes) > 1000, docx_err))

    for desc, okay, err_msg in export_checks:
        status = "PASS" if okay else "FAIL"
        detail = f" — {err_msg}" if err_msg else ""
        print(f"  [{status}] {desc}{detail}")

    # 保存导出文件到 data 目录
    if pdf_bytes:
        pdf_path = os.path.join(DATA_DIR, "_test_export_terminal.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"  [INFO] PDF 已保存到 {pdf_path}")
    if docx_bytes:
        docx_path = os.path.join(DATA_DIR, "_test_export_terminal.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        print(f"  [INFO] Word 已保存到 {docx_path}")

    print("\n" + "=" * 60)
    print("  Round 5 全部测试完成。")
    print("=" * 60)


if __name__ == "__main__":
    test_workflow()
