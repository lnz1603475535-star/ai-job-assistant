"""
终端测试脚本 — Round 1 全流程演示
===================================
验证 5 个核心函数端到端协同工作，不依赖 Streamlit。

运行：python terminal_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from resume_engine import (
    parse_user_info,
    extract_style,
    extract_jd_requirements,
    generate_base_resume,
    customize_for_jd,
)
from core import load_and_index_documents, set_vectorstore
from workflow import run_workflow

# 使用项目自带的样例文件
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")

# 用户经验库
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_full_pipeline():
    """端到端测试：从输入到定制简历的完整流程。"""

    print("=" * 60)
    print("  AI 简历生成器 — Round 1 全流程测试")
    print("=" * 60)

    # ── 第 0 步：索引文档 ──
    print("\n" + "─" * 60)
    print("[0/5] 索引文档（FAISS + BM25 双索引）...")
    print("─" * 60)

    sample_resume_path = os.path.join(SAMPLE_DIR, "resume_zhangsan.txt")
    jd_path = os.path.join(SAMPLE_DIR, "jd_python_senior.txt")

    vs, bm25, chunks = load_and_index_documents({
        "user_experience": [os.path.join(DATA_DIR, "experience_bank.md")],
        "sample_resume": [sample_resume_path],
        "jd": [jd_path],
    })
    set_vectorstore(vs, bm25, chunks)
    print(f"  已索引 {len(chunks)} 个文本块")
    print(f"  FAISS 索引：就绪 | BM25 索引：就绪")

    # ── 第 1 步：解析用户信息 ──
    print("\n" + "─" * 60)
    print("[1/5] 解析用户信息...")
    print("─" * 60)

    # 模拟一段用户口语输入（和样本简历不同的人）
    user_text = """我叫李思，邮箱 lisi@email.com，电话 13800002222。
    技能包括：Python、Django、FastAPI、Docker、MySQL、Redis、Linux。
    工作经历：
    2022年6月到2025年3月，在某互联网公司做 Python 后端开发，负责订单系统从单体拆分为微服务，
    用 FastAPI 重写了核心 API，性能提升了 3 倍，日均处理 200 万订单。
    还搭建了 CI/CD 流水线，用 Docker 容器化部署。
    2020年7月到2022年5月，在某创业公司做全栈开发，用 Django 写后端，Vue 写前端，
    独立负责用户系统和支付模块。
    学历：浙江大学 软件工程 本科 2016-2020"""

    user = parse_user_info(user_text)
    print(f"  姓名：{user.name}")
    print(f"  联系方式：{user.contact}")
    print(f"  技能（{len(user.skills)} 项）：{', '.join(user.skills)}")
    print(f"  工作经历：{len(user.experience)} 段")
    for exp in user.experience:
        print(f"    - {exp.title} @ {exp.company} ({exp.duration})")
        for ach in exp.achievements:
            print(f"      - {ach[:60]}...")
    print(f"  教育背景：{user.education}")

    # ── 第 2 步：提取风格 ──
    print("\n" + "─" * 60)
    print("[2/5] 提取样本简历风格...")
    print("─" * 60)

    style = extract_style(sample_resume_path)
    print(f"  章节结构：{style.structure}")
    print(f"  措辞风格：{style.tone}")
    print(f"  句式特征：{style.format_patterns}")

    # ── 第 3 步：提取 JD 要求 ──
    print("\n" + "─" * 60)
    print("[3/5] 提取 JD 要求...")
    print("─" * 60)

    jd_reqs = extract_jd_requirements(jd_path)
    print(f"  岗位名称：{jd_reqs.title}")
    print(f"  必备要求（{len(jd_reqs.must_have)} 项）：")
    for req in jd_reqs.must_have:
        print(f"    - {req}")
    print(f"  加分项（{len(jd_reqs.nice_to_have)} 项）：")
    for n in jd_reqs.nice_to_have:
        print(f"    - {n}")
    print(f"  关键词：{', '.join(jd_reqs.keywords)}")
    print(f"  隐性偏好：{jd_reqs.hidden_preferences}")

    # ── 第 4 步：生成基础简历 ──
    print("\n" + "─" * 60)
    print("[4/5] 生成基础简历...")
    print("─" * 60)

    base_resume = generate_base_resume(user, style)
    print(base_resume)
    print(f"\n  [简历总长度：{len(base_resume)} 字符]")

    # ── 第 5 步：JD 定制 ──
    print("\n" + "─" * 60)
    print("[5/5] 根据 JD 定制简历...")
    print("─" * 60)

    customized = customize_for_jd(base_resume, jd_reqs)
    print(customized)
    print(f"\n  [定制后简历长度：{len(customized)} 字符]")

    # ── 验证清单 ──
    print("\n" + "=" * 60)
    print("  验证清单")
    print("=" * 60)

    checks = [
        ("用户姓名正确提取", user.name == "李思"),
        ("技能列表非空", len(user.skills) > 0),
        ("工作经历已拆分（2段）", len(user.experience) == 2),
        ("风格结构非空", len(style.structure) > 0),
        ("JD 关键词已提取", len(jd_reqs.keywords) > 0),
        ("JD 必备要求已提取", len(jd_reqs.must_have) > 0),
        ("基础简历已生成", len(base_resume) > 100),
        ("定制简历已生成", len(customized) > 100),
        ("双索引就绪", vs is not None and bm25 is not None),
    ]

    all_pass = True
    for desc, result in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {desc}")

    print("\n" + "=" * 60)
    if all_pass:
        print("  全部通过！Round 1 改造完成。")
    else:
        print("  部分检查未通过，请检查上述 FAIL 项。")
    print("=" * 60)


def test_workflow():
    """Round 2 测试：LangGraph 工作流端到端。"""

    print("=" * 60)
    print("  AI 简历生成器 — Round 2 LangGraph 工作流测试")
    print("=" * 60)

    # ── 第 0 步：索引文档 ──
    print("\n" + "─" * 60)
    print("[0/6] 索引文档（FAISS + BM25 双索引）...")
    print("─" * 60)

    sample_resume_path = os.path.join(SAMPLE_DIR, "resume_zhangsan.txt")
    jd_path = os.path.join(SAMPLE_DIR, "jd_python_senior.txt")

    vs, bm25, chunks = load_and_index_documents({
        "user_experience": [os.path.join(DATA_DIR, "experience_bank.md")],
        "sample_resume": [sample_resume_path],
        "jd": [jd_path],
    })
    set_vectorstore(vs, bm25, chunks)
    print(f"  已索引 {len(chunks)} 个文本块")
    print(f"  FAISS 索引：就绪 | BM25 索引：就绪")

    # ── 运行工作流 ──
    user_text = """我叫李思，邮箱 lisi@email.com，电话 13800002222。
    技能包括：Python、Django、FastAPI、Docker、MySQL、Redis、Linux。
    工作经历：
    2022年6月到2025年3月，在某互联网公司做 Python 后端开发，负责订单系统从单体拆分为微服务，
    用 FastAPI 重写了核心 API，性能提升了 3 倍，日均处理 200 万订单。
    还搭建了 CI/CD 流水线，用 Docker 容器化部署。
    2020年7月到2022年5月，在某创业公司做全栈开发，用 Django 写后端，Vue 写前端，
    独立负责用户系统和支付模块。
    学历：浙江大学 软件工程 本科 2016-2020"""

    print("\n" + "─" * 60)
    print("[1-6/6] 运行 LangGraph 工作流...")
    print("      validate_inputs → extract_style → parse_user → extract_jd → generate_base → customize")
    print("─" * 60)

    result = run_workflow(
        user_text=user_text,
        sample_resume_path=sample_resume_path,
        jd_path=jd_path,
        thread_id="test-round2",
    )

    # ── 提取结果 ──
    user = result.get("user_profile")
    style = result.get("style_profile")
    jd_reqs = result.get("jd_requirements")
    base = result.get("base_resume", "")
    customized = result.get("customized_resume", "")
    errors = result.get("errors", [])

    # ── 打印中间结果 ──
    if errors:
        print(f"\n  [ERROR] 工作流有错误：")
        for e in errors:
            print(f"     - {e}")
    else:
        print(f"\n  [OK] 输入验证通过")

    if user:
        print(f"  [OK] parse_user：{user.name}，{len(user.skills)} 项技能，{len(user.experience)} 段经历")
    if style:
        print(f"  [OK] extract_style：{style.structure[:40]}...")
    if jd_reqs:
        print(f"  [OK] extract_jd：{jd_reqs.title}，{len(jd_reqs.keywords)} 个关键词")

    print(f"  [OK] generate_base：{len(base)} 字符")
    if base:
        print(base[:300] + "..." if len(base) > 300 else base)

    print(f"\n  [OK] customize：{len(customized)} 字符")
    if customized:
        print(customized[:300] + "..." if len(customized) > 300 else customized)

    # ── 验证清单 ──
    print("\n" + "=" * 60)
    print("  验证清单")
    print("=" * 60)

    checks = [
        ("无错误", len(errors) == 0),
        ("parse_user：姓名正确", user is not None and user.name == "李思"),
        ("parse_user：技能非空", user is not None and len(user.skills) > 0),
        ("parse_user：经历已拆分", user is not None and len(user.experience) == 2),
        ("extract_style：结构非空", style is not None and len(style.structure) > 0),
        ("extract_jd：关键词已提取", jd_reqs is not None and len(jd_reqs.keywords) > 0),
        ("extract_jd：必备要求已提取", jd_reqs is not None and len(jd_reqs.must_have) > 0),
        ("generate_base：已生成", len(base) > 100),
        ("customize：已生成", len(customized) > 100),
        ("双索引就绪", vs is not None and bm25 is not None),
    ]

    all_pass = True
    for desc, okay in checks:
        status = "PASS" if okay else "FAIL"
        if not okay:
            all_pass = False
        print(f"  [{status}] {desc}")

    print("\n" + "=" * 60)
    if all_pass:
        print("  全部通过！Round 2 LangGraph 工作流改造完成。")
    else:
        print("  部分检查未通过，请检查上述 FAIL 项。")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
    print("\n\n")
    test_workflow()
