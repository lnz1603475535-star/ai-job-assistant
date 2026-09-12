"""
UI 冒烟测试 — app.py（Streamlit）
=================================
用 Streamlit 官方 AppTest 无头运行 app.py：真实执行脚本、捕获渲染异常。

背景：app.py 是项目最大的文件（1300+ 行），此前**没有任何测试执行过它**——
2026-09-12 审查发现的"审核通过必失败"缺陷，正是藏在这条从未被覆盖的
UI ↔ API 接缝上。本文件补上这层覆盖，不依赖浏览器、不需要 LLM。

运行：python ui_test.py
"""

import io as _io
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, _io.TextIOWrapper):
        _stream.reconfigure(errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from models import JDRequirements, StyleProfile, UserProfile, WorkExperience

APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

_results: list[tuple[str, bool, str]] = []


def check(desc: str, ok: bool, detail: str = "") -> None:
    _results.append((desc, bool(ok), detail))


def _new_app():
    """启动一个干净的 AppTest 实例（每次独立 session）。"""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP_PATH, default_timeout=120)


def _fail_detail(app) -> str:
    return "; ".join(str(e.value)[:200] for e in app.exception)


def _sample_result(customized: str = "# 张三\n\n定制后的简历正文", base: str = "base"):
    """构造一份与 workflow 返回值同构的结果。"""
    return {
        "base_resume": base,
        "customized_resume": customized,
        "user_profile": UserProfile(
            name="李思",
            contact="lisi@email.com",
            skills=["Python", "FastAPI"],
            experience=[
                WorkExperience(
                    company="某公司",
                    title="后端",
                    duration="2022.06-2025.03",
                    achievements=["做了订单系统"],
                )
            ],
            education="浙江大学 本科",
        ),
        "style_profile": StyleProfile(
            structure="个人信息 → 技能 → 经历 → 教育", tone="简洁专业",
            format_patterns="动词开头", is_fallback=False,
        ),
        "jd_requirements": JDRequirements(
            title="Python 后端工程师", must_have=["Python"],
            nice_to_have=[], keywords=["Python", "FastAPI"],
            hidden_preferences="",
        ),
        "notifications": ["⚠️ 测试提醒：解析不完整"],
        "_interrupted": True,
    }


def test_first_screen() -> None:
    print("\n" + "─" * 60)
    print("[1/5] 首屏渲染")
    print("─" * 60)

    app = _new_app()
    app.run()

    check("首屏渲染无异常", len(app.exception) == 0, _fail_detail(app))
    check("标题渲染", any("AI 简历生成器" in t.value for t in app.title))
    check("侧边栏渲染（经验库管理）", len(app.sidebar) > 0)
    check("停留在 Step 1", app.session_state["wizard_step"] == 1)


def test_wizard_steps() -> None:
    print("\n" + "─" * 60)
    print("[2/5] 向导各步骤渲染（Step 1-3、5）")
    print("─" * 60)

    for step, label in [(1, "样本简历"), (2, "JD"), (3, "补充信息"), (5, "下载导出")]:
        app = _new_app()
        app.session_state["wizard_step"] = step
        app.run()
        check(
            f"Step {step}（{label}）渲染无异常",
            len(app.exception) == 0,
            _fail_detail(app),
        )


def test_step4_not_started() -> None:
    print("\n" + "─" * 60)
    print("[3/5] Step 4 · 状态「未开始」")
    print("─" * 60)

    app = _new_app()
    app.session_state["wizard_step"] = 4
    app.run()

    check("渲染无异常", len(app.exception) == 0, _fail_detail(app))
    labels = [b.label for b in app.button]
    check("显示「开始生成简历」按钮", any("开始生成简历" in s for s in labels),
          f"实际按钮 {labels}")


def test_step4_paused() -> None:
    print("\n" + "─" * 60)
    print("[4/5] Step 4 · 状态「断点暂停」（审核基础简历）")
    print("─" * 60)

    app = _new_app()
    app.session_state["wizard_step"] = 4
    app.session_state["workflow_paused"] = True
    app.session_state["paused_result"] = _sample_result()
    app.run()

    check("渲染无异常", len(app.exception) == 0, _fail_detail(app))
    labels = [b.label for b in app.button]
    check(
        "显示「审核通过，继续 JD 定制」按钮",
        any("审核通过" in s for s in labels),
        f"实际按钮 {labels}",
    )
    check(
        "显示基础简历已生成的提示",
        any("基础简历已生成" in str(s.value) for s in app.success),
    )
    check(
        "提醒（⚠️ 通知）被展示",
        any("测试提醒" in str(w.value) for w in app.warning),
    )


def test_step4_completed() -> None:
    print("\n" + "─" * 60)
    print("[5/5] Step 4 · 状态「已完成」+ Step 5 下载")
    print("─" * 60)

    app = _new_app()
    app.session_state["wizard_step"] = 4
    app.session_state["workflow_result"] = _sample_result()
    app.run()

    check("渲染无异常", len(app.exception) == 0, _fail_detail(app))
    check("显示定制简历内容", any(
        "定制后的简历正文" in str(m.value) for m in app.markdown
    ))

    # Step 5：导出缓存生成 + 下载按钮
    app5 = _new_app()
    app5.session_state["wizard_step"] = 5
    app5.session_state["workflow_result"] = _sample_result()
    app5.run()

    check("Step 5 渲染无异常", len(app5.exception) == 0, _fail_detail(app5))
    # AppTest 对 download_button 只有无类型访问器（ElementTree 无该属性），
    # 用 getattr 取标签，类型安全且缺失时为假值
    downloads = [str(getattr(d, "label", "")) for d in app5.get("download_button")]
    check("生成 Markdown 下载按钮", any(".md" in s for s in downloads),
          f"实际 {downloads}")
    check(
        "PDF / Word 导出完成（非错误态）",
        any(".pdf" in s for s in downloads) and any(".docx" in s for s in downloads),
        f"实际 {downloads}（若只有 ⏳/⚠️ 说明导出失败）",
    )

    # 四个导出槽位由 _EXPORT_SLOTS 表驱动，最怕 base/customized 映射错位——
    # 两份文本不同，产出的 bytes 必须不同。
    # 注意 AppTest 的 session_state 不是 dict：.get 会被当成"取名为 get 的键"，
    # 只能用下标，且缺键时抛 KeyError，故这里包一层避免整脚本崩掉
    def _slot(name: str):
        try:
            return app5.session_state[name]
        except (KeyError, AttributeError, TypeError):
            return None

    def _desc(value) -> str:
        if value is None:
            return "缺失"
        return f"{value[0]} {len(value[1])}B" if len(value) > 1 else str(value[0])

    cus_pdf, base_pdf = _slot("export_customized_pdf"), _slot("export_base_pdf")
    check(
        "定制版与基础版 PDF 各自产出且内容不同（表驱动映射未错位）",
        cus_pdf is not None and base_pdf is not None and cus_pdf[1] != base_pdf[1],
        f"定制 {_desc(cus_pdf)} / 基础 {_desc(base_pdf)}",
    )


def main() -> None:
    print("=" * 60)
    print("  AI 简历生成器 — Streamlit UI 冒烟测试（AppTest 无头运行）")
    print("=" * 60)

    test_first_screen()
    test_wizard_steps()
    test_step4_not_started()
    test_step4_paused()
    test_step4_completed()

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
        print(f"  {failed}/{len(_results)} 项未通过。")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
