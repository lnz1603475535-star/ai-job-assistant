"""
AI 简历生成器 - 导出模块（Round 5）
===================================
提供两种格式导出：PDF（fpdf2）、Word（python-docx）。
全部从 Markdown 字符串出发，返回值统一为 (result, error) 元组。
"""

from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from fpdf import FPDF as FPDFType

try:
    import docx as _docx
    from docx.shared import Pt as _Pt
    from docx.shared import Cm as _Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn as _qn
except ImportError:
    _docx = None
    _Pt = None
    _Cm = None
    WD_ALIGN_PARAGRAPH = None
    _qn = None

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

# 微软雅黑字体路径（Windows 11 默认安装）
# TODO: Docker/Linux 部署阶段改为配置项或自动检测，Windows/Linux 字体路径不同
_MSYH_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
# PDF 页面设置
_PDF_MARGIN_LR = 18       # 左右边距 mm
_PDF_MARGIN_T = 0         # 上边距 mm（顶栏从 0 开始）
_PDF_MARGIN_B = 12        # 下边距 mm
_PDF_FONT_SIZE = 10.5     # 正文字号
_PDF_FONT_SIZE_H1 = 18    # 姓名
_PDF_FONT_SIZE_H2 = 13    # 章节标题
_PDF_FONT_SIZE_H3 = 11.5  # 子标题
_PDF_LINE_H = 5.5         # 行高 mm

# PDF 配色
_PDF_ACCENT = (43, 87, 154)       # 深蓝 #2b579a
_PDF_ACCENT_LIGHT = (220, 230, 245)  # 浅蓝背景
_PDF_TEXT_DARK = (50, 50, 50)     # 正文深灰
_PDF_TEXT_MEDIUM = (100, 100, 100)  # 次要文字

# Word 页面设置
_DOCX_MARGIN = 2.54       # 页边距 cm（1 英寸）

# 照片尺寸（mm，标准一寸照比例）
_PHOTO_W = 25
_PHOTO_H = 35


# ============================================================
# Markdown 行级解析
# ============================================================

def _is_h2(line: str) -> bool:
    """判断是否为二级标题 ## xxx"""
    return line.startswith("## ") and not line.startswith("### ")


def _is_h3(line: str) -> bool:
    """判断是否为三级标题 ### xxx"""
    return line.startswith("### ")


def _is_list_item(line: str) -> bool:
    """判断是否为列表项 - xxx 或 * xxx"""
    stripped = line.lstrip()
    return stripped.startswith("- ") or stripped.startswith("* ")


def _is_horizontal_rule(line: str) -> bool:
    """判断是否为分割线 --- 或 ***"""
    s = line.strip()
    return s in ("---", "***", "___") or (len(s) >= 3 and all(c == s[0] for c in s) and s[0] in "-*_")


def _strip_inline_format(text: str) -> str:
    """去除行内 Markdown 格式（粗体/斜体/代码/链接），返回纯文本。"""
    # 图片 ![alt](url) → alt（先处理：否则 ![alt](url) 会被链接正则剥成 !alt）
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # 链接 [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # 行内代码 `code`
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # 粗斜体 ***text***
    text = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", text)
    # 粗体 **text**
    text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)
    # 斜体 *text*
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def _count_leading_spaces(line: str) -> int:
    """计算行首空格数（用于嵌套列表判断）。"""
    return len(line) - len(line.lstrip())


def _extract_resume_name(lines: list[str]) -> tuple[str, int]:
    """从 Markdown 行中提取姓名（第一个 # 一级标题行）。

    LLM 输出不保证第一行一定是 `# 姓名`，逐行查找第一个一级标题。
    返回 (姓名, 该行索引)；找不到返回 ("", 0)——调用方按无姓名处理，
    顶栏只显示求职意向（如有），正文从第一行开始。
    """
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return _strip_inline_format(line[2:].strip()), i
    return "", 0


def _strip_list_item_content(line: str, strip_inline: bool = True) -> str:
    """从列表项行中提取内容文本，正确去除列表标记（- 或 * ）。

    与 lstrip("-* ") 不同，此函数只去除行首空格和第一个列表标记，
    不会误删内容中以 * 或 - 开头的字符。

    Args:
        strip_inline: True 时同时去除行内格式（**粗体** 等，PDF 用）；
            False 时保留行内格式标记（Word 用，后续按标记渲染粗体/斜体）。

    Example:
        "  - **Python** 是主力语言" → "**Python** 是主力语言"
        "- *斜体内容*" → "*斜体内容*"
    """
    stripped = line.lstrip()
    if stripped.startswith("- "):
        content = stripped[2:].strip()
    elif stripped.startswith("* "):
        content = stripped[2:].strip()
    else:
        # 兜底：不是标准列表标记，返回原文
        content = stripped
    return _strip_inline_format(content) if strip_inline else content


# ============================================================
# PDF 导出（fpdf2）
# ============================================================

def _check_font() -> Tuple[bool, str]:
    """检查微软雅黑字体是否可用。"""
    if os.path.exists(_MSYH_FONT_PATH):
        return True, _MSYH_FONT_PATH
    # 尝试备选字体
    for alt in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
        if os.path.exists(alt):
            return True, alt
    return False, ""


def _embed_photo(pdf: FPDFType, photo_path: str):
    """在 PDF 当前页右上角嵌入照片。

    照片尺寸 25×35mm（标准一寸），位置：右上角对齐页边距。
    嵌入后光标保持在页面顶部，不影响后续文字排版。
    """
    x = pdf.w - _PDF_MARGIN_LR - _PHOTO_W
    y = _PDF_MARGIN_T
    try:
        pdf.image(photo_path, x=x, y=y, w=_PHOTO_W, h=_PHOTO_H)
    except (OSError, RuntimeError) as e:
        logger.warning("照片嵌入失败（%s），将跳过：%s", photo_path, _sanitize_error(e), exc_info=True)


def _add_pdf_top_bar(pdf: FPDFType, name: str, job_target: str = ""):
    """绘制顶部深蓝色条幅。姓名在上，求职意向在下，均靠左。"""
    bar_h = 30 if job_target else 24  # 有求职意向时顶栏稍高
    # 深蓝背景条
    pdf.set_fill_color(*_PDF_ACCENT)
    pdf.rect(0, 0, pdf.w, bar_h, style="F")
    # 白色姓名——靠左
    pdf.set_y(5)
    pdf.set_x(_PDF_MARGIN_LR)
    pdf.set_font("msyh", "", _PDF_FONT_SIZE_H1)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, name, new_x="LMARGIN", new_y="NEXT", align="L")
    # 求职意向——靠左，姓名下方
    if job_target:
        pdf.set_x(_PDF_MARGIN_LR)
        pdf.set_font("msyh", "", 10)
        pdf.set_text_color(180, 200, 230)
        pdf.cell(0, 6, f"求职意向：{job_target}", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_text_color(*_PDF_TEXT_DARK)
    pdf.set_y(bar_h + 4)


def _add_pdf_section_header(pdf: FPDFType, title: str):
    """绘制带左侧色条的章节标题。"""
    pdf.ln(3)
    pdf.set_fill_color(*_PDF_ACCENT)
    pdf.set_text_color(*_PDF_ACCENT)
    pdf.set_font("msyh", "", _PDF_FONT_SIZE_H2)
    # 左侧色块
    pdf.rect(_PDF_MARGIN_LR, pdf.get_y() + 1, 3, 6, style="F")
    pdf.set_x(_PDF_MARGIN_LR + 6)
    pdf.cell(0, _PDF_LINE_H + 2, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*_PDF_TEXT_DARK)
    # 底部细线
    y = pdf.get_y()
    pdf.set_draw_color(*_PDF_ACCENT)
    pdf.line(_PDF_MARGIN_LR, y + 1, pdf.w - _PDF_MARGIN_LR, y + 1)
    pdf.ln(3)


def _add_pdf_sub_header(pdf: FPDFType, title: str):
    """绘制子标题（公司-职位），带浅蓝背景。"""
    pdf.set_fill_color(*_PDF_ACCENT_LIGHT)
    pdf.set_font("msyh", "", _PDF_FONT_SIZE_H3)
    pdf.set_text_color(*_PDF_TEXT_DARK)
    x0 = _PDF_MARGIN_LR
    pdf.set_x(x0)
    # 先量宽度再画背景
    text_w = pdf.get_string_width(title) + 4
    pdf.rect(x0, pdf.get_y(), text_w, _PDF_LINE_H + 2, style="F")
    pdf.set_xy(x0 + 2, pdf.get_y() + 1)
    pdf.cell(text_w - 2, _PDF_LINE_H, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def markdown_to_pdf_bytes(md_text: str, photo_path: Optional[str] = None,
                         job_target: str = "") -> Tuple[Optional[bytes], Optional[str]]:
    """将 Markdown 简历文本转换为 PDF bytes。

    Args:
        md_text: Markdown 格式的简历文本。
        photo_path: 可选的照片文件路径（JPG/PNG），放置在首页右上角。
        job_target: 可选，求职意向（如"Python 后端工程师"），显示在顶栏右侧。

    Returns:
        (pdf_bytes, None) 成功时； (None, error_message) 失败时。
        pdf_bytes 保证是 bytes（fpdf2 的 output() 返回 bytearray，
        Streamlit download_button 不接受 bytearray，这里统一转 bytes）。
    """
    if not md_text or not md_text.strip():
        return None, "简历内容为空，无法生成 PDF。"

    # 正文若已含求职意向（LLM 生成时可能已写入），顶栏不再重复添加
    if re.search(r"求职意向", md_text):
        job_target = ""

    # 检查字体
    font_ok, font_path = _check_font()
    if not font_ok:
        logger.error("PDF 导出失败：未找到中文字体文件")
        return None, (
            "PDF 生成失败：未找到中文字体（微软雅黑/黑体/宋体）。"
            "请确认 C:\\Windows\\Fonts 目录下有中文字体文件。"
        )

    try:
        from fpdf import FPDF
    except ImportError:
        return None, "PDF 生成失败：fpdf2 库未安装，请运行 pip install fpdf2。"

    # 验证照片文件
    if photo_path and not os.path.exists(photo_path):
        logger.warning("照片文件不存在：%s，将跳过照片嵌入", photo_path)
        photo_path = None

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=_PDF_MARGIN_B)
        pdf.add_page()
        pdf.add_font("msyh", "", font_path)
        pdf.set_font("msyh", "", _PDF_FONT_SIZE)
        pdf.set_text_color(*_PDF_TEXT_DARK)
        pdf.set_draw_color(*_PDF_ACCENT)

        lines = md_text.strip().splitlines()

        # 提取姓名（第一个 # 一级标题行），其余为正文
        name, name_idx = _extract_resume_name(lines)
        i = name_idx + 1 if name else 0

        # 绘制顶栏（姓名靠左 + 求职意向靠右）
        _add_pdf_top_bar(pdf, name, job_target)

        # 嵌入照片（首页右上角，顶栏上方）
        if photo_path:
            _embed_photo(pdf, photo_path)
            # 照片占满右上角 _PHOTO_H 高，正文下移到照片底部以下，
            # 避免正文（尤其联系方式行）画在照片上
            pdf.set_y(max(pdf.get_y(), _PDF_MARGIN_T + _PHOTO_H + 2))

        # ── 第二遍：渲染正文 ──
        while i < len(lines):
            line = lines[i]

            # 空行
            if not line.strip():
                pdf.ln(_PDF_LINE_H * 0.6)
                i += 1
                continue

            # 分割线
            if _is_horizontal_rule(line):
                pdf.ln(2)
                y = pdf.get_y()
                pdf.set_draw_color(*_PDF_ACCENT)
                pdf.line(_PDF_MARGIN_LR, y, pdf.w - _PDF_MARGIN_LR, y)
                pdf.ln(3)
                i += 1
                continue

            # 二级标题
            if _is_h2(line):
                title = _strip_inline_format(line.lstrip("# ").strip())
                _add_pdf_section_header(pdf, title)
                i += 1
                continue

            # 三级标题
            if _is_h3(line):
                title = _strip_inline_format(line.lstrip("# ").strip())
                _add_pdf_sub_header(pdf, title)
                i += 1
                continue

            # 列表项
            if _is_list_item(line):
                indent = _count_leading_spaces(line)
                pdf.set_font("msyh", "", _PDF_FONT_SIZE)
                content = _strip_list_item_content(line)
                x_offset = _PDF_MARGIN_LR + 4 + min(indent, 8) * 2.5
                bullet = "•" if indent <= 2 else "–"
                available_w = pdf.w - x_offset - _PDF_MARGIN_LR
                pdf.set_x(x_offset - 4)
                pdf.set_text_color(*_PDF_ACCENT)
                pdf.cell(4, _PDF_LINE_H, bullet, new_x="RIGHT", new_y="TOP")
                pdf.set_text_color(*_PDF_TEXT_DARK)
                pdf.multi_cell(available_w, _PDF_LINE_H, content, new_x="LMARGIN", new_y="NEXT")
                i += 1
                continue

            # 普通段落（联系方式等）
            pdf.set_font("msyh", "", _PDF_FONT_SIZE)
            content = _strip_inline_format(line.strip())
            pdf.set_text_color(*_PDF_TEXT_MEDIUM)
            # 联系方式用紧凑格式
            if "@" in content or "|" in content or "电话" in content:
                pdf.set_x(_PDF_MARGIN_LR)
                pdf.cell(pdf.w - _PDF_MARGIN_LR * 2, _PDF_LINE_H, content,
                        new_x="LMARGIN", new_y="NEXT", align="L")
            else:
                pdf.set_text_color(*_PDF_TEXT_DARK)
                pdf.multi_cell(pdf.w - _PDF_MARGIN_LR * 2, _PDF_LINE_H, content,
                              new_x="LMARGIN", new_y="NEXT", align="L")
            i += 1

        pdf_bytes = bytes(pdf.output())
        return pdf_bytes, None

    except Exception as e:
        logger.exception("PDF 生成失败")
        return None, f"PDF 生成失败：{_sanitize_error(e)}"


# ============================================================
# Word 导出（python-docx）
# ============================================================

# 颜色常量
_DOCX_ACCENT_RGB = 0x2B579A
_DOCX_ACCENT_LIGHT_RGB = 0xE8F0FA


def _docx_add_header_bar(doc, name: str, job_target: str = ""):
    """在 Word 文档开头创建深蓝顶栏（段落背景着色）。"""

    para = doc.add_paragraph()
    para.paragraph_format.space_after = _Pt(0)
    para.paragraph_format.space_before = _Pt(0)
    # 段落背景色
    pPr = para._element.get_or_add_pPr()
    shd = _docx.oxml.OxmlElement("w:shd")
    shd.set(_qn("w:fill"), "2B579A")
    shd.set(_qn("w:val"), "clear")
    pPr.append(shd)

    # 姓名——白色大字
    run = para.add_run(name)
    run.bold = True
    run.font.size = _Pt(20)
    run.font.color.rgb = _docx.shared.RGBColor(0xFF, 0xFF, 0xFF)
    run.font.name = "微软雅黑"

    if job_target:
        run = para.add_run(f"\n求职意向：{job_target}")
        run.font.size = _Pt(11)
        run.font.color.rgb = _docx.shared.RGBColor(0xB4, 0xC8, 0xE6)
        run.font.name = "微软雅黑"


def _docx_add_photo(doc, photo_path: str):
    """在 Word 文档顶栏下方右侧插入照片（25×35mm 一寸照，右对齐段落）。

    inline 图片撑起 35mm 行高，后续正文自动排到照片下方，不会与照片重叠。
    照片缺失/损坏时静默跳过（warning 日志），不阻断导出。
    """
    if not os.path.exists(photo_path):
        logger.warning("照片文件不存在：%s，将跳过照片插入", photo_path)
        return
    try:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = para.add_run()
        run.add_picture(photo_path, width=_Cm(2.5), height=_Cm(3.5))
    except Exception as e:
        logger.warning("Word 照片插入失败（%s），将跳过：%s", photo_path, _sanitize_error(e), exc_info=True)


def _docx_add_section_header(doc, title: str):
    """添加带左侧蓝色边框 + 浅蓝背景的章节标题。"""

    para = doc.add_paragraph()
    para.paragraph_format.space_before = _Pt(14)
    para.paragraph_format.space_after = _Pt(4)
    # 浅蓝背景
    pPr = para._element.get_or_add_pPr()
    shd = _docx.oxml.OxmlElement("w:shd")
    shd.set(_qn("w:fill"), "E8F0FA")
    shd.set(_qn("w:val"), "clear")
    pPr.append(shd)
    # 左侧蓝色边框
    pBdr = _docx.oxml.OxmlElement("w:pBdr")
    left = _docx.oxml.OxmlElement("w:left")
    left.set(_qn("w:val"), "single")
    left.set(_qn("w:sz"), "12")
    left.set(_qn("w:space"), "6")
    left.set(_qn("w:color"), "2B579A")
    pBdr.append(left)
    pPr.append(pBdr)
    # 下方细线边框
    bottom = _docx.oxml.OxmlElement("w:bottom")
    bottom.set(_qn("w:val"), "single")
    bottom.set(_qn("w:sz"), "4")
    bottom.set(_qn("w:space"), "1")
    bottom.set(_qn("w:color"), "2B579A")
    pBdr.append(bottom)

    run = para.add_run(title)
    run.bold = True
    run.font.size = _Pt(13)
    run.font.color.rgb = _docx.shared.RGBColor(0x2B, 0x57, 0x9A)
    run.font.name = "微软雅黑"


def _docx_add_sub_header(doc, title: str):
    """添加带浅蓝背景的子标题（公司-职位）。"""

    para = doc.add_paragraph()
    para.paragraph_format.space_before = _Pt(10)
    para.paragraph_format.space_after = _Pt(2)
    # 浅蓝背景
    pPr = para._element.get_or_add_pPr()
    shd = _docx.oxml.OxmlElement("w:shd")
    shd.set(_qn("w:fill"), "E8F0FA")
    shd.set(_qn("w:val"), "clear")
    pPr.append(shd)

    run = para.add_run(title)
    run.bold = True
    run.font.size = _Pt(11.5)
    run.font.color.rgb = _docx.shared.RGBColor(0x2B, 0x57, 0x9A)
    run.font.name = "微软雅黑"


def _docx_add_paragraph_with_format(doc, text: str, style: str | None = None,
                                    left_indent_cm: float | None = None):
    """向 Word 文档添加段落，支持行内粗体/斜体格式。

    Args:
        text: 段落文本（保留 **粗体** / *斜体* 标记，函数内解析）
        style: 可选段落样式名（如 "List Bullet"）
        left_indent_cm: 可选左缩进（cm），用于嵌套列表
    """
    if text is None:
        text = ""

    para = doc.add_paragraph()
    if style:
        para.style = doc.styles[style]
        para.clear()  # 清除样式模板自带的空 run
    if left_indent_cm is not None:
        para.paragraph_format.left_indent = _Cm(left_indent_cm)

    pattern = re.compile(r"(\*{1,3})(.+?)\1")
    last_end = 0
    for match in pattern.finditer(text):
        before = text[last_end:match.start()]
        if before:
            run = para.add_run(before)
            run.font.size = _Pt(11)
            run.font.name = "微软雅黑"
        stars = match.group(1)
        formatted = match.group(2)
        run = para.add_run(formatted)
        run.bold = len(stars) >= 2
        run.italic = len(stars) in (1, 3)
        run.font.size = _Pt(11)
        run.font.name = "微软雅黑"
        last_end = match.end()

    tail = text[last_end:]
    if tail:
        run = para.add_run(tail)
        run.font.size = _Pt(11)
        run.font.name = "微软雅黑"

    return para


def markdown_to_docx_bytes(md_text: str, job_target: str = "",
                           photo_path: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str]]:
    """将 Markdown 简历文本转换为 Word (.docx) bytes。

    Args:
        md_text: Markdown 格式的简历文本。
        job_target: 可选，求职意向，显示在顶栏。
        photo_path: 可选的照片文件路径（JPG/PNG），插入在顶栏下方右侧。

    Returns:
        (docx_bytes, None) 成功时； (None, error_message) 失败时。
    """
    if not md_text or not md_text.strip():
        return None, "简历内容为空，无法生成 Word 文档。"

    # 正文若已含求职意向（LLM 生成时可能已写入），顶栏不再重复添加
    if re.search(r"求职意向", md_text):
        job_target = ""

    if _docx is None:
        return None, "Word 生成失败：python-docx 库未安装，请运行 pip install python-docx。"

    try:
        doc = _docx.Document()

        # 页面设置
        for section in doc.sections:
            section.top_margin = _Cm(0)       # 顶栏从页面顶部开始
            section.bottom_margin = _Cm(_DOCX_MARGIN)
            section.left_margin = _Cm(_DOCX_MARGIN)
            section.right_margin = _Cm(_DOCX_MARGIN)

        # 设置默认字体
        style = doc.styles["Normal"]
        style.font.size = _Pt(11)
        style.font.name = "微软雅黑"
        style.paragraph_format.space_after = _Pt(4)
        style.paragraph_format.line_spacing = 1.5

        lines = md_text.strip().splitlines()

        # 提取姓名（第一个 # 一级标题行），其余为正文
        name, name_idx = _extract_resume_name(lines)
        _docx_add_header_bar(doc, name, job_target)

        # 照片（顶栏下方右侧，与 PDF 右上角位置对齐）
        if photo_path:
            _docx_add_photo(doc, photo_path)

        # ── 渲染正文 ──
        i = name_idx + 1 if name else 0
        while i < len(lines):
            line = lines[i]

            if not line.strip():
                doc.add_paragraph("")
                i += 1
                continue

            if _is_horizontal_rule(line):
                hr_para = doc.add_paragraph()
                hr_para.paragraph_format.space_before = _Pt(6)
                hr_para.paragraph_format.space_after = _Pt(6)
                run = hr_para.add_run("─" * 60)
                run.font.size = _Pt(8)
                run.font.color.rgb = _docx.shared.RGBColor(0xCC, 0xCC, 0xCC)
                i += 1
                continue

            if _is_h2(line):
                title = _strip_inline_format(line.lstrip("# ").strip())
                _docx_add_section_header(doc, title)
                i += 1
                continue

            if _is_h3(line):
                title = _strip_inline_format(line.lstrip("# ").strip())
                _docx_add_sub_header(doc, title)
                i += 1
                continue

            if _is_list_item(line):
                indent = _count_leading_spaces(line)
                content = _strip_list_item_content(line, strip_inline=False)
                _docx_add_paragraph_with_format(
                    doc, content,
                    style="List Bullet",
                    left_indent_cm=1.27 + indent * 0.32,
                )
                i += 1
                continue

            # 普通段落（联系方式等）
            content = line.strip()
            para = _docx_add_paragraph_with_format(doc, content)
            # 联系方式用灰色
            if "@" in content or "|" in content or "电话" in content:
                for run in para.runs:
                    run.font.color.rgb = _docx.shared.RGBColor(0x66, 0x66, 0x66)
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            i += 1

        buf = BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.getvalue(), None

    except Exception as e:
        logger.exception("Word 文档生成失败")
        return None, f"Word 生成失败：{_sanitize_error(e)}"


# ============================================================
# 辅助函数
# ============================================================

def _sanitize_error(exc: Exception) -> str:
    """脱敏异常信息：替换用户目录路径，截断到 200 字符。"""
    msg = str(exc)
    home = os.path.expanduser("~")
    if home and home != "~":
        msg = msg.replace(home, "~")
    return msg[:200]


# ============================================================
# 独立运行冒烟测试
# ============================================================

if __name__ == "__main__":
    SAMPLE_MD = """# 张三

联系方式：zhangsan@email.com | 138-0000-0001

## 专业技能
- Python， Django， FastAPI， Docker， MySQL
- Redis， Linux， Git， CI/CD
- 微服务架构， RESTful API 设计

## 工作经历

### ABC 科技 - 高级后端工程师（2022.06-2025.03）
- 负责订单系统从**单体架构拆分为微服务**，用 FastAPI 重写核心 API
- 性能提升 **3 倍**，日均处理 200 万订单
- 搭建 CI/CD 流水线，用 Docker 容器化部署，部署效率提升 80%

### XYZ 创业 - 全栈开发工程师（2020.07-2022.05）
- 独立负责用户系统和支付模块开发
- 用 Django + Vue 技术栈完成 3 个核心项目

## 教育背景
- 浙江大学 软件工程 本科 2016-2020
"""

    print("=" * 60)
    print("  导出模块冒烟测试")
    print("=" * 60)

    # PDF
    print("\n[1/2] 测试 PDF 导出...")
    pdf_bytes, pdf_err = markdown_to_pdf_bytes(SAMPLE_MD)
    if pdf_err:
        print(f"  ❌ PDF 失败: {pdf_err}")
    else:
        import os
        out_path = os.path.join(os.path.dirname(__file__), "data", "_test_export.pdf")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"  ✅ PDF 生成成功 ({len(pdf_bytes)} bytes) → {out_path}")

    # Word
    print("\n[2/2] 测试 Word 导出...")
    docx_bytes, docx_err = markdown_to_docx_bytes(SAMPLE_MD)
    if docx_err:
        print(f"  ❌ Word 失败: {docx_err}")
    else:
        import os
        out_path = os.path.join(os.path.dirname(__file__), "data", "_test_export.docx")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(docx_bytes)
        print(f"  ✅ Word 生成成功 ({len(docx_bytes)} bytes) → {out_path}")

    print("\n" + "=" * 60)
    print("  冒烟测试完成")
    print("=" * 60)
