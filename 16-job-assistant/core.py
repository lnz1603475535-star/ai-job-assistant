"""
AI 简历生成器 - 核心基础设施
============================
纯基础设施层：LLM 配置、Embedding、文档索引、混合检索。
不包含任何业务逻辑——业务逻辑在 resume_engine.py 中。
"""

import os, re, warnings
from typing import List

import requests

# 抑制依赖库的噪音警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import jieba
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv(find_dotenv())

# ============================================================
# LLM 配置
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)

# ============================================================
# Embedding 模型（懒加载 + 单例模式）
# ============================================================

_embeddings = None

def get_embeddings():
    """获取 embedding 模型实例（单例模式，首次调用时下载）。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

# ============================================================
# 全局向量库状态
# ============================================================

_vectorstore = None
_bm25_index = None
_chunks_text: List[str] = []
_chunks_metadata: List[dict] = []

def set_vectorstore(vs, bm25=None, chunks=None):
    """设置全局向量库实例，同时保存 BM25 索引和 chunk 元数据。"""
    global _vectorstore, _bm25_index, _chunks_text, _chunks_metadata
    _vectorstore = vs
    _bm25_index = bm25
    if chunks is not None:
        _chunks_text = [c.page_content for c in chunks]
        _chunks_metadata = [c.metadata for c in chunks]

# ============================================================
# 文档处理
# ============================================================

def normalize_text(text: str) -> str:
    """文本规范化：修复 PDF/网页提取常见的格式问题。

    - PDF 每行末尾硬换行 → 合并为段落
    - 列表项 / 标题行保留独立换行，不会被合并
    - 多余空行（3 个以上）→ 合并为双换行
    """
    # 统一换行符（Windows \r\n → \n）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        prev_stripped = lines[i - 1].strip() if i > 0 else ""

        # 当前行或上一行是列表项 / 标题 → 保留独立换行
        # \s* 容忍 PDF 解析丢掉空格 / 全角空格（\s 已覆盖 U+3000）
        _list_re = (
            r'^[-*•]\s*'                          # -item  *item  •item（含空格/全角空格）
            r'|^\d+[.、)]\s*'                     # 1.item  1、item  1)item
            r'|^（[一二三四五六七八九十\d]+）'      # （一）（1）
            r'|^[一二三四五六七八九十]+[、．]'      # 一、item  二．item
            r'|^#{1,6}\s'                         # ## heading（标题必须有空格）
        )
        is_special = bool(re.match(_list_re, stripped))
        prev_is_special = bool(re.match(_list_re, prev_stripped))

        if i == 0:
            result.append(line)
        elif stripped == "":
            result.append("")
        elif prev_stripped == "" or is_special or prev_is_special:
            # 新段落 / 列表项 / 标题 —— 保留换行
            result.append(line)
        else:
            # 同一段落继续 → 合并到上一行
            result[-1] = result[-1] + " " + stripped

    text = "\n".join(result)
    # 3 个及以上连续换行 → 双换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file_content(path: str) -> str:
    """加载文件文本内容，自动检测 .txt / .pdf 格式。

    参数：
        path：文件路径（支持 .txt、.pdf 和 .docx）

    返回：
        文件的完整文本内容（PDF 多页用双换行拼接）

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 加密 / 扫描件无文字 / 文件损坏
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf.errors import FileNotDecryptedError, PdfReadError
        except ImportError:
            FileNotDecryptedError = Exception
            PdfReadError = Exception
        try:
            docs = PyPDFLoader(path).load()
        except FileNotDecryptedError:
            raise ValueError("PDF 文件已加密，请移除密码后重新上传。")
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypt" in error_msg:
                raise ValueError("PDF 文件已加密，请移除密码后重新上传。")
            elif "corrupt" in error_msg or "not a pdf" in error_msg:
                raise ValueError("PDF 文件已损坏或格式异常，请检查后重新上传。")
            else:
                raise ValueError(f"PDF 文件读取失败：{str(e)[:100]}")
        text = normalize_text("\n\n".join(d.page_content for d in docs))
        if not text.strip():
            raise ValueError("PDF 可能是扫描件，无法提取文字。请上传含文本的 PDF 或直接粘贴文字内容。")
        return text
    elif ext == ".docx":
        try:
            docs = Docx2txtLoader(path).load()
            text = normalize_text("\n\n".join(d.page_content for d in docs))
            if not text.strip():
                raise ValueError("Word 文件内容为空，请检查后重新上传。")
            return text
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypt" in error_msg or "password" in error_msg:
                raise ValueError("Word 文件已加密，请移除密码后重新上传。")
            elif "corrupt" in error_msg or "not a valid" in error_msg:
                raise ValueError("Word 文件已损坏或格式异常，请检查后重新上传。")
            else:
                raise ValueError(f"Word 文件读取失败：{str(e)[:100]}")
    else:
        try:
            return TextLoader(path, encoding="utf-8").load()[0].page_content
        except UnicodeDecodeError:
            raise ValueError("文件编码不支持，请保存为 UTF-8 编码后重新上传。")


# ── URL 请求 ────────────────────────────────────────────

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_LOGIN_URL_PATTERNS = ["/login", "/auth", "/signin", "auth=", "redirect="]
_LOGIN_CONTENT_KEYWORDS = [
    "用户名", "密码", "验证码", "忘记密码", "username", "password", "captcha",
]


def _smart_decode(raw: bytes, fallback_encodings: list[str]) -> str:
    """多级编码回退 + 乱码检测。"""
    for enc in fallback_encodings:
        if not enc:
            continue
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if text.count("�") > len(text) * 0.01:  # 替换字符太多
            continue
        return text
    raise ValueError("无法识别该网页的文字编码，请尝试直接复制 JD 文字后上传。")


def _check_login_redirect(resp) -> None:
    """检测 302 重定向是否到了登录页。"""
    if resp.history and any(p in resp.url.lower() for p in _LOGIN_URL_PATTERNS):
        raise ValueError(
            "该链接已重定向到登录页面，需要先登录才能查看 JD 内容。"
            "请直接复制 JD 文字粘贴到上传文件，或换一个不需要登录的链接。"
        )


def _check_login_content(text: str) -> None:
    """检测提取内容是否为登录表单。"""
    text_lower = text.lower()
    hits = [kw for kw in _LOGIN_CONTENT_KEYWORDS if kw in text_lower]
    if len(hits) >= 2 and len(text) < 2000:
        raise ValueError(
            f"提取内容疑似登录页面（检测到：{'、'.join(hits[:3])}），"
            "不是招聘 JD。请直接复制 JD 文字后上传文件。"
        )


def _strip_html(html: str) -> str:
    """去除 HTML 标签，提取纯文本正文。"""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"</?(?:br|p|div|li|tr|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


# ── 公开函数 ────────────────────────────────────────────

def fetch_url_content(url: str) -> str:
    """从网页 URL 提取文本内容（自动去 HTML 标签、检测登录页、编码回退）。

    Raises:
        ValueError：请求失败 / 超时 / 编码无法识别 / 内容是登录页
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("链接格式错误，请以 http:// 或 https:// 开头。")

    # 1. HTTP 请求
    try:
        resp = requests.get(
            url, headers=_REQUEST_HEADERS, timeout=15, allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("请求超时，请检查网络连接或换一个链接重试。")
    except requests.exceptions.ConnectionError:
        raise ValueError("无法连接到该网站，请检查链接是否正确。")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "未知"
        messages = {404: "页面不存在", 403: "网站拒绝访问，该页面可能需要登录"}
        raise ValueError(f"{messages.get(status, f'请求失败（HTTP {status}）')}，请检查链接后重试。")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"网络请求失败：{str(e)[:100]}")

    # 2. 解码
    encodings = [resp.apparent_encoding, "utf-8", "gbk", "gb2312"]
    html = _smart_decode(resp.content, encodings)

    if not html.strip():
        raise ValueError("页面内容为空，请检查链接是否正确。")

    # 3. 登录页检测
    _check_login_redirect(resp)

    # 4. HTML → 纯文本
    text = _strip_html(html)
    if not text:
        raise ValueError("未能从页面提取到有效文字，该页面可能为纯图片或需 JavaScript 渲染。")

    # 5. 内容登录检测 + 规范化
    _check_login_content(text)
    return normalize_text(text)


_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _load_file_to_documents(path: str) -> list[Document]:
    """加载文件为 LangChain Document 列表，自动检测 .txt / .pdf / .docx / .md。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pdf", ".docx", ".txt", ".md", ""):
        raise ValueError(f"不支持的文件格式：{ext}。支持的格式：.txt / .pdf / .docx / .md")

    try:
        file_size = os.path.getsize(path)
    except OSError as e:
        raise ValueError(f"文件不存在或无法访问：{os.path.basename(path)}")
    if file_size > _MAX_FILE_SIZE:
        raise ValueError(
            f"文件过大（{file_size / 1024 / 1024:.1f}MB），请压缩后再上传。"
            f"简历文件建议 < 5MB。"
        )

    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext == ".docx":
            return Docx2txtLoader(path).load()
        return TextLoader(path, encoding="utf-8").load()
    except Exception as e:
        error_msg = str(e).lower()
        if "encrypt" in error_msg or "password" in error_msg:
            raise ValueError(f"文件已加密，无法读取：{os.path.basename(path)}")
        elif "corrupt" in error_msg or "not a pdf" in error_msg or "not a valid" in error_msg:
            raise ValueError(f"文件已损坏或格式异常：{os.path.basename(path)}")
        else:
            raise ValueError(f"文件读取失败（{os.path.basename(path)}）：{str(e)[:100]}")


def load_and_index_documents(file_paths: dict[str, list[str]]) -> tuple[object, object, list[Document]]:
    """加载多类型文档，切分，建立 FAISS + BM25 双索引。

    参数：
        file_paths: {"doc_type": ["path1", "path2"], ...}
        例如：{"sample_resume": ["samples/xxx.txt"], "jd": ["jd_python.txt"]}

    返回：
        (vectorstore, bm25_index, chunks)
    """
    if not file_paths:
        raise ValueError("未指定任何文档路径。")

    all_docs = []
    for doc_type, paths in file_paths.items():
        for path in paths:
            docs = _load_file_to_documents(path)
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["doc_type"] = doc_type
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError("所有文档均为空，无法建立索引。请检查文件内容。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "。", "！", "？", "\n", "，", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("文档切分后无有效内容，请检查文件是否只有空白字符。")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 建立 BM25 关键词索引（jieba 中文分词）
    bm25 = BM25Okapi([jieba.lcut(c.page_content) for c in chunks])

    return vectorstore, bm25, chunks

# ============================================================
# 工具定义
# ============================================================

@tool
def search_documents(query: str, k: int = 4) -> str:
    """混合搜索文档数据库（FAISS 语义检索 + BM25 关键词匹配）。
    用于查找用户经历、JD 要求、样本风格等已索引的文档内容。
    query：中文或英文的自然语言搜索词。"""
    if _vectorstore is None:
        return "尚未加载任何文档。"
    if _bm25_index is None:
        return "关键词索引不可用，请重新索引文档。"

    # 防止空查询返回随机结果
    if not query.strip():
        return "未提供搜索词。"

    # FAISS 语义检索（MMR：相关性 + 多样性，避免返回重复内容）
    faiss_docs = _vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=20
    )

    # BM25 关键词检索（jieba 中文分词）
    bm25_scores = _bm25_index.get_scores(jieba.lcut(query))
    if len(bm25_scores) > 0:
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]
    else:
        bm25_top_indices = []

    # RRF 融合：Reciprocal Rank Fusion，k=60
    # score(d) = sum(1/(K + rank_i(d)))，两路检索结果按排名加权融合
    K = 60
    rrf_scores: dict[str, dict[str, float | str]] = {}  # content -> {"score": float, "doc_type": str}

    # FAISS 排名得分（rank 从 0 开始）
    for rank, doc in enumerate(faiss_docs):
        content = doc.page_content
        rrf_scores[content] = {
            "score": 1.0 / (K + rank),
            "doc_type": doc.metadata.get("doc_type", "unknown"),
        }

    # BM25 排名得分，与 FAISS 累加（同一文档被两路都找到时得分更高）
    for rank, idx in enumerate(bm25_top_indices):
        content = _chunks_text[idx]
        score = 1.0 / (K + rank)
        if content in rrf_scores:
            rrf_scores[content]["score"] += score
        else:
            rrf_scores[content] = {
                "score": score,
                "doc_type": _chunks_metadata[idx].get("doc_type", "unknown"),
            }

    # 按 RRF 得分降序排列，取 top-k
    sorted_results = sorted(
        rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True
    )[:k]

    merged = [f"[{meta['doc_type']}] {content}" for content, meta in sorted_results]

    if not merged:
        return "未找到相关文档。"

    return "\n\n---\n\n".join(merged)

# ============================================================
# 通用工具函数
# ============================================================

def parse_experience(level_text: str) -> dict[str, int | str | None]:
    """从水平描述中提取结构化信息。

    支持的格式：
      - "5年" / "5年以上" / "5年+" → {"years": 5, "label": "5年以上"}
      - "2-3年" → {"years": 3, "label": "2-3年"}（取上限）
      - "精通（5年）" → {"years": 5, "label": "精通（5年）"}
      - "精通" / "熟悉" / "了解" → {"years": None, "label": "精通"}

    返回 dict: {"years": int|None, "label": str}
    """
    text = level_text.strip()

    # 1. 匹配 "X-Y年" 或 "X至Y年"（取上限 Y）
    m = re.search(r'(\d+)\s*[-–—至到]\s*(\d+)\s*年?', text)
    if m:
        return {"years": int(m.group(2)), "label": text}

    # 2. 匹配单独数字
    m = re.search(r'(\d+)\s*年?', text)
    if m:
        return {"years": int(m.group(1)), "label": text}

    # 3. 纯文字描述
    return {"years": None, "label": text}

# ============================================================
# Token 预算控制
# ============================================================

class TokenBudget:
    """Token 预算跟踪器，基于字符数估算 Token 使用量。

    估算方法（保守估算）：
    - 中文字符：约 1 token / 1.5 字符
    - 非中文字符（英文、数字、标点、空格）：约 1 token / 3.5 字符
    """

    def __init__(self, max_tokens: int = 8000, warning_threshold: float = 0.7):
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self._used = 0
        self._warning_issued = False

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量。"""
        if not text:
            return 0
        chinese = len(re.findall(r'[\u4e00-\u9fff\uff00-\uffef]', text))
        other = len(text) - chinese
        return int(chinese / 1.5 + other / 3.5) + 1  # +1 安全边界

    def add_usage(self, text: str) -> int:
        """记录 token 使用量，返回本次增加的 token 估算数。"""
        tokens = self.estimate_tokens(text)
        self._used += tokens
        return tokens

    @property
    def used_tokens(self) -> int:
        return self._used

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self._used / self.max_tokens

    def get_warning(self) -> str:
        """超过阈值时返回警告文本，否则返回空字符串。每个预算只警告一次。"""
        ratio = self.usage_ratio
        if ratio >= 0.9:
            self._warning_issued = True
            return (
                f"⚠️ Token 预算即将耗尽（{self._used}/{self.max_tokens}，{ratio:.0%}）。"
                f"请保持回复简洁，优先输出关键内容，尽快给出最终结论。"
            )
        if ratio >= self.warning_threshold and not self._warning_issued:
            self._warning_issued = True
            return (
                f"⚠️ Token 使用量已达 {ratio:.0%}（{self._used}/{self.max_tokens}）。"
                f"请注意控制输出长度。"
            )
        return ""



