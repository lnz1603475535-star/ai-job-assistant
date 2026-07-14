"""
AI 简历生成器 - 核心基础设施
============================
纯基础设施层：LLM 配置、Embedding、文档索引、混合检索。
不包含任何业务逻辑——业务逻辑在 resume_engine.py 中。
"""

import html as _html
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

def set_vectorstore(vs, chunks):
    """设置全局向量库实例，BM25 由 chunks 自动构建，保证三者永远一致。"""
    global _vectorstore, _bm25_index, _chunks_text, _chunks_metadata
    _vectorstore = vs
    if chunks:
        _chunks_text = [c.page_content for c in chunks]
        _chunks_metadata = [c.metadata for c in chunks]
        _bm25_index = BM25Okapi([jieba.lcut(c.page_content) for c in chunks])
    else:
        _chunks_text = []
        _chunks_metadata = []
        _bm25_index = None

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
    """加载文件文本内容，自动检测 .txt / .pdf / .docx / .md。

    委托给 _load_file_to_documents，避免重复文件检测和错误处理逻辑。

    Raises:
        ValueError: 加密 / 扫描件无文字 / 文件损坏
    """
    docs = _load_file_to_documents(path)
    text = normalize_text("\n\n".join(d.page_content for d in docs))
    if not text.strip():
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            raise ValueError("PDF 可能是扫描件，无法提取文字。请上传含文本的 PDF 或直接粘贴文字内容。")
        raise ValueError("文件内容为空，请检查后重新上传。")
    return text


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
    text = _html.unescape(text)
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


def load_and_index_documents(file_paths: dict[str, list[str]]) -> tuple[object, list[Document]]:
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

    for i, doc in enumerate(chunks):
        doc.metadata["_chunk_idx"] = i

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, chunks

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

    # FAISS + BM25 各取 _fetch_k 条进入 RRF 融合，最后截断到 k
    _fetch_k = max(k * 5, min(len(_chunks_text), 20))
    faiss_docs = _vectorstore.similarity_search(query, k=_fetch_k)

    # BM25 关键词检索（jieba 中文分词）
    bm25_scores = _bm25_index.get_scores(jieba.lcut(query))
    if len(bm25_scores) > 0:
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:_fetch_k]
    else:
        bm25_top_indices = []

    # RRF 融合：Reciprocal Rank Fusion，k=60
    # 用 chunk 索引做 key，避免字符串内容微小差异导致融合作废
    K = 60
    rrf_scores: dict[int, float] = {}  # chunk_index → 累计 RRF 得分

    # FAISS 排名得分（用 _chunk_idx 元数据，避免字符串微小差异导致匹配失败）
    for rank, doc in enumerate(faiss_docs):
        idx = doc.metadata.get("_chunk_idx", -1)
        if idx == -1 or idx >= len(_chunks_text):
            continue
        rrf_scores[idx] = 1.0 / (K + rank)

    # BM25 排名得分，与 FAISS 累加
    for rank, idx in enumerate(bm25_top_indices):
        score = 1.0 / (K + rank)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + score

    # 按 RRF 得分降序排列，取 top-k
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    merged = []
    for idx, score in sorted_indices:
        content = _chunks_text[idx]
        doc_type = _chunks_metadata[idx].get("doc_type", "unknown")
        merged.append(f"[{doc_type}] {content}")

    if not merged:
        return "未找到相关文档。"

    return "\n\n---\n\n".join(merged)

# ============================================================
# Token 预算控制（基于 API 返回的真实 token 计数）
# ============================================================

class TokenBudget:
    """Token 预算跟踪器，从 API 响应的 usage_metadata 中提取真实 token 用量。
    用法：在 workflow 中创建 budget 并传给 customize_for_jd。
    LLM 响应的 response_metadata["token_usage"] 包含真实的 input/output token 数。
    """

    def __init__(self, max_tokens: int = 15000, warning_ratio: float = 0.7):
        self.max_tokens = max_tokens
        self.warning_ratio = warning_ratio
        self.input_tokens = 0
        self.output_tokens = 0
        self._warning_issued = False

    def record(self, input_tokens: int = 0, output_tokens: int = 0):
        """累积记录 token 用量。"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @staticmethod
    def _parse_usage(usage: dict) -> tuple[int, int]:
        """从 LLM 响应中提取 (input_tokens, output_tokens)，自动处理各厂商字段名差异。
        只提取不记录，可复用。"""
        if not usage:
            return 0, 0
        input_tokens = (
            usage["prompt_tokens"] if "prompt_tokens" in usage
            else usage.get("input_tokens", 0)
        )
        output_tokens = (
            usage["completion_tokens"] if "completion_tokens" in usage
            else usage.get("output_tokens", 0)
        )
        return input_tokens, output_tokens

    def record_from_response(self, usage: dict):
        """从 LLM 响应中记录 token 用量。"""
        inp, out = self._parse_usage(usage)
        self.input_tokens += inp
        self.output_tokens += out

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self.total_tokens / self.max_tokens

    def get_warning(self) -> str:
        ratio = self.usage_ratio
        if ratio >= 0.9:
            return (
                f"⚠️ Token 预算即将耗尽（{self.total_tokens}/{self.max_tokens}，{ratio:.0%}）。"
                f"请保持回复简洁，优先输出关键内容，尽快给出最终结论。"
            )
        if ratio >= self.warning_ratio and not self._warning_issued:
            self._warning_issued = True
            return (
                f"⚠️ Token 使用量已达 {ratio:.0%}（{self.total_tokens}/{self.max_tokens}）。"
                f"请注意控制输出长度。"
            )
        return ""

    def get_usage_report(self) -> str:
        return (
            f"Token 用量：输入 {self.input_tokens} + 输出 {self.output_tokens}"
            f" = {self.total_tokens} / {self.max_tokens}（{self.usage_ratio:.0%}）"
        )



