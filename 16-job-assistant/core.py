"""
AI 简历生成器 - 核心基础设施
============================
纯基础设施层：LLM 配置、Embedding、文档索引、混合检索。
不包含任何业务逻辑——业务逻辑在 resume_engine.py 中。
"""

import logging
import os
import re
import threading
import warnings

# 抑制依赖库的噪音警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ============================================================
# 日志配置
# ============================================================
import atexit
from logging.handlers import RotatingFileHandler
from typing import Any, ClassVar

import jieba
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
_log_initialized = False

# ============================================================
# 路径常量（单一来源——app.py / api.py 一律从这里导入，
# 避免同一条路径在多个模块各写一遍、改一处漏一处）
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXPERIENCE_BANK_PATH = os.path.join(DATA_DIR, "experience_bank.md")


def sanitize_error(exc: Exception | str) -> str:
    """脱敏异常信息：替换用户目录路径，截断到 200 字符。

    基础设施级工具——放在 core 而非 exporters（原先在 exporters 中定义，
    却被 api.py / app.py 当通用工具跨模块导入，归属错位）。
    """
    msg = str(exc)
    home = os.path.expanduser("~")
    if home and home != "~":
        msg = msg.replace(home, "~")
    return msg[:200]


class SensitiveFilter(logging.Filter):
    """日志敏感信息脱敏：API key、手机号、邮箱。"""

    _patterns: ClassVar[list[tuple[str, str]]] = [
        (r"sk-[a-zA-Z0-9_-]{20,}", "sk-***"),
        (r"1[3-9]\d{9}", "138****0000"),
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "***@***.***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in self._patterns:
                record.msg = re.sub(pattern, replacement, record.msg)
        if isinstance(record.args, dict):
            new_args = {}
            for k, v in record.args.items():
                if isinstance(v, str):
                    for pattern, replacement in self._patterns:
                        v = re.sub(pattern, replacement, v)
                new_args[k] = v
            record.args = new_args
        elif isinstance(record.args, tuple):
            new_args = []
            for a in record.args:
                if isinstance(a, str):
                    for pattern, replacement in self._patterns:
                        a = re.sub(pattern, replacement, a)
                new_args.append(a)
            record.args = tuple(new_args)
        return True


def setup_logging() -> None:
    """配置日志系统（幂等调用——多次调用不会重复注册）。

    - 控制台：WARNING+，简洁格式
    - data/app.log：DEBUG+，含行号定位信息，RotatingFileHandler 10MB×3
    - 第三方库噪音静音（httpx / urllib3 / openai / langchain 等）
    - 敏感信息自动脱敏（API key / 手机号 / 邮箱）
    """
    global _log_initialized
    if _log_initialized:
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台 handler：WARNING+
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    console.addFilter(SensitiveFilter())
    root.addHandler(console)

    # 文件 handler：DEBUG+，含定位信息，自动轮转
    file_handler = RotatingFileHandler(
        os.path.join(DATA_DIR, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s:%(lineno)d %(funcName)s() — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler.addFilter(SensitiveFilter())
    root.addHandler(file_handler)

    # 第三方库静音
    _noisy_libs = [
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "openai._base_client",
        "langchain_core",
        "langchain",
        "langgraph",
        "huggingface_hub",
        "transformers",
        "filelock",
        "fsspec",
        "tqdm",
    ]
    for name in _noisy_libs:
        logging.getLogger(name).setLevel(logging.WARNING)
    # transformers 的 __path__ 废弃警告是 WARNING 级别，提到 ERROR 才能消掉
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _log_initialized = True
    logging.getLogger(__name__).info("日志系统已初始化")

    # 确保程序退出时刷新所有 handler
    atexit.register(logging.shutdown)


load_dotenv(find_dotenv())

# ============================================================
# LLM 配置
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # pyright: ignore[reportArgumentType]——stub 过严，运行时接受 str
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
)

# ============================================================
# Embedding 模型（懒加载 + 单例模式）
# ============================================================

_embeddings = None
# 加载模型耗时且占内存（约 400MB），并发首次调用会各构造一份
_embeddings_lock = threading.Lock()


def get_embeddings():
    """获取 embedding 模型实例（单例模式，首次调用时加载）。"""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:  # 双重检查：等锁期间可能已被别的线程建好
                _embeddings = HuggingFaceEmbeddings(
                    model_name="shibing624/text2vec-base-chinese",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings


# ============================================================
# 检索服务（RetrievalService 类封装，替代模块级全局状态）
# ============================================================

# 分词清洗：只保留"词"字符（汉字 / 字母 / 数字 / 下划线）
# 不过滤的话，jieba 会把空格、标点也切成 token 参与 BM25 打分——
# " " 和 "。" 这类 token 几乎每块都有，会给出与语义无关的分数，
# 实测中曾让一个完全无关的块排到第一位（见 troubleshooting 记录）
_TOKEN_RE = re.compile(r"^\w+$")


def _tokenize(text: str) -> list[str]:
    """jieba 分词 + 清洗：去空格/标点 token，统一小写。

    索引与查询必须走同一个函数，否则大小写/标点处理不一致会导致漏召回
    （原先索引和查询各写一遍 jieba.lcut，"Python" 与 "python" 无法匹配）。
    """
    return [t.lower() for t in jieba.lcut(text) if _TOKEN_RE.match(t)]


class RetrievalService:
    """混合检索服务：FAISS 语义检索 + BM25 关键词检索 + RRF 融合。

    持有向量库与关键词索引的全部状态（#10 重构：原模块级全局变量
    _vectorstore/_bm25_index/_chunks_text/_chunks_metadata 收敛于此）。
    实例化即建立完整双索引，不存在"忘了写入"的中间状态；
    FastAPI 阶段可对每个请求注入独立实例（依赖注入）。
    """

    def __init__(self, vectorstore: Any, chunks: list[Document]):
        self._vectorstore = vectorstore
        if chunks:
            self._chunks_text = [c.page_content for c in chunks]
            self._chunks_metadata = [c.metadata for c in chunks]
            # 全部块都切不出 token（如纯符号文档）时 BM25 无意义，
            # 且 BM25Okapi 以 0 长度语料计算会除零，故置 None 降级为纯语义检索
            tokenized = [_tokenize(c.page_content) for c in chunks]
            self._bm25_index = BM25Okapi(tokenized) if any(tokenized) else None
        else:
            self._chunks_text = []
            self._chunks_metadata = []
            self._bm25_index = None

    @property
    def is_ready(self) -> bool:
        """检索服务是否可用（向量库与语料均就绪；BM25 为可选增强）。"""
        return self._vectorstore is not None and bool(self._chunks_text)

    @property
    def chunk_count(self) -> int:
        """已索引的文本块数量。"""
        return len(self._chunks_text)

    def search(self, query: str, k: int = 4, doc_types: list[str] | None = None) -> str:
        """混合搜索：FAISS 语义 + BM25 关键词 → RRF 融合 → top-k。

        返回格式：每块以 "[doc_type] " 开头，块间以 "\n\n---\n\n" 分隔。
        """
        if not self.is_ready:
            return "尚未加载任何文档。"

        # 防止空查询返回随机结果
        if not query.strip():
            return "未提供搜索词。"

        # 粗排候选数：至少 20 条并随 k 放大，但不超过语料总量
        _fetch_k = min(len(self._chunks_text), max(k * 5, 20))
        faiss_docs = self._vectorstore.similarity_search(query, k=_fetch_k)

        # BM25 关键词检索（中文分词，索引与查询共用 _tokenize）
        bm25_top_indices: list[int] = []
        if self._bm25_index is not None:
            bm25_scores = self._bm25_index.get_scores(_tokenize(query))
            if len(bm25_scores) > 0:
                # 只收 score > 0 的候选。零分块与查询毫无关键词交集，
                # 放进 RRF 会拿到 1/(K+rank+1)，排名靠前时与 FAISS 头名同分，
                # 把真正的语义命中挤出 top-k（实测 top-4 被无关块占掉一半）
                bm25_top_indices = [
                    i
                    for i in sorted(
                        range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i],
                        reverse=True,
                    )
                    if bm25_scores[i] > 0
                ][:_fetch_k]

        # RRF 融合：Reciprocal Rank Fusion，K=60（rank 从 1 起算，对齐标准定义）
        # 用 chunk 索引做 key，避免字符串内容微小差异导致融合作废
        K = 60
        rrf_scores: dict[int, float] = {}  # chunk_index → 累计 RRF 得分

        # FAISS 排名得分（用 _chunk_idx 元数据，避免字符串微小差异导致匹配失败）
        for rank, doc in enumerate(faiss_docs, start=1):
            idx = doc.metadata.get("_chunk_idx", -1)
            if idx == -1 or idx >= len(self._chunks_text):
                continue
            rrf_scores[idx] = 1.0 / (K + rank)

        # BM25 排名得分，与 FAISS 累加
        for rank, idx in enumerate(bm25_top_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (K + rank)

        # 按 RRF 得分降序排列，取 top-k（doc_types 过滤后再截断，不足 k 就返回少一些）
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_indices = [
            (idx, score)
            for idx, score in sorted_indices
            if doc_types is None
            or self._chunks_metadata[idx].get("doc_type") in doc_types
        ][:k]

        merged = []
        for idx, _score in filtered_indices:
            content = self._chunks_text[idx]
            doc_type = self._chunks_metadata[idx].get("doc_type", "unknown")
            merged.append(f"[{doc_type}] {content}")

        if not merged:
            return "未找到相关文档。"

        return "\n\n---\n\n".join(merged)


# 当前活跃检索服务（模块级指针）。FastAPI 阶段用依赖注入替换实例，
# 但保留模块级注册表：Agent 的 @tool 包装（search_documents）与工作流
# 直接调用（search_documents_impl）仍从这里读取当前实例。
_retrieval_service: RetrievalService | None = None


def set_retrieval_service(service: RetrievalService | None) -> None:
    """注册/替换当前检索服务实例（None 表示清空）。"""
    global _retrieval_service
    _retrieval_service = service


def get_retrieval_service() -> RetrievalService | None:
    """获取当前检索服务实例（未注册时返回 None）。"""
    return _retrieval_service


# ============================================================
# 文档处理
# ============================================================


# 列表项 / 标题行的识别（normalize_text 逐行判断用，模块级避免循环内重复构造）
# \s* 容忍 PDF 解析丢掉空格 / 全角空格（\s 已覆盖 U+3000）
_LIST_ITEM_RE = re.compile(
    r"^[-*•]\s*"  # -item  *item  •item（含空格/全角空格）
    r"|^\d+[.、)]\s*"  # 1.item  1、item  1)item
    r"|^（[一二三四五六七八九十\d]+）"  # （一）（1）
    r"|^[一二三四五六七八九十]+[、．]"  # 一、item  二．item
    r"|^#{1,6}\s"  # ## heading（标题必须有空格）
)


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
        is_special = bool(_LIST_ITEM_RE.match(stripped))
        prev_is_special = bool(_LIST_ITEM_RE.match(prev_stripped))

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
            raise ValueError(
                "PDF 可能是扫描件，无法提取文字。请上传含文本的 PDF 或直接粘贴文字内容。"
            )
        raise ValueError("文件内容为空，请检查后重新上传。")
    return text


_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _load_file_to_documents(path: str) -> list[Document]:
    """加载文件为 LangChain Document 列表，自动检测 .txt / .pdf / .docx / .md。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pdf", ".docx", ".txt", ".md", ""):
        raise ValueError(
            f"不支持的文件格式：{ext}。支持的格式：.txt / .pdf / .docx / .md"
        )

    try:
        file_size = os.path.getsize(path)
    except OSError as e:
        raise ValueError(f"文件不存在或无法访问：{os.path.basename(path)}") from e
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
            raise ValueError(f"文件已加密，无法读取：{os.path.basename(path)}") from e
        elif (
            "corrupt" in error_msg
            or "not a pdf" in error_msg
            or "not a valid" in error_msg
        ):
            raise ValueError(f"文件已损坏或格式异常：{os.path.basename(path)}") from e
        else:
            raise ValueError(
                f"文件读取失败（{os.path.basename(path)}）：{str(e)[:100]}"
            ) from e


def load_and_index_documents(file_paths: dict[str, list[str]]) -> RetrievalService:
    """加载多类型文档，切分，建立 FAISS + BM25 双索引，注册为当前检索服务。

    参数：
        file_paths: {"doc_type": ["path1", "path2"], ...}
        例如：{"sample_resume": ["samples/xxx.txt"], "jd": ["jd_python.txt"]}

    返回：
        RetrievalService 实例（已自动注册，search_documents 立即可用，
        调用方无需再手动写入——#10 消除调用方遗漏风险）
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
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "。", "！", "？", "\n", "，", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("文档切分后无有效内容，请检查文件是否只有空白字符。")

    for i, doc in enumerate(chunks):
        doc.metadata["_chunk_idx"] = i

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    service = RetrievalService(vectorstore, chunks)
    set_retrieval_service(service)
    return service


# ============================================================
# 工具定义
# ============================================================


@tool
def search_documents(query: str, k: int = 4, doc_types: list[str] | None = None) -> str:
    """混合搜索文档数据库（FAISS 语义检索 + BM25 关键词匹配）。
    用于查找用户经历、JD 要求、样本风格等已索引的文档内容。

    参数：
        query：中文或英文的自然语言搜索词
        k：返回结果数量，默认 4
        doc_types：只返回指定文档类型（如 ["user_experience"]），None 返回全部类型
    """
    return search_documents_impl(query, k, doc_types)


def search_documents_impl(
    query: str, k: int = 4, doc_types: list[str] | None = None
) -> str:
    """search_documents 的核心实现——普通函数，供工具包装和工作流直接调用。"""
    service = get_retrieval_service()
    if service is None:
        return "尚未加载任何文档。"
    return service.search(query, k, doc_types)


# ============================================================
# Token 预算控制（基于 API 返回的真实 token 计数）
# ============================================================


# 预算提醒分级文案（0=正常，1=警戒，2=接近耗尽）
_TOKEN_NOTICES = {
    0: "",
    1: (
        "⚠️ Token 使用量已达 {ratio:.0%}（{total}/{max}）。请注意控制输出长度，"
        "避免大段铺陈。"
    ),
    2: (
        "⚠️ Token 预算即将耗尽（{total}/{max}，{ratio:.0%}）。请保持回复简洁，"
        "优先输出关键内容，尽快给出最终结论。"
    ),
}
_TOKEN_DANGER_RATIO = 0.9

# 单次定制请求的 token 预算上限（覆盖整条 Agent 链路，含工具调用轮次）。
# 取值依据（2026-09-12 实测两次真实定制）：
#   10140（一轮，未调工具） / 20533（两轮，含 search_documents 检索轮）
# 定得低于典型用量（如原值 15000）会导致预算只在最后一轮才被突破，
# 提醒来不及影响输出，形同虚设；30000 让警戒线落在"跑三轮以上"时，
# 正是需要模型收敛的场景。调整此值即可改变提醒触发时机。
DEFAULT_TOKEN_BUDGET = 30000


class TokenBudget:
    """Token 预算跟踪器：既记账，也**在生成过程中**真正约束模型输出。

    用法（见 resume_engine.JDTokenBudgetMiddleware）：作为 Agent 中间件在
    每次模型调用**前**调 get_warning() 取提醒注入对话、**后**调
    record_from_message() 记账，因此模型能看到预算提醒并据此收敛。

    实例按请求创建（每请求隔离，不做模块级共享）。注意 customize_for_jd
    内部有重试逻辑，重试失败的尝试无 API 响应对象、无法提取用量；
    30% 的 warning_ratio 缓冲足以覆盖这种误差（瞬时网络错误通常未实际扣费）。
    """

    def __init__(
        self, max_tokens: int = DEFAULT_TOKEN_BUDGET, warning_ratio: float = 0.7
    ):
        self.max_tokens = max_tokens
        self.warning_ratio = warning_ratio
        self.input_tokens = 0
        self.output_tokens = 0
        self._notified_level = 0  # 已注入过提醒的最高级别，保证每级只提醒一次

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
            usage["prompt_tokens"]
            if "prompt_tokens" in usage
            else usage.get("input_tokens", 0)
        )
        output_tokens = (
            usage["completion_tokens"]
            if "completion_tokens" in usage
            else usage.get("output_tokens", 0)
        )
        return input_tokens, output_tokens

    def record_from_response(self, usage: dict):
        """从 LLM 响应中记录 token 用量。"""
        inp, out = self._parse_usage(usage)
        self.input_tokens += inp
        self.output_tokens += out

    @classmethod
    def from_usage(cls, usage: dict, **kwargs: Any) -> "TokenBudget":
        """由最终用量构造预算对象（调用方只需展示用量提示时用）。"""
        budget = cls(**kwargs)
        budget.record_from_response(usage)
        return budget

    @property
    def notified_level(self) -> int:
        """已注入过提醒的最高级别（Agent 中间件在图表 state 中携带）。"""
        return self._notified_level

    def restore(
        self, input_tokens: int, output_tokens: int, notified_level: int
    ) -> None:
        """恢复累计状态（中间件从图 state 重建预算对象时用）。"""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self._notified_level = notified_level

    def record_from_message(self, message: Any) -> None:
        """从一条 LangChain 消息中记录 token 用量（Agent 中间件逐轮调用）。

        来源优先级与 resume_engine._extract_agent_token_usage 一致：
        response_metadata（同步 invoke）→ usage_metadata（流式聚合）。
        """
        meta = getattr(message, "response_metadata", None) or {}
        usage = meta.get("token_usage") if "token_usage" in meta else meta.get("usage")
        if not usage:
            usage = getattr(message, "usage_metadata", None)
        if usage:
            self.record_from_response(usage)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self.total_tokens / self.max_tokens

    @property
    def level(self) -> int:
        """当前提醒级别：0=正常，1=已达警戒线，2=接近耗尽。"""
        ratio = self.usage_ratio
        if ratio >= _TOKEN_DANGER_RATIO:
            return 2
        if ratio >= self.warning_ratio:
            return 1
        return 0

    def _notice(self, level: int) -> str:
        return _TOKEN_NOTICES[level].format(
            ratio=self.usage_ratio,
            total=self.total_tokens,
            max=self.max_tokens,
        )

    def get_warning(self) -> str:
        """取需要注入模型的预算提醒；每级只触发一次，无提醒时返回空串。

        由中间件在每次模型调用前调用——这是"控制"而非"报告"：
        提醒进入对话后模型才能真正收敛输出。
        """
        current = self.level
        if current <= self._notified_level:
            return ""
        self._notified_level = current
        return self._notice(current)

    def get_status_notice(self) -> str:
        """给人看的用量提示（UI 通知用），不消耗提醒级别。"""
        return self._notice(self.level)

    def get_usage_report(self) -> str:
        return (
            f"Token 用量：输入 {self.input_tokens} + 输出 {self.output_tokens}"
            f" = {self.total_tokens} / {self.max_tokens}（{self.usage_ratio:.0%}）"
        )
