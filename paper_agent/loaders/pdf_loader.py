"""本地 PDF 加载器：基于 langchain_community 的 PyPDFLoader（内部使用 pypdf）。

完全离线可用，不发起任何网络请求。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

# 静音 pypdf 对损坏文件的解析噪音日志
logging.getLogger("pypdf").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader

from paper_agent.loaders.base import LoadedPaper
from paper_agent.models import NormalizedDocument

# 首行清理：去掉行内序号等噪音（保守处理，避免误删内容）
_HEADER_CLEAN = re.compile(r"^\s*(?:arXiv:\s*\S+\s+)?(\d{4}\.\d{4,5})\s*$")
# 会议/卷册页眉（"Proceedings of ...", "In Proceedings of ...", 日期、页码等）
_VENUE_RE = re.compile(
    r"^(proceedings|in proceedings|proceedings of|\d{1,3}\b|doi[: ]|©|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\b)",
    re.I,
)


def _infer_title(pages_text: str, fallback: str) -> str:
    """从首页文本推断标题：取首个有意义的完整行。"""
    first_page = (pages_text.split("\n\n")[0] if pages_text else "").strip()
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    for ln in lines[:12]:
        if _HEADER_CLEAN.match(ln):
            continue
        if _VENUE_RE.match(ln):
            continue
        if len(ln) >= 8 and not re.fullmatch(r"[\d\s/:\-]+", ln):
            return ln[:200]
    return fallback


def load_pdf(path: str | Path) -> LoadedPaper:
    """解析本地 PDF 文件。损坏/加密文件会抛出清晰异常。"""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件: {pdf_path}")

    try:
        docs = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:  # pypdf 对损坏/加密文件抛各类异常
        raise RuntimeError(f"PDF 解析失败（可能损坏或加密）: {pdf_path}: {exc}") from exc

    if not docs:
        raise RuntimeError(f"PDF 无内容: {pdf_path}")

    pages = [d.page_content or "" for d in docs]
    text = "\n\n".join(pages).strip()
    if len(text) < 50:
        raise RuntimeError(f"PDF 文本提取为空（可能为扫描件）: {pdf_path}")

    title = _infer_title(text, pdf_path.stem)
    doc_id = hashlib.sha256(f"pdf|{str(pdf_path.resolve())}".encode()).hexdigest()[:16]

    return LoadedPaper(
        paper=NormalizedDocument(
            id=doc_id,
            source_type="pdf",
            title=title,
            url=None,
            authors=[],
            abstract=None,
            text=text,
            num_pages=len(pages),
        )
    )
