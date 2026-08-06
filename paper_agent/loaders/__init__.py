"""加载器分发：根据输入形式自动选择 PDF / arXiv / 网页。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from paper_agent.loaders.base import LoadedPaper
from paper_agent.loaders.arxiv_loader import extract_arxiv_id, load_arxiv
from paper_agent.loaders.pdf_loader import load_pdf
from paper_agent.loaders.web_loader import load_web

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

__all__ = ["LoadedPaper", "load_paper", "cache_id_for"]


def load_paper(input_str: str) -> LoadedPaper:
    """统一入口：识别来源并加载论文。

    - 以 .pdf 结尾 → 本地 PDF
    - arXiv ID（如 2404.07143）或含 arxiv.org/abs → arXiv
    - http(s):// → 网页
    """
    s = input_str.strip()
    lower = s.lower()

    if lower.endswith(".pdf") or Path(s).suffix.lower() == ".pdf":
        return load_pdf(s)

    if _ARXIV_ID_RE.match(s) or "arxiv.org/abs/" in lower or "arxiv.org/pdf/" in lower:
        return load_arxiv(s)

    if lower.startswith(("http://", "https://")):
        return load_web(s)

    raise ValueError(
        f"无法识别的输入: {input_str!r}。支持：本地 PDF 路径、arXiv ID（如 2404.07143）、"
        "arXiv 链接、或 http(s):// 网页 URL"
    )


def cache_id_for(input_str: str) -> tuple[str, str]:
    """不加载论文，仅由输入计算 (source_type, cache_id)。

    与各 loader 内部的 id 计算保持一致，用于缓存命中预检。
    """
    s = input_str.strip()
    lower = s.lower()

    if lower.endswith(".pdf") or Path(s).suffix.lower() == ".pdf":
        resolved = str(Path(s).resolve())
        return "pdf", hashlib.sha256(f"pdf|{resolved}".encode()).hexdigest()[:16]

    if _ARXIV_ID_RE.match(s) or "arxiv.org/abs/" in lower or "arxiv.org/pdf/" in lower:
        arxiv_id = extract_arxiv_id(s)
        return "arxiv", hashlib.sha256(f"arxiv|{arxiv_id}".encode()).hexdigest()[:16]

    if lower.startswith(("http://", "https://")):
        return "web", hashlib.sha256(f"web|{s}".encode()).hexdigest()[:16]

    raise ValueError(
        f"无法识别的输入: {input_str!r}。支持：本地 PDF 路径、arXiv ID（如 2404.07143）、"
        "arXiv 链接、或 http(s):// 网页 URL"
    )
