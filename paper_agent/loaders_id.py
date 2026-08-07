"""轻量 ID 计算模块（位于 paper_agent 包顶层，不触发 loaders 包导入）。

只依赖 hashlib/re/arxiv_mcp.core（httpx），**不导入任何 loader**，
避免在缓存命中场景拖入 langchain_community / transformers 等重依赖。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from arxiv_mcp.core import extract_arxiv_id

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


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