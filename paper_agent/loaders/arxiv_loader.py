"""arXiv 加载器 —— 委托给 arxiv_mcp.core，不与 MCP server 重复实现。

策略：
  1. 全文 → arxiv_mcp.core.get_paper_full_text()（PDF + ar5iv 兜底）
  2. 元数据 → arxiv_mcp.core.get_paper_metadata()（Atom API 多域名 + abs HTML 兜底）
  3. PDF 磁盘缓存 → arxiv_mcp.core.download_paper_pdf()
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# arxiv_mcp 是项目根目录的顶层包，未随 paper_agent 一起安装到 site-packages。
# 通过 paper.exe 运行时（工作目录不在 sys.path），需把项目根目录临时加进 sys.path。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from arxiv_mcp.core import (
    download_paper_pdf,
    extract_arxiv_id,
    get_paper_full_text,
    get_paper_metadata,
)

from paper_agent.loaders.base import LoadedPaper
from paper_agent.models import NormalizedDocument

# 当前 arxiv_mcp 不支持的信息通过变量承载（便于后续升级统一）
ARXIV_MCP_MAX_CHARS = 500_000  # 不加截断，图形节点自行控制 token 预算


def load_arxiv(input_str: str) -> LoadedPaper:
    """下载并解析 arXiv 论文（全文 + 元数据）。"""
    arxiv_id = extract_arxiv_id(input_str)

    full = get_paper_full_text(arxiv_id, max_chars=ARXIV_MCP_MAX_CHARS)
    meta = get_paper_metadata(arxiv_id)

    # 同时把 PDF 缓存到本地，方便后续离线访问
    try:
        download_paper_pdf(arxiv_id)
    except Exception:
        pass  # 磁盘缓存失败不阻塞主流程

    doc_id = hashlib.sha256(f"arxiv|{arxiv_id}".encode()).hexdigest()[:16]
    return LoadedPaper(
        paper=NormalizedDocument(
            id=doc_id,
            source_type="arxiv",
            title=full.get("title") or meta.get("title") or f"arXiv:{arxiv_id}",
            url=meta.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}",
            doi=meta.get("doi"),
            published=meta.get("published"),
            authors=meta.get("authors", []),
            abstract=meta.get("abstract"),
            text=full["text"],
            num_pages=None,
        )
    )
