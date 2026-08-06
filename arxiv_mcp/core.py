"""arXiv 数据访问层：检索 / 元数据 / 全文提取。

网络稳健性策略（国内网络适配）：
  1. Atom API 多域名兜底（arxiv.org/api/query → export.arxiv.org/api/query）
  2. 元数据失败降级解析 abs 页面 HTML（arxiv.org/abs/<id>）
  3. PDF 全文失败降级 ar5iv HTML（ar5iv.labs.arxiv.org/html/<id>）
  4. HTTP 429/5xx 自动指数退避重试
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Literal

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS = "https://arxiv.org/abs/{arxiv_id}"
AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_ATOM_API_HOSTS = ["https://arxiv.org/api/query", "https://export.arxiv.org/api/query"]
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

_USER_AGENT = "Mozilla/5.0 (arxiv-mcp/0.1; research tool) python-httpx"
_TIMEOUT = 30.0

# 新格式 (2404.07143) 与 2007 年前老格式 (cmp-lg/9701001) 都支持
_ARXIV_ID_RE = re.compile(r"^((?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{4,7}))(?:v\d+)?$")
_ANY_ID_IN_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z-]+(?:\.[A-Z]{2})?/\d{4,7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)")
_SEARCH_FIELD_RE = re.compile(r"^[a-z]{2}:")
_HEADER_LINE_RE = re.compile(r"^arXiv:\S+\s+\[[^\]]*\]\s+(.+)$")
_ABSTRACT_HEAD_RE = re.compile(r"^abstract\s*[:—\-]?\s*$", re.I)
_SECTION_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Za-z]", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+", re.I)
_PREPRINT_RE = re.compile(r"preprint|under review|for consideration|submitted to", re.I)


def extract_arxiv_id(input_str: str) -> str:
    """从裸 ID 或 abs/pdf 链接中提取 arXiv ID。"""
    s = input_str.strip()
    m = _ARXIV_ID_RE.match(s)
    if m:
        return m.group(1)
    m = _ANY_ID_IN_URL_RE.search(s)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    raise ValueError(f"无法识别的 arXiv 输入（支持裸 ID 如 2404.07143 或 arxiv.org 链接）: {input_str!r}")


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 500, 502, 503, 504)


@retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _get(url: str, *, params: dict | None = None, timeout: float = _TIMEOUT) -> httpx.Response:
    resp = httpx.get(
        url,
        params=params,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Atom API：检索 / 元数据
# ---------------------------------------------------------------------------

def _atom_query(params: dict, timeout: float = 15.0) -> list[dict]:
    """跨域名请求 Atom API，全部失败抛异常。返回 entry 列表。"""
    import xml.etree.ElementTree as ET

    last_exc: Exception | None = None
    for host in _ATOM_API_HOSTS:
        try:
            resp = httpx.get(
                host,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
                timeout=timeout,
            )
            resp.raise_for_status()
            return _parse_atom_entries(resp.text, ET)
        except Exception as exc:  # 换下一个域名
            last_exc = exc
    raise RuntimeError(f"arXiv Atom API 请求失败（{', '.join(_ATOM_API_HOSTS)}）: {last_exc}")


def _parse_atom_entries(xml_text: str, ET) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries: list[dict] = []
    for e in root.findall("atom:entry", {"atom": _ATOM_NS}):
        entries.append(_entry_to_dict(e, ET))
    return entries


def _entry_to_dict(e, ET) -> dict:
    ns = {"atom": _ATOM_NS, "arxiv": _ARXIV_NS}
    id_ = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
    m = re.search(r"abs/([a-z-]+(?:\.[A-Z]{2})?/\d{4,7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)", id_)
    arxiv_id = m.group(1) if m else id_
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # 去掉版本后缀
    pc = e.find("arxiv:primary_category", ns)
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join((e.findtext("atom:title", default="", namespaces=ns) or "").split()) or None,
        "abstract": " ".join((e.findtext("atom:summary", default="", namespaces=ns) or "").split()) or None,
        "authors": [a.text.strip() for a in e.findall("atom:author/atom:name", ns) if a.text and a.text.strip()],
        "published": e.findtext("atom:published", default=None, namespaces=ns),
        "updated": e.findtext("atom:updated", default=None, namespaces=ns),
        "doi": e.findtext("arxiv:doi", default=None, namespaces=ns),
        "journal_ref": e.findtext("arxiv:journal_ref", default=None, namespaces=ns),
        "comment": e.findtext("arxiv:comment", default=None, namespaces=ns),
        "categories": [c.get("term") for c in e.findall("atom:category", ns) if c.get("term")],
        "primary_category": pc.get("term") if pc is not None else None,
    }


def search_papers(query: str, max_results: int = 10, start: int = 0, sort_by: str = "relevance") -> list[dict]:
    """按 arXiv 检索语法搜索论文。"""
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    if max_results < 1 or max_results > 100:
        raise ValueError("max_results 需在 1–100 之间")
    if sort_by not in ("relevance", "submittedDate", "lastUpdatedDate"):
        raise ValueError("sort_by 需为 relevance | submittedDate | lastUpdatedDate")

    # 裸关键词自动转 all:；已含字段前缀（ti: / au: / cat: 等）则原样透传
    search_query = query if _SEARCH_FIELD_RE.match(query) else f"all:{query}"
    entries = _atom_query({
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
    })
    out = []
    for e in entries:
        out.append({
            "arxiv_id": e["arxiv_id"],
            "title": e["title"],
            "authors": e["authors"],
            "published": e["published"],
            "categories": e["categories"],
            "abstract": e["abstract"][:800] + ("…" if e["abstract"] and len(e["abstract"]) > 800 else "") if e["abstract"] else None,
            "abs_url": f"https://arxiv.org/abs/{e['arxiv_id']}",
        })
    return out


# ---------------------------------------------------------------------------
# 元数据兜底：abs 页面 HTML 解析
# ---------------------------------------------------------------------------

def _parse_abs_html(html_text: str) -> dict:
    """从 arxiv.org/abs/<id> 页面 HTML 提取元数据（Atom API 失败时兜底）。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    title = None
    h1 = soup.find("h1", class_="title")
    if h1:
        title = re.sub(r"^Title:\s*", "", h1.get_text(" ", strip=True)).strip() or None

    authors = []
    div_authors = soup.find("div", class_="authors")
    if div_authors:
        authors = [a.get_text(" ", strip=True) for a in div_authors.find_all("a") if a.get_text(" ", strip=True).strip()]

    abstract = None
    block_abs = soup.find("blockquote", class_="abstract")
    if block_abs:
        abstract = re.sub(r"^Abstract:\s*", "", block_abs.get_text(" ", strip=True)).strip() or None

    published = None
    div_date = soup.find("div", class_="dateline")
    if div_date:
        published = div_date.get_text(" ", strip=True).strip() or None

    return {"title": title, "authors": authors, "abstract": abstract, "published": published}


def get_paper_metadata(arxiv_id: str) -> dict:
    """按 arXiv ID 取元数据：Atom API 多域名 → abs HTML 兜底。"""
    arxiv_id = extract_arxiv_id(arxiv_id)
    try:
        entries = _atom_query({"id_list": arxiv_id})
        if entries:
            e = entries[0]
            return {
                "arxiv_id": arxiv_id,
                "title": e["title"],
                "authors": e["authors"],
                "abstract": e["abstract"],
                "published": e["published"],
                "updated": e["updated"],
                "doi": e["doi"],
                "journal_ref": e["journal_ref"],
                "comment": e["comment"],
                "categories": e["categories"],
                "primary_category": e["primary_category"],
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            }
    except Exception:
        pass  # 降级 abs HTML
    resp = _get(ARXIV_ABS.format(arxiv_id=arxiv_id))
    meta = _parse_abs_html(resp.text)
    if not meta["title"]:
        raise RuntimeError(f"arXiv {arxiv_id} 元数据获取失败（Atom API 与 abs 页面均失败）")
    return {
        "arxiv_id": arxiv_id,
        **meta,
        "doi": None, "journal_ref": None, "comment": None, "categories": [], "primary_category": None,
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


# ---------------------------------------------------------------------------
# 全文提取：PDF（pypdf）→ ar5iv HTML 兜底
# ---------------------------------------------------------------------------

def _parse_pdf_bytes(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _parse_ar5iv_html(arxiv_id: str) -> str:
    from bs4 import BeautifulSoup

    resp = _get(AR5IV_HTML.format(arxiv_id=arxiv_id))
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _parse_metadata_from_text(text: str) -> dict:
    """从 PDF 首页文本解析 标题/作者/摘要（尽力而为，与 paper_agent 同源策略）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    published = None
    for ln in lines[:12]:
        m = _HEADER_LINE_RE.match(ln)
        if m:
            published = m.group(1)
            break

    abs_start = None
    for i, ln in enumerate(lines):
        if _ABSTRACT_HEAD_RE.match(ln) or ln.lower().startswith("abstract:"):
            abs_start = i
            break

    header = lines[: abs_start] if abs_start is not None else lines

    abstract = None
    if abs_start is not None:
        abs_lines = []
        for ln in lines[abs_start + 1:]:
            if _SECTION_RE.match(ln):
                break
            abs_lines.append(ln)
        abstract = " ".join(abs_lines).strip()[:2000] or None

    title = None
    authors: list[str] = []
    author_idx = None
    for i, ln in enumerate(header):
        if _EMAIL_RE.search(ln):
            continue
        if _PREPRINT_RE.search(ln) and len(ln) < 40:
            continue
        if re.fullmatch(r"[A-Za-z ]{0,20}", ln) and len(ln) < 16:
            continue
        if ("," in ln or " and " in ln.lower()) and len(ln) > 15:
            author_idx = i
            authors = [
                a.strip()
                for a in re.split(r",\s*|\band\b", ln)
                if a.strip() and len(a.strip()) > 2
            ]
            break

    title_lines = [ln for ln in header[: author_idx if author_idx is not None else 4]
                   if not _HEADER_LINE_RE.match(ln)  # 跳过 "arXiv:2404.07143 [cs.CL] 10 Apr 2024" 页眉
                   and not (_PREPRINT_RE.search(ln) and len(ln) < 40)]
    if author_idx is not None or title_lines:
        title = " ".join(title_lines)[:300] or None
    title = re.sub(r"\s+", " ", title or "").strip() or None

    return {"title": title, "authors": authors, "abstract": abstract, "published": published}


def get_paper_full_text(arxiv_id: str, max_chars: int = 120_000) -> dict:
    """下载并解析全文。返回 {arxiv_id, title, source, chars, truncated, text}。"""
    arxiv_id = extract_arxiv_id(arxiv_id)
    if max_chars < 500 or max_chars > 500_000:
        raise ValueError("max_chars 需在 500–500000 之间")

    pdf_error: str | None = None
    text = ""
    try:
        pdf_resp = _get(ARXIV_PDF.format(arxiv_id=arxiv_id), timeout=60.0)
        text = _parse_pdf_bytes(pdf_resp.content)
    except Exception as exc:
        pdf_error = str(exc)

    source: Literal["pdf", "ar5iv"] = "pdf"
    if len(text) < 500:
        try:
            text = _parse_ar5iv_html(arxiv_id)
            source = "ar5iv"
        except Exception as ar5iv_exc:
            if pdf_error:
                raise RuntimeError(f"arXiv PDF 下载/解析失败: {pdf_error}") from ar5iv_exc
            raise RuntimeError(f"arXiv HTML 解析失败: {ar5iv_exc}") from ar5iv_exc

    if len(text) < 200:
        raise RuntimeError(f"arXiv {arxiv_id} 内容为空，请稍后重试")

    # 标题：Atom 元数据 > abs HTML > PDF 文本解析
    title = None
    try:
        meta = get_paper_metadata(arxiv_id)
        title = meta.get("title")
    except Exception:
        meta = _parse_metadata_from_text(text)
        title = meta.get("title")

    truncated = len(text) > max_chars
    return {
        "arxiv_id": arxiv_id,
        "title": title or f"arXiv:{arxiv_id}",
        "source": source,
        "chars": len(text),
        "truncated": truncated,
        "text": text[:max_chars],
    }


def download_paper_pdf(arxiv_id: str, output_dir: str | None = None) -> dict:
    """下载 PDF 到本地磁盘，返回 {path, size_bytes, bytes}。"""
    arxiv_id = extract_arxiv_id(arxiv_id)
    out_dir = Path(output_dir or "data/pdfs").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = _get(ARXIV_PDF.format(arxiv_id=arxiv_id), timeout=60.0)
    path = out_dir / f"{arxiv_id}.pdf"
    path.write_bytes(resp.content)
    return {
        "arxiv_id": arxiv_id,
        "path": str(path),
        "size_bytes": len(resp.content),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }
