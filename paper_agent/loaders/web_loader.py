"""网页加载器：httpx + BeautifulSoup（stdlib html.parser）。

剔除 script/style/nav/footer，优先取 article/main 正文。
"""

from __future__ import annotations

import hashlib
import re

import httpx
from bs4 import BeautifulSoup

from paper_agent.loaders.base import LoadedPaper
from paper_agent.models import NormalizedDocument

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 paper-agent/0.1"

# 与正文无关的块级标签
_DROP_TAGS = ["script", "style", "nav", "header", "footer", "iframe", "form", "aside", "noscript"]
# 页面导航等常见 class/id
_DROP_PATTERNS = re.compile(r"(nav|menu|sidebar|footer|comment|social|share|cookie|advert)", re.I)


def _drop_unwanted(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(True):
        attrs = tag.attrs or {}  # 畸形 HTML 中部分标签 attrs 为 None
        cls = " ".join(attrs.get("class", [])) or ""
        tag_id = attrs.get("id") or ""
        if tag.name in _DROP_TAGS or _DROP_PATTERNS.search(f"{cls} {tag_id}"):
            tag.decompose()


def _extract_title(soup: BeautifulSoup, url: str) -> str:
    for sel in ("meta[property='og:title']", "meta[name='twitter:title']"):
        node = soup.select_one(sel)
        if node is not None:
            content = node.get("content")
            if content and content.strip():
                return content.strip()[:200]
    title_tag = soup.find("title")
    if title_tag is not None:
        title = title_tag.get_text(strip=True)
        if title:
            return title[:200]
    return url


def load_web(url: str) -> LoadedPaper:
    """抓取网页正文并清洗。"""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"网页访问失败 HTTP {exc.response.status_code}: {url}") from exc
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise RuntimeError(f"网页访问超时或网络错误（可能无法访问该站点）: {url}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    _drop_unwanted(soup)

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    # 折叠多余空行
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) < 100:
        raise RuntimeError(f"网页正文过少，可能被反爬拦截或为 JS 渲染页面: {url}")

    doc_id = hashlib.sha256(f"web|{url}".encode()).hexdigest()[:16]
    return LoadedPaper(
        paper=NormalizedDocument(
            id=doc_id,
            source_type="web",
            title=_extract_title(soup, url),
            url=url,
            authors=[],
            abstract=None,
            text=text,
        )
    )
