"""论文分块管线：结构优先 + 递归兜底。

L1 结构分块：按章节标题正则切分，产出"块 ≈ 一节"，带 section 元数据。
L2 递归兜底：超长 section 用 RecursiveCharacterTextSplitter 再切，短 section 合并。

每个块附 paper_id / title / section / section_index / chunk_index 元数据，
供 RAG 检索时标注出处。
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_agent import config

# ---------------------------------------------------------------------------
# 章节标题检测
# ---------------------------------------------------------------------------

# 已知章节名（大小写不敏感，支持中英文常见变体）
_KNOWN_SECTIONS = (
    r"abstract|摘要",
    r"introduction|引言|intro",
    r"related\s*work|相关工作|literature\s*review|文献综述",
    r"background|背景|preliminar(y|ies)|预备知识",
    r"problem\s*(formulation|definition|statement)|问题(定义|描述|建模)",
    r"method|方法|approach|方法论|methodology|proposed\s*(method|approach|framework|model)|"
    r"model\s*(architecture|design)|模型(架构|设计)|our\s*(method|approach|model)",
    r"experiment|实验|evaluation|评估|results?|结果|empirical\s*(study|evaluation)",
    r"discussion|讨论|analysis|分析|ablation|消融",
    r"conclusion(\s*(and|&)\s*future\s*work)?|结论|summary|总结|concluding\s*remarks|future\s*work|未来工作",
    r"references?|参考文献|bibliography",
    r"appendix|附录|supplementary|补充材料",
    r"acknowledgments?|acknowledgements?|致谢|感谢",
)

# 编译为单一正则，每条独立成行时匹配
# 编号章节标题的长度上限（章节标题通常很短；
# 引言里的 "1. We introduce..."" 贡献列表项是长句，靠长度排除）
_NUMBERED_HEADER_MAX = 60

# 已知章节名（完整行匹配，不区分大小写）
_KNOWN_SECTIONS_RE = re.compile(r"^(?:" + "|".join(_KNOWN_SECTIONS) + r")\s*$", re.IGNORECASE)

# 编号/罗马数字章节标题：1 Title / 1.1 Title / I. Title / (1) Title
# 捕获编号后的标题内容（用于后续首字符检查）
_NUMBERED_SECTION_RE = re.compile(
    r"^\s*"
    r"(?:\d+(?:\.\d+)*|[IVX]+|\(\d+\))"
    r"[\.\s)、．]+"
    r"(.+?)\s*$",
    re.IGNORECASE,
)


def _is_section_header(line: str) -> bool:
    """判断一行是否为章节标题。

    三层守卫：
    1. 已知章节名（Abstract / Introduction / References…）完整行匹配；
    2. 编号章节：排除 4 位年份（"(2023). ..." 引用）与过长行；
    3. 编号后标题首字符若为英文，须大写（排除 "1 and mixer heads" 正文）。
    """
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if _KNOWN_SECTIONS_RE.match(s):
        return True
    # 排除 4 位年份开头（引用："2023). Indeed..."、"2015)."、"2020.")
    if re.match(r"^\d{4}\b", s):
        return False
    if len(s) > _NUMBERED_HEADER_MAX:
        return False
    m = _NUMBERED_SECTION_RE.match(s)
    if not m:
        return False
    rest = m.group(1).strip()
    first = rest[0] if rest else ""
    # 标题首字符须为 ASCII 大写字母：
    #   "3.1 Infini-attention" ✓ / "IV. Results" ✓
    #   "1 and mixer heads" ✗ / "V. (6)" ✗（正文或公式引用）
    return bool(first.isascii() and first.isupper())


# ---------------------------------------------------------------------------
# 噪声行过滤（页码 / 页眉 / 孤立数字）
# ---------------------------------------------------------------------------

_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")             # 孤立页码
_ARXIV_HEADER_RE = re.compile(r"^arXiv:\S+\s+\[")    # arXiv 页眉
_PREPRINT_LINE_RE = re.compile(
    r"(preprint|under review|for consideration|submitted to|"
    r"published as a conference paper at|ICLR|ICML|NeurIPS|ACL|EMNLP|NAACL|AAAI|CVPR|ICCV|ECCV)",
    re.I,
)


def _is_noise(line: str) -> bool:
    """排除页码、页眉、预印本声明等噪声行。"""
    s = line.strip()
    if not s:
        return False
    if _PAGE_NUM_RE.match(s):
        return True
    if _ARXIV_HEADER_RE.match(s):
        return True
    # PDF 页脚常见：第 X 页 / Page X of Y
    if re.match(r"^(第\s*\d+\s*页|Page\s+\d+\s+of\s+\d+)$", s, re.I):
        return True
    return False


def _normalize_section_name(line: str) -> str:
    """归一化章节名：去编号前缀、截断过长。"""
    s = line.strip()
    # 去编号前缀 "1. ", "1.1 ", "(1) ", "I. " 等
    s = re.sub(r"^(?:\d+(?:\.\d+)*[\.\s)、．]+|[IVX]+\.\s*|\(\d+\)\s*)", "", s)
    return s.strip()[:80]


# ---------------------------------------------------------------------------
# 两级分块管线
# ---------------------------------------------------------------------------


def _split_by_sections(lines: list[str]) -> list[dict]:
    """L1：按章节标题将行切分为 section 列表。

    Returns:
        [{section: str, section_index: int, text: str}, ...]
        若无任何章节标题，返回 []（调用方走全文递归兜底）。
    """
    sections: list[dict] = []
    current_lines: list[str] = []
    current_section = ""
    section_idx = 0
    found_header = False

    for ln in lines:
        if _is_noise(ln):
            continue
        if _is_section_header(ln):
            found_header = True
            # 保存上一节
            if current_lines:
                body = "\n".join(current_lines).strip()
                if len(body) > 30:  # 丢弃空节/极短节
                    sections.append({
                        "section": _normalize_section_name(current_section) if current_section else "(摘要前)",
                        "section_index": section_idx,
                        "text": body,
                    })
                    section_idx += 1
                current_lines = []
            current_section = ln.strip()
        else:
            current_lines.append(ln)

    # 最后一节
    if current_lines:
        body = "\n".join(current_lines).strip()
        if len(body) > 30:
            sections.append({
                "section": _normalize_section_name(current_section) if current_section else "(文末)",
                "section_index": section_idx,
                "text": body,
            })

    # 无任何章节标题 → 视为无结构文档，交由调用方全文兜底
    return sections if found_header else []


def _merge_short_sections(sections: list[dict]) -> list[dict]:
    """合并过短 section 到相邻节，避免噪声块。"""
    if not sections:
        return sections
    merged: list[dict] = []
    buf: dict | None = None
    for sec in sections:
        if len(sec["text"]) < config.SECTION_MIN_CHARS:
            if merged:
                # 合并到上一个
                merged[-1]["text"] += f"\n\n{sec['section']}\n{sec['text']}"
                merged[-1]["section"] += f" + {sec['section']}"
            elif buf is None:
                buf = sec  # 暂存，等下一节
            else:
                buf["text"] += f"\n\n{sec['section']}\n{sec['text']}"
                buf["section"] += f" + {sec['section']}"
        else:
            if buf is not None:
                sec["text"] = f"{buf['section']}\n{buf['text']}\n\n{sec['section']}\n{sec['text']}"
                sec["section"] = f"{buf['section']} + {sec['section']}"
                buf = None
            merged.append(sec)
    if buf is not None:
        if merged:
            merged[-1]["text"] += f"\n\n{buf['section']}\n{buf['text']}"
            merged[-1]["section"] += f" + {buf['section']}"
        else:
            merged.append(buf)  # 全部为短节时，累积块作为唯一一节
    return merged


def _split_long_section(sec: dict, paper_id: str, title: str) -> list[Document]:
    """L2：超长 section 用 RecursiveCharacterTextSplitter 再切。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    docs: list[Document] = []
    for ci, chunk in enumerate(splitter.split_text(sec["text"])):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "paper_id": paper_id,
                "title": title,
                "section": sec["section"],
                "section_index": sec["section_index"],
                "chunk_index": ci,
            },
        ))
    return docs


def chunk_paper(text: str, paper_id: str, title: str) -> list[Document]:
    """对一篇论文全文执行两级分块，返回 LangChain Document 列表。

    Args:
        text: 论文全文。
        paper_id: 论文唯一 ID（cache id）。
        title: 论文标题。

    Returns:
        带元数据的 Document 列表，可直接写入向量库。
    """
    lines = text.splitlines()
    sections = _split_by_sections(lines)

    # 如果结构分块完全失败（无章节标题），整篇用递归分块兜底
    if not sections:
        fake_sec = {"section": "(全文)", "section_index": 0, "text": text}
        return _split_long_section(fake_sec, paper_id, title)

    sections = _merge_short_sections(sections)

    docs: list[Document] = []
    for sec in sections:
        if len(sec["text"]) <= config.SECTION_MAX_CHARS:
            docs.append(Document(
                page_content=sec["text"],
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "section": sec["section"],
                    "section_index": sec["section_index"],
                    "chunk_index": 0,
                },
            ))
        else:
            docs.extend(_split_long_section(sec, paper_id, title))
    return docs


# ---------------------------------------------------------------------------
# 论文级摘要块（薄索引）
# ---------------------------------------------------------------------------


def make_paper_abstract_doc(
    paper_id: str, title: str, authors: list[str] | None = None,
    abstract: str | None = None, source_type: str = "",
) -> Document:
    """为论文级检索生成一个摘要块（论文级索引）。

    嵌入 title + authors + abstract，type="paper_abstract" 与正文块区分。
    跨论文检索时用它做第一道筛网 — 先定位"哪几篇相关"再下沉正文。
    """
    parts = [f"标题: {title}"]
    if authors:
        parts.append(f"作者: {', '.join(authors)}")
    if abstract:
        parts.append(f"摘要: {abstract}")
    return Document(
        page_content="\n".join(parts),
        metadata={
            "paper_id": paper_id,
            "title": title,
            "source_type": source_type,
            "type": "paper_abstract",
            "section": "(论文摘要)",
            "section_index": -1,
            "chunk_index": 0,
        },
    )
