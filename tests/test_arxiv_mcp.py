"""arxiv_mcp 核心逻辑单元测试（全部离线，不发起网络请求）。"""

from __future__ import annotations

import pytest

from arxiv_mcp import core


# ---------------------------------------------------------------------------
# extract_arxiv_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_str,expected", [
    ("2404.07143", "2404.07143"),
    ("2404.07143v2", "2404.07143"),
    ("https://arxiv.org/abs/2404.07143", "2404.07143"),
    ("https://arxiv.org/pdf/2404.07143", "2404.07143"),
    ("http://arxiv.org/abs/cmp-lg/9701001", "cmp-lg/9701001"),
    ("cmp-lg/9701001", "cmp-lg/9701001"),
    ("arxiv.org/abs/2302.13971", "2302.13971"),
])
def test_extract_arxiv_id(input_str, expected):
    assert core.extract_arxiv_id(input_str) == expected


@pytest.mark.parametrize("bad", ["", "hello world", "http://example.com/x", "123", "arXiv:2404.07143"])
def test_extract_arxiv_id_invalid(bad):
    with pytest.raises(ValueError):
        core.extract_arxiv_id(bad)


# ---------------------------------------------------------------------------
# Atom API XML 解析
# ---------------------------------------------------------------------------

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2404.07143v1</id>
    <title>   Leave No Context Behind   </title>
    <summary>This work introduces an efficient method.</summary>
    <published>2024-04-10T16:18:42Z</published>
    <updated>2024-05-01T00:00:00Z</updated>
    <author><name> Tsendsuren Munkhdalai </name></author>
    <author><name> Manaal Faruqui </name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1234/x</arxiv:doi>
    <arxiv:journal_ref xmlns:arxiv="http://arxiv.org/schemas/atom">ACL 2024</arxiv:journal_ref>
    <link title="pdf" href="http://arxiv.org/pdf/2404.07143v1" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cmp-lg/9701001v1</id>
    <title>Old Style Paper</title>
    <summary>An old paper.</summary>
    <published>1997-01-01T00:00:00Z</published>
    <updated>1997-01-01T00:00:00Z</updated>
    <author><name> Old Author </name></author>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_atom_entries():
    import xml.etree.ElementTree as ET

    entries = core._parse_atom_entries(_SAMPLE_ATOM, ET)
    assert len(entries) == 2

    e = entries[0]
    assert e["arxiv_id"] == "2404.07143"
    assert e["title"] == "Leave No Context Behind"
    assert e["abstract"] == "This work introduces an efficient method."
    assert e["authors"] == ["Tsendsuren Munkhdalai", "Manaal Faruqui"]
    assert e["published"] == "2024-04-10T16:18:42Z"
    assert e["doi"] == "10.1234/x"
    assert e["journal_ref"] == "ACL 2024"
    assert e["categories"] == ["cs.CL", "cs.AI"]
    assert e["primary_category"] == "cs.CL"

    # 老式 ID
    assert entries[1]["arxiv_id"] == "cmp-lg/9701001"


# ---------------------------------------------------------------------------
# abs 页面 HTML 解析（Atom API 兜底路径）
# ---------------------------------------------------------------------------

_SAMPLE_ABS_HTML = """<html><body>
  <h1 class="title mathjax">Title: Some Interesting Paper</h1>
  <div class="authors">Alice Bob and Charlie Dave</div>
  <blockquote class="abstract mathjax">Abstract: We do things.</blockquote>
  <div class="dateline">[Submitted on 12 Jun 2024]</div>
</body></html>"""


def test_parse_abs_html():
    meta = core._parse_abs_html(_SAMPLE_ABS_HTML)
    assert meta["title"] == "Some Interesting Paper"
    assert meta["abstract"] == "We do things."
    assert "Submitted on 12 Jun 2024" in (meta["published"] or "")


# ---------------------------------------------------------------------------
# PDF 首页文本元数据解析（全文兜底路径）
# ---------------------------------------------------------------------------

_SAMPLE_FIRST_PAGE = """arXiv:2404.07143v1  [cs.CL]  10 Apr 2024
Leave No Context Behind:
Efficient Infinite Context Transformers with Infini-attention
Tsendsuren Munkhdalai, Manaal Faruqui and Siddharth Gopal
Google
tsendsuren@google.com
Abstract
This work introduces an efficient method to scale Transformer-based
Large Language Models to infinitely long inputs with bounded memory.
1 Introduction
..."""


def test_parse_metadata_from_text():
    meta = core._parse_metadata_from_text(_SAMPLE_FIRST_PAGE)
    assert meta["title"] == "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
    assert meta["authors"] == ["Tsendsuren Munkhdalai", "Manaal Faruqui", "Siddharth Gopal"]
    assert "infinitely long inputs" in (meta["abstract"] or "")
    assert "10 Apr 2024" in (meta["published"] or "")


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def test_search_papers_param_validation():
    with pytest.raises(ValueError):
        core.search_papers("")
    with pytest.raises(ValueError):
        core.search_papers("x", max_results=0)
    with pytest.raises(ValueError):
        core.search_papers("x", max_results=101)
    with pytest.raises(ValueError):
        core.search_papers("x", sort_by="bogus")


def test_get_paper_full_text_param_validation():
    with pytest.raises(ValueError):
        core.get_paper_full_text("2404.07143", max_chars=100)
    with pytest.raises(ValueError):
        core.get_paper_full_text("2404.07143", max_chars=600_000)
