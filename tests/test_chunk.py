"""分块器测试：两级分块（结构优先 + 递归兜底）。"""

from __future__ import annotations

import pytest

from paper_agent.chunk import (
    _is_section_header,
    _merge_short_sections,
    _normalize_section_name,
    _split_by_sections,
    chunk_paper,
    make_paper_abstract_doc,
)


class TestSectionHeader:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("1. Introduction", True),
            ("2.1 Model Architecture", True),
            ("3.1.1 Sub-sub section", True),
            ("IV. Results", True),
            ("(1) Data", True),
            ("6 Conclusion", True),
            ("3 Method", True),
            ("Abstract", True),
            ("Related Work", True),
            ("Conclusion and Future Work", True),
            ("acknowledgments", True),
            # 假阳性应排除
            ("1. We introduce a practical and yet powerful attention mechanism - X", False),
            ("2023). Indeed, scaling LLMs to longer sequences", False),
            ("2015).", False),
            ("2020.", False),
            ("1 and mixer heads with score close to 0.5.", False),
            ("V. (6)", False),
            ("This is a regular long body sentence in the paper", False),
        ],
    )
    def test_header_detection(self, line, expected):
        assert _is_section_header(line) is expected

    def test_header_ignores_page_numbers_and_short(self):
        assert _is_section_header("") is False
        assert _is_section_header("42") is False
        assert _is_section_header("a") is False


class TestChunking:
    def test_chunk_paper_with_sections(self):
        # 每节正文 >150 字符，避免被短节合并
        abstract = "We study the problem of long-context language modeling with efficient attention. " * 4
        intro = "Transformers struggle with quadratic attention costs as sequences grow longer. " * 4
        method = "We propose Infini-attention that compresses past information into memory. " * 4
        concl = "Our approach enables scaling to infinitely long context with bounded memory. " * 4
        text = (
            f"Abstract\n{abstract}\n\n"
            f"1 Introduction\n{intro}\n\n"
            f"2 Method\n{method}\n\n"
            f"3 Conclusion\n{concl}\n"
        )
        docs = chunk_paper(text, "abc123", "Test Paper")
        sections = {d.metadata["section"] for d in docs}
        assert "Abstract" in sections
        assert "Introduction" in sections
        assert "Method" in sections
        # 每个块都带出处元数据
        for d in docs:
            assert d.metadata["paper_id"] == "abc123"
            assert d.metadata["title"] == "Test Paper"
            assert "section" in d.metadata

    def test_chunk_paper_no_sections_falls_back(self):
        """无章节标题 → 递归分块兜底。"""
        text = "This is a plain text document without any section headers. " * 200
        docs = chunk_paper(text, "id1", "Plain Doc")
        assert len(docs) > 1  # 被递归拆成多块
        assert docs[0].metadata["section"] == "(全文)"

    def test_long_section_is_split_l2(self):
        """超长 section 走 L2 递归分块。"""
        body = ("word " * 50 + "\n") * 60  # ~3000 词
        text = f"1 Introduction\n{body}"
        docs = chunk_paper(text, "id2", "Long Paper")
        # L1 分出 1 节，L2 拆成多块
        assert len(docs) > 1
        assert all(d.metadata["section"] == "Introduction" for d in docs)

    def test_short_sections_merged(self):
        """过短 section 与相邻节合并。"""
        sections = [
            {"section": "A", "section_index": 0, "text": "x" * 300},
            {"section": "B", "section_index": 1, "text": "short"},
            {"section": "C", "section_index": 2, "text": "y" * 300},
        ]
        merged = _merge_short_sections(sections)
        assert len(merged) == 2
        # B 被合并进 A
        assert "B" in merged[0]["section"]

    def test_normalize_section_name(self):
        assert _normalize_section_name("1. Introduction") == "Introduction"
        assert _normalize_section_name("2.1 Model") == "Model"
        assert _normalize_section_name("IV. Results") == "Results"
        assert _normalize_section_name("(1) Data") == "Data"


class TestAbstractDoc:
    def test_make_paper_abstract_doc(self):
        doc = make_paper_abstract_doc(
            paper_id="p1", title="T", authors=["A", "B"], abstract="Abs", source_type="arxiv"
        )
        assert doc.metadata["type"] == "paper_abstract"
        assert doc.metadata["paper_id"] == "p1"
        assert "T" in doc.page_content
        assert "A" in doc.page_content
        assert "Abs" in doc.page_content
