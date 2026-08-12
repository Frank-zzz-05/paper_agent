"""BM25 单元测试：tokenize（中英混合）+ 打分排序。"""

from __future__ import annotations

from paper_agent.bm25 import BM25, tokenize


def test_tokenize_english():
    assert tokenize("Retrieval-Augmented Generation") == ["retrieval", "augmented", "generation"]
    assert tokenize("RAG 2024") == ["rag", "2024"]


def test_tokenize_chinese_bigrams():
    # 中文 → 字符 bigram
    toks = tokenize("检索增强")
    assert "检索" in toks and "索增" in toks and "增强" in toks


def test_tokenize_mixed():
    toks = tokenize("attention 机制")
    assert "attention" in toks
    assert "机制" in toks


def test_bm25_scores_relevant_first():
    corpus = [
        "retrieval augmented generation combines a retriever with a generator",
        "transformer attention is all you need",
        "检索增强生成结合检索器与生成器",
    ]
    bm = BM25(corpus)

    hits = bm.top_k("retrieval generator", k=3)
    assert hits and hits[0][0] == 0  # 英文文档命中

    hits_zh = bm.top_k("检索增强", k=3)
    assert hits_zh and hits_zh[0][0] == 2  # 中文文档命中

    hits_mixed = bm.top_k("attention 检索增强", k=3)
    idxs = [i for i, _ in hits_mixed]
    assert 2 in idxs and 1 in idxs  # 中英各自命中


def test_bm25_empty_and_miss():
    bm = BM25([])
    assert bm.top_k("anything", k=3) == []

    bm2 = BM25(["only one document"])
    assert bm2.top_k("nonexistenttermxyz", k=3) == []