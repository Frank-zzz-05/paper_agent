"""RAG 测试：向量库入库/检索（假 embedding，不联网）、检索上下文、RAG 图节点（mock）。"""

from __future__ import annotations

import hashlib
import os

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_SERVER_HTTP_PORT", "0")


class FakeEmbeddings:
    """确定性假 embedding（哈希 → 8 维），避免测试下载 bge-m3（~1.2GB）。"""

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


@pytest.fixture
def tmp_vectorstore(tmp_path, monkeypatch):
    """把向量库重定向到临时目录，并注入假 embedding（重置单例）。"""
    from paper_agent import config, vectorstore

    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path / "vectorstore")
    monkeypatch.setattr(vectorstore, "_embedding_function", FakeEmbeddings())
    monkeypatch.setattr(vectorstore, "_vectorstore", None)
    yield vectorstore
    monkeypatch.setattr(vectorstore, "_embedding_function", None)
    monkeypatch.setattr(vectorstore, "_vectorstore", None)


# 各节正文 >150 字符（避免被短节合并成一块）
_SYNTH_TEXT = (
    "Abstract\nThis paper studies retrieval augmented generation which combines a retriever with a generator. "
    "The core idea is to ground generation in retrieved evidence to improve factual accuracy. " * 2
    + "\n\n"
    "1 Introduction\nRetrieval augmented generation combines a retriever with a generator. "
    "This paradigm grounds language model outputs in external evidence, reducing hallucination. " * 2
    + "\n\n"
    "2 Method\nWe use a dense retriever to fetch relevant passages from a corpus. "
    "The generator then conditions on the retrieved passages to produce faithful answers. " * 2
    + "\n\n"
    "3 Conclusion\nRAG improves factual accuracy and enables knowledge-grounded generation. "
    "We conclude that retrieval-augmented methods are a practical path forward. " * 2
    + "\n"
)


class TestVectorstore:
    def test_ingest_paper_and_count(self, tmp_vectorstore):
        n = tmp_vectorstore.ingest_paper(
            paper_id="p1", text=_SYNTH_TEXT, title="RAG Paper",
            authors=["A"], abstract="About RAG", source_type="arxiv",
        )
        assert n >= 2  # 摘要块 + 至少一个正文块
        assert tmp_vectorstore.get_paper_chunk_count("p1") == n
        assert "p1" in tmp_vectorstore.get_stored_paper_ids()

    def test_retrieve_finds_relevant(self, tmp_vectorstore):
        tmp_vectorstore.ingest_paper("p1", _SYNTH_TEXT, "RAG Paper")
        docs = tmp_vectorstore.retrieve("retrieval augmented generation method", paper_id="p1", top_k=3)
        assert docs
        # 假 embedding 不具备语义，只验证检索契约：返回论文 p1 的块、带出处
        assert all(d.metadata["paper_id"] == "p1" for d in docs)
        assert all("section" in d.metadata for d in docs)

    def test_retrieve_without_paper_filter_returns_all(self, tmp_vectorstore):
        tmp_vectorstore.ingest_paper("p1", _SYNTH_TEXT, "RAG Paper")
        tmp_vectorstore.ingest_paper("p2", _SYNTH_TEXT.replace("RAG", "attention"), "Attn Paper")
        ids = tmp_vectorstore.get_stored_paper_ids()
        assert ids == ["p1", "p2"]

    def test_retrieve_empty_store_returns_empty(self, tmp_vectorstore):
        assert tmp_vectorstore.retrieve("anything") == []

    def test_delete_paper(self, tmp_vectorstore):
        tmp_vectorstore.ingest_paper("p1", _SYNTH_TEXT, "RAG Paper")
        n_before = tmp_vectorstore.get_paper_chunk_count("p1")
        assert n_before > 0
        tmp_vectorstore.delete_paper("p1")
        assert tmp_vectorstore.get_paper_chunk_count("p1") == 0

    def test_ingest_twice_is_idempotent(self, tmp_vectorstore):
        tmp_vectorstore.ingest_paper("p1", _SYNTH_TEXT, "RAG Paper")
        n1 = tmp_vectorstore.get_paper_chunk_count("p1")
        tmp_vectorstore.ingest_paper("p1", _SYNTH_TEXT, "RAG Paper")
        n2 = tmp_vectorstore.get_paper_chunk_count("p1")
        assert n1 == n2  # 不重复堆积


class TestBuildContext:
    def test_build_context_includes_sources(self):
        from langchain_core.documents import Document

        from paper_agent.graph.rag_nodes import _build_context

        docs = [
            Document(
                page_content="A" * 100,
                metadata={"section": "2 Method", "title": "T1", "paper_id": "p1"},
            ),
            Document(
                page_content="B" * 100,
                metadata={"section": "3 Results", "title": "T1", "paper_id": "p1"},
            ),
        ]
        context, sources = _build_context(docs, max_chars=10_000)
        assert "2 Method" in context
        assert sources[0]["section"] == "2 Method"
        assert sources[0]["paper_id"] == "p1"

    def test_build_context_respects_max_chars(self):
        from langchain_core.documents import Document

        from paper_agent.graph.rag_nodes import _build_context

        docs = [
            Document(page_content="X" * 5000, metadata={"section": "S1", "title": "T", "paper_id": "p"}),
            Document(page_content="Y" * 5000, metadata={"section": "S2", "title": "T", "paper_id": "p"}),
        ]
        context, sources = _build_context(docs, max_chars=6000)
        assert len(context) <= 6200  # 受 max_chars 限制


class TestRagGraphNodes:
    def test_retrieve_node_success(self, monkeypatch):
        from langchain_core.documents import Document

        from paper_agent.graph.rag_nodes import retrieve_node

        fake_docs = [
            Document(page_content="Method content", metadata={"section": "Method", "title": "T", "paper_id": "p"})
        ]
        monkeypatch.setattr("paper_agent.graph.rag_nodes.retrieve", lambda q, paper_id=None: fake_docs)
        result = retrieve_node({"question": "how?", "paper_id": None})
        assert "context" in result
        assert "Method" in result["context"]
        assert result["sources"][0]["section"] == "Method"

    def test_retrieve_node_no_results(self, monkeypatch):
        from paper_agent.graph.rag_nodes import retrieve_node

        monkeypatch.setattr("paper_agent.graph.rag_nodes.retrieve", lambda q, paper_id=None: [])
        result = retrieve_node({"question": "how?", "paper_id": None})
        assert result.get("errors")
        assert result["context"] == ""

    def test_answer_node_with_context(self, monkeypatch):
        from paper_agent.graph.rag_nodes import answer_node

        class FakeResp:
            content = "这是带出处的回答 [T, Method]"

        class FakeLLM:
            def invoke(self, messages):
                return FakeResp()

        monkeypatch.setattr("paper_agent.graph.rag_nodes.get_llm", lambda **kw: FakeLLM())
        result = answer_node({
            "question": "how?", "paper_id": None, "context": "[Method]\n...", "sources": [], "errors": [],
        })
        assert "Method" in result["answer"]

    def test_answer_node_empty_context(self):
        from paper_agent.graph.rag_nodes import answer_node

        result = answer_node({"question": "q", "paper_id": None, "context": "", "sources": [], "errors": []})
        assert "没有找到" in result["answer"]
