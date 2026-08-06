"""图测试：用假 LLM（monkeypatch）离线验证节点接线、并行分支与错误传播。"""

from __future__ import annotations

from paper_agent.graph.build import build_paper_graph
from paper_agent.models import PaperExtraction, PaperSummary


def _fake_summary():
    return PaperSummary(title="T", summary="S", key_points=["p1", "p2"], keywords=["k"])


def _fake_extraction():
    return PaperExtraction(
        research_question="Q",
        method="M",
        experiment_results="R",
        contributions=["c1"],
        core_innovations=["i1"],
    )


def _patch_llm(monkeypatch, summary, extraction):
    import paper_agent.graph.nodes as nodes

    def fake_structured_invoke(llm, schema, system, text):
        if schema is PaperSummary:
            return summary
        if schema is PaperExtraction:
            return extraction
        raise AssertionError(f"unexpected schema: {schema}")

    monkeypatch.setattr(nodes, "structured_invoke", fake_structured_invoke)
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: None)


def test_graph_success_both_branches(monkeypatch, sample_pdf):
    summary, extraction = _fake_summary(), _fake_extraction()
    _patch_llm(monkeypatch, summary, extraction)

    result = build_paper_graph().invoke(
        {"source": str(sample_pdf), "options": {"summary": True, "extract": True, "lang": "zh"}}
    )
    assert result["status"] == "done"
    assert result["summary"] == summary
    assert result["extraction"] == extraction
    assert result["errors"] == []
    assert result["paper"].source_type == "pdf"


def test_graph_summary_only(monkeypatch, sample_pdf):
    summary, extraction = _fake_summary(), _fake_extraction()
    _patch_llm(monkeypatch, summary, extraction)

    result = build_paper_graph().invoke(
        {"source": str(sample_pdf), "options": {"summary": True, "extract": False, "lang": "zh"}}
    )
    assert result["summary"] == summary
    assert result.get("extraction") is None


def test_graph_extract_only(monkeypatch, sample_pdf):
    summary, extraction = _fake_summary(), _fake_extraction()
    _patch_llm(monkeypatch, summary, extraction)

    result = build_paper_graph().invoke(
        {"source": str(sample_pdf), "options": {"summary": False, "extract": True, "lang": "zh"}}
    )
    assert result.get("summary") is None
    assert result["extraction"] == extraction


def test_graph_load_error_propagates(monkeypatch):
    import paper_agent.graph.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm", lambda **kw: None)

    result = build_paper_graph().invoke(
        {"source": "definitely_missing_file.pdf", "options": {"summary": True, "extract": True, "lang": "zh"}}
    )
    assert result["status"] == "error"
    assert result["errors"]
    assert result.get("summary") is None
    assert result.get("extraction") is None


def test_graph_llm_error_collected(monkeypatch, sample_pdf):
    import paper_agent.graph.nodes as nodes

    def boom(llm, schema, system, text):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(nodes, "structured_invoke", boom)
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: None)

    result = build_paper_graph().invoke(
        {"source": str(sample_pdf), "options": {"summary": True, "extract": True, "lang": "zh"}}
    )
    assert result["status"] == "error"
    assert len(result["errors"]) == 2  # 两个并行分支各记一条错误
