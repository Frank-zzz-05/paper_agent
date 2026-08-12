"""CLI 测试：metadata-only 离线真实；完整 read 用假图，验证缓存 round-trip。"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from paper_agent.cli import app
from paper_agent.models import NormalizedDocument, PaperExtraction, PaperSummary

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_auto_ingest(monkeypatch):
    """所有 CLI 测试阻断自动入库、向量库清空与真实 LLM（bge-m3 / DeepSeek）。"""
    from paper_agent import cli

    monkeypatch.setattr(cli, "_auto_ingest", lambda **kw: None)
    monkeypatch.setattr("paper_agent.vectorstore.clear_all", lambda: 0)
    # 结构化记忆更新用假实现，避免真实 LLM 调用
    monkeypatch.setattr(
        cli, "_update_memory",
        lambda existing, q, a: {
            "facts": (existing.get("facts") or []) + [f"事实:{q}"],
            "preferences": existing.get("preferences") or [],
            "answered": (existing.get("answered") or []) + [{"q": q, "a": a}],
        },
    )


def test_read_metadata_only_offline(sample_pdf):
    result = runner.invoke(app, ["read", str(sample_pdf), "--no-summary", "--no-extract", "--no-cache"])
    assert result.exit_code == 0
    assert "来源" in result.stdout
    assert "标题" in result.stdout


def test_read_invalid_input_exits_1():
    result = runner.invoke(app, ["read", "garbage_input_xyz"])
    assert result.exit_code == 1
    assert "无法识别" in result.stderr or "无法识别" in result.stdout


def test_read_corrupt_pdf_exits_1(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    result = runner.invoke(app, ["read", str(bad)])
    assert result.exit_code == 1


def test_full_read_cache_roundtrip(monkeypatch, tmp_cache, sample_pdf):
    from paper_agent import cli
    from paper_agent.loaders import cache_id_for

    _, pid = cache_id_for(str(sample_pdf))  # 缓存键与 CLI 计算一致
    fake_paper = NormalizedDocument(
        id=pid, source_type="pdf", title="Fake Title", text="Fake text content"
    )
    fake_summary = PaperSummary(title="Fake Title", summary="S", key_points=["p"], keywords=["k"])
    fake_extraction = PaperExtraction(
        research_question="Q", method="M", experiment_results="R",
        contributions=["c"], core_innovations=["i"],
    )
    calls = {"n": 0}

    class FakeGraph:
        def invoke(self, state):
            calls["n"] += 1
            return {
                "paper": fake_paper,
                "summary": fake_summary,
                "extraction": fake_extraction,
                "errors": [],
                "status": "done",
            }

    monkeypatch.setattr("paper_agent.graph.build.build_paper_graph", lambda: FakeGraph())

    # 第一次 read → 调图 → 写缓存
    r1 = runner.invoke(app, ["read", str(sample_pdf), "--output", "json"])
    assert r1.exit_code == 0, r1.stderr
    assert f'"{pid}"' in r1.stdout
    assert calls["n"] == 1

    # 第二次 read → 命中缓存，不再调图
    r2 = runner.invoke(app, ["read", str(sample_pdf), "--output", "json"])
    assert r2.exit_code == 0
    assert calls["n"] == 1
    assert "命中缓存" in (r2.stderr or "")

    # list / show / clear-cache
    rl = runner.invoke(app, ["list"])
    assert rl.exit_code == 0
    assert "Fake Title" in rl.stdout

    rs = runner.invoke(app, ["show", pid])
    assert rs.exit_code == 0
    assert "research_question" in rs.stdout

    rc = runner.invoke(app, ["clear-cache"])
    assert rc.exit_code == 0
    assert "1" in rc.stdout

    rl2 = runner.invoke(app, ["list"])
    assert "暂无缓存" in rl2.stdout


def test_ask_no_papers_exits_1(monkeypatch):
    """向量库为空时 ask 给出清晰错误。"""
    from paper_agent import vectorstore

    monkeypatch.setattr(vectorstore, "get_stored_paper_ids", lambda: [])
    result = runner.invoke(app, ["ask", "question"])
    assert result.exit_code == 1
    assert "向量库中没有论文" in result.stderr or "向量库中没有论文" in result.stdout


def test_delete_paper(monkeypatch, tmp_cache, sample_pdf):
    """delete 从缓存和向量库中彻底删除某篇论文（mock 向量库）。"""
    from paper_agent import cache, cli, vectorstore

    # 建一条缓存（直接写，绕过真实图）
    cache.save(
        "del-1",
        meta={"title": "Delete Me Paper", "source_type": "pdf"},
        text="text",
        summary=None,
        extraction=None,
    )
    assert cache.get("del-1") is not None

    deleted = []
    stored = ["del-1"]
    monkeypatch.setattr(vectorstore, "get_stored_paper_ids", lambda: list(stored))
    monkeypatch.setattr(vectorstore, "delete_paper", lambda pid: (stored.remove(pid), deleted.append(pid)))

    # 按 ID 删除
    r1 = runner.invoke(app, ["delete", "del-1"])
    assert r1.exit_code == 0, r1.stderr
    assert deleted == ["del-1"]
    assert cache.get("del-1") is None
    assert "已删除" in r1.stdout

    # 再删一次 → 找不到，exit 1
    r2 = runner.invoke(app, ["delete", "del-1"])
    assert r2.exit_code == 1
    assert "找不到" in r2.stderr or "找不到" in r2.stdout


def test_delete_paper_by_title(monkeypatch, tmp_cache):
    """delete 支持按标题子串匹配。"""
    from paper_agent import cache, cli, vectorstore

    cache.save(
        "t-1",
        meta={"title": "Retentive Network Rocks", "source_type": "arxiv"},
        text="text",
        summary=None,
        extraction=None,
    )
    deleted = []
    stored = ["t-1"]
    monkeypatch.setattr(vectorstore, "get_stored_paper_ids", lambda: list(stored))
    monkeypatch.setattr(vectorstore, "delete_paper", lambda pid: (stored.remove(pid), deleted.append(pid)))

    r = runner.invoke(app, ["delete", "retentive"])
    assert r.exit_code == 0, r.stderr
    assert deleted == ["t-1"]
    assert cache.get("t-1") is None
    assert "已删除" in r.stdout


def test_ask_with_mock_rag(monkeypatch, tmp_cache):
    """ask 全链路（mock RAG 图）：输出回答与出处。"""
    from paper_agent import cli, vectorstore

    monkeypatch.setattr(vectorstore, "get_stored_paper_ids", lambda: ["p1"])

    class FakeRagGraph:
        def invoke(self, state):
            return {
                "question": state["question"],
                "paper_id": state["paper_id"],
                "history": state.get("history", ""),
                "context": "[Method]\n...",
                "sources": [{"section": "Method", "title": "T", "paper_id": "p1", "excerpt": "..."}],
                "answer": "这是回答 [T, Method]",
                "errors": [],
            }

    monkeypatch.setattr("paper_agent.graph.rag_build.build_rag_graph", lambda: FakeRagGraph())
    result = runner.invoke(app, ["ask", "问题", "--paper", "p1"])
    assert result.exit_code == 0, result.stderr
    assert "这是回答" in result.stdout
    assert "Method" in result.stdout


def test_ask_multi_turn_history(monkeypatch, tmp_cache):
    """多轮：第一问无记忆，第二问按相关性拼入；结构化记忆落盘。"""
    from paper_agent import cli, config, conversations, vectorstore

    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_cache.parent / "conversations")
    monkeypatch.setattr(vectorstore, "get_stored_paper_ids", lambda: ["p1"])

    seen_histories = []

    class FakeRagGraph:
        def invoke(self, state):
            seen_histories.append(state.get("history", ""))
            return {
                "question": state["question"],
                "paper_id": state["paper_id"],
                "history": state.get("history", ""),
                "context": "...",
                "sources": [],
                "answer": f"回答: {state['question']}",
                "errors": [],
            }

    monkeypatch.setattr("paper_agent.graph.rag_build.build_rag_graph", lambda: FakeRagGraph())

    # 第一问：无记忆
    r1 = runner.invoke(app, ["ask", "方法是什么", "--paper", "p1"])
    assert r1.exit_code == 0
    assert seen_histories == [""]

    # 结构化记忆已落盘（answered 含本轮）
    h = conversations.get_history("p1")
    assert h["answered"][0]["q"] == "方法是什么"

    # 第二问：相关记忆被选中拼入 prompt
    r2 = runner.invoke(app, ["ask", "方法的局限呢", "--paper", "p1"])
    assert r2.exit_code == 0
    assert "方法是什么" in seen_histories[1]

    # 已累积 2 条 answered
    h = conversations.get_history("p1")
    assert len(h["answered"]) == 2

    # --reset 清空记忆
    r3 = runner.invoke(app, ["ask", "新对话", "--paper", "p1", "--reset"])
    assert r3.exit_code == 0
    assert seen_histories[2] == ""  # reset 后无记忆
    h = conversations.get_history("p1")
    assert len(h["answered"]) == 1  # 只剩新的一轮

    # --no-history 不读取也不写记忆
    r4 = runner.invoke(app, ["ask", "test", "--paper", "p1", "--no-history"])
    assert r4.exit_code == 0
    assert seen_histories[3] == ""
    assert len(conversations.get_history("p1")["answered"]) == 1  # 未追加
