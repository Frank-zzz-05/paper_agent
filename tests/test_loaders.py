"""加载器测试：PDF 离线真实、arXiv/网页用 mock，不依赖网络与 LLM。"""

from __future__ import annotations

import pytest

from paper_agent.loaders import cache_id_for, load_paper
from paper_agent.loaders.arxiv_loader import extract_arxiv_id


def test_pdf_loader_offline(sample_pdf):
    loaded = load_paper(str(sample_pdf))
    p = loaded.paper
    assert p.source_type == "pdf"
    assert p.title
    assert p.num_pages == 1
    assert len(p.text) > 200


def test_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paper(str(tmp_path / "nope.pdf"))


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    with pytest.raises(RuntimeError, match="PDF"):
        load_paper(str(bad))


def test_cache_id_for_pdf(sample_pdf):
    st, cid = cache_id_for(str(sample_pdf))
    assert st == "pdf"
    assert len(cid) == 16


def test_cache_id_for_arxiv_and_web():
    assert cache_id_for("2404.07143")[0] == "arxiv"
    assert cache_id_for("https://arxiv.org/abs/2302.13971")[0] == "arxiv"
    assert cache_id_for("https://example.com/a")[0] == "web"


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="无法识别"):
        load_paper("garbage_input_xyz")


def test_extract_arxiv_id():
    assert extract_arxiv_id("2404.07143") == "2404.07143"
    assert extract_arxiv_id("2404.07143v2") == "2404.07143"
    assert extract_arxiv_id("https://arxiv.org/abs/2302.13971") == "2302.13971"
    assert extract_arxiv_id("https://arxiv.org/pdf/2302.13971v3") == "2302.13971"
    assert extract_arxiv_id("cmp-lg/9701001") == "cmp-lg/9701001"  # 老格式
    with pytest.raises(ValueError):
        extract_arxiv_id("https://example.com/not-arxiv")


def test_arxiv_loader_mocked(monkeypatch):
    """Mock arxiv_mcp.core 函数（打补丁在 arxiv_loader 模块的本地引用上）。"""
    from paper_agent.loaders import arxiv_loader

    mock_text = "Abstract\nThis is the abstract of a mock paper.\n1 Introduction\nSome body text.\n2 Method\nMore text.\n" * 30

    monkeypatch.setattr(arxiv_loader, "get_paper_full_text", lambda arxiv_id, max_chars=500_000: {
        "arxiv_id": arxiv_id, "title": "Mock Title", "source": "pdf",
        "chars": len(mock_text), "truncated": False, "text": mock_text,
    })
    monkeypatch.setattr(arxiv_loader, "get_paper_metadata", lambda arxiv_id: {
        "arxiv_id": arxiv_id, "title": "Mock Title", "authors": ["A", "B"],
        "abstract": "This is the abstract.", "published": "2024-01-01",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    })
    monkeypatch.setattr(arxiv_loader, "download_paper_pdf", lambda arxiv_id, output_dir=None: {})

    loaded = arxiv_loader.load_arxiv("2404.07143")
    p = loaded.paper
    assert p.source_type == "arxiv"
    assert p.title == "Mock Title"
    assert p.authors == ["A", "B"]
    assert len(p.text) > 500


def test_web_loader_mocked(monkeypatch):
    from paper_agent.loaders import web_loader

    html = (
        "<html><head><title>Test Article Title</title>"
        "<meta property='og:title' content='OG Test Title'></head>"
        "<body><nav>menu</nav><article>"
        + ("<p>Meaningful article content sentence.</p>" * 30)
        + "</article></body></html>"
    )

    class FakeResp:
        status_code = 200
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(web_loader.httpx, "get", lambda *a, **k: FakeResp())
    loaded = load_paper("https://example.com/article")
    p = loaded.paper
    assert p.source_type == "web"
    assert p.title == "OG Test Title"
    assert "menu" not in p.text  # nav 已被剔除
    assert len(p.text) > 100
