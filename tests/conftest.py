"""pytest 共享夹具：确保项目根目录可导入、生成离线样例 PDF、隔离缓存。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_make_sample_pdf():
    spec = importlib.util.spec_from_file_location(
        "make_sample_pdf", str(TESTS_DIR / "make_sample_pdf.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """离线样例 PDF（不存在则生成）。"""
    make = _load_make_sample_pdf()
    pdf = TESTS_DIR / "data" / "sample.pdf"
    if not pdf.exists():
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(make.build_pdf())
    return pdf


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """把缓存文件重定向到临时目录，隔离真实 data/cache.json。"""
    from paper_agent import config

    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(config, "CACHE_FILE", cache_file)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return cache_file
