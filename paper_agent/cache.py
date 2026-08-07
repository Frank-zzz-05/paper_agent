"""磁盘 JSON 缓存：data/cache.json。

条目结构：
{
  "id": str,
  "meta": {...NormalizedDocument 内容, 不含 text},
  "text": str,            # 全文（供后续 RAG 阶段复用）
  "summary": {...} | None,
  "extraction": {...} | None,
  "read_at": "ISO 时间"
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from paper_agent import config


def _read() -> dict[str, dict]:
    if not config.CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(config.CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data: dict[str, dict]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(config.CACHE_FILE)


def get(paper_id: str) -> dict | None:
    return _read().get(paper_id)


def save(
    paper_id: str,
    meta: dict,
    text: str,
    summary: dict | None = None,
    extraction: dict | None = None,
) -> None:
    data = _read()
    existing = data.get(paper_id, {})
    merged = {
        "id": paper_id,
        "meta": meta,
        "text": text,
        "summary": summary if summary is not None else existing.get("summary"),
        "extraction": extraction if extraction is not None else existing.get("extraction"),
        "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    data[paper_id] = merged
    _write(data)


def list_entries(limit: int | None = None) -> list[dict]:
    entries = sorted(_read().values(), key=lambda e: e.get("read_at", ""), reverse=True)
    return entries[:limit] if limit else entries


def delete(paper_id: str) -> bool:
    """从缓存中删除某篇论文。返回是否实际删除。"""
    data = _read()
    if paper_id not in data:
        return False
    del data[paper_id]
    _write(data)
    return True


def clear() -> int:
    data = _read()
    count = len(data)
    if data:
        _write({})
    return count
