"""多轮对话历史管理（development-plan §12）：按论文隔离，磁盘持久化。

存储：data/conversations/{paper_id}.json
{
  "summary": str | None,   # LLM 滚动摘要（压缩后保留的旧上下文）
  "turns": [{"role": "user"|"assistant", "content": str}, ...]
}

只依赖 stdlib（json / pathlib），不经 paper_agent 包顶层导入，保持轻量。
LLM 滚动压缩（Tier B）由调用方注入 summarize_fn，本模块不直接调 LLM。
"""

from __future__ import annotations

import json
from typing import Callable

from paper_agent import config


def _path(paper_id: str) -> object:
    return config.CONVERSATIONS_DIR / f"{paper_id}.json"


def get_history(paper_id: str) -> dict:
    """读取某篇论文的对话历史。返回 {"summary": str|None, "turns": [...]}。"""
    try:
        data = json.loads(_path(paper_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"summary": None, "turns": []}
    return {
        "summary": data.get("summary"),
        "turns": data.get("turns", []),
    }


def save(paper_id: str, history: dict) -> None:
    """写入完整历史（含 summary），原子写。"""
    config.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(f"{paper_id}.tmp")
    tmp.write_text(
        json.dumps({"summary": history.get("summary"), "turns": history.get("turns", [])},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(_path(paper_id))


def append_turn(paper_id: str, user: str, assistant: str, summarize_fn: Callable[[str], str] | None = None) -> None:
    """问答完成后追加一轮（user + assistant）。

    传 summarize_fn 时在追加后**立即整理**（高水位压缩），
    保证下次提问的起点历史已低于预算（留出余量，不会临时挤爆）。
    """
    history = get_history(paper_id)
    history.setdefault("turns", []).append({"role": "user", "content": user})
    if assistant:
        history["turns"].append({"role": "assistant", "content": assistant})
    if summarize_fn is not None:
        history = trim_history(history, summarize_fn)
    save(paper_id, history)


def clear(paper_id: str) -> bool:
    """删除某篇论文的对话历史。返回是否实际删除。"""
    p = _path(paper_id)
    if p.exists():
        p.unlink()
        return True
    return False


def clear_all() -> int:
    """清空全部对话历史。返回删除的文件数。"""
    if not config.CONVERSATIONS_DIR.exists():
        return 0
    count = 0
    for p in config.CONVERSATIONS_DIR.glob("*.json"):
        p.unlink()
        count += 1
    return count


# ---------------------------------------------------------------------------
# 压缩：滑动窗口（Tier A）+ LLM 滚动摘要（Tier B）
# ---------------------------------------------------------------------------


def _turns_to_text(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = "用户" if t.get("role") == "user" else "助手"
        lines.append(f"{role}：{t.get('content', '')}")
    return "\n".join(lines)


def _occupied_chars(summary: str, turns: list[dict]) -> int:
    """当前历史占用：旧摘要 + 本轮次内容的总字符数。"""
    return len(summary) + sum(len(x.get("content", "")) for x in turns)


def trim_history(
    history: dict,
    summarize_fn: Callable[[str], str] | None = None,
) -> dict:
    """将历史裁剪到高水位内，返回可持久化/拼 prompt 的 {"summary", "turns"}。

    两级压缩 + 高水位余量：
    - Tier A 滑动窗口：turns 超过 RAG_HISTORY_MAX_TURNS → 把超出的最旧轮次划入待压缩池（零成本）。
    - Tier B 滚动摘要：历史占用超过 **预算 85%（高水位）** 时，把"旧摘要 + 待压缩池 + 最旧问答对"
      折成一段 LLM 滚动摘要（单次调用），把占用压回高水位之下，给下一轮留 15% 余量。
    - 兜底：仍超硬预算 → 从最旧问答对丢弃。

    summarize_fn 注入 LLM 压缩（失败/空串时安全降级为直接丢弃，不阻塞）。
    """
    summary = history.get("summary") or ""
    turns = list(history.get("turns") or [])
    hard = config.RAG_HISTORY_MAX_CHARS
    high_water = int(hard * config.RAG_HISTORY_COMPRESS_THRESHOLD)

    # ---- Tier A：滑动窗口，超出的最旧轮进待压缩池 ----
    max_turns = config.RAG_HISTORY_MAX_TURNS
    condense: list[dict] = []
    if len(turns) > max_turns:
        condense = turns[: len(turns) - max_turns]
        turns = turns[len(turns) - max_turns:]

    # ---- 高水位检查：占用 > 预算 85% 才主动压缩（留 15% 余量） ----
    if _occupied_chars(summary, turns) > high_water:
        # 把最旧问答对继续让出，直到占用回到高水位之下
        while len(turns) >= 2 and _occupied_chars(summary, turns) > high_water:
            condense.extend(turns[:2])
            turns = turns[2:]
        # Tier B：折叠"旧摘要 + 待压缩池"为 LLM 滚动摘要（单次调用）
        if condense and summarize_fn is not None:
            merged = f"{summary}\n{_turns_to_text(condense)}".strip()
            try:
                condensed = summarize_fn(merged)
                if condensed:
                    summary = condensed
            except Exception:
                pass  # 压缩失败 → 旧轮直接丢弃，安全降级

    # ---- 兜底：硬预算仍溢出（摘要异常膨胀等病态）→ 丢弃最旧问答对 ----
    while len(turns) >= 2 and _occupied_chars(summary, turns) > hard:
        turns = turns[2:]

    return {"summary": summary, "turns": turns}


def format_history(summary: str | None, turns: list[dict]) -> str:
    """把 (summary, turns) 格式化为拼进 prompt 的历史段文本。为空返回 ""。"""
    parts: list[str] = []
    if summary:
        parts.append(f"（对话摘要）{summary}")
    text = _turns_to_text(turns)
    if text:
        parts.append(text)
    return "\n".join(parts)