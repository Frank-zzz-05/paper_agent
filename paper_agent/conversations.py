"""多轮对话记忆（development-plan §12）：按论文隔离，磁盘持久化，结构化压缩。

存储：data/conversations/{paper_id}.json
{
  "facts":        ["已确认事实", ...],
  "preferences":  ["用户偏好/关注点", ...],
  "answered":     [{"q": "...", "a": "...", "t": "..."}, ...]
}

设计：不用滑动窗口保留原始轮次，而是由 LLM 把每轮问答**压缩**进结构化槽位；
拼 prompt 时按与当前问题的相关性**选择**命中项（而非全量），受 token 预算约束。
"""

from __future__ import annotations

import json
from typing import Callable

from paper_agent import config

_EMPTY = {"facts": [], "preferences": [], "answered": []}


def _path(paper_id: str) -> object:
    return config.CONVERSATIONS_DIR / f"{paper_id}.json"


def get_history(paper_id: str) -> dict:
    """读取某篇论文的结构化记忆。旧格式（summary/turns）不再兼容，返回空。"""
    try:
        data = json.loads(_path(paper_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"facts": [], "preferences": [], "answered": []}
    if not isinstance(data, dict) or "turns" in data:  # 旧格式
        return {"facts": [], "preferences": [], "answered": []}
    return {
        "facts": list(data.get("facts") or []),
        "preferences": list(data.get("preferences") or []),
        "answered": list(data.get("answered") or []),
    }


def save(paper_id: str, memory: dict) -> None:
    """写入结构化记忆，原子写。"""
    config.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(f"{paper_id}.tmp")
    tmp.write_text(
        json.dumps(
            {"facts": memory.get("facts", []), "preferences": memory.get("preferences", []),
             "answered": memory.get("answered", [])},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(_path(paper_id))


def update_memory(
    paper_id: str,
    question: str,
    answer: str,
    update_fn: Callable[[dict, str, str], dict | None],
) -> dict:
    """问答后把本轮并入结构化记忆（LLM 压缩）。

    update_fn(existing, question, answer) -> 新记忆 dict；返回 None 表示更新失败，
    此时安全降级：仅把本轮追加进 answered（不崩溃）。
    """
    memory = get_history(paper_id)
    try:
        updated = update_fn(memory, question, answer)
    except Exception:
        updated = None
    if not updated or not isinstance(updated, dict):
        updated = dict(memory)
        updated.setdefault("answered", []).append(
            {"q": question, "a": answer, "t": "turn"}
        )
    # 兜底：answered 有界（防 LLM 超发）
    updated["answered"] = (updated.get("answered") or [])[-config.RAG_MEMORY_MAX_ANSWERED:]
    save(paper_id, updated)
    return updated


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
# 相关性选择：按当前问题挑选命中记忆，非全量
# ---------------------------------------------------------------------------

# 常见中文功能词/疑问词：字符 bigram 切分易让"什么/是什"等产生假命中，从查询 token 中剔除
_STOPWORDS = {
    "的", "了", "是", "吗", "呢", "啊", "吧", "和", "与", "在", "有", "就", "都", "而", "及", "或",
    "个", "这", "那", "其", "中",
    "什么", "怎么", "如何", "为什么", "哪个", "哪里", "何时", "是否", "是什", "怎样", "咋样", "咋",
    "一个", "这个", "那个", "一些", "这些", "那些", "这样", "那样", "以及", "还是",
}


def _tokens(text: str) -> set[str]:
    from paper_agent.bm25 import tokenize

    return set(tokenize(text)) - _STOPWORDS


def _overlap_score(text: str, q_tokens: set[str]) -> int:
    if not q_tokens:
        return 0
    return len(_tokens(text) & q_tokens)


def select_history(memory: dict, question: str, max_tokens: int | None = None) -> str:
    """按相关性选择记忆拼接为 prompt 段（受 token 预算约束）。无命中返回 ""。

    对每个 事实/偏好/已回答 计算与问题的 token 重叠（剔除停用词），
    按分数降序累积到预算上限。
    """
    from paper_agent.tokens import estimate_tokens

    budget = max_tokens if max_tokens is not None else config.RAG_HISTORY_MAX_TOKENS
    q_tokens = _tokens(question)
    if not q_tokens:
        return ""

    items: list[tuple[int, str, str]] = []  # (score, label, text)
    for f in memory.get("facts", []):
        if f:
            items.append((_overlap_score(f, q_tokens), "事实", str(f)))
    for p in memory.get("preferences", []):
        if p:
            items.append((_overlap_score(p, q_tokens), "偏好", str(p)))
    for t in memory.get("answered", []):
        text = f"问：{t.get('q', '')} 答：{t.get('a', '')}"
        items.append((_overlap_score(f"{t.get('q','')} {t.get('a','')}", q_tokens), "已回答", text))

    picked = [it for it in items if it[0] > 0]
    picked.sort(key=lambda x: x[0], reverse=True)

    parts: list[str] = []
    used = 0
    for score, label, text in picked:
        tok = estimate_tokens(text)
        if used + tok > budget:
            break
        parts.append(f"- [{label}] {text}")
        used += tok
    return "\n".join(parts)