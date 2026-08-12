"""结构化对话记忆测试：存储 + LLM 更新 + 相关性选择。"""

from __future__ import annotations

from paper_agent import config, conversations


def test_get_save_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    pid = "p1"

    # 初始为空
    assert conversations.get_history(pid) == {"facts": [], "preferences": [], "answered": []}

    conversations.save(pid, {"facts": ["F1"], "preferences": ["P1"], "answered": [{"q": "Q1", "a": "A1"}]})
    h = conversations.get_history(pid)
    assert h["facts"] == ["F1"]
    assert h["answered"][0]["q"] == "Q1"

    assert conversations.clear(pid) is True
    assert conversations.get_history(pid) == {"facts": [], "preferences": [], "answered": []}
    assert conversations.clear(pid) is False


def test_update_memory_with_llm(tmp_path, monkeypatch):
    """update_memory 用 update_fn（LLM）合并新问答进结构化记忆。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    conversations.save("p1", {"facts": ["旧事实"], "preferences": [], "answered": []})

    def fake_update(existing, q, a):
        return {
            "facts": existing["facts"] + [f"新事实: {q}"],
            "preferences": existing["preferences"] + ["关注方法"],
            "answered": existing["answered"] + [{"q": q, "a": a}],
        }

    result = conversations.update_memory("p1", "方法是什么", "用了 Transformer", fake_update)
    assert "新事实: 方法是什么" in result["facts"]
    assert "关注方法" in result["preferences"]
    assert result["answered"][-1]["q"] == "方法是什么"

    # 已持久化
    h = conversations.get_history("p1")
    assert "新事实: 方法是什么" in h["facts"]


def test_update_memory_fallback_on_failure(tmp_path, monkeypatch):
    """update_fn 失败 → 安全降级：仅把本轮追加进 answered，不崩溃。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    conversations.save("p1", {"facts": ["F"], "preferences": [], "answered": []})

    def failing(existing, q, a):
        raise RuntimeError("LLM down")

    result = conversations.update_memory("p1", "Q", "A", failing)
    h = conversations.get_history("p1")
    assert h["facts"] == ["F"]  # 原有保留
    assert h["answered"][-1]["q"] == "Q"


def test_update_memory_answered_capped(tmp_path, monkeypatch):
    """answered 有界：超过上限裁剪。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_MEMORY_MAX_ANSWERED", 2)

    def fake_update(existing, q, a):
        return {"facts": [], "preferences": [], "answered": existing["answered"] + [{"q": q, "a": a}]}

    for i in range(5):
        conversations.update_memory("p1", f"Q{i}", f"A{i}", fake_update)
    h = conversations.get_history("p1")
    assert len(h["answered"]) == 2
    assert h["answered"][-1]["q"] == "Q4"  # 保留最近


def test_select_history_relevance():
    """按相关性选择：只有与问题重叠的记忆被选中。"""
    memory = {
        "facts": ["Attention 机制的核心是缩放点积注意力", "数据集是 SQuAD"],
        "preferences": ["偏好简明回答"],
        "answered": [{"q": "Transformer 有几层？", "a": "12 层"}, {"q": "数据集是什么？", "a": "SQuAD"}],
    }
    out = conversations.select_history(memory, "Attention 机制是什么？")
    assert "Attention" in out
    assert "点积" in out
    assert "SQuAD" not in out  # 无关记忆不选中
    assert "偏好" not in out    # 无关偏好不选中

    # 完全不相关 → 空串（不全量拼接）
    assert conversations.select_history(memory, "完全无关话题xyz") == ""


def test_select_history_budget(tmp_path, monkeypatch):
    """超过 token 预算 → 截断（按相关性排序优先保留高分项）。"""
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TOKENS", 10)
    memory = {
        "facts": ["Attention 机制 A", "Attention 机制 B", "Attention 机制 C"],
        "preferences": [],
        "answered": [],
    }
    out = conversations.select_history(memory, "Attention 机制")
    assert out  # 至少选中一个
    assert "C" not in out  # 预算截断，未全量


def test_old_format_ignored(tmp_path, monkeypatch):
    """旧格式（summary/turns）不兼容 → 按空记忆处理。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    config.CONVERSATIONS_DIR.mkdir()
    (config.CONVERSATIONS_DIR / "p1.json").write_text(
        '{"summary": "s", "turns": [{"role": "user", "content": "q"}]}', encoding="utf-8"
    )
    assert conversations.get_history("p1") == {"facts": [], "preferences": [], "answered": []}