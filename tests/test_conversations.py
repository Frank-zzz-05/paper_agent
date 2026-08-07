"""对话历史模块测试：存储 + 滑动窗口 + 滚动摘要 + 格式化。"""

from __future__ import annotations

from paper_agent import config, conversations


def test_append_get_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    pid = "p1"

    # 初始为空
    assert conversations.get_history(pid) == {"summary": None, "turns": []}

    conversations.append_turn(pid, "问题1", "回答1")
    h = conversations.get_history(pid)
    assert h["turns"] == [{"role": "user", "content": "问题1"}, {"role": "assistant", "content": "回答1"}]

    conversations.append_turn(pid, "问题2", "回答2")
    h = conversations.get_history(pid)
    assert [t["role"] for t in h["turns"]] == ["user", "assistant", "user", "assistant"]
    assert h["turns"][-1]["content"] == "回答2"

    # clear 后消失
    assert conversations.clear(pid) is True
    assert conversations.get_history(pid) == {"summary": None, "turns": []}
    assert conversations.clear(pid) is False


def test_trim_sliding_window(tmp_path, monkeypatch):
    """Tier A：超过 MAX_TURNS 轮只保留最近 N 轮，不调 LLM。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TURNS", 4)

    turns = []
    for i in range(10):
        turns.append({"role": "user", "content": f"q{i}"})
        turns.append({"role": "assistant", "content": f"a{i}"})

    called = {"n": 0}
    def summarize_fn(text):
        called["n"] += 1
        return "压缩摘要"

    result = conversations.trim_history({"summary": None, "turns": turns}, summarize_fn=summarize_fn)
    # 保留最近 4 轮 = 2 问 2 答
    assert len(result["turns"]) == 4
    assert result["turns"][0]["content"] == "q8"
    # 被丢弃的旧轮 + 旧摘要未超预算 → 不触发 LLM
    assert called["n"] == 0


def test_trim_rolling_summary(tmp_path, monkeypatch):
    """Tier B：占用 > 高水位（85%）→ 调 LLM 生成滚动摘要，留出余量。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TURNS", 2)
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_CHARS", 400)  # 高水位 = 340
    # 保留的最近 2 轮要超过高水位（340）→ 每轮 ~195 字符
    turns = []
    for i in range(10):
        turns.append({"role": "user", "content": f"问题{i} " + "字" * 190})
        turns.append({"role": "assistant", "content": f"回答{i} " + "字" * 190})

    calls = {"n": 0, "text": ""}
    def summarize_fn(text):
        calls["n"] += 1
        calls["text"] = text
        return "滚动摘要：前几轮关于方法的讨论。"

    result = conversations.trim_history(
        {"summary": "旧摘要", "turns": turns}, summarize_fn=summarize_fn
    )
    assert calls["n"] == 1
    assert "滚动摘要" in result["summary"]
    # 压缩后占用回到高水位之下（余量保证）
    assert len(result["turns"]) <= 2


def test_trim_below_high_water_no_llm(tmp_path, monkeypatch):
    """占用 < 高水位 → 只做滑动窗口，不调 LLM（零成本）。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TURNS", 4)
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_CHARS", 400)  # 高水位 = 340

    # 10 轮短问答：滑动后 4 轮仅 ~80 字符，远低于高水位
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 5

    calls = {"n": 0}
    def summarize_fn(text):
        calls["n"] += 1
        return "摘要"

    result = conversations.trim_history({"summary": None, "turns": turns}, summarize_fn=summarize_fn)
    assert calls["n"] == 0
    assert len(result["turns"]) == 4  # 只保留最近 4 轮


def test_trim_summarize_failure_downgrades(tmp_path, monkeypatch):
    """LLM 压缩失败 → 安全降级为直接丢弃旧轮，不崩溃。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TURNS", 2)
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_CHARS", 100)  # 高水位 = 85

    # 每轮 50 字符，保留 2 轮 = 100 > 85 → 触发压缩（但压缩失败）
    turns = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y" * 50}] * 5

    def failing(text):
        raise RuntimeError("LLM down")

    result = conversations.trim_history({"summary": None, "turns": turns}, summarize_fn=failing)
    assert result["summary"] == ""
    assert len(result["turns"]) <= 2


def test_append_turn_eager_trim(tmp_path, monkeypatch):
    """append_turn 传 summarize_fn 时立即按高水位整理，存储有界。"""
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_TURNS", 2)
    monkeypatch.setattr(config, "RAG_HISTORY_MAX_CHARS", 400)  # 高水位 = 340

    # 先塞满 8 轮长对话（每轮 ~195 字符，滑动后保留的 2 轮将超高水位 340）
    hist = {"summary": None, "turns": [
        {"role": "user", "content": f"q{i}" + "字" * 190} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}" + "字" * 190}
        for i in range(8)
    ]}
    conversations.save("p1", hist)

    calls = {"n": 0}
    def summarize_fn(text):
        calls["n"] += 1
        return "滚动摘要"

    # 追加一轮（内容足够长，滑动后保留的 2 轮超高水位）→ 触发高水位压缩
    conversations.append_turn("p1", "新问题" + "字" * 190, "新回答" + "字" * 190, summarize_fn=summarize_fn)
    assert calls["n"] == 1
    h = conversations.get_history("p1")
    assert h["summary"] == "滚动摘要"
    assert len(h["turns"]) <= 4  # 有界：原始 + 追加后不会无限增长


def test_format_history():
    # 空历史 → 空串
    assert conversations.format_history(None, []) == ""
    # 摘要 + 轮次
    out = conversations.format_history("旧摘要", [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ])
    assert "旧摘要" in out
    assert "用户：问题" in out
    assert "助手：回答" in out