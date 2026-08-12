"""token 计数与截断：优先 tiktoken 精确计数，降级语言感知启发式。

问题：`len(text)//3` 对中文（1 字符 ≈ 1.5 token）严重低估，对英文高估。
方案：tiktoken（cl100k_base，DeepSeek/OpenAI 兼容近似的）精确计数；
      不可用时按语言感知估算（中文 1 字符 ≈ 1.5 token，英文 1 token ≈ 4 字符）。

懒加载：tiktoken 只在首次计数时导入，不拖慢 CLI 启动。
"""

from __future__ import annotations

import re

from paper_agent import config

_CJK_RE = re.compile(r"[一-鿿]")

_encoder = None
_encoder_attempted = False


def _get_encoder():
    """惰性获取 tiktoken 编码器（单例）。失败返回 None。"""
    global _encoder, _encoder_attempted
    if not _encoder_attempted:
        _encoder_attempted = True
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def estimate_tokens(text: str) -> int:
    """估算文本 token 数。tiktoken 精确；否则语言感知启发式。"""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    n = len(text)
    if _cjk_ratio(text) >= 0.5:
        return int(n * 1.5) + 1
    return max(1, int(n / 4) + 1)


def _head_char_budget(text: str, max_tokens: int) -> int:
    """语言感知的字符预算（tiktoken 不可用时的降级）。"""
    if _cjk_ratio(text) >= 0.5:
        return int(max_tokens / 1.5)
    return max_tokens * 4


def truncate_head(text: str, max_tokens: int) -> str:
    """保留开头 max_tokens 个 token。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    enc = _get_encoder()
    if enc is not None:
        try:
            return enc.decode(enc.encode(text)[:max_tokens])
        except Exception:
            pass
    return text[:_head_char_budget(text, max_tokens)]


def truncate_head_tail(text: str, max_tokens: int) -> str:
    """头尾截断：保留开头与结尾各 max_tokens//2，中间省略。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    half = max_tokens // 2
    head = truncate_head(text, half)
    tail = truncate_head(text[::-1], half)[::-1]
    return head + config.TRUNCATE_ELLIPSIS + tail