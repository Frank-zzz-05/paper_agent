"""纯 Python BM25 关键词检索（零依赖）。

- 中文：字符 bigram 切分（不用 jieba，零依赖）
- 英文/数字：小写词切分
- BM25 公式：Robertson-Sparck Jones（k1=1.5, b=0.75）

用于与 bge-m3 向量检索组成双路混合检索：向量捕捉语义、BM25 捕捉精确关键词命中。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_CJK = "一-鿿"
# 捕获组：re.split 返回 [非中文段, 中文段, 非中文段, ...] 交替
_CJK_SEG_RE = re.compile(rf"([{_CJK}]+)")
_CJK_CHAR_RE = re.compile(rf"[{_CJK}]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """tokenize：英文按小写词，中文按字符 bigram。"""
    tokens: list[str] = []
    for seg in _CJK_SEG_RE.split(text.lower()):
        if not seg:
            continue
        if _CJK_CHAR_RE.search(seg):
            chars = _CJK_CHAR_RE.findall(seg)
            if len(chars) == 1:
                tokens.append(chars[0])
            else:
                tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            tokens.extend(_WORD_RE.findall(seg))
    return tokens


class BM25:
    """BM25 检索器。corpus 为文档字符串列表。"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus
        self.n = len(corpus)
        self.doc_lens = [len(tokenize(d)) for d in corpus]
        self.avgdl = sum(self.doc_lens) / self.n if self.n else 0.0
        self.df: Counter = Counter()                 # term → 出现文档数
        self.postings: dict[str, dict[int, int]] = {}  # term → {doc_idx: tf}
        for i, doc in enumerate(corpus):
            tfs = Counter(tokenize(doc))
            for term, tf in tfs.items():
                self.df[term] += 1
                self.postings.setdefault(term, {})[i] = tf

    def scores(self, query: str) -> list[float]:
        """返回每个文档的 BM25 分数。"""
        scores = [0.0] * self.n
        for term in Counter(tokenize(query)):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in self.postings.get(term, {}).items():
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else tf + 1
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数) 倒序取前 k（去掉 0 分）。"""
        scored = self.scores(query)
        order = sorted(range(self.n), key=lambda i: scored[i], reverse=True)
        return [(i, scored[i]) for i in order[:k] if scored[i] > 0]