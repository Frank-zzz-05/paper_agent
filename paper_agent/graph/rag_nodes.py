"""RAG 图节点：retrieve + answer。

拓扑：
    START → retrieve → answer → END

retrieve: 问题 embedding → 向量检索 top-k（可选 paper_id 过滤）
answer:   检索块 + 出处 → DeepSeek 生成带引用的回答
"""

from __future__ import annotations

import sys
from typing import TypedDict

from paper_agent import config
from paper_agent.llm import get_llm
from paper_agent.vectorstore import retrieve


def _progress(msg: str) -> None:
    print(f"→ {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    question: str
    paper_id: str | None        # None = 跨论文检索
    max_context_chars: int
    context: str                # 拼接后的检索上下文
    sources: list[dict]         # [{section, title, paper_id, excerpt}, ...]
    answer: str
    errors: list[str]


# ---------------------------------------------------------------------------
# 构建检索上下文
# ---------------------------------------------------------------------------


def _build_context(docs: list, max_chars: int) -> tuple[str, list[dict]]:
    """将检索到的 Document 列表拼接为 LLM 上下文，同时提取出处。

    Returns:
        (context_string, sources_list)
    """
    parts: list[str] = []
    sources: list[dict] = []
    total = 0
    seen = set()

    for doc in docs:
        meta = doc.metadata
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)

        # 去重：相同内容只保留一次
        key = content[:80]
        if key in seen:
            continue
        seen.add(key)

        chunk_len = len(content)
        if total + chunk_len > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                content = content[:remaining] + "…"
            else:
                break

        section = meta.get("section", "?")
        title = meta.get("title", "?")
        paper_id = meta.get("paper_id", "?")

        parts.append(f"[{section}] ({title})\n{content}")
        sources.append({
            "section": section,
            "title": title,
            "paper_id": paper_id,
            "excerpt": content[:300] + ("…" if len(content) > 300 else ""),
        })
        total += chunk_len

    return "\n\n---\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# 回答提示词
# ---------------------------------------------------------------------------

_RAG_SYSTEM_ZH = """你是一名严谨的学术论文阅读助手。请基于提供的论文片段回答用户问题。

规则：
1. 回答**必须标注出处**：每条引用用方括号注明来源，格式 [标题, 章节]。
   示例："本文提出了一种新的注意力机制 [Infini-attention, 3. Method]"
2. 仅基于提供的片段回答；如果片段中没有相关信息，明确说"提供的论文内容中没有找到相关信息"，不要编造。
3. 回答使用中文，简洁准确，3-5 段。
4. 如果涉及多条证据，分别标注各自出处。"""

_RAG_SYSTEM_EN = """You are a rigorous academic paper reading assistant. Answer the user's question based solely on the provided paper excerpts.

Rules:
1. You **must cite sources**: annotate each claim with [Title, Section] in brackets.
   Example: "The authors propose a novel attention mechanism [Infini-attention, 3. Method]"
2. Only answer based on provided excerpts; if the information is not in the excerpts, explicitly state "The provided paper content does not contain this information." Do not fabricate.
3. Be concise and accurate, 3-5 paragraphs in English.
4. If multiple pieces of evidence are used, cite each source separately."""


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    """检索相关论文块。"""
    _progress("检索相关段落…")
    question = state["question"]
    paper_id = state.get("paper_id")
    max_chars = state.get("max_context_chars", config.RAG_MAX_CONTEXT_CHARS)

    try:
        docs = retrieve(question, paper_id=paper_id)
    except Exception as exc:
        return {"errors": [f"检索失败: {exc}"], "context": "", "sources": []}

    if not docs:
        return {"errors": ["向量库中没有找到相关内容（可能是尚未导入论文，请先运行 `paper read` 或 `paper import`）"],
                "context": "", "sources": []}

    context, sources = _build_context(docs, max_chars)
    return {"context": context, "sources": sources}


def answer_node(state: RAGState) -> dict:
    """基于检索上下文生成回答。"""
    if state.get("errors"):
        return {"answer": ""}
    if not state.get("context"):
        return {"answer": "（向量库中没有找到相关论文内容。请先使用 `paper read` 或 `paper import` 导入论文。）"}

    _progress("生成回答…")
    question = state["question"]
    context = state["context"]

    system = _RAG_SYSTEM_ZH  # 默认中文
    human = f"论文片段：\n\n{context}\n\n问题：{question}\n\n请基于以上论文片段回答问题，并标注出处。"

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": human}])
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer.strip()}
    except Exception as exc:
        return {"errors": [f"回答生成失败: {exc}"]}
