"""LangGraph 节点：load_documents / summarize / extract / finalize。

拓扑为静态图 + 并行分支，节点自守卫：load 失败置 status="error" 后，
summarize/extract 直接透传，finalize 汇总。
"""

from __future__ import annotations

import sys

from paper_agent import config
from paper_agent.graph.state import AgentState
from paper_agent.llm import get_llm, structured_invoke
from paper_agent.loaders import load_paper
from paper_agent.models import PaperExtraction, PaperSummary
from paper_agent.prompts import (
    build_extraction_human,
    build_extraction_system,
    build_summary_human,
    build_summary_system,
)


def _progress(msg: str) -> None:
    """节点进度打印到 stderr（不影响 stdout 的结果输出）。"""
    print(f"→ {msg}", file=sys.stderr, flush=True)


def _head(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _head_tail(text: str, max_chars: int) -> str:
    """头尾截断：保留开头（引言/方法）与结尾（实验/结论）。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + config.TRUNCATE_ELLIPSIS + text[-half:]


def load_documents(state: AgentState) -> dict:
    _progress("加载论文…")
    try:
        loaded = load_paper(state["source"])
        return {"paper": loaded.paper, "status": "loaded"}
    except Exception as exc:
        return {"errors": [f"加载失败: {exc}"], "status": "error"}


def summarize(state: AgentState) -> dict:
    if state["status"] != "loaded" or not state["options"].get("summary", True):
        return {}
    _progress("生成摘要与要点…")
    paper = state["paper"]
    lang = state["options"].get("lang", "zh")
    text = _head(paper.text, config.SUMMARY_TRUNCATE_CHARS)
    try:
        llm = get_llm(temperature=config.DEEPSEEK_TEMPERATURE_SUMMARY)
        summary = structured_invoke(
            llm,
            PaperSummary,
            build_summary_system(lang),
            build_summary_human(paper.title, paper.abstract, text),
        )
        return {"summary": summary}
    except Exception as exc:
        return {"errors": [f"摘要生成失败: {exc}"]}


def extract(state: AgentState) -> dict:
    if state["status"] != "loaded" or not state["options"].get("extract", True):
        return {}
    _progress("抽取结构化信息…")
    paper = state["paper"]
    lang = state["options"].get("lang", "zh")
    text = _head_tail(paper.text, config.MAX_LLM_CHARS)
    try:
        llm = get_llm(temperature=config.DEEPSEEK_TEMPERATURE_EXTRACT)
        extraction = structured_invoke(
            llm,
            PaperExtraction,
            build_extraction_system(lang),
            build_extraction_human(paper.title, text),
        )
        return {"extraction": extraction}
    except Exception as exc:
        return {"errors": [f"结构化抽取失败: {exc}"]}


def finalize(state: AgentState) -> dict:
    status = "error" if state.get("errors") else "done"
    return {"status": status}
