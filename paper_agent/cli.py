"""CLI：paper read / list / show / clear-cache / ask / import。"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer

from paper_agent import cache, config
from paper_agent.graph.build import build_paper_graph
from paper_agent.graph.rag_build import build_rag_graph
from paper_agent.graph.rag_nodes import RAGState
from paper_agent.loaders import cache_id_for, load_paper
from paper_agent.models import PaperExtraction, PaperSummary

app = typer.Typer(
    name="paper",
    help="论文阅读智能体：输入 PDF / arXiv / 网页，输出摘要+要点与结构化信息。",
    no_args_is_help=True,
    add_completion=False,
)


def _fatal(msg: str) -> typer.Exit:
    typer.secho(f"错误: {msg}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _validate_lang(lang: str) -> str:
    return lang if lang in ("zh", "en") else "zh"


def _auto_ingest(*, paper_id: str, text: str, title: str, authors: list[str], abstract: str | None, source_type: str) -> None:
    """自动将论文分块写入向量库。失败静默（不阻塞主流程）。"""
    try:
        from paper_agent import vectorstore as _vs
        _vs.ingest_paper(
            paper_id=paper_id,
            text=text,
            title=title,
            authors=authors,
            abstract=abstract,
            source_type=source_type,
        )
    except Exception:
        pass  # 向量库入库失败不阻塞 read 主流程


def _entry_models(entry: dict) -> tuple[dict, Optional[PaperSummary], Optional[PaperExtraction]]:
    meta = entry.get("meta", {})
    summary = PaperSummary.model_validate(entry["summary"]) if entry.get("summary") else None
    extraction = (
        PaperExtraction.model_validate(entry["extraction"]) if entry.get("extraction") else None
    )
    return meta, summary, extraction


def _print_text(meta: dict, summary: Optional[PaperSummary], extraction: Optional[PaperExtraction]) -> None:
    out: list[str] = []
    out.append(f"标题: {meta.get('title', '?')}")
    src = meta.get("source_type", "?")
    out.append(f"来源: {src}")
    if meta.get("url"):
        out.append(f"链接: {meta['url']}")
    if meta.get("authors"):
        out.append(f"作者: {', '.join(meta['authors'])}")
    if meta.get("published"):
        out.append(f"日期: {meta['published']}")
    if meta.get("num_pages"):
        out.append(f"页数: {meta['num_pages']}")

    if summary:
        out.append("\n── 摘要 ──")
        out.append(summary.summary)
        if summary.key_points:
            out.append("\n── 要点 ──")
            out.extend(f"- {k}" for k in summary.key_points)
        if summary.keywords:
            out.append(f"\n── 关键词 ──\n{', '.join(summary.keywords)}")

    if extraction:
        out.append("\n── 研究问题 ──")
        out.append(extraction.research_question)
        out.append("\n── 方法 ──")
        out.append(extraction.method)
        if extraction.dataset:
            out.append(f"\n── 数据集 ──\n{extraction.dataset}")
        out.append("\n── 实验结果 ──")
        out.append(extraction.experiment_results)
        if extraction.contributions:
            out.append("\n── 贡献 ──")
            out.extend(f"- {c}" for c in extraction.contributions)
        if extraction.core_innovations:
            out.append("\n── 核心创新点 ──")
            out.extend(f"- {c}" for c in extraction.core_innovations)
        if extraction.limitations:
            out.append(f"\n── 局限 ──\n{extraction.limitations}")
        if extraction.conclusions:
            out.append(f"\n── 结论 ──\n{extraction.conclusions}")

    typer.echo("\n".join(out))


def _print_json(meta: dict, summary: Optional[PaperSummary], extraction: Optional[PaperExtraction]) -> None:
    payload = {
        "meta": meta,
        "summary": summary.model_dump() if summary else None,
        "extraction": extraction.model_dump() if extraction else None,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def read(
    input: str = typer.Argument(..., help="本地 PDF 路径 / arXiv ID（如 2404.07143）/ 网页 URL"),
    no_summary: bool = typer.Option(False, "--no-summary", help="跳过摘要与要点"),
    no_extract: bool = typer.Option(False, "--no-extract", help="跳过结构化信息抽取"),
    output: str = typer.Option("text", "--output", help="输出格式: text | json"),
    lang: str = typer.Option("zh", "--lang", help="输出语言: zh | en"),
    no_cache: bool = typer.Option(False, "--no-cache", help="忽略缓存，强制重新分析"),
):
    """阅读一篇论文：加载 → 摘要+要点 → 结构化信息。"""
    lang = _validate_lang(lang)
    if output not in ("text", "json"):
        raise _fatal(f"无效 --output: {output!r}（可选 text/json）")

    try:
        _source_type, paper_id = cache_id_for(input)
    except ValueError as exc:
        raise _fatal(str(exc))

    need_summary = not no_summary
    need_extract = not no_extract
    entry = None if no_cache else cache.get(paper_id)

    # 缓存快速路径
    if entry is not None:
        meta, csum, cext = _entry_models(entry)
        has_summary = csum is not None
        has_extract = cext is not None
        if (not need_summary or has_summary) and (not need_extract or has_extract):
            typer.secho("→ 命中缓存", fg=typer.colors.GREEN, err=True)
            if output == "json":
                _print_json(meta, csum if need_summary else None, cext if need_extract else None)
            else:
                _print_text(meta, csum if need_summary else None, cext if need_extract else None)
            return

    # 执行图（只跑缓存缺失的部分）
    options = {
        "summary": need_summary and (entry is None or not entry.get("summary")),
        "extract": need_extract and (entry is None or not entry.get("extraction")),
        "lang": lang,
    }
    result = build_paper_graph().invoke({"source": input, "options": options})

    if result.get("errors"):
        for e in result["errors"]:
            typer.secho(f"错误: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    meta = result["paper"].model_dump(exclude={"text"})
    summary = result.get("summary")
    extraction = result.get("extraction")

    # 合并缓存中已有的部分
    if entry is not None:
        _, csum, cext = _entry_models(entry)
        summary = summary or csum
        extraction = extraction or cext

    # 用 LLM 从全文提取的真实标题覆盖启发式推断的标题
    if summary is not None:
        meta["title"] = summary.title

    # 无论是否 --no-cache 都写缓存（--no-cache 仅跳过"读取"，属于强制刷新）
    cache.save(
        paper_id,
        meta=meta,
        text=result["paper"].text,
        summary=summary.model_dump() if summary else None,
        extraction=extraction.model_dump() if extraction else None,
    )

    # 自动入库到向量库（失败不阻塞主流程）
    _auto_ingest(
        paper_id=paper_id,
        text=result["paper"].text,
        title=meta["title"],
        authors=meta.get("authors", []),
        abstract=meta.get("abstract"),
        source_type=meta.get("source_type", ""),
    )

    if output == "json":
        _print_json(meta, summary, extraction)
    else:
        _print_text(meta, summary, extraction)


@app.command("list")
def list_papers(
    limit: int = typer.Option(10, "--limit", help="最多显示条数"),
):
    """列出已缓存的论文。"""
    entries = cache.list_entries(limit)
    if not entries:
        typer.echo("（暂无缓存，用 `paper read <输入>` 开始阅读）")
        return
    typer.echo(f"{'ID':<18}  {'标题':<40} {'来源':<6}  阅读时间")
    typer.echo("-" * 90)
    for e in entries:
        meta = e.get("meta", {})
        title = (meta.get("title") or "?")[:40]
        typer.echo(f"{e['id']:<18}  {title:<40} {meta.get('source_type', '?'):<6}  {e.get('read_at', '')}")


@app.command("show")
def show(paper_id: str = typer.Argument(..., help="论文 ID（见 paper list）")):
    """显示缓存的某篇论文结果（JSON）。"""
    entry = cache.get(paper_id)
    if entry is None:
        raise _fatal(f"缓存中不存在该论文: {paper_id}")
    meta, summary, extraction = _entry_models(entry)
    _print_json(meta, summary, extraction)


@app.command("clear-cache")
def clear_cache():
    """清空论文缓存。"""
    n = cache.clear()
    typer.echo(f"已清除 {n} 条缓存")


def _resolve_paper_ref(ref: str, stored_ids: set[str], entries: list[dict]) -> str | None:
    """把 `--paper` 参数解析为向量库中的 paper_id。

    优先精确匹配 ID；否则按标题子串（忽略大小写）匹配。
    `entries` 已按 read_at 倒序（最近入库优先），返回首个命中的 ID。
    返回 None 表示无法解析。
    """
    if ref in stored_ids:
        return ref
    for e in entries:
        if e["id"] not in stored_ids:
            continue
        title = e.get("meta", {}).get("title") or ""
        if ref.lower() in title.lower():
            return e["id"]
    return None


@app.command()
def ask(
    question: str = typer.Argument(..., help="要问的问题"),
    paper_id: Optional[str] = typer.Option(
        None, "--paper", "-p",
        help="限定某篇论文（ID 或标题，不指定则默认检索最近入库的一篇）",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="检索块数"),
):
    """基于已入库论文进行问答（RAG）：检索相关段落 + 生成带出处回答。

    默认问答范围是最近入库的一篇论文；用 --paper 指定 ID 或标题可检索之前的论文。
    """
    try:
        from paper_agent.vectorstore import get_stored_paper_ids
    except ImportError as exc:
        raise _fatal(f"缺少 RAG 依赖: {exc}\n请运行: pip install sentence-transformers chromadb langchain-chroma")

    # 检查向量库是否有数据
    paper_ids = get_stored_paper_ids()
    if not paper_ids:
        raise _fatal("向量库中没有论文。请先使用 `paper read` 阅读论文（会自动入库），或 `paper import` 批量导入。")

    stored_ids = set(paper_ids)
    cache_entries = cache.list_entries()  # 按 read_at 倒序 = 最近入库优先

    # 默认范围：最近入库的一篇论文
    latest_id = next((e["id"] for e in cache_entries if e["id"] in stored_ids), None)

    if paper_id:
        resolved = _resolve_paper_ref(paper_id, stored_ids, cache_entries)
        if resolved is None:
            typer.secho(
                f"⚠ 找不到论文 {paper_id!r}（不在向量库中），默认改为检索最近入库的论文",
                fg=typer.colors.YELLOW, err=True,
            )
            paper_id = latest_id
        else:
            paper_id = resolved
    else:
        if latest_id:
            typer.secho(f"→ 默认检索最近入库的论文: {latest_id}", fg=typer.colors.BLUE, err=True)
        paper_id = latest_id

    state: RAGState = {
        "question": question,
        "paper_id": paper_id,
        "max_context_chars": config.RAG_MAX_CONTEXT_CHARS,
        "context": "",
        "sources": [],
        "answer": "",
        "errors": [],
    }

    result = build_rag_graph().invoke(state)

    if result.get("errors"):
        for e in result["errors"]:
            typer.secho(f"错误: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    typer.echo(answer)

    # 显示出处
    if sources:
        typer.echo("\n── 参考出处 ──")
        shown: set[tuple[str, str]] = set()
        for s in sources:
            key = (s["title"], s["section"])
            if key not in shown:
                shown.add(key)
                typer.echo(f"  [{s['title']}] {s['section']}")


@app.command("import")
def import_papers(
    inputs: list[str] = typer.Argument(..., help="要导入的论文（PDF 路径/arXiv ID/URL，可多个）"),
):
    """批量导入论文到向量库（不运行摘要/抽取，仅入库）。"""
    try:
        from paper_agent import vectorstore as vs
    except ImportError as exc:
        raise _fatal(f"缺少 RAG 依赖: {exc}\n请运行: pip install sentence-transformers chromadb langchain-chroma")

    success = 0
    for inp in inputs:
        typer.secho(f"→ 导入: {inp}", fg=typer.colors.BLUE, err=True)
        try:
            loaded = load_paper(inp)
            p = loaded.paper
            n = vs.ingest_paper(
                paper_id=p.id,
                text=p.text,
                title=p.title,
                authors=p.authors,
                abstract=p.abstract,
                source_type=p.source_type,
            )
            # 同时写缓存（后续 paper read 可命中）
            cache.save(
                p.id,
                meta={"title": p.title, "source_type": p.source_type, "url": p.url,
                      "authors": p.authors, "doi": p.doi, "published": p.published,
                      "num_pages": p.num_pages},
                text=p.text,
                summary=None,
                extraction=None,
            )
            typer.secho(f"  ✔ {p.title[:60]} — {n} 个块已入库", fg=typer.colors.GREEN, err=True)
            success += 1
        except Exception as exc:
            typer.secho(f"  ✘ 导入失败: {exc}", fg=typer.colors.RED, err=True)

    typer.echo(f"\n完成：{success}/{len(inputs)} 篇论文已导入向量库。")


if __name__ == "__main__":
    # Windows 下统一 UTF-8 输出，避免 GBK 控制台/管道乱码
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    app()
