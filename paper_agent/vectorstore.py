"""向量库管理：embedding（bge-m3）+ ChromaDB 本地持久化。

关键设计：
- 单例 embedding function（bge-m3 约 1.2GB，只加载一次）
- 初始化显式传 embedding_function，禁用 chromadb 内置默认（防联网下载 ONNX）
- hf-mirror.com 国内镜像下载模型
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document

from paper_agent import config
from paper_agent.bm25 import BM25
from paper_agent.chunk import chunk_paper, make_paper_abstract_doc

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

# 在 import HuggingFaceEmbeddings 之前设置 HF 镜像（国内网络适配）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_embedding_function: Embeddings | None = None
_vectorstore: Chroma | None = None
_bm25_bundle: tuple | None = None   # (version, BM25, docs) —— 语料变更时重建
_reranker = None                    # bge-reranker-v2-m3 单例（False = 不可用）


def get_embedding_function() -> Embeddings:
    """获取 bge-m3 embedding 单例。

    首次调用时加载模型（可加 --trust-remote-code），
    后续调用直接返回已加载实例。
    """
    global _embedding_function
    if _embedding_function is None:
        # 优先用 langchain_huggingface（细粒度控制），
        # 其次用 langchain_community（已装）。
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

        device = _pick_device()
        _embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={
                "device": device,
                "trust_remote_code": False,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )
    return _embedding_function


def _pick_device() -> str:
    """选择推理设备：优先 CUDA，其次 MPS，最后 CPU。"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_vectorstore() -> Chroma:
    """获取 Chroma 向量库单例。

    使用 PersistentClient + 显式 embedding_function，
    chromadb 不会尝试下载默认 ONNX 模型。
    """
    global _vectorstore
    if _vectorstore is None:
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="papers",
            embedding_function=get_embedding_function(),
            persist_directory=str(config.VECTORSTORE_DIR),
        )
    return _vectorstore


def ingest_paper(
    paper_id: str,
    text: str,
    title: str,
    authors: list[str] | None = None,
    abstract: str | None = None,
    source_type: str = "",
) -> int:
    """将一篇论文分块后写入向量库。

    Args:
        paper_id: 论文唯一 ID。
        text: 全文。
        title: 标题。
        authors: 作者列表。
        abstract: 摘要。
        source_type: 来源类型（pdf/arxiv/web）。

    Returns:
        写入的块数（含论文摘要块）。
    """
    # 先删旧数据，避免重复
    delete_paper(paper_id)

    docs: list[Document] = []

    # 论文级摘要块（跨论文检索第一道筛网）
    docs.append(make_paper_abstract_doc(paper_id, title, authors, abstract, source_type))

    # 正文分块
    docs.extend(chunk_paper(text, paper_id, title))

    if not docs:
        return 0

    get_vectorstore().add_documents(docs)
    return len(docs)


def delete_paper(paper_id: str) -> None:
    """从向量库中删除某篇论文的所有块。"""
    vs = get_vectorstore()
    try:
        # Chroma 的 get 方法获取所有文档 ID
        results = vs.get(where={"paper_id": paper_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
    except Exception:
        pass  # 首次使用时 collection 可能尚不存在


def _get_bm25_bundle() -> tuple | None:
    """惰性构建/重建 BM25 索引（语料变更时按 collection 计数重建）。

    Returns:
        (version, BM25, docs) 或 None（空库/出错）。
    """
    global _bm25_bundle
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
    except Exception:
        return None
    if _bm25_bundle is not None and _bm25_bundle[0] == count:
        return _bm25_bundle
    try:
        results = vs.get(include=["documents", "metadatas"])
    except Exception:
        return None
    docs: list[Document] = []
    ids = results.get("ids", [])
    texts = results.get("documents") or []
    metas = results.get("metadatas") or []
    for cid, text, meta in zip(ids, texts, metas):
        d = Document(page_content=text or "", metadata={**(meta or {}), "chunk_id": cid})
        docs.append(d)
    _bm25_bundle = (count, BM25([d.page_content for d in docs]), docs)
    return _bm25_bundle


def _get_reranker():
    """惰性加载 bge-reranker-v2-m3（CPU 可跑）。失败置 False，之后跳过。"""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(config.RAG_RERANK_MODEL, device=_pick_device())
        except Exception:
            _reranker = False
    return _reranker if _reranker else None


def retrieve(
    query: str,
    paper_id: str | None = None,
    top_k: int = config.RAG_TOP_K,
    fetch_k: int = config.RAG_FETCH_K,
) -> list[Document]:
    """检索相关块（双路混合 + 重排）。

    1. bge-m3 向量检索 RAG_HYBRID_K 块（语义）
    2. BM25 关键词检索 RAG_HYBRID_K 块（精确命中）
    3. 合并去重 → bge-reranker-v2-m3 重排 → 按分数取 top_k（淘汰低相关块）

    reranker 不可用时回退合并结果（保持可用性）。

    Args:
        query: 用户问题。
        paper_id: 限定某篇论文（None = 跨论文检索）。
        top_k: 返回块数。
        fetch_k: 保留参数（MMR 时代的 fetch_k，现由 RAG_HYBRID_K 替代）。

    Returns:
        相关 Document 列表（含 section 元数据），按重排分数降序。
    """
    vs = get_vectorstore()

    # 检查 collection 是否有数据
    try:
        count = vs._collection.count()
    except Exception:
        return []
    if count == 0:
        return []

    candidates: list[Document] = []

    # ---- 1) 向量检索（plain similarity；多样性交给重排） ----
    try:
        kwargs: dict = {"k": min(config.RAG_HYBRID_K, count)}
        if paper_id:
            kwargs["filter"] = {"paper_id": paper_id}
        candidates.extend(vs.similarity_search(query, **kwargs))
    except Exception:
        pass

    # ---- 2) BM25 关键词检索 ----
    bundle = _get_bm25_bundle()
    if bundle:
        _ver, bm25, bdocs = bundle
        for idx, _score in bm25.top_k(query, config.RAG_HYBRID_K):
            d = bdocs[idx]
            if paper_id and d.metadata.get("paper_id") != paper_id:
                continue
            candidates.append(d)

    if not candidates:
        return []

    # ---- 3a) 合并去重 ----
    merged: list[Document] = []
    seen: set[str] = set()
    for d in candidates:
        key = d.metadata.get("chunk_id") or d.page_content[:80]
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)

    # ---- 3b) 重排（可用时）----
    reranker = _get_reranker()
    if reranker is not None and merged:
        try:
            scores = reranker.predict([(query, d.page_content) for d in merged])
            ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
            ranked = [(d, s) for d, s in ranked if s >= config.RAG_RERANK_MIN_SCORE]
            return [d for d, _ in ranked[:top_k]]
        except Exception:
            pass  # 重排失败 → 回退合并结果

    return merged[:top_k]


def get_stored_paper_ids() -> list[str]:
    """获取向量库中所有论文 ID。"""
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
    except Exception:
        return []
    if count == 0:
        return []
    results = vs.get(include=["metadatas"])
    ids_set: set[str] = set()
    for m in (results.get("metadatas") or []):
        pid = m.get("paper_id")
        if pid:
            ids_set.add(pid)
    return sorted(ids_set)


def clear_all() -> int:
    """清空向量库全部论文。返回删除的块数。"""
    vs = get_vectorstore()
    try:
        results = vs.get(include=[])
        ids = results.get("ids", [])
        if ids:
            vs.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def get_paper_chunk_count(paper_id: str) -> int:
    """获取某篇论文的块数。"""
    vs = get_vectorstore()
    try:
        results = vs.get(where={"paper_id": paper_id})
        return len(results.get("ids", []))
    except Exception:
        return 0
