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

from paper_agent import config
from paper_agent.chunk import chunk_paper, make_paper_abstract_doc

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

# 在 import HuggingFaceEmbeddings 之前设置 HF 镜像（国内网络适配）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_embedding_function: Embeddings | None = None
_vectorstore: Chroma | None = None


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


def retrieve(
    query: str,
    paper_id: str | None = None,
    top_k: int = config.RAG_TOP_K,
    fetch_k: int = config.RAG_FETCH_K,
) -> list[Document]:
    """检索相关块。

    Args:
        query: 用户问题。
        paper_id: 限定某篇论文（None = 跨论文检索）。
        top_k: 返回块数。
        fetch_k: 实际获取块数（MMR 去重前）。

    Returns:
        相关 Document 列表（含 section 元数据）。
    """
    vs = get_vectorstore()

    # 检查 collection 是否有数据
    try:
        count = vs._collection.count()
    except Exception:
        return []
    if count == 0:
        return []

    search_kwargs: dict = {"k": min(top_k, count)}
    filter_dict: dict | None = None
    if paper_id:
        filter_dict = {"paper_id": paper_id}

    # 尝试 MMR 检索（去重 + 多样性），失败回退 similarity
    try:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={**search_kwargs, "fetch_k": min(fetch_k, count), **(dict(filter=filter_dict) if filter_dict else {})},
        )
        return list(retriever.invoke(query))
    except Exception:
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        return list(vs.similarity_search(query, **search_kwargs))


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


def get_paper_chunk_count(paper_id: str) -> int:
    """获取某篇论文的块数。"""
    vs = get_vectorstore()
    try:
        results = vs.get(where={"paper_id": paper_id})
        return len(results.get("ids", []))
    except Exception:
        return 0
