"""arXiv MCP server（FastMCP 3.x，stdio 传输）。

运行：E:/Miniconda/envs/langchain/python.exe -m arxiv_mcp.server
"""

from __future__ import annotations

# 在导入 fastmcp 前设置，抑制启动 banner 与 PyPI 更新检查（stdio 服务器不需要）
import os

os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")

from fastmcp import FastMCP

from arxiv_mcp import __version__
from arxiv_mcp import core

mcp = FastMCP(
    "arxiv",
    version=__version__,
    instructions=(
        "检索并读取 arXiv 论文。可用的工具："
        "search_papers 按关键词检索（返回摘要与链接）；"
        "get_paper_metadata 按 arXiv ID 取元数据；"
        "get_paper_full_text 下载并解析全文文本（供摘要/信息抽取）；"
        "download_paper_pdf 把 PDF 保存到本地磁盘。"
    ),
)


@mcp.tool(description="按 arXiv 检索语法搜索论文，返回标题/作者/日期/分类/摘要截断。")
def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[dict]:
    """按关键词检索 arXiv 论文。

    Args:
        query: arXiv 检索语法，如 "retrieval augmented generation"、
            字段限定 "ti:rag"、"au:hinton"、"cat:cs.CL AND abs:agent"。
        max_results: 返回条数（1–100，默认 10）。
        sort_by: 排序 relevance | submittedDate | lastUpdatedDate。
    """
    return core.search_papers(query, max_results=max_results, sort_by=sort_by)


@mcp.tool(description="按 arXiv ID 获取论文元数据（标题/作者/摘要/发布日期/DOI/分类/期刊引用）。")
def get_paper_metadata(arxiv_id: str) -> dict:
    """获取单篇论文的完整元数据。

    Args:
        arxiv_id: arXiv ID（如 "2404.07143"）或 abs/pdf 链接。
    """
    return core.get_paper_metadata(arxiv_id)


@mcp.tool(description="下载 PDF 并解析论文全文文本（供摘要与信息抽取；PDF 失败自动降级 ar5iv HTML）。")
def get_paper_full_text(arxiv_id: str, max_chars: int = 120_000) -> dict:
    """下载并提取论文全文，返回纯文本。

    Args:
        arxiv_id: arXiv ID（如 "2404.07143"）或 abs/pdf 链接。
        max_chars: 返回文本上限字符数（默认 120_000，约 40K token）。
    """
    return core.get_paper_full_text(arxiv_id, max_chars=max_chars)


@mcp.tool(description="把 arXiv 论文 PDF 下载保存到本地磁盘，返回文件路径。")
def download_paper_pdf(arxiv_id: str, output_dir: str | None = None) -> dict:
    """下载 PDF 到本地。

    Args:
        arxiv_id: arXiv ID（如 "2404.07143"）或 abs/pdf 链接。
        output_dir: 保存目录（默认 data/pdfs/，相对当前工作目录）。
    """
    return core.download_paper_pdf(arxiv_id, output_dir=output_dir)


if __name__ == "__main__":
    mcp.run()
