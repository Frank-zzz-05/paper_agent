"""arXiv MCP server —— 让 Claude 检索 / 读取 arXiv 论文。

设计：本地 stdio 传输，FastMCP 3.x 框架。零鉴权（arXiv 公开数据）。
工具面（小表面，一动作一工具）：
  - search_papers          按关键词/字段检索论文列表
  - get_paper_metadata      按 arXiv ID 取元数据（标题/作者/摘要/日期/分类）
  - get_paper_full_text     下载 PDF 并解析全文（ar5iv HTML 兜底）
  - download_paper_pdf      下载 PDF 到本地磁盘
"""

__version__ = "0.1.0"
