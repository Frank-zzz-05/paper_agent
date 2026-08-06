"""AgentState：LangGraph 图的共享状态。"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from paper_agent.models import NormalizedDocument, PaperExtraction, PaperSummary

Status = Literal["pending", "loading", "loaded", "error", "done"]


class AgentState(TypedDict):
    source: str                                     # CLI 输入：路径 / arXiv id / URL
    options: dict                                   # {"summary": bool, "extract": bool, "lang": str}
    paper: Optional[NormalizedDocument]             # 归一化论文（加载后填充）
    summary: Optional[PaperSummary]                 # 摘要结果
    extraction: Optional[PaperExtraction]           # 结构化抽取结果
    errors: Annotated[list[str], operator.add]      # 累积错误（append reducer）
    status: Status                                  # 节点执行状态
