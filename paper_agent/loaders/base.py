"""加载器统一接口。"""

from __future__ import annotations

from dataclasses import dataclass

from paper_agent.models import NormalizedDocument


@dataclass
class LoadedPaper:
    """一次加载的产物：归一化文档 + 原始文本。"""

    paper: NormalizedDocument
