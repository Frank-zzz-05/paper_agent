"""Pydantic 数据模型：归一化论文文档 + LLM 输出 schema。

字段 description 使用中文，作为 function-calling 结构化输出的语义来源。
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["pdf", "arxiv", "web"]


def _flatten(items) -> list[str]:
    """展平嵌套列表；dict 转 JSON 字符串。"""
    out: list[str] = []
    for it in items:
        if isinstance(it, list):
            out.extend(_flatten(it))
        elif isinstance(it, dict):
            out.append(json.dumps(it, ensure_ascii=False))
        else:
            s = str(it).strip()
            if s:
                out.append(s)
    return out


def _split_to_list(value):
    """把 LLM 返回的列表字段统一转为 list[str]。

    LLM 常见输出形式：合法 JSON 数组 / 含换行的 JSON 数组字符串 /
    "1. x\n2. y" / "- x\n- y"。逐级容错。
    """
    if isinstance(value, str):
        s = value.strip()
        # 1) 合法 JSON 数组
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return _flatten(arr)
            except Exception:
                pass
            # 2) 形如 ["a...", "b..."] 但元素含原始换行（非严格 JSON）→ 按引号提取
            quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
            if quoted:
                out = [q.replace("\n", " ").strip() for q in quoted]
                return [x for x in out if x]
        # 3) 按换行/分号/编号/项目符号拆分（不拆连字符单词，[-*] 需后随空白）
        parts = re.split(r"[\n\r]+|;\s*|(?:\d+[\.\、．]\s*)|•\s*|[-*]\s+", s)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, list):
        return _flatten(value)
    return value


class NormalizedDocument(BaseModel):
    """三种来源解析后的统一论文对象。"""

    id: str = Field(description="来源哈希，用于缓存去重")
    source_type: SourceType
    title: str
    url: str | None = None
    doi: str | None = None
    published: str | None = Field(default=None, description="发布日期，arXiv 提供")
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = Field(default=None, description="摘要（arXiv 元数据提供），可作为总结种子")
    text: str = Field(description="全文正文")
    num_pages: int | None = None


class PaperSummary(BaseModel):
    """摘要与要点。"""

    title: str = Field(description="论文标题")
    summary: str = Field(description="摘要：2-4 段，覆盖动机、方法、结论")
    key_points: list[str] = Field(description="要点：5-8 条")
    keywords: list[str] = Field(default_factory=list, description="关键词")

    @field_validator("key_points", "keywords", mode="before")
    @classmethod
    def _coerce_list_fields(cls, v):
        return _split_to_list(v)

    @field_validator("key_points", mode="after")
    @classmethod
    def _ensure_key_points_plural(cls, v: list[str]) -> list[str]:
        """LLM 偶尔把 5-8 条要点合并成一条长句，按句子边界拆开。"""
        if len(v) == 1 and len(v[0]) > 60:
            parts = re.split(r"(?<=[。！？!?；;])\s*", v[0])
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
            if len(parts) > 1:
                return parts
        return v


class MemoryTurn(BaseModel):
    """多轮记忆中的一条已回答问答。"""

    q: str = Field(description="用户问题")
    a: str = Field(description="回答摘要")


class ConversationMemory(BaseModel):
    """结构化多轮记忆（development-plan §12 结构化压缩）。

    替代"原始轮次 + 滚动摘要"：LLM 把对话压缩为三类结构化信息，
    prompt 时按相关性选择而非全量拼接。
    """

    facts: list[str] = Field(
        default_factory=list, description="已确认事实（论文信息、结论，去重去冗余）"
    )
    preferences: list[str] = Field(
        default_factory=list, description="用户偏好/关注点（反复追问的主题、输出偏好）"
    )
    answered: list[MemoryTurn] = Field(
        default_factory=list, description="已回答问题列表（保留最近 N 条）"
    )

    @field_validator("facts", "preferences", mode="before")
    @classmethod
    def _coerce_list_fields(cls, v):
        return _split_to_list(v)


class PaperExtraction(BaseModel):
    """结构化信息抽取。"""

    research_question: str = Field(description="研究问题")
    method: str = Field(description="方法")
    dataset: str | None = Field(default=None, description="使用的数据集（无则留空）")
    experiment_results: str = Field(description="实验结果")
    limitations: str | None = Field(default=None, description="局限（无则留空）")
    contributions: list[str] = Field(description="贡献/意义：论文做了什么、价值何在")
    core_innovations: list[str] = Field(
        description=(
            "核心创新点（3-5 条）：与已有工作的本质差异——问题定义、方法设计、"
            "技术路径上'新在哪里'，须具体可独立成句，避免与贡献泛泛重复"
        )
    )
    conclusions: str | None = Field(default=None, description="结论/启示")

    @field_validator("contributions", "core_innovations", mode="before")
    @classmethod
    def _coerce_list_fields(cls, v):
        return _split_to_list(v)
