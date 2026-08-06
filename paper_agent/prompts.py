"""提示词构建。默认中文输出，lang="en" 时切换英文。"""

from __future__ import annotations

_OUT_LANG = {"zh": "中文", "en": "English"}


def _lang_instr(lang: str) -> str:
    return _OUT_LANG.get(lang, "中文")


def build_summary_system(lang: str = "zh") -> str:
    return (
        "你是一名严谨的学术论文阅读助手。请基于给定论文全文，用"
        f"{_lang_instr(lang)}输出摘要与要点。"
        "必须忠于原文，不得编造原文不存在的观点或数据。"
        "输出结构：\n"
        "1. summary：2-4 段，依次覆盖研究动机、采用的方法、主要结论；\n"
        "2. key_points：必须 5-8 条要点；每一条是一个独立、完整、简短的句子"
        "（20-45 字），一个要点一条，严禁把多条要点合并成一条长句；\n"
        "3. keywords：3-6 个关键词。"
    )


def build_summary_human(title: str | None, abstract: str | None, text: str) -> str:
    parts = []
    if title:
        parts.append(f"标题：{title}")
    if abstract:
        parts.append(f"摘要（作者原文，仅供参考）：{abstract}")
    parts.append(f"正文全文：\n{text}")
    return "\n\n".join(parts)


def build_extraction_system(lang: str = "zh") -> str:
    return (
        "你是一名严谨的学术论文分析助手。请基于给定论文全文，用"
        f"{_lang_instr(lang)}抽取以下结构化信息：\n"
        "- research_question：研究问题（一句话）\n"
        "- method：方法（概述，含关键技术与设计思路）\n"
        "- dataset：使用的数据集（无则为 null）\n"
        "- experiment_results：实验结果（关键指标与对比）\n"
        "- limitations：局限（无则为 null）\n"
        "- contributions：贡献/意义——论文做了什么、价值何在（3-5 条）\n"
        "- core_innovations：核心创新点（3-5 条）——与已有工作的本质差异，"
        "'新在哪里、凭什么有效'，须先对比相关工作再提炼，每条具体可独立成句，"
        "避免与 contributions 泛泛重复\n"
        "- conclusions：结论/启示（无则为 null）\n"
        "必须忠于原文，不得编造。"
    )


def build_extraction_human(title: str | None, text: str) -> str:
    parts = []
    if title:
        parts.append(f"标题：{title}")
    parts.append(f"正文全文：\n{text}")
    return "\n\n".join(parts)
