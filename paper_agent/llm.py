"""DeepSeek LLM 工厂 + 结构化输出。

结构化输出降级链：
1. function_calling（with_structured_output 默认，已验证支持）
2. json_mode（with_structured_output method="json_mode"，DeepSeek 要求 prompt 含 "json"）
3. 普通 prompt + 提取 JSON 块 + json.loads + model_validate

禁用 strict=True（会切 beta API 且要求 schema 全字段必填）。
"""

from __future__ import annotations

import json
import re

from typing import Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from paper_agent import config

TModel = TypeVar("TModel", bound=BaseModel)

_JSON_MODE_HINT = (
    "请严格输出一个 JSON 对象（不要输出 Markdown 代码块或其他文字），"
    "包含键：research_question, method, dataset, experiment_results, limitations, "
    "contributions, core_innovations, conclusions。dataset/limitations/conclusions 无内容时为 null。"
)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def get_llm(*, temperature: float, model: str | None = None) -> ChatDeepSeek:
    """构造 ChatDeepSeek 实例。

    注意：`ChatDeepSeek` 读取 `DEEPSEEK_API_BASE`（非 `DEEPSEEK_BASE_URL`），
    见 config.py 说明。
    """
    return ChatDeepSeek(
        model=model or config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        api_base=config.DEEPSEEK_API_BASE,
        temperature=temperature,
        max_tokens=config.DEEPSEEK_MAX_TOKENS,
        timeout=config.DEEPSEEK_TIMEOUT,
        max_retries=config.DEEPSEEK_MAX_RETRIES,
    )


def _coerce(result, schema: Type[TModel]) -> TModel:
    """把结构化输出结果统一转为目标 Pydantic 模型。"""
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema.model_validate(result)
    if isinstance(result, str):
        payload = _extract_json_object(result)
        return schema.model_validate(payload)
    if hasattr(result, "content") and isinstance(result.content, str):
        payload = _extract_json_object(result.content)
        return schema.model_validate(payload)
    raise ValueError(f"无法解析结构化输出: {type(result)}")


def _extract_json_object(text: str) -> dict:
    """从文本中提取 JSON 对象；兼容代码块包裹。"""
    m = _CODE_FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 不是对象: {type(payload)}")
    return payload


def structured_invoke(
    llm: ChatDeepSeek, schema: Type[TModel], system: str, text: str
) -> TModel:
    """通用结构化输出，带降级链。仅当三级全部失败时抛 RuntimeError。"""
    # 1. function_calling
    try:
        result = llm.with_structured_output(schema, method="function_calling").invoke(
            [SystemMessage(content=system), HumanMessage(content=text)]
        )
        return _coerce(result, schema)
    except Exception:
        pass

    # 2. json_mode
    try:
        result = llm.with_structured_output(schema, method="json_mode").invoke(
            [SystemMessage(content=system + "\n\n" + _JSON_MODE_HINT), HumanMessage(content=text)]
        )
        return _coerce(result, schema)
    except Exception:
        pass

    # 3. 普通 prompt + 手动解析
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=text)])
        return _coerce(response.content, schema)
    except Exception as exc:
        raise RuntimeError(f"结构化输出失败（三级降级均失败，schema={schema.__name__}）: {exc}") from exc
