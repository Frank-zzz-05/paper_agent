"""全局配置：加载 .env、定义常量。

注意：`ChatDeepSeek` 读取的是环境变量 `DEEPSEEK_API_BASE`（默认
`https://api.deepseek.com/v1`），`.env` 中的 `DEEPSEEK_BASE_URL` 会被忽略。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（paper_agent 的上级目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env（存在才加载，不覆盖已设置的环境变量）
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ---- DeepSeek ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE_SUMMARY = 0.3
DEEPSEEK_TEMPERATURE_EXTRACT = 0.0
DEEPSEEK_MAX_TOKENS = 4096
DEEPSEEK_TIMEOUT = 60.0
DEEPSEEK_MAX_RETRIES = 3

# ---- Token 预算 ----
# deepseek-chat 上下文 64K；估算 len(text)//3（粗估，偏保守），为输出预留 → 输入上限 50K
INPUT_TOKEN_BUDGET = 50_000
MAX_LLM_CHARS = INPUT_TOKEN_BUDGET * 3          # ≈50K token
SUMMARY_TRUNCATE_CHARS = 40_000 * 3             # ≈40K token，摘要只喂开头（标题/摘要/引言优先）
TRUNCATE_ELLIPSIS = "\n\n[... 中间内容省略 ...]\n\n"  # 抽取用头尾拼接的省略标记

# ---- 数据目录 ----
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_FILE = DATA_DIR / "cache.json"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# ---- RAG / Embedding ----
# BAAI/bge-m3：多语言（中英）、1024 维、输入上限 8192 token
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# 分块：结构优先 + 递归兜底
CHUNK_SIZE = 800         # RecursiveCharacterTextSplitter chunk_size（token 近似）
CHUNK_OVERLAP = 100      # 重叠量
SECTION_MAX_CHARS = 1000  # L1 结构分块后，超此长度的 section 进入 L2 递归分块
SECTION_MIN_CHARS = 150   # 短于此长度的 section 与相邻节合并
# RAG 检索
RAG_TOP_K = 5             # 默认检索块数
RAG_FETCH_K = 10          # 实际获取块数（去重/合并后取 top_k）
# RAG 回答 token 预算
RAG_MAX_CONTEXT_CHARS = 12_000   # 检索上下文上限（≈4K token）
RAG_LLM_MAX_TOKENS = 2048        # 回答最大 token
