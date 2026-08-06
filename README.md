<div align="center">

# 📄 Paper Agent

**论文阅读智能体 — LangGraph + 多模型 LLM**

输入论文来源，自动输出**中文摘要 + 要点**与**结构化信息抽取**（研究问题 / 方法 / 数据集 / 实验结果 / 局限 / 贡献 / 核心创新点 / 结论），并可基于向量库对已入库论文做 **RAG 问答**。

附带一个 **arXiv MCP server**，让 Claude 直接在对话中检索 / 读取 arXiv 论文。

<br/>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-graph-orange.svg)](https://www.langchain.com/langgraph)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-LLM-4B5563.svg)](https://www.deepseek.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-compatible-fff.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ 特性

- **多来源输入**：本地 PDF（离线）、arXiv ID（自动下载 + 解析元数据）、网页 URL
- **双阶段 LangGraph 流水线**：`load → summarize ∥ extract → finalize`（并行分支）
- **结构化抽取**：研究问题 / 方法 / 数据集 / 实验结果 / 局限 / 贡献 / **核心创新点** / 结论
- **RAG 问答**：bge-m3 多语言 embedding + ChromaDB，带出处（章节）的回答
- **磁盘缓存**：`data/cache.json`，重复读取命中，可 `--no-cache` 强制刷新
- **JSON 输出**：`--output json` 供程序化消费
- **arXiv MCP server**：让 Claude 对话中直接检索 / 读取论文

## 📦 目录结构

```
paper_agent/
├── paper_agent/            # 核心包
│   ├── cli.py              # typer 命令行入口
│   ├── config.py           # 配置 / 常量 / token 预算
│   ├── loaders/            # 加载器：PDF / arXiv / Web
│   ├── graph/              # LangGraph 流水线（摘要 / 抽取 / RAG）
│   ├── llm.py              # DeepSeek 调用（function_calling → json 降级）
│   ├── cache.py            # 磁盘缓存
│   ├── chunk.py            # 结构优先分块
│   └── vectorstore.py      # ChromaDB 向量库
├── arxiv_mcp/              # arXiv MCP server（FastMCP）
├── tests/                  # 离线单测
├── docs/                   # 需求与开发计划
├── .mcp.json               # MCP server 注册
└── pyproject.toml
```

## 🚀 快速开始

```bash
# 1. 安装依赖（建议 conda 环境）
conda activate langchain
pip install -e .

# 2. 配置 .env（见下方「配置」）
cp .env.example .env

# 3. 本地 PDF（离线）
paper read tests/data/sample.pdf

# 4. arXiv（自动下载 + 解析标题/作者/摘要）
paper read 2404.07143

# 5. 网页文章
paper read "https://www.ruanyifeng.com/blog/2024/01/weekly-issue-286.html"

# 6. JSON 输出（供程序化消费）
paper read 2404.07143 --output json
```

> 也可不用 `paper` 命令，直接 `python -m paper_agent.cli ...`。

## 🧩 常用选项

| 选项 | 说明 |
|---|---|
| `--no-summary` | 跳过摘要与要点 |
| `--no-extract` | 跳过结构化信息抽取 |
| `--output text\|json` | 输出格式（默认 `text`） |
| `--lang zh\|en` | 输出语言（默认中文） |
| `--no-cache` | 忽略缓存强制重新分析（结果仍会更新缓存） |

## 💬 RAG 问答

`paper read` 成功后自动将论文分块写入向量库；`paper ask` 检索相关段落并生成**带出处（章节）的回答**。

```bash
# 跨论文检索
paper ask "Infini-attention 如何工作？"

# 限定单篇追问
paper ask "方法的局限是什么？" --paper <id>

# 批量灌库（不运行摘要，只写入向量库）
paper import paper1.pdf paper2.pdf 2404.07143

# 缓存管理
paper list
paper show <id>
paper clear-cache
```

| 技术点 | 方案 |
|---|---|
| Embedding | `BAAI/bge-m3`（中英多语言，1024 维），本地 GPU/CUDA，`HF_ENDPOINT=https://hf-mirror.com` |
| 向量库 | ChromaDB 本地持久化 `data/vectorstore/`，语义检索 + `paper_id` 元数据过滤 |
| 分块 | 结构优先（按章节切分，带 section 出处）+ 递归兜底（超长节切分） |
| 回答 | DeepSeek 生成，**必须标注出处** `[标题, 章节]` |

## ⚙️ 配置（`.env`）

复制 `.env.example` 为 `.env`，填好密钥。

### 选择 LLM 后端

通过 `LLM_PROVIDER` 切换，支持多种模型：

| `LLM_PROVIDER` | 后端 | 使用的环境变量 |
|---|---|---|
| `deepseek`（默认） | DeepSeek | `DEEPSEEK_API_KEY` `DEEPSEEK_API_BASE` `DEEPSEEK_MODEL` |
| `openai` | OpenAI | `OPENAI_API_KEY` `OPENAI_API_BASE` `OPENAI_MODEL` |
| `anthropic` | Anthropic Claude | `ANTHROPIC_API_KEY` `ANTHROPIC_MODEL` |
| `openai_compatible` | 任意 OpenAI 兼容服务（Qwen/GLM/Moonshot/Ollama 等） | `OPENAI_API_KEY` `OPENAI_API_BASE` `OPENAI_MODEL` |

```ini
# 默认 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE=https://api.deepseek.com   # 官方推荐，不加 /v1

# 例：切换阿里云百炼 Qwen
# LLM_PROVIDER=openai_compatible
# OPENAI_API_KEY=sk-...
# OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
# OPENAI_MODEL=qwen-plus

# 例：本地 Ollama
# LLM_PROVIDER=openai_compatible
# OPENAI_API_KEY=ollama
# OPENAI_API_BASE=http://localhost:11434/v1
# OPENAI_MODEL=qwen2.5:7b

# LangSmith 遥测（可选）
# LANGCHAIN_API_KEY=lsv2_...
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=paper_agent
```

> **关于 `DEEPSEEK_API_BASE`**：DeepSeek 官方推荐 `https://api.deepseek.com`（不加 `/v1`）。`/v1` 只是为兼容 OpenAI SDK 提供的路径别名，与模型/API 版本无关；LangChain 的 `ChatDeepSeek` 会自行拼接请求路径，故不加 `/v1` 更正确。

## 🏗️ 架构

- **加载器** `paper_agent/loaders/`：本地 PDF（pypdf）/ arXiv（委托 `arxiv_mcp.core`，无重复代码）/ 网页（httpx + BeautifulSoup）
- **图** `paper_agent/graph/`：LangGraph 静态图 `load → summarize ∥ extract → finalize`（并行分支）
- **LLM** `paper_agent/llm.py`：多后端工厂（DeepSeek / OpenAI / Anthropic / OpenAI 兼容），结构化输出带 function_calling → json_mode → 手动解析三级降级
- **CLI** `paper_agent/cli.py`：typer 命令 `read / list / show / clear-cache / ask / import`
- **缓存** `paper_agent/cache.py`：`data/cache.json` 磁盘缓存，重复读取命中
- **RAG** `paper_agent/{chunk,vectorstore}.py` + `graph/rag_*.py`：bge-m3 本地 GPU embedding + ChromaDB 向量库

## 🧪 测试

```bash
python -m pytest tests/ -q
```

## 🤖 arXiv MCP server

项目根目录 `.mcp.json` 注册了一个本地 stdio MCP server（FastMCP 3.x），让 Claude 直接检索 / 读取 arXiv 论文，无需 `paper read`。

**首次使用**：在项目内运行 `claude`，批准 `arxiv` 服务器（或 `claude mcp add` 后重启）。之后会话自动可用。

```bash
claude mcp list    # 查看状态
```

| 工具 | 功能 |
|---|---|
| `search_papers` | 按 arXiv 检索语法搜论文（返回标题/作者/日期/分类/摘要） |
| `get_paper_metadata` | 按 ID 取元数据（标题/作者/摘要/DOI/分类/期刊引用） |
| `get_paper_full_text` | 下载 PDF 解析全文（默认截断 120K 字符 ≈40K token） |
| `download_paper_pdf` | 下载 PDF 到 `data/pdfs/` |

**网络稳健性**（适配国内）：Atom API 多域名兜底（arxiv.org → export.arxiv.org）→ abs 页面 HTML 兜底；PDF 全文失败自动降级 ar5iv HTML；HTTP 429/5xx 指数退避重试。

**独立运行 / 调试**：

```bash
python -m arxiv_mcp            # 启动 stdio 服务器
python -c "from arxiv_mcp import core; print(core.search_papers('all:rag'))"  # 直接调逻辑
```

源码：`arxiv_mcp/core.py`（数据层）/ `arxiv_mcp/server.py`（FastMCP 工具面）/ `tests/test_arxiv_mcp.py`（离线单测）。

## 🗺️ 路线图

- ✅ P0–P4：加载器 / 摘要 / 结构化抽取（含核心创新点）/ CLI / 缓存 / 测试
- ✅ arXiv MCP server：检索 / 元数据 / 全文 / 下载四个工具（`arxiv_mcp/`）
- ✅ P5：RAG 问答 `paper ask`（bge-m3 embedding 本地 GPU + ChromaDB 向量库 + 自动入库）
- ⏳ P6：FastAPI Web 界面（SSE 流式）

详细需求与方案见 [docs/requirements.md](docs/requirements.md) 和 [docs/development-plan.md](docs/development-plan.md)。

## 📄 License

[MIT](LICENSE) © 2026 Frank-zzz-05