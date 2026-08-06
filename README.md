# Paper Agent — 论文阅读智能体

基于 **LangGraph + DeepSeek** 的命令行论文阅读工具：输入论文来源，自动输出**中文摘要+要点**与**结构化信息抽取**（研究问题 / 方法 / 数据集 / 实验结果 / 局限 / 贡献 / **核心创新点** / 结论）。

同时提供 **arXiv MCP server**（`arxiv_mcp/`），让 Claude 直接在对话中检索 / 读取 arXiv 论文。

## 快速开始

```bash
# 使用 langchain conda 环境
conda activate langchain

# 环境检查（依赖已装好，.env 已配好 DeepSeek + LangSmith key）
python -c "import langchain,langgraph,langchain_deepseek,pypdf,typer;print('deps ok')"

# 本地 PDF（离线）
paper read tests/data/sample.pdf

# arXiv（下载 + 自动解析标题/作者/摘要）
paper read 2404.07143

# 网页文章
paper read "https://www.ruanyifeng.com/blog/2024/01/weekly-issue-286.html"

# JSON 输出（供程序化消费）
paper read 2404.07143 --output json

# 缓存管理
paper list
paper show <id>
paper clear-cache

# RAG 问答（基于已入库论文）
paper ask "这篇论文解决了什么问题？"          # 检索全部已入库论文
paper ask "压缩记忆机制是什么？" --paper <id>  # 限定单篇论文
paper import *.pdf  <arxiv_id>  <url>         # 批量入库（不读，只灌向量库）
```

也可不用 `paper` 命令，直接：`python -m paper_agent.cli ...`

## 常用选项

| 选项 | 说明 |
|---|---|
| `--no-summary` | 跳过摘要与要点 |
| `--no-extract` | 跳过结构化信息抽取 |
| `--output text\|json` | 输出格式（默认 text） |
| `--lang zh\|en` | 输出语言（默认中文） |
| `--no-cache` | 忽略缓存强制重新分析（结果仍会更新缓存） |

## 架构

- **加载器** `paper_agent/loaders/`：本地 PDF（pypdf）/ arXiv（委托 `arxiv_mcp.core`，无重复代码）/ 网页（httpx + BeautifulSoup）
- **图** `paper_agent/graph/`：LangGraph 静态图 `load → summarize ∥ extract → finalize`（并行分支）
- **LLM** `paper_agent/llm.py`：DeepSeek `deepseek-chat`，结构化输出带 function_calling → json_mode → 手动解析三级降级
- **CLI** `paper_agent/cli.py`：typer 命令 `read / list / show / clear-cache / ask`
- **缓存** `paper_agent/cache.py`：`data/cache.json` 磁盘缓存，重复读取命中
- **RAG** `paper_agent/{chunk,vectorstore}.py` + `graph/rag_*.py`：bge-m3 多语言 embedding（本地 GPU）+ ChromaDB 向量库（`data/vectorstore/`），`paper read` 自动入库，`paper ask` 带出处问答

## RAG 问答（P5）

`paper read` 成功后自动把论文分块写入向量库；`paper ask` 检索相关段落并生成**带出处（章节）的回答**。

```bash
paper ask "Infini-attention 如何工作？"          # 跨论文检索
paper ask "方法的局限是什么？" --paper <id>       # 单篇追问
paper import paper1.pdf paper2.pdf 2404.07143   # 批量灌库（不运行摘要）
```

| 技术点 | 方案 |
|---|---|
| Embedding | `BAAI/bge-m3`（多语言中英，1024 维），本地 GPU/CUDA，`HF_ENDPOINT=https://hf-mirror.com` |
| 向量库 | ChromaDB 本地持久化 `data/vectorstore/`，语义检索 + `paper_id` 元数据过滤 |
| 分块 | 结构优先（按章节切分，带 section 出处）+ 递归兜底（超长节切分） |
| 回答 | DeepSeek 生成，**必须标注出处** `[标题, 章节]` |

## 配置（`.env`）

```ini
DEEPSEEK_API_KEY=...                    # 必填
DEEPSEEK_API_BASE=https://api.deepseek.com/v1   # ChatDeepSeek 读取此变量（非 DEEPSEEK_BASE_URL）
LANGCHAIN_API_KEY=lsv2_...              # LangSmith 可选（免费）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=paper_agent
```

## 测试

```bash
python -m pytest tests/ -q
```

## 路线图

- ✅ P0–P4：加载器 / 摘要 / 结构化抽取（含核心创新点）/ CLI / 缓存 / 测试
- ✅ arXiv MCP server：检索 / 元数据 / 全文 / 下载四个工具（`arxiv_mcp/`）
- ✅ P5：RAG 问答 `paper ask`（bge-m3 embedding 本地 GPU + ChromaDB 向量库 + 自动入库）
- ⏳ P6：FastAPI Web 界面（SSE 流式）

详细需求与方案见 [docs/requirements.md](docs/requirements.md) 和 [docs/development-plan.md](docs/development-plan.md)。

---

## arXiv MCP server

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
E:/Miniconda/envs/langchain/python.exe -m arxiv_mcp            # 启动 stdio 服务器
E:/Miniconda/envs/langchain/python.exe -c "from arxiv_mcp import core; print(core.search_papers('all:rag'))"  # 直接调逻辑
```

源码：`arxiv_mcp/core.py`（数据层）/ `arxiv_mcp/server.py`（FastMCP 工具面）/ `tests/test_arxiv_mcp.py`（离线单测）。
