# 论文阅读智能体 —— 开发方案

> 对应需求见 [requirements.md](requirements.md) · 实现状态见各阶段验证命令

## 1. 项目结构

```
paper_agent/
├── __init__.py
├── config.py            # load_dotenv、模型名、token 预算常量
├── models.py            # Pydantic：NormalizedDocument, PaperSummary, PaperExtraction
├── llm.py               # get_llm() 工厂 + 结构化输出 runnable（带降级链）
├── prompts.py           # SUMMARY_PROMPT, EXTRACTION_PROMPT
├── loaders/
│   ├── __init__.py      # load_paper(input, out_dir) 分发器
│   ├── base.py          # LoadedPaper dataclass
│   ├── pdf_loader.py    # PyPDFLoader (langchain_community)
│   ├── arxiv_loader.py  # httpx + pypdf + arXiv Atom API 元数据
│   └── web_loader.py    # httpx + BeautifulSoup（stdlib parser）
├── graph/
│   ├── state.py         # AgentState TypedDict
│   ├── nodes.py         # load_documents, summarize, extract, finalize
│   ├── build.py         # build_paper_graph() -> CompiledStateGraph
│   ├── rag_nodes.py     # RAG：retrieve / answer（带出处）
│   └── rag_build.py     # build_rag_graph() -> retrieve → answer
├── cache.py             # JSON 磁盘缓存 (data/cache.json)
├── chunk.py             # 两级分块：结构优先 + 递归兜底（带 section 出处）
├── vectorstore.py       # bge-m3 embedding + ChromaDB 本地持久化（GPU）
└── cli.py               # typer：read / list / show / clear-cache / ask / import
│   ├── core.py          #   数据层：Atom API 检索/元数据、PDF/ar5iv 全文、多域名兜底
│   ├── server.py        #   FastMCP 工具面（search/metadata/full_text/download_pdf）
│   └── __main__.py      #   python -m arxiv_mcp 入口
tests/                   # test_loaders / test_graph / test_cli / test_arxiv_mcp
docs/                    # 需求文档 + 本文件
data/                    # pdfs/ + cache.json（gitignore）
```

## 2. LangGraph 图设计

**State**（TypedDict + `Annotated` append-reducer）：`source, source_type, options, paper: NormalizedDocument, chunks, summary, extraction, errors, status`。

**拓扑**（静态图 + 并行分支，节点自守卫）：

```
START → load_documents →┬→ summarize ─┐
                        └→ extract   ─┴→ finalize → END
```

- `summarize`/`extract` 依据 `options` 开关决定是否调 LLM，关闭时透传（近零开销），无需 conditional edges。
- `load_documents` 捕获加载异常写入 `errors` 并置 `status="error"`；下游节点检查后跳过。
- **单次 LLM 调用默认**（`deepseek-chat` 64K 上下文，典型论文 5–15K token）：输入预算 50K（`len(text)//3` 估算），超限摘要截断前 40K；抽取 map-reduce 合并留到 Phase 4。
- 本阶段**不用 checkpointer**（一次性读论文管道）。

## 3. 加载器要点

- **PDF**：`PyPDFLoader`（community，已装），逐页 join，title 取首页/文件名；无需拆分器。
- **arXiv**：委托 `arxiv_mcp.core`（与 MCP server 共享实现，零重复代码）。全文 `get_paper_full_text()`（PDF+pypdf → ar5iv 兜底）、元数据 `get_paper_metadata()`（Atom API 双域名 → abs HTML 兜底）、PDF 磁盘缓存 `download_paper_pdf()`。
- **网页**：httpx + bs4 stdlib parser，剔除 script/style/nav/footer，优先取 article/main。
- 分发规则：`.pdf` → PDF；`^\d{4}\.\d{4,5}$` 或含 `arxiv.org/abs/` → arXiv；`http(s)://` → Web；否则报错。
- **零新增依赖**（httpx / pypdf / bs4 / tenacity 均已装）。

## 4. DeepSeek 集成（`llm.py`）

```python
ChatDeepSeek(model="deepseek-chat", api_key=..., api_base="https://api.deepseek.com/v1",
             temperature=0.3, max_tokens=4096, timeout=60, max_retries=3)
```

- 用 `deepseek-chat`（V3），不用 `deepseek-reasoner`（R1：慢、忽略 temperature、不适合结构化输出）。
- **结构化输出**：`with_structured_output(PaperExtraction, method="function_calling")`（已对照源码验证支持）。**禁用 `strict=True`**（切 beta API 且要求字段全必填）。降级链：function_calling → `json_mode`（prompt 须含字面词 "json"）→ 普通 prompt + `json.loads` + `model_validate`。

## 5. 提示词与 schema

- `PaperSummary`：`title, summary, key_points[list], keywords[list]`
- `PaperExtraction`：`research_question, method, dataset|None, experiment_results, limitations|None, contributions[list], core_innovations[list], conclusions|None` —— 字段用中文 `description=` 携带语义（function-calling 依赖它）。其中 `core_innovations`（核心创新点，3–5 条）与 `contributions`（贡献/意义）区分：提示词要求模型先对比 related work 再提炼"新在哪里"，每条具体可独立成句，避免与贡献泛泛重复。
- 默认中文输出，`--lang en` 切换。
- CLI text 输出含 `## 核心创新点` 小节（紧随 `## 贡献`）。

## 6. 环境配置（`.env`）

```ini
DEEPSEEK_API_KEY=...                     # 已配
DEEPSEEK_BASE_URL=https://api.deepseek.com  # 已配（ChatDeepSeek 忽略此变量）
DEEPSEEK_API_BASE=https://api.deepseek.com/v1  # 新增：ChatDeepSeek 读取此变量
LANGCHAIN_API_KEY=lsv2_...               # 已配（LangSmith）
LANGCHAIN_TRACING_V2=true                # 新增：开启 tracing
LANGCHAIN_PROJECT=paper_agent            # 新增：trace 项目名
```

## 7. 里程碑与验证

| 阶段 | 内容 | 验证 |
|---|---|---|
| **P0** 骨架+文档 | 建包、config、写 `docs/` 两份文档、追加 `.env`、`.gitignore` | `python -c "import langchain,langgraph,langchain_deepseek,pypdf,typer"` 通过；`--help` 可用 |
| **P1** 加载器 | 三个 loader + 分发器；`read --no-summary --no-extract` 打印元数据 | 本地 PDF（离线）+ arXiv ID + 网页 URL 均出 title/非空 text |
| **P2** 摘要 | llm.py、prompts、summarize 节点，接线 load→summarize→finalize | `read 2404.07143 --no-extract` 出连贯中文摘要+要点；本地 PDF 离线可用 |
| **P3** 结构化抽取 | PaperExtraction、extract 节点、并行边、`--output json` | `--output json` 输出合法 Pydantic JSON；json_mode 降级形状一致 |
| **P4** CLI 完善 | cache.py、list/show、token 守卫+map-reduce、token 流式、pyproject console script、pytest | cache 命中、`pytest -q` 绿、损坏 PDF/不可达 URL 优雅报错 exit 1 |
| **P5 ✅** | RAG 问答（`ask`）+ 论文语料库（实现完成，见 §11） | `read` 自动入库、`paper ask` 带出处问答、`paper import` 批量入库、`pytest tests/ -q` 全绿（GPU embed） |
| **P6 延后** | FastAPI Web：POST /read、GET /papers/{id}、SSE 流式 | — |
| **MCP ✅** | arXiv MCP server（`arxiv_mcp/`）：search / metadata / full_text / download 四工具；FastMCP 3.x stdio；多域名 + abs HTML + ar5iv 三级兜底 | 36 个 pytest 全绿；真实网络 4 工具端到端通过；`claude mcp list` 识别 |

## 8. 关键坑点备忘（已对照安装源码验证）

1. `with_structured_output` 在 langchain-deepseek 1.0.1 支持；`strict=True` 会切 beta API 且要求字段全必填 → 避免。
2. `.env` 原 `DEEPSEEK_BASE_URL` 被 ChatDeepSeek 忽略 → 新增 `DEEPSEEK_API_BASE`。
3. 模型参数是 `model=`（不是 `model_name=`）。
4. DeepSeek JSON 模式要求 prompt 含字面词 "json"。
5. LangGraph 1.1 导入：`from langgraph.graph import StateGraph, START, END`。
6. Pydantic v2：用 `model_dump_json()` / `model_dump()`。
7. Windows 下用 `E:/Miniconda/envs/langchain/python.exe` 调用（正斜杠）。

## 9. 端到端验证

```bash
E:/Miniconda/envs/langchain/python.exe -c "import langchain,langgraph,langchain_deepseek,pypdf,typer;print('deps ok')"
E:/Miniconda/envs/langchain/python.exe -m paper_agent.cli read tests/data/sample.pdf            # 本地 PDF（离线）
E:/Miniconda/envs/langchain/python.exe -m paper_agent.cli read 2404.07143 --output json         # arXiv → JSON
E:/Miniconda/envs/langchain/python.exe -m paper_agent.cli read "https://www.ruanyifeng.com/blog/2024/01/weekly-issue-286.html"  # 网页
E:/Miniconda/envs/langchain/python.exe -m paper_agent.cli list
E:/Miniconda/envs/langchain/python.exe -m paper_agent.cli show <id>
E:/Miniconda/envs/langchain/python.exe -m pytest tests/ -q
```

> 激活 conda 环境后可省略前缀：`conda activate langchain && paper read ...`
> 注：arXiv Atom API（export.arxiv.org）在国内网络可能超时，元数据已改为从 PDF 首页文本解析。

### 10. arXiv MCP server（已实现）

**目标**：让 Claude 直接在对话中检索 / 读取 arXiv（解决 `paper read` 的 arXiv 依赖在对话流中不可达的问题）。

- **形态**：本地 stdio（FastMCP 3.x，Python），项目根 `.mcp.json` 注册；个人工具零部署摩擦，后续可平滑升级 MCPB / 远程 HTTP。
- **工具面**（小表面，一动作一工具）：
  - `search_papers(query, max_results, sort_by)` — arXiv 检索语法，裸关键词自动补 `all:`
  - `get_paper_metadata(arxiv_id)` — Atom API 多域名 → abs HTML 兜底
  - `get_paper_full_text(arxiv_id, max_chars)` — PDF(pypdf) → ar5iv HTML 兜底，默认 120K 字符
  - `download_paper_pdf(arxiv_id, output_dir)` — 存盘 `data/pdfs/`
- **paper_agent 集成**：`arxiv_loader.py` 已重构为薄封装层，直接调用 `arxiv_mcp.core` 的 `get_paper_full_text` + `get_paper_metadata` + `download_paper_pdf`，消除了原有的 httpx/pypdf/解析重复代码（~180 行 → ~50 行）。`paper read <arxiv_id>` 端到端验证通过。
- **网络稳健性**：Atom API 双域名（arxiv.org/api/query → export.arxiv.org/api/query）→ abs 页面 BeautifulSoup 解析 → ar5iv；HTTP 429/5xx tenacity 指数退避。
- **坑点**：FastMCP 3.x `Client` 用 `StdioTransport(command, args, cwd)` 启动（无 `command` kwarg）；启动 banner 用 `FASTMCP_SHOW_SERVER_BANNER=false`、PyPI 更新检查用 `FASTMCP_CHECK_FOR_UPDATES=off` 抑制（须在 import fastmcp 前设置）；Atom 的 category/primary_category 取 `term` 属性而非文本；旧格式 arXiv ID（`cmp-lg/9701001`）需单独正则。
- **验证**：`E:/Miniconda/envs/langchain/python.exe -m pytest tests/`（36 个含新增 17 个离线单测）；真实网络端到端 4 工具全部通过；`claude mcp list` 识别 `arxiv` 服务器。

### 11. RAG 问答（P5 ✅ 已实现）

> 决策（2026-08-06）：embedding=`bge-m3` · 向量库=`chromadb` 本地 · 分块=结构优先+递归兜底 · 进库=被动（read 自动）+主动（import） · 推理设备=GPU（CUDA）

**实现（2026-08-06）**：`paper_agent/chunk.py`（两级分块）、`paper_agent/vectorstore.py`（bge-m3 embedding + ChromaDB）、`graph/rag_nodes.py` + `graph/rag_build.py`（retrieve→answer 图）、CLI 新增 `ask` / `import`，`read` 自动入库。GPU 已安装 CUDA torch，embedding 自动选 cuda。端到端验证：单篇追问、跨篇检索、`--paper` 过滤均出带出处回答；74 个 pytest 全绿。

#### 11.1 定位：一个论文语料库，两个查询入口

RAG 补足 `read` 一次性管道的三个缺口：**单篇追问**（`paper ask` 多轮问答，不重读全文）、**超长论文全覆盖**（>50K token 分块后全篇可查，替代"截断前 40K"）、**跨论文检索**（已读论文变为可联合查询的语料库，走向文献综述）。

设计为**一个库、两个入口**：
- 语料库 = 已入库论文的向量库（`data/vectorstore/`）。
- 单篇追问 = 全库检索 + `where={"paper_id": <id>}` 过滤。
- 跨篇检索 = 全库检索。

#### 11.2 库的三层结构

1. **论文级索引**（薄）：`paper_id / title / authors / 摘要` —— 检索第一道筛网，先定位"哪几篇相关"再下沉正文。
2. **正文分块**（主体）：按**章节**切块，每块带 `section 标题 / 序号 / 页` —— 回答"引用出处"的原料。
3. **结构化抽取结果**（复用管道）：`PaperExtraction` 进库，语义级查询（"哪些论文提出新的注意力机制"）直接命中，比全文向量检索准。

#### 11.3 Embedding：bge-m3（定稿）

- **否决 `text-embedding-3`**：OpenAI API 国内不可达 + 破坏 NFR-3 离线（问答半条离线不算真离线）；DeepSeek 无 embeddings API。
- **否决 `bge-small-zh-v1.5`**：偏中文，英文正文检索弱；论文英文 + 查询中文 → 选多语言 **`BAAI/bge-m3`**（M3 = Multi-linguality，维度 1024，输入上限 8192 token）。
- 落地：`HF_ENDPOINT=https://hf-mirror.com` 下载（~1.2GB，huggingface.co 不可达）；sentence-transformers 加载（新依赖，torch 2.13 CPU 已装）。

#### 11.4 向量库：chromadb 本地持久化（定稿）

- `PersistentClient(path="data/vectorstore/")` → 落成**本地目录**，零常驻进程、零运维，备份 = 拷贝目录。
- **选 chromadb 而非 faiss**：原生元数据过滤 `where={"paper_id": ...}` 是单篇/跨篇切换的关键；Milvus/Qdrant 是要起服务的，个人库规模杀鸡用牛刀（`langchain-milvus`/`pymilvus` 虽已装，不启用）。
- **关键坑**：chromadb 默认 embedding 首次使用联网下载 ONNX 模型（国内卡死）→ 初始化必须显式传 `embedding_function=bge_m3`，禁用内置默认。
- `.gitignore` 追加 `data/vectorstore/`（当前只忽略 `data/pdfs/` 与 `data/cache.json`）。

#### 11.5 分块管线：结构优先 + 递归兜底

对比结论：**纯 RecursiveCharacterTextSplitter 不够** —— 忽略章节结构、同主题上下文被拆散、无 section 出处。论文是层级文档，章节标题是免费且最强的语义边界 → 以**结构分块为主**、递归分块兜底，**两层管线**：

| 层 | 方法 | 作用 | 关键参数 |
|---|---|---|---|
| L1 结构分块（主） | 按章节标题正则切分（`\n\s*\d+\.\s+Title`、`Abstract`、`References` 等） | 产出"块 = 一节"，带 section 元数据，天然语义完整 | 自定义 heading 正则 |
| L2 递归兜底 | `RecursiveCharacterTextSplitter`（langchain_text_splitters 1.1.1 已装） | 超长 section（>1000 token）切到目标大小；公式/表格按行优先断 | `chunk_size=800, chunk_overlap=100`，separators `["\n\n","\n"," ",""]` |

- 过短 section（<150 token）与相邻节**合并成一块**（保留双节标题），避免噪声块。
- 公式保护：默认分隔符先切 `\n\n` → `\n`，避免从行内公式中间切断。

#### 11.6 进库两种方式

- **被动（默认）**：`paper read` 成功 → 自动分块入库（复用 load 结果，零额外 LLM 调用）。库 = 读过的所有论文。
- **主动**：`paper import <输入…>` 批量入库不读（为"本周要精读 5 篇，先灌库"场景）。

#### 11.7 ask 子图

```
START → retrieve（query embedding → 向量检索 top-k，可选 paper_id 过滤）
     → answer（检索块 + 命中块结构化字段 + 出处 → DeepSeek 生成带引用回答）→ END
```

- 回答**必须标注出处**（section 标题）——论文 agent 的核心价值，防模型编造章节。
- 低分/无命中 → 明确"库中没有相关内容"，不硬编。

#### 11.8 里程碑与验证

| 子阶段 | 内容 | 验证 |
|---|---|---|
| P5a 分块器 | `chunk.py` + 单测（L1/L2） | 对 `data/pdfs/2404.07143.pdf` 输出带 section 的块 |
| P5b 入库 | embedding（bge-m3）+ chromadb + `read` 自动入库 + gitignore | `read` 后 `data/vectorstore/` 有数据 |
| P5c ask | retrieve/answer 节点 + CLI `paper ask` | 单篇追问带出处、跨篇检索正确、`pytest -q` 绿 |
| P5d import | 批量入库命令 | 一批 PDF 入库后可跨篇检索 |

新增依赖：`sentence-transformers`、`chromadb`（`langchain-text-splitters 1.1.1`、`transformers 5.3.0`、`torch 2.13.0+cpu` 已装）。`BAAI/bge-m3` 模型需 hf-mirror 下载一次。

#### 11.9 坑点备忘（新增）

1. chromadb 默认 embedding 联网下载 → 初始化显式传 `embedding_function`。
2. bge-m3 下载设 `HF_ENDPOINT=https://hf-mirror.com`。
3. `data/vectorstore/` 进 gitignore，防止索引文件入库。

### 12. 上下文管理：`paper ask` 多轮对话（✅ 已实现 2026-08-11）

> 决策（2026-08-11）：按论文隔离 · 磁盘持久化 · **结构化压缩替代滑动窗口** · 按相关性选择

**实现**：`paper_agent/conversations.py`（结构化记忆：事实/偏好/已回答列表 + 相关性选择）、`rag_nodes.answer` 拼记忆段、CLI `ask` 增加 `--reset` / `--no-history`，`clear-cache` / `delete` 连带清对话记忆。90 个 pytest 全绿。

**背景**：`paper ask` 原为**单次无状态**问答。为同一篇论文维护多轮记忆，让追问能引用前文。

#### 12.1 记忆范围：按论文隔离

- 记忆以 `paper_id` 为键隔离。`ask` 作用域是"最近入库一篇"或 `--paper` 指定篇，记忆天然按论文归属。
- 切换论文 → 记忆切换（互不污染）；`--paper` 指定 A 篇不会带出 B 篇前文。

#### 12.2 存储：磁盘持久化（关键约束）

- 每次 `paper ask` 是**独立进程**，内存记忆无意义 → 必须落盘。
- 方案：`data/conversations/{paper_id}.json`，结构 `{"facts": [...], "preferences": [...], "answered": [{"q","a"}, ...]}`。
- `data/conversations/` 在 `data/` 下（整体 gitignore）。
- 命令面：`paper ask <q> [--paper <id>] [--reset]`（`--reset` 清空该篇记忆）。

#### 12.3 结构化压缩（不用滑动窗口）

- **不用滑动窗口、不用滚动摘要**：保留原始轮次是浪费，且"最近 N 轮 + 摘要"割裂语义。
- 每轮回答后调 LLM 把本轮问答**压缩**进结构化槽位（`ConversationMemory` schema，Pydantic 约束）：
  - `facts`：已确认事实（论文信息、结论），LLM 合并去重、删除过时；
  - `preferences`：用户偏好/关注点（反复追问的主题、输出偏好）；
  - `answered`：已回答问题（保留最近 10 条，Python 侧兜底裁剪）。
- 更新失败 → 安全降级：仅把本轮追加进 `answered`，不阻塞主流程。

#### 12.4 拼 prompt：按相关性选择，而非全量

- `rag_nodes.answer` 的 prompt 在"检索块 + 出处"之外追加一段 `记忆`。
- **选择**：对本轮问题做 tokenize（BM25 同款：英文词 + 中文 bigram），剔除停用词
  （"什么/怎么/如何/的/了"等），与每条 事实/偏好/已回答 计算 token 重叠；
  只选重叠 >0 的项，按分数降序累积到 `RAG_HISTORY_MAX_TOKENS`（默认 4000）预算。
- 无关历史不进入 prompt（不全量拼接）——既省 token 又避免噪声。

#### 12.5 预算

- 记忆段 token 预算 `RAG_HISTORY_MAX_TOKENS = 4000`，用 `tokens.estimate_tokens`（tiktoken 精确计数）约束。
- 检索 `retrieve` 仍只基于**当前问题**（记忆不走向量检索），记忆仅作 answer 上下文。

#### 12.6 坑点备忘（新增）

1. 相关性选择要**剔除中文停用词**（字符 bigram 会让"什么/是什"等产生假命中）。
2. 清空时机：`paper clear-cache` **连带清 `conversations/`**，`--keep-vectorstore` 同理；`paper delete <id>` 连带删该篇记忆。
3. 并发：单用户 CLI，无需锁；`save` 用原子 tmp+replace（与 cache.py 一致）。
4. 成本护栏：记忆更新是每轮 1 次 LLM 调用（结构化输出），失败静默降级。

#### 12.7 里程碑与验证

| 子阶段 | 内容 | 验证 |
|---|---|---|
| C1 存储 | `conversations.py`（读写/清空）+ gitignore | 两轮 ask 后 `data/conversations/{pid}.json` 有 facts/answered |
| C2 结构化压缩 | `_update_memory` LLM 更新（事实/偏好/已回答） | 追问记忆落到 answered；失败降级不崩溃 |
| C3 相关性选择 | `select_history` 按重叠 + 预算选择 | 相关记忆进 prompt、无关不拼；`--reset` 清空 |
| C4 CLI | `--reset` / `--no-history` 开关 | `pytest -q` 全绿 |

**新增依赖**：无（复用现有 LLM；磁盘 JSON 用 stdlib）。

### 12b. 检索质量与 token 预算（✅ 已实现 2026-08-11）

> 三个改进：真实 token 计数 · 双路检索（向量+BM25）· reranker 重排

#### 12b.1 token 计数：tiktoken 精确 + 语言感知降级（`paper_agent/tokens.py`）

- 问题：`len(text)//3` 对中文（1 字符 ≈ 1.5 token）严重低估、对英文高估。
- 方案：优先 **tiktoken**（cl100k_base，DeepSeek/OpenAI 兼容近似）精确计数，懒加载不拖启动；
  不可用时按语言感知降级（中文 1 字符 ≈ 1.5 token，英文 1 token ≈ 4 字符）。
- 应用：`graph/nodes.py` 摘要/抽取截断（`truncate_head` / `truncate_head_tail`）、
  `conversations.py` 记忆预算，全部改为 token 精确计数。
- 常量：`INPUT_TOKEN_BUDGET=50K`、`SUMMARY_TOKEN_BUDGET=40K`（原 `*3` 字符预算移除）。

#### 12b.2 双路检索：bge-m3 向量 + BM25 关键词（`paper_agent/bm25.py`）

- MMR 只做多样性去重、精度一般；向量检索对精确关键词命中（公式/术语/编号）弱。
- 方案：向量 top-k（语义）+ BM25 top-k（精确命中，纯 Python 零依赖：英文按词、中文按字符 bigram）→ 合并去重。
- 常量：`RAG_HYBRID_K=10`（各路取 top-k 合并）。

#### 12b.3 reranker 重排：bge-reranker-v2-m3

- 合并结果用 `BAAI/bge-reranker-v2-m3`（与 embedding 同源，CPU 可跑，懒加载单例）打分重排。
- **按重排分数降序截断** 12K 上下文（替代原"按拼接顺序截断"）；分数 < `RAG_RERANK_MIN_SCORE=0.1` 淘汰低相关块。
- reranker 不可用（离线/下载失败）→ 回退合并结果，保持可用性。

#### 12b.4 里程碑与验证

| 子阶段 | 内容 | 验证 |
|---|---|---|
| R1 tokens | `tokens.py` + nodes/conversations 接入 | 中文文本 token 计数不再低估；`pytest` 绿 |
| R2 BM25 | `bm25.py` + 双路合并 | 关键词命中进候选；BM25 单测绿 |
| R3 rerank | bge-reranker-v2-m3 接入 + 分数截断 | 重排后上下文按分数排序；reranker 不可用时回退 |

**新增依赖**：`tiktoken`（已装）；reranker 模型 `BAAI/bge-reranker-v2-m3` 需 hf-mirror 下载一次（懒加载，首次 ask 时下载）。
