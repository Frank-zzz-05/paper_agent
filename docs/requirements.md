# 论文阅读智能体 —— 需求文档

> 版本：v1.0（2026-08-06） · 技术方案见 [development-plan.md](development-plan.md)

## 1. 项目目标

构建一个命令行论文阅读智能体，用户输入论文来源（本地 PDF / arXiv 链接 / 网页文章），智能体自动解析全文并输出**中文摘要+要点**与**结构化信息抽取**（研究问题、方法、数据集、实验结果、局限、贡献、核心创新点、结论）。

## 2. 用户与使用场景

- **使用者**：论文阅读者 / 研究者，希望通过 CLI 快速理解一篇论文，无需人工通读全文。
- **典型流程**：`paper read 2404.07143` → 几秒内得到摘要+要点+结构化字段；支持 `--output json` 供程序化消费。

## 3. 功能需求

### FR-1 论文来源（三种，自动识别分发）

| 来源 | 输入形式 | 说明 |
|---|---|---|
| 本地 PDF | 文件路径（`.pdf`） | 完全离线可用，不依赖网络 |
| arXiv | ID（如 `2404.07143`）或 `arxiv.org/abs/…` URL | 自动下载 PDF + 抓取元数据（标题/摘要/作者/日期） |
| 网页文章 | `http(s)://` URL | 抓取正文并清洗（剔除导航/脚本） |

### FR-2 摘要与要点

- 输出字段：`title`、`summary`（摘要，2–4 段，覆盖动机/方法/结论）、`key_points`（5–8 条要点）、`keywords`。
- 语言默认中文，可切换英文（`--lang en`）。

### FR-3 结构化信息抽取

- 输出字段：`research_question`（研究问题）、`method`（方法）、`dataset`（数据集，可空）、`experiment_results`（实验结果）、`limitations`（局限，可空）、`contributions`（贡献列表）、`conclusions`（结论/启示，可空）。
- 以 JSON 输出，Pydantic schema 约束，可直接 `model_validate`。

### FR-3a 核心创新点提炼

- 输出字段：`core_innovations`（核心创新点，列表）。
- **与 `contributions` 区分**：`contributions` 指论文的贡献/意义（做了什么、价值何在）；`core_innovations` 指**新颖性本质**——论文在问题定义、方法设计、技术路径上与已有工作的**本质差异**（"新在哪里、凭什么有效"），通常来自对 related work 的对比。每条创新点需简洁、具体、可独立成句（如"提出 X 机制替代传统 Y，解决了 Z 场景下的 W 问题"）。
- 抽取提示词需引导模型：先对比相关工作，再提炼 3–5 条真正的创新点，避免罗列泛泛的功能描述。

### FR-4 结果缓存

- 每次成功读取持久化到 `data/cache.json`；`paper list` 浏览历史，`paper show <id>` 回看，`paper clear-cache` 清空。重复读取同一来源命中缓存。

### FR-5 CLI 命令面

- `paper read <INPUT> [--no-summary] [--no-extract] [--output text|json] [--lang zh|en] [--no-cache]`
- `paper list [--limit N]` / `paper show <ID>` / `paper clear-cache`

### FR-6 RAG 问答与论文语料库（P5，设计已定稿）

- `paper ask "<问题>" [--paper <ID|标题>]`：对已入库论文做向量检索问答；默认检索**最近入库**的一篇，指定 `--paper`（ID 或标题）可检索之前的论文；回答**标注出处**（章节）。
- `paper import <输入…>`：批量分块入库（不生成摘要），供后续检索。
- `paper read` 成功后**自动入库**（被动积累，默认开启）。
- 存储：`data/vectorstore/`（chromadb 本地持久化，gitignore）。设计细节见 development-plan.md §11。
- 运行进度打印到 stderr（`→ loading… / → summarizing…`）
- 错误场景优雅报错：损坏/加密 PDF、arXiv 限流（提示稍后重试）、URL 404、DeepSeek API 故障，均输出清晰信息并以 exit code 1 退出。

## 4. 非功能需求

- **NFR-1 运行环境**：conda env `langchain`（Python 3.12），`E:/Miniconda/envs/langchain/python.exe` 运行，零新增依赖。
- **NFR-2 成本**：单篇论文通常 2 次 LLM 调用（摘要+抽取），使用 `deepseek-chat`（低价高速）；token 预算受控（输入上限 50K token，超限摘要截断前 40K）。
- **NFR-3 离线能力**：本地 PDF 路径完全离线可用。
- **NFR-4 可观测性**：LangSmith tracing 默认开启（`.env` 已配 `LANGCHAIN_API_KEY`）。
- **NFR-5 可维护性**：模块化（loaders / graph / llm / prompts / cli 分层），为后续 RAG、Web 阶段复用。

## 5. 约束与边界（本阶段不做）

- ⏳ **RAG 问答**（`paper ask`）——设计已定稿（development-plan.md §11：bge-m3 + chromadb + 分块管线），尚未实现。
- ✅ **多轮对话上下文**（`paper ask` 同一论文维护历史）——development-plan.md §12 已实现：按论文隔离 + 磁盘持久化 + token 预算 + 两级压缩（滑动窗口 + LLM 滚动摘要）。
- ❌ 多论文对比、文献综述。
- ❌ Web 界面（P6）。
- ⏳ 向量数据库——方案定稿（chromadb 本地 `data/vectorstore/`），尚未实现；checkpointer 多轮对话、账户/鉴权本阶段不做。

## 6. 技术栈

`langchain 1.2.12` · `langgraph 1.1.2` · `langchain-deepseek 1.0.1`（`deepseek-chat`）· `typer 0.24.1` · `pypdf 6.10.2` · `beautifulsoup4 4.14.3` · `httpx 0.28.1` · `langsmith 0.10.15` · `pydantic 2.12.5` · `pytest 9.0.3` · `langchain-text-splitters 1.1.1` · `transformers 5.3.0` · `torch 2.13.0+cpu` · `sentence-transformers`（P5 新增）· `chromadb`（P5 新增）· 模型 `BAAI/bge-m3`（hf-mirror 下载）

## 7. 验收标准

1. `paper read` 对本地 PDF / arXiv ID / 网页 URL 三类输入均输出摘要+要点+全部结构化字段（含**核心创新点**）。
2. `--output json` 输出为合法 JSON，键与 Pydantic schema 完全一致。
3. 本地 PDF 断网可运行；重复读取命中缓存。
4. 损坏 PDF、无效 URL、无网环境给出清晰错误且 exit 1。
5. 核心创新点 ≥1 条、与贡献字段内容不重复（抽样人工核验 2 篇）。
6. `pytest tests/ -q` 全绿；LangSmith 后台可见每次 run 的 trace。
7. `paper ask` 单篇/跨篇返回带出处回答；库已建时离线可问答；`data/vectorstore/` 已 gitignore。
