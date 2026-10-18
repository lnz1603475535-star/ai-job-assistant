# AGENTS.md — AI 求职助手 · 项目交接与协作约定

> **这份文件是什么**：从 Claude Code 的 24 篇记忆（`C:\Users\李晓杰\.claude\projects\C--Users----\memory\`）、3 份会话记录与 git 历史整理出的单一事实源。任何 AI 编码代理接手，**先读完本文件再动手**。
>
> **怎么维护**：进度变了改 §5；新决策写 §4；踩了新坑追加 §8；待办变更改 §9。纯过程性叙述（"今天聊了什么"）不要往里写。
>
> **原始细节**：§13 是原始记忆索引，需要故事级细节时回查。

---

## 0. 三十秒接续

| 项 | 值 |
|---|---|
| 代码目录 | `C:\Users\李晓杰\ai-job-assistant\16-job-assistant` |
| 当前阶段 | 第 13 课（FastAPI）**代码与消化均已完成**（2026-09-03 消化完，2026-10-11 api.py 通读收尾） |
| 下一步 | **第 14 课 PostgreSQL，尚未开工**（代码里只有 4 处"第 14 课迁移"TODO，依赖清单里没有 psycopg/sqlalchemy） |
| 开工第一件事 | 先讲概念（ER 图 / 为什么 PostgreSQL / PostgresSaver 是什么），概念立住再动代码 |
| 硬约束 | 改完必须 `ruff check .` + `npx pyright .` + 测试**三绿**；**不自动 commit**，等用户确认 |
| git 状态 | `main` 与 `origin/main` 同步，工作区干净 |

用户说"继续"时的默认动作：读本文件 → `git status` → 跑 `python regression_test.py`（零 LLM、秒级）确认基线 → 按 §5 的下一步开工。

---

## 1. 用户与目标

- 中国矿业大学 **AI + 采矿工程交叉学科**硕士在读（2028 届）；本科山西农业大学计算机（2020.09–2024.06）。
- 编程基础：Python / Java / JavaScript / Vue，偏后端。**不碰 Azure，不要推荐 Azure 方案。**
- 目标：**AI 应用工程师**实习 → 校招。暑期实习投递窗口 **2026.12–2027.03**（大厂 3 月集中开放），秋招 2027.08–10。
- 这个项目**不是练习项目**，是要写进简历的生产级项目，面试官要能打开链接直接用。

---

## 2. 怎么和他协作（直接影响每次回答的形态）

1. **讲课从基础概念讲起**（初学者视角）：先讲"是什么 / 为什么 / 原理"，可配最小可跑示例；再回到项目代码做"回头看"。
   - 反面做法：每段都绑定项目代码逐行讲——信息密度过高，概念没立住就上应用细节，吸收率低（2026-08-30 明确要求改掉）。
2. **不要每段布置练习**：练习他自己安排（"练习留到最后做"）。提问验证保留，但放在概念讲完之后，不打断讲解节奏。
3. **三层检验**（新课/新模块的终点）：
   - 第 1 层 做出来：功能跑通；
   - 第 2 层 验完整：全链路（测试全 PASS + 前端可用）；
   - 第 3 层 讲出来：白纸画架构图，能讲清每个节点的输入输出与选型理由。
4. **他会追问 why**：每个技术决策都要能讲出不选别的原因，否则就是"教程教的"，面试不加分。
5. **日期只用硬证据**：跨天对话里不要用"今天/明天"，拿不到证据就写区间（如 08-07~08-14）或"约"，**绝不猜**。
6. 全中文交流。

---

## 3. 项目速览

### 结构
`ai-job-assistant/` 是 LangChain 12 课 + 后端补丁课的代码仓库，16 个章节目录（`01-hello-agent` … `16-job-assistant`）。
**只有 `16-job-assistant/` 是简历项目本体**，其余是课程练习存档。

### 核心文件（`16-job-assistant/`）
| 文件 | 职责 | 规模 |
|---|---|---|
| `app.py` | Streamlit 5 步向导 UI、经验库管理、SSE 消费、本地回退 | 约 1460 行，项目最大文件 |
| `api.py` | FastAPI 服务，11 条路由（生成/定制/导出/流式/用户 CRUD/面试） | 约 730 行 |
| `core.py` | 检索（FAISS+BM25+RRF）、embedding、TokenBudget、日志、LLM 客户端 | 约 700 行 |
| `workflow.py` | LangGraph 7 节点工作流 + checkpoint 单例 | 约 490 行 |
| `resume_engine.py` | 5 个 LLM 引擎函数 + Agent 定制 + 流式生成器 | 约 520 行 |
| `exporters.py` | Markdown → PDF（fpdf2）/ Word（python-docx） | 约 750 行 |
| `prompts.py` / `models.py` | 提示词模板 7 个 / Pydantic 数据模型 | — |
| `terminal_test.py` / `api_test.py` / `regression_test.py` / `ui_test.py` | 四套测试 | 见 §6.4 |

### 技术栈
Python 3.12 · DeepSeek（`deepseek-flash`）· LangChain 1.3.1 · LangGraph 1.2.0 · FAISS · BM25(rank_bm25) · jieba · HuggingFace Embeddings · Streamlit · FastAPI · fpdf2 / python-docx
升级路径：+ PostgreSQL → + Docker → + 免费部署。

### 运行
```powershell
cd C:\Users\李晓杰\ai-job-assistant\16-job-assistant
streamlit run app.py                  # 前端 8501
uvicorn api:app --port 8000           # 后端 8000（前端优先走 API，失败回退本地）
python terminal_test.py               # 端到端（会调 LLM）
python regression_test.py             # 零 LLM 缺陷回归，秒级
ruff check .                          # 风格
npx pyright .                         # 类型（与 Pylance 同源）
```

### 仓库
`https://github.com/lnz1603475535-star/ai-job-assistant`，分支 `main`。

---

## 4. 架构与关键机制（讲 why 的素材）

### 4.1 一句话
**三输入 → 一输出**：样本简历（定风格）+ 用户信息（定内容）+ JD（定方向）→ 定制简历。

### 4.2 七节点工作流
```
validate_inputs → extract_style → extract_jd → parse_user → generate_base → check_parsed → customize
```
- `extract_jd` 原本在 `parse_user` 之后，**2026-08-01 前移**：两者无依赖，且 `parse_user` 需要 JD 关键词做按需检索。
- `WorkflowState` 是 `TypedDict(total=False)`：节点只写部分键，键存在性由图执行顺序保证。
- `notifications: Annotated[list[str], operator.add]`：节点追加提醒，**两个消费者**——Agent（调整策略）和 UI（`st.warning`）。只接一头是历史踩坑（§8 #12）。
- 断点：`interrupt_after=["check_parsed"]`（2026-07-17 从 `generate_base` 前移，让用户看到完整解析提醒再审核）；恢复必须先 `get_state` 确认 checkpoint 存在，再 `invoke(None)`，异常透传不包裹。
- 单例统一叫 `get_*`（`get_checkpointer` / `get_embeddings` / `get_workflow` / `get_retrieval_service`）：`build_*` 暗示"每次新建"，会让人误判性能特征。

### 4.3 检索（混合 + 融合）
- 双路召回 + **RRF 融合**：`score = 1/(60 + FAISS排名) + 1/(60 + BM25排名)`。
- FAISS 路用 `similarity_search`，**不用 MMR**（MMR 按多样性排序，与 RRF"rank 越小越相关"的假设冲突，§8 #1）。
- 融合 key 用 **chunk 整数索引**，不用字符串（PDF 解析的换行差异会让同一 chunk 被当成两个，§8 #2）。
- BM25 候选必须 `score > 0` 过滤，否则零分块靠 rank=0 挤占名额（§8 #21）。
- 候选池 `fetch_k = k * 5`（上限 20）；中文用 jieba 分词，**不要回退成 `.split()`**。
- 全局状态已收敛为 `RetrievalService` 类 + 模块级注册表；`load_and_index_documents` 自动注册，消除调用方遗漏。

### 4.4 按需检索（架构升级，非原计划）
经验库 > 5000 字符时：JD 关键词 → `search_documents(doc_types=["user_experience"])` → 只喂 top-10 块（≈5000 字符），**token 恒定不随经验库增长**。
三兜底：检索空 → 回退全文；库 ≤ 5000 → 跳过检索；检索异常 → 回退全文。
⚠️ **经验库头部（姓名/联系方式/学历）必须无条件保留**（`_experience_head()`，≤800 字符）——它与 JD 关键词零交集，被检索替换过导致姓名邮箱全空（§8 #21）。
实现形态：核心逻辑是普通函数 `_search_documents_impl`，`@tool` 只是薄包装（StructuredTool 不能直接调用，§8 #13）。

### 4.5 定制与重试
- `customize_for_jd` 返回 `(text, token_usage, failed)` 元组，不再依赖模块级全局。
- 瞬时错误（超时/连接/限流/5xx，**按异常类型判断，不按字符串匹配**）→ 指数退避重试 1 次；非瞬时 → 直接降级基础简历。
- **流式路径无重试**（中途重跑体验差），失败在任务层降级。
- TokenBudget 作为 **Agent 中间件**在每轮模型调用前注入提醒、调用后累积；预算 `DEFAULT_TOKEN_BUDGET = 30000`（实测一轮 10140 / 两轮 20533，原 15000 永远不会触发，§8 #23）。

### 4.6 SSE 流式（第 13 课重点）
- 异步任务模型：`POST /stream` 后台线程执行（与 HTTP 生命周期解耦）→ `GET /stream/{task_id}` 逐 token → `GET /result/{task_id}` 轮询兜底。**前端中断不影响任务**。
- 事件投递：`queue.Queue`（生产端直接 put）+ 消费端 `get_nowait()` + `asyncio.sleep(0.05)`。
  **刻意不碰事件循环**——第一版 `asyncio.Queue + call_soon_threadsafe` 在 TestClient 下彻底失效（§8 #26）。
- 空闲 15 秒发 SSE 心跳注释行，防代理掐断（公网必需）。
- `stream_mode=["messages","values"]` 是组合模式，输出 `(mode, data)` 元组，解包方式与单模式不同（§8 #18）。

### 4.7 方案 B：生成与定制同进程
generate 也走 `POST /api/resume/generate`，用后端返回的 `thread_id`；后端不可用或 404 **一律回退本地** `resume_workflow`。
原因：Streamlit 进程与 uvicorn 进程各有自己的 MemorySaver，跨进程找不到 thread_id（§8 #21）。
两条路径返回值必须**同构**（`_merge_customize_result` + 键集合断言），否则走 API 成功后基础简历显示为空。

### 4.8 基础设施
- **日志**：`RotatingFileHandler`（10MB×3）+ 双通道分级（控制台 WARNING+ / 文件 DEBUG+）+ 定位信息 + 第三方库静音 + `SensitiveFilter` 脱敏（API key / 手机号 / 邮箱）。入口在 `app.py main()` 与 `terminal_test.py`，新模块只 `logging.getLogger(__name__)`。
- **LLM 客户端**：全项目只在 `core.py` 构造一次，`timeout=120.0` + `max_retries=1`。SDK 默认 read=600s×重试 2 次 ≈ 最坏挂 30 分钟，且全程占一个线程池槽位（§8 #28）。
- **内存治理**：checkpoint LRU 50 / 定制任务注册表 LRU 20 / 面试会话 LRU 50，三处原本都"永不回收"。

---

## 5. 进度（学到哪、做到了哪）

### 已完结
- **LangChain 12 课**（Phase 1–5）全部学完，代码全部跑通。
- **第 12 课简历生成器 Round 1–5** 全部交付：数据模型+核心函数 → LangGraph 工作流 → Streamlit UI → RRF/PDF/Token 预算 → PDF/Word 导出+照片+求职意向。
- **第 13 课 FastAPI**：5 个核心端点（实际 11 条路由）+ SSE 流式 + 前端异步化 + 全局状态重构，2026-08-05 代码完成，**2026-09-03 消化完毕**（A→F 板块全过，SSE 时序图白纸画出，口试 5/5）。
- **类型标注修复**：2026-09-09 pyright 96 → 0，ruff 全过（§8 #20）。
- **全面代码审查 + 方案 B**：2026-09-12，三个 P0 缺陷修复，新增 regression_test / ui_test（§8 #21）。
- **api.py 走读**：2026-09-09 至 2026-10-11 完成；末次讨论确认"第 13 课新知识分在 api.py / api_test.py / app.py 客户端段 / 依赖与日志四处"。

### 当前档位
**第 14 课 PostgreSQL 未开工。** 代码内仅有 4 处 TODO 指向它（`api.py` 第 14 / 64 / 146 / 641 行附近），`requirements.txt` 无数据库依赖。

### 第 14 课范围
- 四张表：用户 / JD / 简历 / 对话历史；画 ER 图、建索引、写慢查询优化。
- `MemorySaver` → `PostgresSaver`：checkpoint 持久化，进程重启不丢。
- 上下文管理：模拟面试多轮对话的前提（同 thread_id 跨请求保持记忆）。
- 顺带：模型微调（LoRA / QLoRA）嵌入本课，见 §9。

---

## 6. 代码规范与工作流（写代码前必读）

### 6.1 写之前
1. 先翻 §9 确认这次改动与后续计划不冲突；grep `TODO/HACK/FIXME`。
2. 多文件改动先想清架构，**先计划再动手**。
3. 不提前优化——数据驱动，别猜。

### 6.2 写的时候
4. 每行都要有理由；判断条件要严谨，不靠巧合。
5. **新功能必须同步加错误处理**（不能只写 happy path）：
   - 用户输入错误 → 中文友好提示 + 告诉怎么修；
   - 系统错误 → 分类，区分可重试/不可重试；
   - 意外异常 → 兜底捕获，截断信息，不把 traceback 抛给用户。
6. 命名要让调用方的"预期"正确：取一个叫 `get_*`，造一个才叫 `build_*`。
7. 写完自查四项：正则边界（空串/特殊字符）、异常后变量状态、类型标注一致、换行符（Win `\r\n` vs `\n`）。

### 6.3 写完之后（八个角度各过一遍）
调用方怎么用 / 数据怎么坏 / 调用链怎么断 / 外部系统怎么坑 / 平台差异怎么炸 / 用户怎么困惑 / 类型契约怎么骗人 / 重复之处怎么骗人。
- **重复必须并排看**：两处判据不一致，线性阅读在结构上就看不见。做法：对同一份共享数据画表——谁在读、读几次、各自判据。
- **把语义约束降维成断言**：机器检测不了"两处必须一致"，但可以写成测试（如"两条路径键集合相同"）。这是唯一能长期不复发的手段。
- **改不改的判据**：它防的风险（概率 × 后果）是否大于它引入的维护面。边际时默认不做；被批评"过度设计"后也别矫枉过正。

### 6.4 验证（三条全绿才算完）
```powershell
ruff check .                 # 风格
npx pyright .                # 类型（Pylance 同源，零成本）
python regression_test.py    # 63 断言，零 LLM，秒级
```
- 测试规模：`terminal_test.py` 14/14（端到端，调 LLM）、`api_test.py` 20/20（TestClient）、`regression_test.py` 63 断言（零 LLM）、`ui_test.py` 19 断言（Streamlit AppTest 无头跑 app.py）。
- **测试通过 ≠ 目标路径生效**：兜底 except 会把功能失效吞成假阳性。端到端要同时验证"走了哪条分支"（查日志/中间值）。
- **exit code 0 ≠ 没 bug**：关键输出要检查数值是否在合理量级（曾把 `1781607138.90s` 当正常计时）。
- `terminal_test.py` 跟 `app.py` 走相同路径；**只保留当前架构的测试，删旧 Round 残留**。
- 改了检索/预算/导出，先跑 regression_test。
- UI 相关改动 terminal_test 覆盖不到，必须 AppTest 或人工点验。

### 6.5 提交
- **改完不自动 commit**，等用户确认。
- commit 前最后看一遍 diff；消息写"做了什么 + 为什么"（中文，前缀 `fix:` / `refactor:` / `test:` / `docs:`）。
- 小步提交，一次改动一个主题，便于回滚。

### 6.6 类型标注约定
- `Any` = "先别检查"（部署前补精确类型时的占位符）；`object` = "严格检查用法"，会传染给调用方。当前全项目用 `Any`。
- 第三方库 stub 缺陷用**文件级 pyright 压制 + 注释写明重开条件**（部署前统一补标注时重开）。
- `AIMessage.content` 是 `str | list[内容块]`：必须 `isinstance(content, str)` 窄化，**不要用 `str(content)` 兜底**（会把内容块列表变成 Python repr 混进业务数据）。

---

## 7. 不要做的事（负面清单）

- 不要推荐 Azure。
- 不要回退 `similarity_search` → MMR，不要回退 jieba → `.split()`。
- 不要用字符串当 RRF 的 key。
- 不要把 LLM 超时/重试改回 SDK 默认。
- 不要在 `asyncio.to_thread` 里做"等待"（占共享线程池，池子只有 `min(32, cpu+4)`）。
- 不要为方便测试而让某条分支长期不被覆盖——那是给 bug 上锁。
- 不要写"这次先这样，回头再补"的错误处理。
- 不要在没验证真实触发路径前就断言"这里有 bug"（曾误报 2 条）。

---

## 8. 踩坑索引（28 条，面试素材）

| # | 问题 | 结论 |
|---|---|---|
| 1 | FAISS MMR 按多样性排序，与 RRF 的"rank 小=相关"冲突 | 改回 `similarity_search`；双路融合本身已提供多样性 |
| 2 | RRF 用 `page_content` 字符串做 key，PDF 换行差异导致同一 chunk 去重失败 | 改用 chunk 整数索引 |
| 3 | 文件不存在时异常处理嵌套抛错，触发递归栈溢出 | 格式检查移出 try；`getsize` 单独包 `OSError`；删掉"捕获再原样抛出" |
| 4 | 索引路径没过 `normalize_text`，PDF 硬换行污染向量 | 所有加载路径统一过同一个 normalize |
| 5 | `MemorySaver()` 每次新建，断点恢复完全失效 | 模块级懒加载单例 |
| 6 | `get(a) or get(b) or 0` 把合法的 0 当缺失 | 用 `"key" in dict` 判断存在性 |
| 7 | 5 个引擎函数错误处理不一致，线上挂了查不到日志 | 统一兜底 + 每个 except 加 `logger.exception()` |
| 8 | 用了废弃的 `langchain.agents.create_agent` | 迁到 `langgraph.prebuilt.create_react_agent`，全链路统一 langgraph 运行时 |
| 9 | Agent 多轮 tool-call 的 usage 被逐条累加，计数虚高 | 倒序取最后一条带 usage 的消息 |
| 10 | 全局向量库无锁 + 建索引不写全局（调用方漏写就静默失败） | 收敛为 `RetrievalService` + 自动注册（第 13 课落地） |
| 11 | 脱敏过滤器把双层 for 写进生成器，args 被放大 3 倍 | 显式双层循环，同一元素叠加应用所有 pattern |
| 12 | 引擎层兜底后形成静默错误链，用户拿到空洞简历且无提示 | 加 `notifications` + `check_parsed` 节点，**Agent 与 UI 两端都要接** |
| 13 | `@tool` 是 StructuredTool 不能直接调用，异常被兜底吞掉→按需检索从未生效 | 拆 `_search_documents_impl` 普通函数 + 薄 `@tool` 包装 |
| 14 | checkpoint 反序列化未注册 `models.StyleProfile`，未来版本会被阻止 | 升级 langgraph 或换 PostgresSaver 时注册 `allowed_msgpack_modules` |
| 15 | Streamlit widget key 与 session_state key 同名，实例化后不能赋值 | widget key 与逻辑键分离 |
| 16 | fpdf2 `output()` 返回 `bytearray`，`download_button` 只收 `bytes` | `bytes(pdf.output())` + 测试加 `isinstance` 断言 |
| 17 | 审查时对"自认为熟悉"的文件跳过细读，漏掉 5 处不同步 | 每个文件当陌生代码重审；文案/编号类问题只能读原文发现 |
| 18 | 组合 `stream_mode` 输出是 `(mode, data)`，按单模式解包→零 token 事件且静默降级 | 按 `(mode, data)` 解包；流式测试必须断言"token 事件数 > 0" |
| 19 | 流式聚合后 usage 在 `usage_metadata` 属性，不在 `response_metadata` | 提取函数双源兜底 |
| 20 | pyright 96 条：`object` 标注 + 第三方 stub 缺陷 + 6 条真问题 | 先分四类再动手；机械替换与真修复分开 |
| 21 | 三个 P0：跨进程 thread_id 丢失 / 按需检索丢姓名学历 / BM25 零分挤占名额 | 方案 B + 保留头部区 + `score > 0`；补 regression/ui 测试 |
| 22 | 给单例加锁反而死锁（同一把不可重入锁嵌套获取） | 锁外取依赖；用守护线程 + 超时断言把死锁锁进测试 |
| 23 | Token 预算 15000 低于真实用量，警戒线永远事后才触发 | 按实测校准为 30000，并在注释写依据 |
| 24 | 抽取带类型标注的函数 = 悄悄收紧类型契约（测试全绿，pyright 报 14 条） | 类型检查纳入改完的验证动作 |
| 25 | `queue.Queue.put` → `asyncio.Queue.put` 写法相同但语义不同（后者不 await 等于不执行） | 换库/换类型时逐个调用点复查，不能全局替换了事 |
| 26 | SSE 轮询等待占用进程级共享线程池，20 条连接能卡住"生成简历" | 改 `queue.Queue` + `get_nowait`（不碰事件循环）；加心跳 |
| 27 | 错误分类有两份且已分叉（缺认证分支）、`encoding` 分支因 `codec≠encode` 几乎死掉 | 抽纯函数 + 唯一分类表 + 守卫断言 |
| 28 | SDK 默认 read=600s × 重试 2 次，交互式服务等于没超时 | `timeout=120` + `max_retries=1`；默认值要主动查 |

> 完整故事（现象→诊断→修复→收益）在 `C:\Users\李晓杰\.claude\projects\C--Users----\memory\project-troubleshooting-log.md`。

---

## 9. 待办与路线图

### 近期（下一步）
1. **第 14 课 PostgreSQL**：四张表 + ER 图 + 索引 + 慢查询 → `PostgresSaver` → 面试多轮上下文持久化。
2. **候选改造（2026-08-14 提出，开工时定夺）**：检索服务依赖注入——现在 `api.py` 每次 generate 重建索引并**替换全局单实例**，多用户并发会互相覆盖（A 的检索结果张冠李戴）。方案：`Depends` per-request 注入独立 `RetrievalService`；同时把 `load_and_index_documents` 包进 `asyncio.to_thread`（embedding 走 CPU 数秒，未包会冻住事件循环）。**第 16 课部署前必须解决**，本地单人阶段不疼。
3. 模型微调（LoRA / QLoRA）嵌入第 14 课：unsloth/peft + bitsandbytes，Qwen-0.5B 级别，50 条数据；要能画图讲 LoRA 原理与 QLoRA 省显存。
4. **部署前必须处理**：§8 #26 线程池耦合、#14 msgpack 注册、无认证、`.env.example`。
5. **第 13 课可选补课（不阻塞第 14 课）**：2026-10-11 的结论是第 13 课新知识分布在 4 处——`api.py`（已通读）、`api_test.py`（TestClient 测端点 / SSE / 404 / 422）、`app.py` 客户端段（`_parse_sse_lines` / `_consume_customize_sse` / `_poll_stream_result` / multipart 上传 / 本地回退）、依赖与日志。后两处**未确认过完**，想更稳可以补。
6. **eval_data 素材积累**：目标 30–50 条，目前 0 条。RAGAS 阶段要用，平时每跑通一次就存一条，别等回头补。

### 后续课程
- **第 15 课 Docker**：Dockerfile（slim + 非 root + 缓存清理）+ compose（FastAPI + PostgreSQL + Streamlit）。
- **第 16 课 部署**：HF Spaces / Render / Railway 择一，数据库可配 Supabase；准备系统设计问答（"1000 人同时用"、"LLM 慢怎么优化"），讲概念不动手。
- **阶段 2.5 AI 专项**：Reranker（BAAI/bge-reranker-base，粗排 20 → 精排 4，**要接进按需检索路径**）、RAGAS 评测（ContextRecall / Faithfulness / AnswerCorrectness，**覆盖按需检索**）、Token 三层降级、内容一致性校验（技能差集 + 事实对比 + 差异高亮，差异数据复用给 RAGAS）、模拟面试多 Agent。
- **阶段 3 模拟面试**：Supervisor + 3 面试官（Tech/项目/HR），MVP 先做单 Agent 三模式；语音 = faster-whisper（STT）+ edge-tts（TTS）。
- **阶段 4 算法**：LeetCode Hot 100，每天 1 题 medium，三遍法（硬憋 15-40 分钟 → 看题解 → 合上默写 AC → 隔天重做 → 一周后口述），2026.12 前两遍。

### 已砍（别再提）
Query 改写（Agent 本身就是改写器）· 图片 OCR（场景概率极低）· 联网搜索 JD（反爬 + 时效差）· 结构化文档解析（PyPDFLoader 够用）· HITL 分级审批（现有断点已够）· K8s 实操（改到第 16 课讲概念）。

### 长期低优先级
PDF/Word 排版细节优化（顶栏/间距/字体，功能已闭环）· mypy（部署前统一加标注时）· Makefile（第 15 课）· structlog / OpenTelemetry（大概率不需要）。

---

## 10. 量化数据（简历与面试用）

### 节点耗时
| 节点 | 2026-09-12 实测 | 2026-10-10（deepseek-flash） |
|---|---|---|
| extract_style | 3.6s | 2.9s |
| extract_jd | 3.3s | 2.4–3.5s |
| parse_user | 2.0s | 1.5–1.6s |
| generate_base | 5.7s | 3.6–15.1s（旧模型最慢 57.0s） |
| customize | 15.2s | 10.3–11.3s（旧模型最慢 55.0s） |

### Token
- 单次定制：**10331 / 10551**（占 30000 预算的 34–35%）。
- 一轮（未调检索）10140；两轮（含检索轮）20533。
- SSE 一次流式定制：13 个 `messages/partial` 事件 / 1263 字符 / prompt 6577 + completion 6729。
- 同步定制（同 thread 二次）：prompt 8123 + completion 3221。

### 检索质量
构造 FAISS 语义 top-4 应为 `[20,21,22,23]` 的场景：修复前 `[20,0,21,1]`（一半名额被零分块占用），修复后严格 `[20,21,22,23]`。

### 测试
`terminal_test` 14/14 · `api_test` 20/20 · `regression_test` 63 断言（零 LLM，秒级）· `ui_test` 19 断言（AppTest）。

---

## 11. 运行环境与已知风险

- **Windows 11 + PowerShell**。含中文的 `.ps1` 必须存成 **UTF-8 带 BOM**，否则 PowerShell 5.1 按 GBK 读会语法错误、脚本根本不执行（已真实发生过）。
- **DSH 文件沙箱**：只有"会话工作区"内的文件可写。项目必须作为工作区打开（`C:\Users\李晓杰\ai-job-assistant`），否则只能读不能改。
- 本文件被 DSH 自动加载；`CLAUDE.md` 也兼容但不需要重复维护。

---

## 12. 侧线项目（与主项目无关，被问到时才展开）

| 项目 | 状态 | 关键路径 |
|---|---|---|
| **综述论文**《煤炭产品碳足迹中甲烷量化与不确定性》 | 写作计划书 **V4.4** 已完成（2026-09-23 两轮补丁），建议冻结协议开工 | Word：桌面 `甲烷量化与不确定性综述_写作计划书V4.4.docx`；生成脚本 `C:\Users\李晓杰\make_plan_docx_v44.py`（改内容改脚本重跑）。目标期刊 IJCST（主投）/ JCLP（冲刺）/ RCR |
| **教材术语审校**《智慧矿山导论》 | 12–14 章已完成（26 条建议 + 9 条核对） | 工具在 `C:\Users\李晓杰\`（`term_audit.py` 等）；页码 = PDF 页 − 10 |

综述论文的关键方法约束（避免重复讨论）：P1/P2/P3 可比性分级、Study/Dataset/Case 三层独立性（当前 N_study=2、N_case=19，不得当独立研究合成）、ΔCF_CH₄ ≠ ΔPCF_total、双证据流（A 甲烷量化 / B PCF-LCA / C 接口）、双 Gate（第 5 周末、第 8 周）。

---

## 13. 原始记忆索引

目录：`C:\Users\李晓杰\.claude\projects\C--Users----\memory\`（`MEMORY.md` 为索引，共 24 个文件；下表按主题归并，未逐条列全）

| 文件 | 内容 |
|---|---|
| `user-ai-beginner.md` | 学历、目标、项目定位 |
| `career-roadmap.md` | **唯一计划文件**：完整技术路线 + 时间线 + 已砍清单 |
| `langchain-learning-progress.md` | 逐课进度与每次改造的细节 |
| `langchain-12-lesson-plan.md` | 三层检验方法论 |
| `teaching-style-preference.md` | 讲课方式偏好 |
| `pre-coding-checklist.md` | 写代码前清单（本文件 §6 的来源） |
| `always-add-error-handling.md` / `prevent-small-defects.md` / `quality-over-speed.md` / `lesson-code-generation-lesson.md` | 质量准则与教训 |
| `project-troubleshooting-log.md` | **28 条完整踩坑故事**（43KB，面试素材） |
| `project-metrics-tracking.md` | 量化指标与实测数据 |
| `unplanned-features.md` | 计划外交付清单（面试"你做过哪些超出需求的优化"） |
| `logging-system-upgrade.md` / `customize-retry-logic.md` / `terminal-test-convention.md` | 单项机制设计 |
| `resume-builder-upgrade-plan.md` | 历史技术细节（已并入 career-roadmap） |
| `ragas-eval-checklist.md` | 检索/生成改动后要补的评测项 |
| `date-evidence-rule.md` | 日期标注规则 |
| `claude-code-effort-config.md` | 仅 Claude Code 相关，迁移后可忽略 |
| `review-paper-methane-pcf.md` / `textbook-terminology-review.md` | 侧线项目 |
