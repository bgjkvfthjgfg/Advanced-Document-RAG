# Advanced Document RAG with Reranking and Evidence-Grounded Evaluation

## 🔥 项目亮点（Highlights）

- 📄 基于页级（page-level）的文档解析，实现细粒度证据定位  
- 🔍 构建 Retrieval + Rerank Pipeline，显著提升检索质量  
- 🧠 引入 Evidence-Grounded Generation，有效降低 hallucination  
- 📊 基于 QASPER 构建结构化 benchmark，并进行人工评测  
- ⚙️ 实现从数据处理到评测分析的完整 RAG 工程闭环  

在 benchmark 实验中，系统准确率从 0.56 提升至 0.90（rerank + 大模型），验证了方法有效性

## 1. 项目概述（Project Overview）

本项目实现了一个面向复杂技术文档（如论文 PDF）的检索增强问答系统（Retrieval-Augmented Generation, RAG），重点在于构建一个**完整、可评测、可复现的工程化 RAG Pipeline**，而非简单的问答演示系统。

在技术文档问答场景中，传统 RAG 系统通常面临以下问题：

- 检索粒度过粗，难以定位精确证据（evidence）
- 检索结果排序不合理，关键信息未被优先利用
- 大模型生成阶段容易产生 hallucination（幻觉）
- 缺乏系统化 benchmark，难以评估优化是否有效

针对上述问题，本项目从系统设计角度出发，构建了包含以下模块的完整流程：

- 基于页级（page-level）的文档解析与文本构建
- 基于 embedding 的文本检索（text-grounded retrieval）
- 引入 definition-aware 的重排序机制（reranking）
- 构建证据约束生成（evidence-grounded generation）流程
- 设计结构化 benchmark，对系统效果进行量化评估

在实验部分，基于 QASPER 数据集构建了一个包含 50 条样本的 benchmark，并对以下系统配置进行对比：

- baseline_3B（基础检索 + 小模型）
- rerank_3B（加入重排序）
- rerank_14B（重排序 + 大模型）

实验结果表明：

- 引入 rerank 后，检索质量与最终答案准确率均有显著提升
- 在 evidence-grounded 场景下，大模型表现更加稳定
- 系统整体 hallucination 明显下降

本项目不仅实现了一个完整的 Document RAG 系统，还构建了从数据处理到评测分析的完整闭环，具备较强的工程实践价值与扩展能力。

---

## 2. 核心功能与技术亮点（Key Features）

本项目围绕文档级 RAG 系统的工程化实现，设计并实现了多个关键模块。

### 2.1 页级文档解析（Page-level Document Parsing）

系统首先对原始 PDF 文档进行结构化处理：

- 将 PDF 转换为页级图片（page-level images）
- 构建页级 metadata（doc_id / page_id / text / image_path）
- 形成统一的 page-level 数据表示

相比整文或段落级处理方式，页级粒度更适合技术文档中局部信息的精确定位。

---

### 2.2 文本检索（Text-grounded Retrieval）

在检索阶段，系统基于 embedding 构建文本检索能力：

- 使用 sentence-transformers 生成语义向量
- 在 page-level corpus 上执行 top-k 检索
- 返回候选 evidence 用于后续处理

该模块提供语义级召回能力，并作为后续 rerank 与生成模块的输入基础。

---

### 2.3 重排序机制（Reranking）

针对初始检索结果排序不准确的问题，引入 rerank 模块进行优化。

重排序策略综合考虑：

- 语义相似度（cosine similarity）
- 关键词匹配（keyword overlap）
- 定义类问题增强（definition-aware bonus）

其中，definition-aware 机制针对 “What is …” 等问题类型进行优化，使包含定义句的文本更容易被提升至高排名位置。

该模块能够有效减少“检索到了但未被利用”的问题。

---

### 2.4 证据约束生成（Evidence-Grounded Generation）

在生成阶段，系统将检索得到的 evidence 显式输入语言模型：

- 将 top evidence 拼接为上下文
- 构建包含问题与证据的 prompt
- 约束模型基于 evidence 进行回答

同时支持不同模型规模的对比（3B / 14B），用于分析模型能力对结果的影响。

该机制显著降低了 hallucination，并提升了答案的可解释性。

---

### 2.5 Benchmark 评测体系（Evaluation Pipeline）

本项目构建了完整的 benchmark 评测流程，而非仅展示 demo。

评测设计包括：

- 基于 QASPER 构建 50 条测试子集
- 对比三种系统配置：
  - baseline_3B
  - rerank_3B
  - rerank_14B
- 引入人工标注指标：
  - answer_correct（答案正确性）
  - grounding_valid（证据一致性）
  - error_type（错误类型）

该评测体系使系统优化具备可量化依据。

---

### 2.6 工程闭环设计（End-to-End Pipeline）

系统整体流程如下：

PDF → Page Parsing → Retrieval → Rerank → Generation → Benchmark → Analysis

该设计形成从数据处理到评测分析的完整闭环，使系统具备：

- 可复现性（完整数据与流程可重现）
- 可评测性（benchmark 支持定量分析）
- 可扩展性（支持向更复杂 RAG 系统演进）

为后续扩展至更复杂 RAG 系统提供基础。

## 3. 项目结构（Project Structure）

本项目采用分层组织结构，将数据处理、模型推理与评测流程解耦，以保证系统的可复现性与可扩展性。

整体目录结构如下：

DeepSeek-VL2-Multimodal-RAG/
├── assets/                         # README展示与结果截图
├── configs/                        # 配置文件
├── data/
│   ├── raw_pdfs/                   # 原始 PDF 文档
│   ├── page_images/                # 页级图片
│   ├── metadata/                   # 页级 metadata 与文本数据
│   ├── index/                      # 文本检索索引
│   └── benchmark/                  # benchmark 数据（QASPER 子集等）
├── notebooks/                      # 实验分析与调试
├── results/                        # benchmark 输出与评测结果
├── scripts/                        # 各阶段运行脚本
├── src/                            # 核心模块实现
├── requirements.txt
└── README.md

---

### 3.1 数据层（data/）

数据层用于管理从原始文档到检索语料的完整中间过程：

- raw_pdfs：原始技术文档（PDF）
- page_images：PDF 转换后的页级图片
- metadata：页级结构化数据（doc_id / page_id / text 等）
- index：embedding 检索索引
- benchmark：评测所使用的数据集

该分层设计保证数据处理过程清晰且可复现。

---

### 3.2 脚本层（scripts/）

scripts 目录提供完整 pipeline 的执行入口，每个阶段均对应独立脚本：

- pdf_to_images.py：PDF 转页级图片
- build_metadata.py：构建页级 metadata
- build_page_texts.py：提取页级文本
- build_text_index.py：构建检索索引
- query_text_index.py：检索测试
- run_text_grounded_rag.py：主流程运行
- run_qasper_text_grounded.py：benchmark 推理
- test_qasper_retrieval.py：检索评测
- summarize_qasper_retrieval.py：结果统计

该设计降低了使用门槛，并便于逐阶段调试。

---

### 3.3 结果层（results/）

results 目录用于保存所有实验输出与评测结果：

- 检索 benchmark 输出
- generation benchmark 输出
- 人工标注结果（CSV）
- 最终统计指标（metrics）

例如：

- benchmark_results_qasper.csv
- final_metrics.txt

该目录是实验闭环的最终落点。

---

### 3.4 核心模块（src/）

src 目录用于存放核心逻辑实现，包括：

- 检索模块（retrieval）
- 重排序模块（rerank）
- 生成相关逻辑（generation）
- 数据处理工具函数

与 scripts 层相比，src 更偏底层实现与模块复用。

---

### 3.5 结构设计原则

本项目在结构设计上遵循以下原则：

1. 数据与代码分离  
   提高可复现性与可维护性

2. Pipeline 解耦  
   各阶段独立，可单独测试

3. 结果可追踪  
   所有实验输出均落地保存

4. 可扩展性  
   支持后续扩展至多模态或更复杂 RAG 系统

---

## 4. 系统流程（Pipeline）

本项目采用分阶段的 RAG Pipeline 设计，将整个系统拆分为五个主要阶段：

Query → Retrieval → Rerank → Generation → Evaluation

---

### 4.1 文档处理阶段（Document Processing）

该阶段负责将原始 PDF 转换为结构化数据。

主要流程：

1. 读取原始 PDF 文档  
2. 转换为页级图片（page-level images）  
3. 构建页级 metadata  
4. 提取页级文本（page-level text）  

输出数据包括：

- doc_id（文档标识）
- page_id（页号）
- text（文本内容）
- image_path（可选）

该阶段为后续检索提供基础语料。

---

### 4.2 检索阶段（Retrieval）

在检索阶段，系统根据输入问题召回相关 evidence。

主要步骤：

1. 对问题进行 embedding 编码  
2. 在 page-level corpus 上执行向量检索  
3. 返回 top-k 候选文本片段  

特点：

- 基于语义相似度
- 支持跨段落信息检索
- 提供初始 evidence 集合

---

### 4.3 重排序阶段（Rerank）

为提升 evidence 质量，对候选结果进行重排序。

重排序依据包括：

- 语义相似度（cosine similarity）
- 关键词匹配（keyword overlap）
- 定义类问题增强（definition-aware bonus）

该阶段的目标是：

- 提升关键证据排序
- 降低无关信息干扰

---

### 4.4 生成阶段（Generation）

在生成阶段，系统基于 evidence 生成最终答案。

流程：

1. 将 top evidence 拼接为上下文  
2. 构建 prompt（问题 + evidence）  
3. 输入语言模型生成答案  

实验中对比了不同配置：

- baseline_3B
- rerank_3B
- rerank_14B

重点评估：

- 答案正确性
- 是否基于 evidence（grounding）

---

### 4.5 评测阶段（Evaluation）

评测阶段用于量化系统性能。

方法：

- 基于 QASPER 子集进行测试
- 对输出进行人工标注

标注维度包括：

- answer_correct（答案正确性）
- grounding_valid（证据一致性）
- error_type（错误类型）

---

### 4.6 Pipeline 总结

该系统构建了一个完整的 RAG 闭环：

- 从文档解析开始
- 经检索与重排序优化 evidence
- 最终生成 grounded answer
- 并通过 benchmark 进行定量评估

相比简单问答系统，该 pipeline 具备：

- 可解释性（retrieval + rerank 可分析）
- 可评测性（完整 benchmark）
- 可扩展性（支持后续系统升级）

## 5. 数据准备（Data Preparation）

本项目的数据准备流程主要包括两部分：

1. 原始 PDF 文档处理（用于构建检索语料）
2. QASPER benchmark 构建（用于评测系统效果）

---

### 5.1 原始 PDF 文档处理

系统以技术论文 PDF 作为主要数据来源，通过多阶段处理构建 page-level 检索语料。

#### （1）PDF 转页级图片

将 PDF 转换为页级图片，用于：

- 文档可视化检查
- 后续多模态扩展预留接口

执行：

    python scripts/pdf_to_images.py

输出：

- `data/page_images/` 下的页级图片文件

---

#### （2）构建页级 Metadata

为每一页构建结构化信息，包括：

- doc_id（文档标识）
- page_id（页号）
- image_path（图片路径）
- pdf_path（原始路径）

执行：

    python scripts/build_metadata.py

输出：

- `data/metadata/metadata.jsonl`

---

#### （3）提取页级文本

从 PDF 或 OCR 结果中提取每一页文本，形成 page-level corpus。

执行：

    python scripts/build_page_texts.py

输出：

- 包含 text 字段的 metadata 数据

---

#### （4）构建检索索引

基于 page-level 文本构建 embedding 索引，用于语义检索。

执行：

    python scripts/build_text_index.py

输出：

- `data/index/` 中的向量索引

---

### 5.2 QASPER Benchmark 构建

为实现系统评测，基于 QASPER 数据集构建 benchmark。

---

#### （1）构建子集（50 条）

从原始数据中抽取样本，构建小规模测试集。

执行：

    python scripts/build_qasper_subset.py

输出：

- `data/benchmark/qasper_subset.jsonl`

---

#### （2）构建 Benchmark Corpus

从 QASPER 文档中提取文本，构建用于检索的语料库。

执行：

    python scripts/build_qasper_corpus.py

输出：

- `data/benchmark/qasper_corpus.jsonl`

---

#### （3）数据规模说明

当前 benchmark 设置：

- 样本数：50 条
- 每条包含：
  - question（问题）
  - gold answer（参考答案）
  - 文档内容（用于检索）

该规模适用于：

- 工程验证（engineering validation）
- 模型对比分析（model comparison）
- 错误分析（error analysis）

---

### 5.3 数据设计说明

本项目在数据设计上遵循以下原则：

1. 小规模高质量  
   优先保证标注质量，而非追求数据规模

2. 页级检索粒度  
   提升技术文档中局部信息的定位能力

3. text-grounded 优先  
   当前聚焦文本证据，为后续多模态扩展打基础

4. 可复现性  
   所有中间数据均保存，支持完整复现

---

## 6. 运行方式（How to Run）

本项目支持按阶段运行，也支持执行完整 pipeline。

---

### 6.1 环境准备

建议使用：

- Python 3.10+
- GPU 环境（推荐）

安装依赖：

    pip install -r requirements.txt

---

### 6.2 数据处理流程

按顺序执行：

    python scripts/pdf_to_images.py
    python scripts/build_metadata.py
    python scripts/build_page_texts.py
    python scripts/build_text_index.py

完成后将得到：

- page-level corpus
- embedding 检索索引

---

### 6.3 检索测试

用于验证检索效果：

    python scripts/query_text_index.py

输出：

- 输入问题
- top-k 检索结果

---

### 6.4 运行 RAG 主流程

执行完整 pipeline：

    python scripts/run_text_grounded_rag.py --question "Your question here"

输出包括：

- 初始检索结果
- rerank 后 evidence
- 最终生成答案

---

### 6.5 运行 Benchmark 推理

执行 benchmark：

    python scripts/run_qasper_text_grounded.py

功能：

- 自动运行 retrieval + rerank + generation
- 输出每条样本预测结果

---

### 6.6 检索 Benchmark

评测 rerank 对检索效果的影响：

    python scripts/test_qasper_retrieval.py
    python scripts/summarize_qasper_retrieval.py

输出：

- hit@k
- 平均命中排名
- improved count

---

### 6.7 Generation Benchmark（人工评测）

评测文件：

    results/benchmark_results_qasper.csv

标注字段：

- answer_correct
- grounding_valid
- error_type

---

### 6.8 指标统计

执行：

    python - << 'PY'
    import pandas as pd

    df = pd.read_csv('results/benchmark_results_qasper.csv')

    print("Accuracy:")
    print(df.groupby('system')['answer_correct'].mean())

    print("\nGrounding:")
    print(df.groupby('system')['grounding_valid'].mean())

    print("\nError Breakdown:")
    print(df.groupby(['system','error_type']).size().unstack(fill_value=0))
    PY

---

### 6.9 输出结果

最终结果包括：

- accuracy（准确率）
- grounding rate（证据一致性）
- error distribution（错误分布）

建议保存至：

    results/final_metrics.txt

用于结果复现与记录

## 7. Benchmark 设计（Evaluation Design）

为避免系统停留在功能演示层面，本项目构建了结构化 benchmark，用于对 RAG 系统的各个模块进行定量评估与分析。

---

### 7.1 评测目标

benchmark 主要围绕以下问题展开：

1. rerank 是否提升检索质量  
2. 检索优化是否带来最终答案质量提升  
3. 模型规模对结果的影响  
4. 系统错误主要来源于哪些模块  

---

### 7.2 数据集选择

benchmark 基于 QASPER（Question Answering on Scientific Papers）数据集构建。

选择该数据集的原因包括：

- 面向论文场景，符合技术文档 RAG 任务
- 问题类型以定义、解释、分析为主
- 需要跨段落理解与证据整合

---

### 7.3 Benchmark 设置

构建小规模高质量测试集：

- 样本数量：50 条
- 每条样本包含：
  - question（问题）
  - gold answer（参考答案）
  - 对应文档内容

该规模适用于工程验证与误差分析。

---

### 7.4 对比系统配置

benchmark 中对比以下三种系统：

1. baseline_3B  
   - 原始 retrieval（无 rerank）  
   - 小模型（3B）

2. rerank_3B  
   - 引入 rerank  
   - 小模型（3B）

3. rerank_14B  
   - 引入 rerank  
   - 大模型（14B）

该设置可分别分析：

- rerank 的影响（baseline → rerank_3B）
- 模型规模的影响（rerank_3B → rerank_14B）

---

### 7.5 Retrieval Benchmark

用于评估检索阶段效果。

评测方法：

- 判断 gold evidence 是否出现在 top-k 检索结果中
- 对比 rerank 前后的变化

评测指标包括：

- hit@k
- 命中排名（hit rank）
- improved count（是否提升）

该部分用于验证 rerank 对 evidence 排序的优化效果。

---

### 7.6 Generation Benchmark

用于评估完整 RAG 系统性能。

采用人工标注方式进行评测。

标注维度包括：

1. answer_correct  
   - 是否正确回答问题

2. grounding_valid  
   - 是否基于 evidence 作答

3. error_type  
   - 错误类型（用于分析系统问题）

---

### 7.7 指标定义

基于标注结果计算以下指标：

1. Accuracy（准确率）

    正确回答数 / 总样本数

2. Grounding Rate（证据一致率）

    基于 evidence 的回答数 / 总样本数

3. Error Breakdown（错误分布）

    各类错误类型统计分布

---

### 7.8 Benchmark 设计总结

该 benchmark 具备以下特点：

- 分离检索与生成评测  
- 引入人工标注提高可靠性  
- 支持误差分析与模块定位  
- 结构清晰，易于扩展  

为系统优化提供了可量化依据。

---

## 8. 实验结果（Results）

本章节对 benchmark 实验结果进行汇总，并分析系统性能变化。

---

### 8.1 实验设置

对以下三种系统进行对比：

- baseline_3B
- rerank_3B
- rerank_14B

测试数据为 50 条 QASPER 样本，所有结果均经过人工标注。

---

### 8.2 Accuracy（答案正确率）

- baseline_3B：0.56
- rerank_3B：0.70
- rerank_14B：0.90

分析：

- 引入 rerank 后，准确率提升约 14%
- 在 rerank 基础上使用更大模型，准确率进一步提升至 0.90

---

### 8.3 Grounding Rate（证据一致率）

- baseline_3B：0.84
- rerank_3B：0.90
- rerank_14B：1.00

分析：

- rerank 提供更可靠 evidence
- 大模型在证据约束生成中表现更稳定
- rerank_14B 基本消除了 hallucination

---

### 8.4 错误类型分析（Error Breakdown）

baseline_3B：

- hallucination 较多
- partial_answer 较多
- 检索质量影响明显

rerank_3B：

- hallucination 数量下降
- 错误更多表现为 partial_answer

rerank_14B：

- 正确率显著提升
- 错误数量明显减少
- 几乎无 hallucination

---

### 8.5 检索与生成的协同作用

实验结果表明系统性能提升来自两个方面：

1. 检索优化（Rerank）  
   提升关键 evidence 排序

2. 模型能力提升（Model Scaling）  
   提升理解与生成能力

整体性能变化呈现：

baseline_3B → rerank_3B → rerank_14B

逐步提升趋势。

---

### 8.6 结果总结

实验结果验证：

- rerank 对 RAG 系统具有显著提升作用  
- evidence-grounded 机制有效减少 hallucination  
- 模型规模对复杂语义任务仍有重要影响  

---

### 8.7 可视化结果

项目中保存关键评测结果截图：

- 检索 benchmark：  
  assets/28_qasper_retrieval_benchmark_summary.png

- 生成 benchmark：  
  assets/29_qasper_generation_benchmark_summary.png

用于结果展示与分析说明。

## 9. 分析与讨论（Discussion）

在完成 benchmark 实验后，可以从系统结构与实验结果两个角度对本项目进行分析。

---

### 9.1 系统优势

#### （1）检索过程具备可解释性

相比纯 embedding 检索，本项目引入 rerank 模块：

- 结合语义相似度与关键词信息
- 针对定义类问题进行专门优化

使得检索结果不仅更准确，而且具备一定可解释性。

---

#### （2）生成过程受证据约束

通过将检索得到的 evidence 显式输入模型：

- 限制模型生成空间
- 提高答案与原文的一致性（grounding）

实验中 grounding rate 的提升验证了该机制的有效性。

---

#### （3）评测体系完整

项目构建了从 retrieval 到 generation 的完整评测流程：

- 分离不同阶段进行分析
- 引入人工标注提高评测可靠性
- 通过 error_type 进行误差分析

使系统优化具备明确方向。

---

### 9.2 系统局限性

尽管当前系统已具备完整 pipeline，但仍存在一定局限：

#### （1）Benchmark 规模较小

- 当前仅包含 50 条样本
- 更适用于工程验证而非统计结论

---

#### （2）检索方法单一

- 当前仅使用 dense retrieval
- 未引入 BM25 或 hybrid retrieval

---

#### （3）Rerank 依赖规则设计

- 基于 similarity 与 heuristic bonus
- 尚未引入学习型 reranker

---

#### （4）多模态能力尚未充分利用

当前系统主要基于文本证据：

- 未对图表、结构信息进行深度建模
- 多模态能力仍有扩展空间

---

### 9.3 错误来源分析

结合 error_type，可将错误划分为：

1. Retrieval Issue  
   - 关键 evidence 未被召回或排序较低

2. Generation Failure  
   - 模型未正确利用 evidence

3. Partial Answer  
   - 回答不完整

4. Hallucination  
   - 输出与 evidence 不一致

实验结果表明：

- rerank 可显著减少 retrieval issue  
- 大模型可降低 hallucination 与 generation failure  

---

### 9.4 对 RAG 系统的启示

本项目得到以下经验：

1. 检索质量决定上限  
   evidence 不准确将直接影响生成结果  

2. rerank 是高性价比优化手段  
   在不更换模型的情况下即可带来明显提升  

3. 模型规模仍然重要  
   在复杂语义任务中，大模型优势明显  

4. benchmark 是必要组件  
   缺乏评测将难以判断优化效果  

---

### 9.5 项目定位

本项目定位为：

- 工程化 Document RAG 系统实现  
- 强调可复现与可评测  
- 为更复杂 RAG 系统提供基础  

---

## 10. 未来工作（Future Work）

在当前系统基础上，可以从多个方向进一步扩展。

---

### 10.1 检索能力优化

- 引入 BM25 检索
- 构建 hybrid retrieval（BM25 + embedding）
- 使用 query expansion 提升召回能力

---

### 10.2 学习型 Reranker

- 引入 cross-encoder 模型
- 使用监督数据训练 rerank 模型
- 替代当前规则驱动方法

---

### 10.3 更复杂的 RAG Pipeline

扩展为多阶段检索流程：

- multi-hop retrieval
- query decomposition
- 动态检索策略

逐步向更复杂 RAG 系统演进。

---

### 10.4 自动化评测

减少人工标注成本：

- 引入 LLM-as-a-judge
- 使用语义相似度指标
- 构建自动化 evaluation pipeline

---

### 10.5 多模态扩展

扩展到真正多模态场景：

- 利用图像与图表信息
- 支持图文联合检索
- 引入视觉模型

---

### 10.6 Benchmark 扩展

- 增加样本规模
- 引入更多问题类型
- 构建标准数据划分

---

### 10.7 工程优化

- 推理性能优化
- 模型加载优化
- 模块解耦
- 构建简单 UI 或 API

---

### 10.8 总结

本项目提供了一个基础的 Document RAG 实现框架：

- 向上可扩展为更复杂 RAG 系统  
- 向外可扩展为多模态 RAG  
- 向下可优化为高性能工程系统  

为后续研究与工程实践提供基础。