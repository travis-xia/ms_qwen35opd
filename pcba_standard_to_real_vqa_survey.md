# 跨域工业视觉问答 / PCBA 缺陷理解 / 标准文档到真实工厂图像迁移调研

调研日期：2026-06-29  
资料范围：公开论文、arXiv、官方 benchmark 页面、官方 GitHub、项目主页。  
任务边界：只做公开资料调研和方法归纳；不假设任何本地实现方案；不提出具体参赛方案。

## 0. 结论摘要

PCBA Standard-to-Real VQA 不是单纯的工业异常检测，也不是通用 VQA。它同时要求模型处理：

- standard-to-real domain shift：标准文档、标准图示、真实工厂图像之间的跨域对齐。
- 工业知识 grounding：元件、焊接、装配、缺陷类型、原因、处置建议必须受规则或标准约束。
- 小目标密集视觉：PCBA 元件密集、缺陷细微、局部证据分散，通用 VLM 容易漏检和误数。
- VQA 输出：不仅要判断是否异常，还要回答类别、数量、属性、位置、原因和处理建议。

现有工作可以覆盖部分能力，但端到端覆盖“标准文档/标准图示 -> 真实 PCBA 图像 -> 可解释 VQA”的公开方案仍很少。最接近的公开方向是 `MMAD`、`MANTA`、`UniPCB / PCB-GPT`、`AD-Copilot`、`Triad`、`LogicQA`、`Qwen2.5-VL` 这几条线的组合。

## 1. 资料来源和访问日期

所有网页访问日期均为 2026-06-29。

优先来源：

- MVTec AD 官方页面：[https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- MVTec LOCO AD 官方页面：[https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- MVTec AD 2 官方页面：[https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- VisA / SPot-the-Difference 官方 GitHub：[https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- Real-IAD 官方项目页：[https://realiad4ad.github.io/Real-IAD/](https://realiad4ad.github.io/Real-IAD/)
- MMAD arXiv：[https://arxiv.org/abs/2410.09453](https://arxiv.org/abs/2410.09453)
- MMAD 官方 GitHub：[https://github.com/jam-cc/MMAD](https://github.com/jam-cc/MMAD)
- MANTA arXiv：[https://arxiv.org/abs/2412.04867](https://arxiv.org/abs/2412.04867)
- MANTA 官方项目页：[https://grainnet.github.io/MANTA](https://grainnet.github.io/MANTA)
- UniPCB arXiv：[https://arxiv.org/abs/2601.19222](https://arxiv.org/abs/2601.19222)
- AnomalyGPT arXiv：[https://arxiv.org/abs/2308.15366](https://arxiv.org/abs/2308.15366)
- AnomalyGPT GitHub：[https://github.com/CASIA-IVA-Lab/AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
- Myriad arXiv：[https://arxiv.org/abs/2310.19070](https://arxiv.org/abs/2310.19070)
- WinCLIP arXiv：[https://arxiv.org/abs/2303.14814](https://arxiv.org/abs/2303.14814)
- AnomalyCLIP arXiv：[https://arxiv.org/abs/2310.18961](https://arxiv.org/abs/2310.18961)
- AdaCLIP arXiv：[https://arxiv.org/abs/2407.15795](https://arxiv.org/abs/2407.15795)
- EfficientAD arXiv：[https://arxiv.org/abs/2303.14535](https://arxiv.org/abs/2303.14535)
- SimpleNet arXiv：[https://arxiv.org/abs/2303.15140](https://arxiv.org/abs/2303.15140)
- Reverse Distillation arXiv：[https://arxiv.org/abs/2201.10703](https://arxiv.org/abs/2201.10703)
- Triad arXiv：[https://arxiv.org/abs/2503.13184](https://arxiv.org/abs/2503.13184)
- EIAD arXiv：[https://arxiv.org/abs/2503.14162](https://arxiv.org/abs/2503.14162)
- OmniAD arXiv：[https://arxiv.org/abs/2505.22039](https://arxiv.org/abs/2505.22039)
- IAD-GPT arXiv：[https://arxiv.org/abs/2510.16036](https://arxiv.org/abs/2510.16036)
- IAD-R1 arXiv：[https://arxiv.org/abs/2508.09178](https://arxiv.org/abs/2508.09178)
- AD-Copilot arXiv：[https://arxiv.org/abs/2603.13779](https://arxiv.org/abs/2603.13779)
- LogicQA arXiv：[https://arxiv.org/abs/2503.20252](https://arxiv.org/abs/2503.20252)
- FabGPT arXiv：[https://arxiv.org/abs/2407.10810](https://arxiv.org/abs/2407.10810)
- WaferSAGE arXiv：[https://arxiv.org/abs/2604.27629](https://arxiv.org/abs/2604.27629)
- Qwen2-VL arXiv：[https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
- Qwen2.5-VL arXiv：[https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
- LLaVA arXiv：[https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- LLaVA-OneVision arXiv：[https://arxiv.org/abs/2408.03326](https://arxiv.org/abs/2408.03326)
- PairTally arXiv：[https://arxiv.org/abs/2509.13939](https://arxiv.org/abs/2509.13939)
- UltraVR arXiv：[https://arxiv.org/abs/2606.05576](https://arxiv.org/abs/2606.05576)
- ManuRAG arXiv：[https://arxiv.org/abs/2601.15434](https://arxiv.org/abs/2601.15434)
- Fine-Grained RAG for VQA / KU-RAG arXiv：[https://arxiv.org/abs/2502.20964](https://arxiv.org/abs/2502.20964)

注：未检索到公开网页中可稳定访问的 `ASUS-NTU PCBA Standard-to-Real Grand Challenge` 官方论文或挑战主页；因此本报告只把它作为问题背景，不引用本地 proposal 内容。

## 2. Benchmark 和数据集脉络

### 2.1 MVTec AD

任务：unsupervised industrial anomaly detection / localization。  
数据：15 类对象和纹理，5000+ 高分辨率图像；训练集为正常样本，测试集包含正常和异常；提供 pixel-precise anomaly annotations。  
价值：工业异常检测标准基准，适合比较 anomaly score 和 anomaly map。  
局限：不是 VQA；没有标准文档、元件语义、缺陷原因和处置建议。

对 PCBA VQA 的意义：可作为传统 IAD baseline 的引用，但与 PCBA 小目标、元件级 grounding、标准迁移仍有明显距离。

### 2.2 MVTec LOCO AD

任务：logical constraints anomaly detection。  
数据：5 类工业场景，3644 张图；包含 structural anomalies 和 logical anomalies。  
价值：引入位置非法、必要对象缺失、逻辑约束违反等异常。  
局限：类别少；仍是 anomaly localization，不是开放式 VQA。

对 PCBA VQA 的意义：很重要。PCBA 中“元件缺失、元件位置错误、数量错误、装配规则违反”更像 logical anomaly，而不只是划痕或污染。

### 2.3 VisA

任务：visual anomaly detection。  
数据：12 个子集、10821 张图，包含 9621 正常样本和 1200 异常样本；其中 4 个 PCB 子集，包含 transistor、capacitor、chip 等复杂结构。  
价值：比 MVTec AD 更接近 PCB 的复杂结构和多实例场景。  
局限：仍是 anomaly detection / segmentation 数据，不是 VQA；PCB 子集是 bare PCB 风格，不等同于复杂 PCBA。

对 PCBA VQA 的意义：PCB/PCBA 方向必须引用的公开数据源之一。

### 2.4 Real-IAD

任务：real-world multi-view industrial anomaly detection。  
数据：来自真实生产线；论文摘要称 150K 高分辨率图像、30 类对象；官方项目页提供 1024 版本、raw high-resolution 版本和 JSON split。  
价值：强调 real-world、multi-view、真实生产线分布。  
局限：没有标准文档到真实图像的 VQA 任务。

对 PCBA VQA 的意义：domain shift、multi-view、真实工厂数据设定很相关。

### 2.5 MVTec AD 2

任务：advanced unsupervised industrial anomaly detection。  
数据：8 个新 anomaly detection scenarios，8000+ 高分辨率图像；测试集包含公开和非公开两部分，非公开部分通过 evaluation server 评估。  
价值：更接近真实工业难点，包括光照变化、复杂透明/反光物体、小缺陷等。  
局限：仍不覆盖 VQA、标准规则、元件语义。

对 PCBA VQA 的意义：可用于论证传统 IAD 逐渐从干净 benchmark 走向真实生产分布，但仍缺少 language/standard grounding。

### 2.6 MMAD

任务：MLLM for industrial anomaly detection benchmark。  
数据：8366 张工业图像、39672 个问题、38 个产品类别、244 个缺陷类别；来源包括 MVTec AD、MVTec LOCO AD、VisA、GoodsAD。  
任务：7 个子任务，包括 anomaly discrimination、defect classification、defect localization、defect description、defect analysis、object classification、object analysis。  
方法：GPT-4V 辅助生成 semantic annotations 和 multiple-choice QA；人工过滤；评测多个 MLLM。  
重要结论：GPT-4o 平均 74.9%，仍低于工业要求；模型对 object-related questions 明显好于 defect-related questions；normal template image 的帮助有限；RAG 和 expert agent 能提升但不解决根本问题。

对 PCBA VQA 的意义：极高。MMAD 是工业 MLLM/VQA 相关工作里必须引用的 benchmark。

### 2.7 MANTA

任务：visual-text anomaly detection for tiny objects。  
数据：137.3K+ multi-view images，38 类，5 个典型域；8.6K anomalous images 有 pixel-level annotations；每个对象 5 个视角。  
文本：Declarative Knowledge 覆盖 what/why/how；Constructivist Learning 含 2K image-text multiple-choice questions 和答案解释。  
价值：小目标、多视角、视觉-文本、原因和视觉特征解释。  
局限：不是 PCBA 专项；标准文档 grounding 不强。

对 PCBA VQA 的意义：很高。PCBA 元件和缺陷高度符合 tiny object、多视角、多实例和 what/why/how 需求。

### 2.8 UniPCB / PCB-GPT

任务：open-ended PCB quality inspection benchmark。  
数据：6581 张 PCB/PCBA 图像、23359 个 bilingual QA pairs，覆盖 BPCB 和 PCBA，三个 annotation scenarios，14 种 inspection subtasks。  
核心挑战：dense patterns、defect co-occurrence、subtle cues、multi-modality、BPCB 到 PCBA 的分布变化。  
方法：统一 defect/component taxonomy；构建 domain knowledge entry，包括 definition、common causes、potential impact、standardized phrasing；用 Qwen2.5-VL-72B-Instruct 生成 QA，并进行结构校验和人工 review。PCB-GPT 基于 Qwen2.5-VL-7B-Instruct，三阶段训练：concept alignment、instruction tuning、GRPO。  
价值：目前最接近 PCBA VQA 的公开 benchmark 和方法。  
局限：论文仍是 2026 arXiv；公开数据和模型可用性需以实际 release 为准；与“标准文档/图示到真实工厂图像”的 challenge 仍不完全等价。

对 PCBA VQA 的意义：最高，必须重点引用。

## 3. 方法 Taxonomy

### 3.1 传统检测/分割式工业缺陷方法

代表方法：

- PCB defect detection with YOLO / Faster R-CNN / Mask R-CNN / segmentation networks
- VR-YOLO: viewpoint robustness for PCB defect detection
- Scale-aware tile inference with topology-aware merging for high-resolution PCB defect detection
- DeepPCB、HRIPCB、PCB-Bank、PCB-Defect 等 PCB 数据源相关检测工作

核心思想：

把缺陷或元件作为显式检测对象，训练 detector / segmenter 输出 bbox、mask、class label。对于高分辨率 PCB，常见做法是 tile-based inference、overlap tiling、global NMS、boundary merging。

数据和监督信号：

- bbox / mask / class label
- PCB 缺陷类别，如 missing hole、rat bite、open circuit、short circuit、burr、spurious copper、soldering issue 等
- 有些方法使用 GAN 或合成缺陷扩增数据

模型结构或训练策略：

- YOLO 系列、Faster R-CNN、Mask R-CNN、U-Net/DeepLab、Transformer detector
- 高分辨率 PCB 常用 patch/tile 训练与推理，保证训练-推理尺度一致
- 小目标用 attention、multi-scale feature、tiling、topology-aware merging

优点：

- 定位、分类、计数直接，工程可控。
- 输出结构化，适合自动评估。
- 对已知缺陷类别效果好。

局限：

- 类别闭集，遇到新 defect type 或新元件封装会退化。
- 不会自然回答原因、影响、处理建议。
- 不理解标准文档和规则。
- 标注成本高，跨产品迁移弱。

对 PCBA Standard-to-Real VQA 的适配程度：

中等。适合作为视觉专家、局部证据提取器、计数器或候选区域生成器，但不能单独完成 open-ended VQA 和 standard grounding。

### 3.2 Anomaly Detection / Anomaly Localization

代表方法：

- PatchCore
- Reverse Distillation
- EfficientAD
- SimpleNet
- MVTec AD、MVTec LOCO AD、VisA、Real-IAD、MVTec AD 2 上的主流 IAD 方法

核心思想：

多数学派以 normal-only 或少量正常样本为训练条件，学习正常分布；测试时根据 reconstruction error、feature distance、teacher-student discrepancy、memory bank nearest neighbor 或 synthetic anomaly discriminator 得到 anomaly score 和 anomaly map。

数据和监督信号：

- 训练：正常图像为主。
- 评测：image-level label、pixel-level mask。
- 通常不需要异常样本训练，或只用 synthetic anomaly。

模型结构或训练策略：

- memory bank：PatchCore 类方法存储正常 patch feature。
- teacher-student：EfficientAD、Reverse Distillation。
- feature-space synthetic anomaly：SimpleNet。
- global + local：EfficientAD 用 local student-teacher 处理局部异常，用 autoencoder 捕捉 logical anomaly。

优点：

- 对 defect sample 稀缺的工业场景很适合。
- 异常定位成熟，推理速度可控。
- 不需要定义大量缺陷类别。

局限：

- 输出是 anomaly score/map，不是语义解释。
- 难以区分“元件 A 异常”还是“焊点 B 异常”。
- 对标准规则和知识 grounding 很弱。
- logical anomalies、数量错误、布局错误仍难。

对 PCBA Standard-to-Real VQA 的适配程度：

中等。适合提供 anomaly prior 和候选异常区域；需要与 VLM、元件检测、规则检索结合才能回答 VQA。

### 3.3 VLM Zero-shot / Prompting 方法

代表方法：

- WinCLIP
- APRIL-GAN
- SAA / SAA+
- AnomalyCLIP
- AdaCLIP
- GlocalCLIP
- AdaptCLIP 类方法

核心思想：

用 CLIP 或其他 vision-language pretrained model 的图文对齐能力，把 abnormal / normal prompt 与图像或 patch feature 对齐，进行 zero-shot 或 few-shot anomaly classification/segmentation。部分方法学习 object-agnostic anomaly prompt，避免过度依赖对象类别语义。

数据和监督信号：

- zero-shot：不使用目标域训练样本，只用 prompt。
- few-shot：使用少量正常参考图。
- prompt learning：可用辅助 anomaly dataset 学习通用 normal/abnormal prompt。

模型结构或训练策略：

- CLIP image encoder + text encoder。
- window / patch feature 与 text prompt 比较。
- normal reference memory bank。
- learnable static / dynamic prompts。
- SAM/DINOv2 等 foundation models 作为 segmentation 或 feature extractor。

优点：

- 跨类别泛化强于传统闭集检测。
- 标注需求低。
- 自然支持文本条件和 open-vocabulary defect terms。

局限：

- CLIP 的自然图像语义不等于工业细粒度异常语义。
- 对 PCBA 小元件、焊点、丝印、走线的细粒度视觉弱。
- 输出通常仍是 anomaly score/map，不是完整 VQA。
- 对 counting、cause、handling 支持弱。

对 PCBA Standard-to-Real VQA 的适配程度：

中高。适合做 zero/few-shot baseline、异常候选定位、文本 prompt 对齐；但必须补充 PCBA 领域知识和结构化 VQA 训练。

### 3.4 VLM / MLLM Fine-tuning / Instruction Tuning

代表方法：

- AnomalyGPT
- Myriad
- EIAD
- Triad
- OmniAD
- IAD-GPT
- IAD-R1 / Anomaly-R1
- AD-Copilot
- UniPCB / PCB-GPT

核心思想：

把工业异常检测转成 multimodal instruction following 或 VQA。模型不仅判断是否异常，还生成缺陷描述、位置、类型、原因、影响、建议等文本。训练上通常包括 caption、QA、CoT、bbox/mask grounding、LoRA/SFT/RL。

数据和监督信号：

- 公开 IAD 图像 + mask/bbox + GPT-generated captions/QA。
- 人工过滤的 defect QA。
- 正常参考图或 expert anomaly map。
- 有些方法引入 manufacturing process、domain knowledge、structured rewards。

模型结构或训练策略：

- LLaVA / MiniGPT-4 / Qwen2.5-VL / InternVL 等 backbone。
- LoRA instruction tuning。
- 视觉专家 anomaly map 引导 attention 或视觉 token。
- bbox/mask grounding module。
- CoT / structured output。
- GRPO / RL with verifiable rewards。
- 多阶段 curriculum：concept alignment -> instruction tuning -> RL。

优点：

- 最接近工业 VQA 输出形式。
- 能覆盖 defect description、defect analysis、object analysis。
- 方便引入标准术语和结构化输出。

局限：

- 容易过拟合公开 IAD 数据或 GPT-generated QA。
- 视觉定位和计数仍不稳定。
- 工业原因和处置建议常来自文本先验，未必有视觉证据 grounding。
- 多图细粒度比较仍弱。

对 PCBA Standard-to-Real VQA 的适配程度：

高。当前最主要的相关技术路线。UniPCB / PCB-GPT、AD-Copilot、Triad 对 PCBA VQA 的启发最大。

### 3.5 Retrieval-Augmented VQA / Standards Knowledge Grounding

代表方法：

- MMAD 中的 RAG 和 expert agent
- MANTA 的 Declarative Knowledge
- UniPCB 的 defect/component knowledge base
- LogicQA 的 VLM-generated checklist
- ManuRAG
- KU-RAG / Fine-Grained RAG for VQA
- ReAG

核心思想：

对需要外部领域知识的问题，先检索标准、规则、工艺知识、缺陷定义、正常模板或图文知识单元，再让 VLM/MLLM 在检索内容和图像证据上回答问题。

数据和监督信号：

- 标准文档、工艺说明、缺陷定义、处置手册。
- 正常样例图、标准图示、模板图。
- image-text knowledge units。
- checklist、rubric、structured QA。

模型结构或训练策略：

- text-only RAG：按元件/缺陷类别检索文本规则。
- multimodal RAG：检索图文知识单元或正常参考图。
- expert agent：调用 anomaly detector / segmenter 输出视觉证据。
- critic / reranker：过滤无关 passage。
- structured prompt：强制回答引用规则或输出 JSON。

优点：

- 与 standard-to-real 问题天然匹配。
- 能支撑 defect cause、impact、handling recommendation。
- 可减少模型凭空编造领域知识。

局限：

- 检索到规则不等于能正确落到图像证据。
- 标准图示、真实图像、文本条款之间对齐困难。
- RAG 对小目标视觉错误无能为力。
- 如果标准文档解析错误，后续回答会系统性错误。

对 PCBA Standard-to-Real VQA 的适配程度：

很高。标准文档到真实图像迁移必须有 knowledge grounding，但不能只做 RAG；需要视觉定位、元件识别和规则验证共同工作。

### 3.6 Domain Adaptation / Domain Generalization

代表方法：

- Real-IAD
- MVTec AD 2
- Robust Distribution Alignment for IAD under Distribution Shift
- Domain-independent detection of known anomalies
- Syn-to-real / sim-to-real industrial parts classification
- AdaptCLIP / AdaCLIP 类跨域 prompt 方法

核心思想：

处理真实工业环境中的分布变化：光照、视角、相机、产线、材质、产品版本、标准图示与真实图像差异、BPCB 到 PCBA 的装配变化。

数据和监督信号：

- 多域、多视角、多光照、多设备数据。
- sparse nominal data。
- unlabeled target data。
- synthetic source + real target。
- target normal reference images。

模型结构或训练策略：

- feature distribution alignment。
- memory bank alignment。
- pseudo-label / self-training。
- domain randomization。
- prompt adaptation。
- cross-attention comparison encoder。

优点：

- 直面真实工厂落地的核心问题。
- 与 standard-to-real 迁移强相关。
- 可以减少对目标域标注的依赖。

局限：

- 公开工作主要研究 image-level / pixel-level AD，不研究标准文档和 VQA。
- 标准图示到真实 PCBA 的跨模态、跨风格、跨尺度迁移尚未成熟。
- 多数方法没有工业规则推理能力。

对 PCBA Standard-to-Real VQA 的适配程度：

高。domain shift 是核心难点，但需要和 VLM instruction tuning、RAG、component grounding 结合。

### 3.7 Teacher-Student / Distillation / Ensemble

代表方法：

- EfficientAD
- Reverse Distillation
- Myriad 的 vision expert guidance
- Triad 的 vision expert-guided visual tokenizer
- AD-Copilot 的 comparison encoder 和 multi-stage training

核心思想：

用强视觉模型、正常样本 teacher 或 IAD expert 为 MLLM 提供局部异常证据。teacher-student 用预测失败定位异常；expert-guided LMM 用 anomaly map 或 ROI 引导语言模型关注异常区域。

数据和监督信号：

- 正常图像。
- teacher feature。
- expert anomaly map。
- mask/bbox。
- caption / QA / CoT。

模型结构或训练策略：

- frozen teacher + trained student。
- anomaly map visualization 输入 MLLM。
- expert ROI tokenizer。
- comparison encoder 对比 query 和 normal reference。
- 多模型 ensemble。

优点：

- 工业视觉稳定性强。
- 可改善 MLLM 对细微异常区域的关注。
- 工程上容易和已有 AOI/IAD 模块结合。

局限：

- 级联误差明显：expert 漏了，MLLM 很难补救。
- anomaly map 不等于元件语义。
- teacher-student 本身不能解释原因和处置。

对 PCBA Standard-to-Real VQA 的适配程度：

中高。适合作为视觉前端或 evidence module，但需要上层 VQA/规则模块解释结果。

### 3.8 Counting and Small-Object Reasoning

代表方法：

- MANTA
- PairTally
- UltraVR
- UniPCB 的 counting / localization subtasks
- Qwen2.5-VL dynamic resolution / grounding
- high-resolution PCB tile inference work

核心思想：

解决高分辨率图像中小目标、密集对象、细粒度类别和数量统计问题。PCBA 中同一图像可能有大量电容、电阻、IC、connector、焊点和局部缺陷，单纯 resize 到普通 VLM 输入会丢失关键信息。

数据和监督信号：

- bbox / mask / point annotation。
- high-resolution image QA。
- count labels。
- evidence chain / intermediate reasoning label。
- tile-level annotations。

模型结构或训练策略：

- high-resolution tiling。
- dynamic resolution visual tokens。
- bbox/point grounding。
- evidence-grounded reasoning。
- detector-assisted counting。
- local crop verification。

优点：

- 命中 PCBA VQA 的主要视觉瓶颈。
- 可用结构化指标评估，例如 count accuracy、bbox F1、IoU。

局限：

- 通用 VLM 对密集小目标计数普遍不可靠。
- 小目标局部 crop 可能丢失全局上下文。
- 多类相似封装容易混淆。

对 PCBA Standard-to-Real VQA 的适配程度：

很高。计数和小目标定位不是附属问题，而是 PCBA VQA 的核心问题。

### 3.9 Cause & Handling / Actionable Inspection Reasoning

代表方法：

- MMAD 的 defect analysis
- MANTA 的 what/why/how knowledge
- FabGPT
- WaferSAGE
- Triad 的 manufacturing process
- UniPCB 的 common causes、potential impact、reinspection recommendations
- VELM / Detect, Classify, Act

核心思想：

从“检测到异常”走向“解释异常为什么发生、有什么影响、如何处理”。这需要视觉证据、缺陷类型、工艺知识、标准规则和处置流程共同参与。

数据和监督信号：

- defect type -> root cause / impact / action 的专家知识。
- 工艺流程文本。
- rubric / checklist。
- synthetic VQA pairs。
- human-reviewed QA。

模型结构或训练策略：

- LMM + domain corpus instruction tuning。
- RAG over process knowledge。
- rubric-guided RL。
- CoT with manufacturing process。
- structured answer template。

优点：

- 贴近真实质检报告和生产线决策。
- 能回答 challenge 中的缺陷原因和处理建议问题。

局限：

- 公开数据少，很多结果依赖私有工艺数据。
- 因果关系常是文本先验，不一定由图像证据直接支持。
- 处置建议可能与具体厂内 SOP 有冲突。

对 PCBA Standard-to-Real VQA 的适配程度：

很高，但必须加约束：回答应基于标准/规则和图像证据，避免泛泛生成。

## 4. 方法对比表

| Method / Paper | Year | Task | Data type | Model backbone | Training signal | Handles domain shift? | Handles standards/rules? | Handles counting? | Handles cause/handling? | Relevance to PCBA VQA |
|---|---:|---|---|---|---|---|---|---|---|---|
| MVTec AD | 2019 | IAD benchmark | high-res industrial images, masks | N/A | normal train, anomaly masks for eval | 弱 | 否 | 否 | 否 | 中，经典 baseline |
| MVTec LOCO AD | 2022 | logical AD benchmark | structural + logical anomalies | N/A | normal train, pixel GT | 部分 | 部分，逻辑约束 | 部分 | 否 | 高，规则异常相关 |
| VisA | 2022 | visual anomaly benchmark | 12 subsets, 4 PCB subsets | N/A | image/mask labels | 部分 | 否 | 弱 | 否 | 高，PCB 公开数据相关 |
| Reverse Distillation | 2022 | AD/localization | normal images | teacher encoder + student decoder | one-class distillation | 部分 | 否 | 否 | 否 | 中，视觉专家 |
| EfficientAD | 2023 | fast AD/localization | MVTec/LOCO/VisA | lightweight teacher-student + AE | normal-only distillation | 部分 | 弱，能处理部分 logical anomaly | 弱 | 否 | 中，快速异常前端 |
| SimpleNet | 2023 | AD/localization | industrial AD datasets | pretrained extractor + adapter + discriminator | feature-space synthetic anomalies | 部分 | 否 | 否 | 否 | 中 |
| WinCLIP | 2023 | zero/few-shot AD | MVTec AD, VisA | CLIP | text prompt + window features + few normal shots | 是 | 弱 | 弱 | 否 | 中高 |
| APRIL-GAN | 2023 | VAND zero/few-shot AD | VAND / VisA / MVTec | CLIP + linear layers/memory | prompt + few-shot memory | 是 | 弱 | 弱 | 否 | 中高 |
| SAA+ | 2023 | zero-shot anomaly segmentation | VisA/MVTec | SAM + CLIP/foundation models | multimodal prompting | 是 | 弱 | 弱 | 否 | 中 |
| AnomalyCLIP | 2023/2024 | zero-shot AD | 17 AD datasets | CLIP | object-agnostic prompt learning | 是 | 否 | 弱 | 否 | 中高 |
| AdaCLIP | 2024 | zero-shot AD | 14 industrial/medical AD datasets | CLIP + hybrid prompts | auxiliary annotated AD data | 是 | 否 | 弱 | 否 | 中高 |
| Real-IAD | 2024 | real-world multi-view IAD benchmark | 150K high-res images, 30 objects | N/A | sample-level/mask eval | 是 | 否 | 弱 | 否 | 高，真实工厂域 |
| MMAD | 2024/2025 | industrial MLLM benchmark | 8366 images, 39672 MCQ | evaluated GPT-4o/Gemini/LLaVA/InternVL etc. | GPT-4V-generated QA + human filtering | 评测覆盖 | 部分，RAG/domain knowledge | 弱 | 是，defect analysis | 很高，必引 |
| AnomalyGPT | 2023/2024 | IAD dialogue | simulated anomaly + text | LVLM / MiniGPT-4 style | synthetic anomaly text, prompt learner | 弱 | 弱 | 弱 | 弱 | 高但泛化有限 |
| Myriad | 2023 | LMM + vision experts for IAD | MVTec, VisA, PCB Bank | LMM + IAD expert | anomaly map guidance + instruction | 部分 | 弱 | 弱 | 弱 | 高，专家引导路线 |
| MANTA | 2024/2025 | tiny-object visual-text AD | 137.3K multi-view images, text QA | benchmark + baseline | mask + visual-text QA + what/why/how | 是 | 部分 | 部分 | 是 | 很高，小目标和解释相关 |
| FabGPT | 2024 | wafer defect knowledge query | SEM wafer data | customized LMM | multimodal corpus + wafer knowledge | 私有域 | 是，工艺知识 | 弱 | 是，root cause | 中高，半导体知识参考 |
| LogicQA | 2025 | logical anomaly reasoning | LOCO + SEM data | VLM-generated checklist | few-shot, training-free QA checklist | 部分 | 是，逻辑约束 | 是 | 弱 | 很高，规则和数量异常相关 |
| EIAD | 2025 | explainable IAD | DDQA | MLLM + defect localization module | real defect QA + localization | 部分 | 弱 | 弱 | 部分 | 高 |
| Triad | 2025 | LMM IAD + manufacturing process | InstructIAD | LLaVA AnyRes + vision expert tokenizer | instruction tuning + CoT-M | 部分 | 是，manufacturing process | 弱 | 是 | 高 |
| OmniAD | 2025 | anomaly detection and understanding | MMAD/IAD datasets | MLLM reasoner | SFT + GRPO rewards | 部分 | 弱 | 弱 | 是 | 高 |
| IAD-GPT | 2025 | MLLM for IAD | MVTec/VisA | CLIP + MLLM + mask fusion | abnormal prompt, text-guided enhancer | 部分 | 弱 | 弱 | 部分 | 高 |
| IAD-R1 | 2025 | reasoning-enhanced IAD | Expert-AD / multiple AD benchmarks | VLM post-training framework | PA-SFT + SC-GRPO | 部分 | 弱 | 弱 | 部分 | 高 |
| AD-Copilot | 2026 | IAD assistant via visual in-context comparison | Chat-AD, MMAD-BBox | MLLM + comparison encoder | caption/VQA/localization, multi-stage | 是 | 弱 | 部分 | 部分 | 很高，多图比较相关 |
| UniPCB / PCB-GPT | 2026 | open-ended PCB quality inspection | 6581 images, 23359 bilingual QA | Qwen2.5-VL-7B + LoRA/GRPO | caption, QA, bbox, CoT, structured rewards | 是，含 BPCB/PCBA shift | 是，defect/component KB | 是 | 是 | 最高，最接近 PCBA VQA |
| Qwen2.5-VL | 2025 | general high-res VLM | large-scale general data | dynamic-resolution ViT + Qwen LLM | pretrain + SFT | 通用 | 否 | 部分 | 否 | 适合 backbone |
| PairTally | 2025 | fine-grained visual counting benchmark | high-res two-category images | benchmark | count labels | 部分 | 否 | 是 | 否 | 中高，计数诊断 |
| UltraVR | 2026 | ultra-resolution VQA | CCTV/RS/WSI/IAD | benchmark | evidence chain labels | 是 | 否 | 是 | 否 | 高，小目标证据链相关 |
| WaferSAGE | 2026 | wafer VQA / root cause | wafer maps | Qwen3-VL 4B | synthetic VQA + rubric-guided RL | 私有/专域 | 是 | 部分 | 是 | 中高，工艺解释参考 |
| ManuRAG | 2026 | manufacturing QA RAG | text/images/formulas/tables | multimodal RAG | retrieval + QA | 部分 | 是 | 否 | 部分 | 中高，标准文档 QA 参考 |
| KU-RAG | 2025 | knowledge-based VQA | text snippets + entity images | MLLM + vector DB | fine-grained retrieval | 部分 | 是 | 弱 | 部分 | 高，图文知识检索参考 |

## 5. 对 PCBA Standard-to-Real VQA 的关键观察

### 5.1 现有方法普遍不足的能力

1. 标准图示 / 标准文档到真实图像的 grounding 不足。

大多数 IAD 方法只看真实图像，不处理标准文档；大多数 RAG 方法只处理文本，不解决图像证据落地。真正需要的是“标准条款/标准图示/元件定义/真实图像局部证据”的联合对齐。

2. 元件级别的视觉语义仍弱。

Anomaly map 只能告诉哪里不正常，不能可靠判断这是 capacitor、resistor、IC、connector、solder joint、silkscreen 还是 trace，也不能自然关联 refdes、BOM 或标准规则。

3. 小目标密集计数不可靠。

PCBA 的“几个元件缺失”“某类元件数量是否正确”“缺陷数量是多少”对 VLM 是硬问题。PairTally、UltraVR、MANTA 都说明通用模型在细粒度小目标计数上仍有明显短板。

4. 多图精细比较能力不足。

Standard-to-real 场景通常需要比较标准图、正常模板、真实图、局部 crop。MMAD 发现 normal template 对一些模型帮助有限甚至可能伤害性能；AD-Copilot 正是针对这个问题引入 comparison encoder。

5. 原因和处置建议缺少可靠 grounding。

FabGPT、Triad、WaferSAGE、UniPCB 都在做 root cause / impact / recommendation，但很多原因来自文本知识或模板。PCBA VQA 中应避免“看图猜工艺原因”，需要明确把原因限定为“可能原因”并绑定视觉证据和标准规则。

6. 评估协议不足。

工业 VQA 不能只用 LLM-as-judge。bbox、count、yes/no、defect type、component type、rule violation 应尽量结构化评估。开放解释可用 LLM judge，但必须配合 evidence 或 rule citation。

### 5.2 已经很常见、不适合作为主要创新的方向

以下方向已经很常见，单独作为 ACM MM Grand Challenge technical paper 的主要创新不够强：

- 单纯把 CLIP prompt 用于 anomaly detection。
- 单纯用 SAM + CLIP / DINOv2 做 anomaly segmentation。
- 单纯把 anomaly map 可视化后喂给 MLLM。
- 单纯对 Qwen/LLaVA/InternVL 做 LoRA instruction tuning。
- 单纯用 GPT-4V/GPT-4o 从 mask 自动生成 QA。
- 单纯 RAG 一份缺陷知识库。
- 单纯 normal reference / few-shot comparison。
- 单纯 teacher-student 或 PatchCore/EfficientAD 做缺陷定位。
- 单纯 high-resolution tiling，除非结合元件语义、跨域对齐或 VQA 评估。

这些都可以作为系统组件，但不宜包装成核心贡献。

### 5.3 与 standard-to-real industrial VQA 最贴合的方向

1. PCB/PCBA 统一 taxonomy + structured VQA。

UniPCB 的 unified defect/component taxonomy、14 subtasks、structured output 和三阶段训练，直接贴近 PCBA VQA。

2. 多图 fine-grained comparison。

AD-Copilot 证明普通 MLLM 独立编码多图不够，comparison encoder / cross-attention 对细微差异检测有价值。standard-to-real 问题更需要标准图和真实图的显式比较。

3. 规则 / 标准 grounding。

LogicQA 的 checklist 思路、MMAD/UniPCB 的 domain knowledge、ManuRAG/KU-RAG 的 multimodal retrieval 都与标准文档 VQA 相关。

4. 小目标定位与计数。

MANTA、UltraVR、PairTally 和 high-resolution PCB tiling 工作都说明，应把 dense small-object reasoning 作为核心能力评估。

5. Cause & handling with evidence。

MANTA、Triad、FabGPT、WaferSAGE、UniPCB 都支持原因/影响/建议，但真正难点是让回答受图像证据、标准规则和工艺知识共同约束。

## 6. ACM MM Grand Challenge Technical Paper 必引相关工作

### 6.1 Benchmark / Dataset 必引

- MVTec AD：经典 unsupervised IAD benchmark。
- MVTec LOCO AD：logical anomalies，与规则/数量/位置约束相关。
- VisA：含 PCB 子集，复杂结构和多实例。
- Real-IAD：真实生产线、多视角、大规模。
- MVTec AD 2：更接近真实工业难点和公开/私有测试 server。
- MMAD：工业 MLLM/VQA benchmark，7 个工业质检子任务。
- MANTA：tiny object、multi-view、visual-text anomaly、what/why/how。
- UniPCB：PCB/PCBA open-ended quality inspection benchmark，最贴近 PCBA VQA。

### 6.2 IAD Baseline 必引

- PatchCore：memory bank patch feature baseline，虽早于 2022 但仍是工业 AD 经典基线。
- Reverse Distillation：teacher-student one-class embedding。
- EfficientAD：fast student-teacher + global AE，工业部署相关。
- SimpleNet：feature-space synthetic anomaly + discriminator。

### 6.3 VLM-based AD 必引

- WinCLIP：zero/few-shot CLIP anomaly classification/segmentation。
- APRIL-GAN / SAA+：VAND challenge 技术报告，foundation model prompting。
- AnomalyCLIP：object-agnostic prompt learning。
- AdaCLIP / AdaptCLIP：hybrid prompt adaptation，跨域 zero-shot anomaly detection。

### 6.4 MLLM Industrial Inspection 必引

- AnomalyGPT：早期把 LVLM 用于 IAD dialogue。
- Myriad：vision expert guidance for IAD。
- EIAD：defect QA + localization 的 explainable IAD。
- Triad：vision expert-guided tokenizer + manufacturing process。
- OmniAD：visual/textual reasoning + GRPO。
- IAD-GPT：prompt/mask fusion 的 MLLM IAD。
- IAD-R1 / Anomaly-R1：reasoning/RL post-training。
- AD-Copilot：visual in-context comparison 和 MMAD-BBox。
- UniPCB / PCB-GPT：PCB 专项 open-ended VQA。

### 6.5 Knowledge / Process / Actionable Reasoning 必引

- MMAD 的 RAG 和 Expert Agent。
- LogicQA：logical anomaly checklist。
- MANTA：what/why/how declarative knowledge。
- FabGPT：wafer defect knowledge query 和 root cause。
- WaferSAGE：rubric-guided wafer defect VQA / root cause。
- ManuRAG / KU-RAG：manufacturing QA 和 fine-grained multimodal retrieval。

### 6.6 General VLM Backbone 必引

- LLaVA / LLaVA-1.5 / LLaVA-OneVision：visual instruction tuning 和多图/视频能力。
- Qwen2-VL / Qwen2.5-VL：dynamic resolution、grounding、document/diagram understanding。
- InternVL2.5：open-source multimodal scaling、multi-image、grounding。

## 7. 直接结论

PCBA Standard-to-Real VQA 的真正难点不是“有没有一个强 VLM”，而是以下能力的组合：

- 标准文档、标准图示、真实 PCBA 图像之间的精确 grounding。
- 细粒度元件识别、缺陷定位和小目标计数。
- 基于标准规则的判断，而不是泛化描述。
- 能输出结构化答案，并支持 bbox/count/type 等可验证指标。
- 对原因和处置建议进行有证据约束的推理。
- 能处理 BPCB/PCBA、标准图/真实图、不同光照/视角/设备之间的 domain shift。

最接近的公开相关工作是 `UniPCB / PCB-GPT` 和 `MMAD`。`MANTA`、`LogicQA`、`AD-Copilot`、`Triad` 分别补足小目标、规则逻辑、多图比较和工艺推理视角。传统 IAD 方法仍应引用，但更适合作为视觉专家模块或 baseline，而不是完整 PCBA VQA 方案。

