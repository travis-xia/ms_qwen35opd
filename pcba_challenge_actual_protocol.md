# PCBA Standard-to-Real Challenge: Proposal 与实际发布情况对照

更新时间：2026-06-29  
用途：整理论文写作时应采用的真实赛题协议，避免直接复述 proposal 中后来没有完全执行的设计。

## 信息来源

- 本地 proposal：`21_PCBA_Standard_to_Real_Chall.md`
- 官网：[ASUS-NTU PCBA VQA Challenge](https://sites.google.com/cmlab.csie.ntu.edu.tw/asus-ntu-pcba-vqa-challenge/home)
- Codabench API：[competition 16306](https://www.codabench.org/api/competitions/16306/)
- Hugging Face 数据集页：[aimmifm/PCBA_Standard-to-Real_Challenge](https://huggingface.co/datasets/aimmifm/PCBA_Standard-to-Real_Challenge)

注意：Codabench 的 participant/submission 数量和 HF 的 downloads/likes 属于抓取时快照，后续可能变化。这里更关注数据组织、评测协议和提交格式。

## 一句话结论

proposal 描述的是一个更完整的双-suite、public+hidden、多指标综合评测设计；实际 Codabench 发布版本是一个统一的 mixed-domain, mixed-task public evaluation：公开训练集 8,200 题，公开测试集 8,200 题，测试集不暴露 task type，提交一个 `submission.csv`，排行榜只显示单一 Overall Score。

因此论文中不宜写成“严格按照 proposal 的两个 suite 分别建模”，更准确的表述是：

> The released benchmark evaluates a unified mixed-domain PCBA VQA setting where task annotations are hidden at test time. The method therefore needs to infer the latent task family and answer format from each question-image pair, while jointly handling standards-derived knowledge QA and real-world visual inspection QA.

## Proposal 中的设计

proposal 的核心设定如下：

| 项目 | proposal 描述 |
| --- | --- |
| 任务定位 | Standard-to-Real cross-domain VQA for SMT-PCBA inspection |
| 数据来源 | Standard corpus + Real-world corpus |
| 真实缺陷类别 | Missing Component, Insufficient Solder, Standing, Flipped Component, Wrong Polarity |
| 任务家族 | Cause & Handling, Factuality, Quantitative Reasoning, Attribute Reasoning |
| public 数据量 | Table 1 写 public dataset 共 16,000 questions，每个子任务约 2,000 |
| split | 每个 suite 有 Train/Test/Hidden，public Train:Test = 2:1 |
| suite 设计 | Suite 1: Standards -> Real-world，Standard:Real = 3:1；Suite 2: Real-world -> Real-world，Standard:Real = 1:5 |
| hidden | 每种 question type 约 500 hidden questions |
| final score | `0.8 * public test + 0.2 * hidden` |
| 提交格式 | proposal 写 single JSON prediction file |
| MCQ 指标 | Accuracy |
| defect existence 指标 | Accuracy + F1 |
| counting 指标 | Accuracy + normalized MAE |
| 总分聚合 | 各 question family 等权 |

proposal 的主旨非常清楚：标准知识、真实工厂图像、可执行工业决策之间存在跨域落差，模型应能从标准规则泛化到真实 PCBA 图像。

## Codabench 实际协议

Codabench competition 标题为：

> ACM MM 2026 PCBA Standard-to-Real Grand Challenge

Codabench 描述为：

- Cross-domain visual question answering for real-world PCBA manufacturing inspection.
- 真实任务强调 industrial standards + real production-line imagery。
- 模型需要理解 manufacturing rules、context-dependent defects、possible causes 和 actionable answers。

### 实际 public split

Codabench Overview 页给出的公开数据规模：

| Split | Number of Questions |
| --- | ---: |
| Train | 8,200 |
| Test | 8,200 |
| Total | 16,400 |

这与 proposal 中 “public dataset 16,000 questions” 以及 “Train:Test = 2:1” 不一致。实际公开评测是 1:1 的 8,200/8,200。

### 实际任务家族

Codabench 页面把任务概括成四类：

1. Standard-based Knowledge QA
2. Factuality
3. Quantitative Reasoning
4. Attribute Reasoning

实际 evaluation server 内部使用以下 task groups：

| Internal task group | 含义 |
| --- | --- |
| `standard_knowledge` | 标准知识、缺陷原因、处理决策等 |
| `component_type` | 元件类型识别 |
| `mount_side` | 装配面识别 |
| `defect_existence` | 缺陷是否存在 |
| `defect_type` | 缺陷类型识别 |
| `count_component` | 元件数量计数 |
| `count_pin_lead` | pin/lead 数量计数 |
| `attribute_reasoning` | 形状等属性推理 |

关键点：`vqa_test_public.json` 不暴露 task-type annotations。官方在服务器端内部处理 task grouping。

这意味着参赛方法不能假设测试时知道题型；论文应强调 hidden task annotation 下的 task-agnostic answering 或 latent task inference。

### 实际提交格式

Codabench 要求提交一个 `.zip`，根目录中只包含一个 `submission.csv`：

```text
your_submission.zip
└── submission.csv
```

CSV 必须有两列：

```csv
qid,answer
1,B
2,12
3,E
```

实际规则：

- `qid` 必须与 `vqa_test_public.json` 中的 question id 完全匹配。
- `answer` 只包含最终答案。
- 多选题提交选项 key，例如 `A`, `B`, `C`, `D`, `E`, `F`。
- 计数题提交普通整数字符串，例如 `12`。
- 每道测试题必须有一行。
- 缺失 qid、重复 qid、未知 qid 会导致 invalid submission。
- `submission.csv` 不能放在额外子目录里。

这与 proposal 中的 JSON prediction file 不一致。

### 实际评测指标

Codabench 实际指标如下：

| Task group | Actual metric |
| --- | --- |
| `standard_knowledge` | Accuracy |
| `component_type` | Accuracy |
| `mount_side` | Accuracy |
| `defect_type` | Accuracy |
| `attribute_reasoning` | Accuracy |
| `defect_existence` | Binary F1 on anomaly-positive class |
| `count_component` | Normalized MAE |
| `count_pin_lead` | Normalized MAE |

计数任务实际使用：

```text
count_score_t = max(0, 1 - MAE_t / C_t)
```

固定归一化常数：

```text
C_count_component = 138
C_count_pin_lead = 260
```

总体分数：

```text
Overall Score = 100 * sum_t (w_t * s_t)
w_t = n_t / N
```

也就是说，实际总分是按 evaluation set 中各 task 的样本比例加权，而不是 proposal 中的 question family 等权。实际 leaderboard 只显示一个 `Overall Score`，范围为 0-100。

### 实际 phase 和 leaderboard

Codabench API 显示：

| 项目 | 实际值 |
| --- | --- |
| Phase name | Public Evaluation |
| Start | 2026-05-14 |
| End | 2026-06-04 |
| Status | Previous |
| `is_final_phase` | true |
| Max submissions per day | 5 |
| Max submissions per person | 100 |
| Leaderboard | Public Leaderboard |
| Leaderboard column | Overall Score |

因此，Codabench 当前公开协议不是 proposal 中明确的“两阶段 public development + final hidden-test phase”。页面只说明 organizers may additionally run the same scoring protocol on private evaluation data for verification or post-competition analysis。

写论文时建议谨慎表述 hidden/private evaluation：可以说 official server may conduct private verification，但不要把 proposal 里的 `0.8 public + 0.2 hidden` 当成实际公开 ranking protocol 来写。

## Hugging Face 实际文件组织

HF 数据集是 public 但 gated：

| 项目 | 值 |
| --- | --- |
| Dataset id | `aimmifm/PCBA_Standard-to-Real_Challenge` |
| Private | false |
| Gated | manual |
| Last modified | 2026-06-19 |

公开 API 暴露的关键 JSON 文件：

```text
Train/Standard/standard_mm_vqa_train_public.json
Train/RealWorld/realworld_mm_vqa_train_public.json
Test/vqa_test_public.json
```

目录结构大致为：

```text
Train/
  Standard/
    images/
    standard_mm_vqa_train_public.json
  RealWorld/
    images/
    realworld_mm_vqa_train_public.json
Test/
  images/
  vqa_test_public.json
```

API 文件数快照：

| Path group | File count from HF API |
| --- | ---: |
| `Train/RealWorld` | 6,601 |
| `Train/Standard` | 1,837 |
| `Test` | 7,792 |

注意：这些是文件数量，不是 question 数量；question 数量应以 Codabench Overview 中的 8,200/8,200 为准。

## Proposal 与实际发布的关键差异

| 维度 | Proposal | 实际 Codabench/HF |
| --- | --- | --- |
| public 总题数 | 16,000 | 16,400 |
| public split | Train:Test = 2:1 | Train 8,200 / Test 8,200 |
| suite | 两个 suite：S->R 与 R->R | 统一 public evaluation，页面不暴露 suite |
| test task type | proposal 未强调隐藏 | 明确不暴露 task-type annotations |
| hidden | 每类约 500，final score 含 hidden | Codabench 页面只公开 Public Evaluation；private evaluation 仅作为可能的验证/分析 |
| final score | 0.8 public + 0.2 hidden | public leaderboard 单一 Overall Score |
| 提交格式 | JSON | zip 内根目录 `submission.csv` |
| defect existence | Accuracy + F1 | F1 only |
| counting | Accuracy + normalized MAE | normalized MAE only |
| 总分聚合 | 各 family 等权 | 按 task 样本比例加权 |
| counting normalization | proposal 称后续公布 R | 实际固定为 138 和 260 |
| 标准/真实组织 | proposal 描述 corpus 和 suite 比例 | HF 实际按 `Train/Standard` 与 `Train/RealWorld` 组织，Test 统一放在 `Test` |

## 对论文叙述的直接影响

### 1. 不要把方法写成两个 suite 的专用系统

实际测试是 unified mixed-domain mixed-task setting。更合理的论文表述是：

> We address the released benchmark as a unified mixed-domain VQA problem with hidden task annotations, where each question-image pair may require standards-grounded reasoning, fine-grained real-image perception, counting, or attribute recognition.

### 2. 方法必须强调 latent task inference

因为 test JSON 不给 task type，方法设计应包括：

- 从 question/options 判断答案格式；
- 区分 MCQ 与 integer counting；
- 对 defect existence 使用 F1-aware calibration；
- 对 counting 使用 normalized-MAE-aware calibration；
- 对 standard_knowledge 与 real-world visual questions 走不同但共享参数的推理路径。

### 3. Standard-to-Real 主旨仍然成立

虽然实际协议没有公开两个 suite，数据来源仍然明确分成：

- standards-derived VQA；
- real-world VQA。

论文中仍应围绕 standards-derived knowledge 到 real-world PCBA inspection 的泛化来叙述。只是不要声称自己在官方公开评测中分别优化了 Suite 1 / Suite 2。

### 4. 更适合的论文问题定义

建议把 task formulation 写成：

```text
Given an image I, a question q, and optional answer choices O,
predict an answer a under a hidden task family t.
The training set exposes standards-derived and real-world subsets,
whereas the public test set is a unified mixture without task labels.
```

其中 hidden task family 包括：

```text
t in {
  standard_knowledge,
  component_type,
  mount_side,
  defect_existence,
  defect_type,
  count_component,
  count_pin_lead,
  attribute_reasoning
}
```

### 5. 方法创新最好贴合以下三点

1. standards-derived knowledge 与 real-world visual evidence 的对齐；
2. hidden task type 下的 task-agnostic VQA；
3. metric-aware constrained decoding。

如果使用多教师 on-policy distillation，可以自然写成：

> A standards teacher captures normative industrial knowledge, a real-image teacher captures visual inspection evidence, and a student learns a unified inspection-state representation under task-agnostic, metric-aware decoding.

这种表述比“两个 teacher ensemble”更贴合组织者主旨，也更符合实际发布协议。
