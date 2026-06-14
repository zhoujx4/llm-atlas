---
title: Tool Use / Function Calling 训练
---

# Tool Use / Function Calling 训练

> **一句话**：教模型在合适的时机、以正确的 schema 发起工具调用并消化返回结果；数据来源经历了自监督标注（*Toolformer*, 2023）→ 真实 API 大规模标注（*ToolLLM*, 2023）→ 执行验证合成（*APIGen*, 2024）三代演进。
>
> 前置阅读：[SFT 总览](/sft/)、[Chat Template](/sft/chat-template)、[Loss Masking](/sft/loss-masking)

## 直觉与动机

模型权重里的知识是静态且有限的：算不准大数乘法、不知道此刻的库存、改不了数据库里的一行记录。工具调用把这些能力外包给确定性系统，模型只需要做三个决策：**要不要调用、调用哪个、参数填什么**。

从生成的角度看，一次 function call 是受约束的结构化生成：函数名必须落在给定工具集内，参数必须满足 JSON schema 的类型与必填约束。但格式正确只是底线，真正困难的在决策层——BFCL 专设 relevance detection 类目考察"没有合适函数时应当拒绝调用"；τ-bench 则显示即使 SOTA 函数调用模型在带政策约束的多轮场景里成功率也不足 50%（GPT-4o），且行为高度不稳定（retail 域 pass^8 < 25%）。

为什么值得专门训练：仅靠 prompt 注入 schema，中小模型的格式遵循与参数抽取都不可靠；而 APIGen 的结果显示，用 6 万条经执行验证的数据训练，7B 模型在 BFCL 上可超过若干 GPT-4 版本，1B 模型超过 GPT-3.5-Turbo 和 Claude-3 Haiku——工具调用是少数"小模型经针对性训练可逆袭"的能力。

![Toolformer：模型在文本生成中自主决定调用问答、计算器、翻译、维基检索等 API](/papers/tool-use/toolformer-overview.png)

> 图源：Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, arXiv:2302.04761（用于学习注解，版权归原作者）

## 方法与公式

### 数据格式与训练目标

一条多轮工具调用样本在 [chat template](/sft/chat-template) 中通常包含四类轮次：

- **system**：注入工具定义（JSON schema 列表）；
- **user**：自然语言请求 $x$；
- **assistant**：结构化调用 $y_{\text{call}} = (\text{name}, \text{args})$，或最终自然语言回答；
- **tool**：工具返回 $o$（非模型生成）。

SFT 只对 assistant 轮计算 loss，tool 返回轮与 system/user 一样 mask 掉：

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t} m_t \log \pi_\theta(y_t \mid x, y_{<t}), \qquad m_t = \mathbb{1}\big[\, y_t \in \text{assistant 轮} \,\big]
$$

### 数据从哪来：三条路线

**路线一：自监督标注（Toolformer, Meta, 2023）。** 不依赖人工演示，让模型自己学会"在文本的哪个位置插入哪个 API 调用"：用 few-shot prompt 在纯文本语料中采样候选调用位置与参数，实际执行 API 得到结果 $r_i$，再用"是否降低后续 token 的加权困惑度"过滤——仅当插入"调用 + 结果"比不插入、或只插入调用更有助于预测后文时才保留：

$$
\min\big(L_i(\varepsilon),\; L_i(c_i)\big) \;-\; L_i(c_i \oplus r_i) \;\ge\; \tau_f
$$

其中 $L_i(\cdot)$ 是以给定前缀计算的位置 $i$ 之后 token 的加权交叉熵，$c_i$ 为 API 调用、$\varepsilon$ 为空串。过滤后的语料用于继续预训练。6.7B 的 GPT-J 经此训练（工具含计算器、问答、搜索、翻译、日历）在多个零样本任务上超过 175B 的 GPT-3，且不损害语言建模能力（NeurIPS 2023 Oral）。

**路线二：真实 API + LLM 搜索标注（ToolLLM, 2023）。** 从 RapidAPI Hub 收集 16,464 个真实 RESTful API（49 个类别），用 ChatGPT 生成单工具/多工具指令，再为指令标注解路径。关键技术 DFSDT（深度优先搜索决策树）：相比 ReAct 的单链生成，允许回溯与多分支扩展，显著扩大搜索空间、提高复杂指令的标注成功率。微调出的 ToolLLaMA 性能可比 ChatGPT，并能零样本泛化到未见过的 API（ICLR 2024 Spotlight）。

![ToolLLM 全流程：API 收集 → 指令生成 → 解路径标注，再训练 ToolLLaMA 并用 ToolEval 评估](/papers/tool-use/toolllm-pipeline.png)

> 图源：Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*, arXiv:2307.16789（用于学习注解，版权归原作者）

**路线三：执行验证合成（APIGen, Salesforce, 2024）。** 把数据正确性做成流水线属性而非抽检属性，三级分层验证：① 格式检查（JSON 可解析、字段完整）；② 真实执行（3,673 个可执行 API 实际运行）；③ 语义验证（执行结果与查询意图一致）。产出 xlam-function-calling-60k 数据集（人工抽检 600 条正确率 > 95%）。后续 **APIGen-MT**（2025）扩展到多轮：先由 LLM 评审委员会迭代生成带 ground-truth 动作序列的任务蓝图，再通过模拟人-agent 交互展开为完整轨迹；据此训练的 xLAM-2-fc-r 系列在 τ-bench 与 BFCL 多轮场景超过 GPT-4o 与 Claude 3.5。

### 偏好优化补充

SFT 之后可用 [DPO](/dpo/dpo) 类方法继续打磨：以正确调用为 $y_w$、幻觉参数或选错函数的调用为 $y_l$ 构造偏好对；"该拒绝时礼貌拒绝" vs "强行编造调用"是另一类高价值偏好对，直接对应 BFCL 的 relevance/hallucination 考点。

## 与 baseline 对比

| 维度 | Toolformer (2023) | ToolLLM (2023) | APIGen / APIGen-MT (2024/2025) |
| --- | --- | --- | --- |
| API 来源 | 5 个通用工具 | 16,464 个真实 REST API | 3,673 个可执行 API |
| 标注方式 | 模型自插入 + 困惑度过滤 | ChatGPT 生成 + DFSDT 搜索 | LLM 生成 + 三级验证 |
| 质量信号 | 后文困惑度下降 | ToolEval 自动评估 | 格式 / 执行 / 语义逐级过滤 |
| 多轮支持 | 否（调用嵌入文本） | 多步解路径 | MT 版含完整多轮人机对话 |
| 训练形式 | 继续预训练 | SFT | SFT |
| 适用场景 | 少量通用工具的无监督习得 | 海量 API 的泛化 | 高精度 function calling |

## 实现要点

```python
# 构造一条 tool-use SFT 样本（伪代码）
msgs = [
    {"role": "system",    "content": render_tools(schemas)},            # mask
    {"role": "user",      "content": x},                                # mask
    {"role": "assistant", "tool_calls": [{"name": f, "arguments": a}]}, # 算 loss
    {"role": "tool",      "content": exec_result},                      # mask！
    {"role": "assistant", "content": final_answer},                     # 算 loss
]
input_ids, labels = apply_chat_template(msgs)  # mask 位置 labels = -100
```

- **tool 轮必须 mask**：让模型学习预测工具输出既无意义（输出来自外部系统）又有害（污染生成分布），与 [Loss Masking](/sft/loss-masking) 中 mask user 轮同理。
- **schema 增广**：随机打乱工具列表顺序、变换函数/参数命名风格、混入无关工具，逼模型读 schema 而不是记位置。
- **负样本不可少**：混入"无合适工具应直接回答或反问"的样本，否则模型见到工具列表就倾向于调用。
- **并行调用支持**：BFCL 的 parallel 类目要求一次输出多个调用，模板与解析器需支持 tool_calls 数组。
- **与通用数据混训**：纯工具数据微调会损伤通用对话能力，按比例混入通用 SFT 数据（见 [数据构造](/sft/data-construction)）。

## 评测

- **BFCL**（UC Berkeley Gorilla 团队）：V1 约 2,000 条问题-函数-答案对，覆盖 simple / multiple / parallel / parallel-multiple 与 relevance detection；评估用 AST 匹配（函数名 + 参数 + 类型校验）与可执行验证两种方式。V2（2024-08）引入用户贡献的 live 数据抑制过拟合，V3（2024-09）引入多轮多步交互，V4 转向整体 agentic 评估（agentic web search、记忆管理、格式敏感性）。
- **τ-bench**（Sierra, 2024）：LM 模拟用户 + 领域 API + 政策约束的动态多轮对话（retail / airline 两域），通过比对最终数据库状态与标注目标判定成功；pass^k（k 次试验全部成功）专门衡量行为稳定性。后续 **τ²-bench**（2025）升级为 dual-control：模拟用户也能操作共享环境（新增 Telecom 域，按 Dec-POMDP 建模），agent 还需引导用户做操作，各模型性能相比单控设置显著下降。

## 调参与实践经验

- **三类典型 failure 及对策**：幻觉参数（编造对话中不存在的取值）→ 加参数可溯源校验、补抽取类数据；漏调用（该查询时直接瞎答）→ 补"必须调用"正例；过度调用（能直接回答却调用）→ 补 relevance 负例。
- **格式敏感性是真实风险**：同一模型换一种 schema 渲染方式成绩可能明显波动（BFCL V4 专设此项），训练期 schema 格式增广是性价比最高的缓解手段。
- **看 pass^k 而非只看 pass@1**：线上 agent 要求"每次都对"，τ-bench 的结论是单次成功率与多次全对率之间存在巨大落差，后者才反映可部署性。
- **质量重于数量**：60k 级别的执行验证数据（APIGen）足以让 7B 模型达到一线水平；可执行验证比堆量更重要。
- 单步调用能力打牢后，多轮长程任务的进一步提升通常要靠 [Agentic RL](/agent/agentic-rl/)。

## 参考文献

- Schick et al., 2023. *Toolformer: Language Models Can Teach Themselves to Use Tools.* arXiv:2302.04761
- Qin et al., 2023. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.* arXiv:2307.16789
- Liu et al., 2024. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets.* arXiv:2406.18518
- Salesforce AI Research, 2025. *APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay.* arXiv:2504.03601
- Patil et al., 2025. *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models.* ICML 2025
- Yao et al., 2024. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* arXiv:2406.12045
- Barres et al., 2025. *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment.* arXiv:2506.07982
- Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629
