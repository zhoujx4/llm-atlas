---
title: 全量微调
---

# 全量微调（Full Fine-Tuning）

> **一句话**：更新模型全部参数的标准 SFT 做法——效果上限最高、对数据最敏感，但显存开销也最大（Adam 下约为参数量的 16 倍字节）。论文：*Training Language Models to Follow Instructions with Human Feedback*（InstructGPT，2022）。
> 代表工作年份：2022（InstructGPT）· 机构/团队：OpenAI · 会议/来源：arXiv:2203.02155（全量微调为通用工程实践，此处以确立后训练范式的标志性工作为锚）
>
> 前置阅读：[SFT 总览](/sft/)、[符号约定](/guide/notation)；对照 [LoRA](/lora/lora)

## 直觉与动机

预训练模型是个 next-token 续写器：给它 "解释一下快速排序" 它可能接着补一道考题，而不是直接讲解。它具备讲解快排所需的全部知识，缺的是"接到指令就以助手身份作答并适时停止"这个行为。全量微调（Full Fine-Tuning，FFT）就是在「指令 → 期望回答」数据上更新模型的**每一个参数**，把输出分布整体搬到"按指令对话"的子空间。

为什么用全量而不是 [PEFT](/lora/lora)？取舍框架大致如下：

- **效果上限**：全量微调能改动所有权重，表达能力最强，是注入大规模新能力（如把通用模型转成强代码/数学模型）时的首选。LoRA 等低秩方法在"风格对齐"类任务上往往能逼近全量，但在需要大幅改变模型行为、注入大量新知识时通常略逊。
- **显存与算力**：全量微调要为每个参数保存梯度和优化器状态，显存是最大瓶颈；LoRA 把可训练参数压到 <1%，单卡即可微调中等模型。
- **部署与多任务**：全量微调每个任务产出一整套权重；LoRA 产出几十 MB 的 adapter，可热插拔、可多任务共享底座。

经验法则：**有充足算力、追求效果上限、要做大规模能力注入或要持续迭代基座模型 → 全量微调；算力受限、任务多、以风格/领域适配为主 → 优先 [LoRA/QLoRA](/lora/qlora)。**

## 方法与公式

训练目标是标准自回归 NLL，只在回答 token 上计 loss：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

形式上与预训练相同，区别全在工程细节：

- **Loss mask**：prompt $x$ 的 token 进 attention 但不算 loss，否则模型会去学"生成问题"而非"回答问题"。多轮对话里只对 assistant 轮算 loss，见 [Loss Masking](/sft/loss-masking)。
- **数据格式**：用 [Chat Template](/sft/chat-template) 把多轮对话拼成带 special token 的单一序列，再做 [Packing](/sft/packing) 提升吞吐。
- **学习率**：远小于预训练，典型 $1\text{e-}5 \sim 2\text{e-}5$，因为我们只想微调而非重塑。
- **epoch 数**：通常 1~3 遍，远少于预训练的"海量数据单遍"，过多 epoch 极易过拟合与遗忘。

**显存构成**是全量微调的核心约束。以参数量 $N$、混合精度 + Adam 优化器为例，每个参数大致需要：

| 组成 | 精度 | 字节/参数 |
| --- | --- | --- |
| 参数（bf16 副本） | bf16 | 2 |
| 梯度 | bf16 | 2 |
| Adam 一阶动量 $m$ | fp32 | 4 |
| Adam 二阶动量 $v$ | fp32 | 4 |
| 参数 fp32 主副本 | fp32 | 4 |

合计约 **16 字节/参数**，即一个 7B 模型仅"状态"就需约 112 GB——远超单张 80GB 卡，所以全量微调几乎总要配合 ZeRO / FSDP 把这些状态切分到多卡。这还没算上激活值（activations），后者由 batch 大小、序列长度和是否开启 gradient checkpointing 决定。

## 与 baseline 对比

| 维度 | 全量微调 | LoRA |
| --- | --- | --- |
| 可训练参数 | 100% | 通常 < 1% |
| 优化器状态显存 | 约 12 字节/全部参数 | 仅 adapter 参数 |
| 总显存（7B 级） | 数张 80GB 卡 + ZeRO/FSDP | 常可单卡 |
| 效果上限 | 最高 | 接近，能力注入类略逊 |
| 多任务切换 | 需整套权重 | 换 adapter 即可 |
| 灾难性遗忘风险 | 较高 | 较低（底座冻结） |
| 训练速度 | 慢（更新全部参数） | 快 |

与**继续预训练（continual pretraining）**的区别也值得厘清：继续预训练用自由文本、不加 loss mask、学习率介于预训练与 SFT 之间，目的是注入领域知识或扩展语种；全量 SFT 用结构化对话、加 loss mask、学习率更小，目的是塑造指令遵循行为。工程上常见"先继续预训练补知识，再 SFT 学对话"的两段式。

## 实现要点

最小训练循环（含 loss masking 的核心思路）：

```python
for batch in dataloader:
    # input_ids: [B, T] 已用 chat template 拼好并 packing
    # labels:    [B, T] prompt / 非 assistant 段已置为 -100（忽略）
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
    # CrossEntropy 默认 ignore_index=-100，自动跳过被 mask 的位置
    loss = cross_entropy(out.logits[:, :-1].flatten(0, 1),
                         batch["labels"][:, 1:].flatten())
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

- **labels 的构造**：把不参与 loss 的位置（prompt、system、padding、上一轮 assistant 之外的所有内容按策略）设为 `-100`，交由 `CrossEntropyLoss(ignore_index=-100)` 处理。这是全量 SFT 最容易写错、且静默生效的地方——错了 loss 照样下降，但学到的行为是错的。
- **分布式策略**：单机多卡或模型放得下时用 ZeRO-2/FSDP（切梯度+优化器状态）；放不下时上 ZeRO-3（连参数也切）或张量/流水并行。bf16 优于 fp16，数值更稳、无需 loss scaling。
- **gradient checkpointing**：用计算换显存，激活值显存大幅下降，代价是多一次前向，长序列训练几乎必开。
- **现成实现**：HF TRL 的 `SFTTrainer`、LLaMA-Factory、axolotl、Megatron-LM 等都内置了模板拼接、packing、loss mask 与分布式封装。

## 调参与实践经验

- **学习率**：$1\text{e-}5 \sim 2\text{e-}5$ 是常见区间；模型越大、数据越少越要往小取。lr 偏大最典型的症状是灾难性遗忘和复读。
- **warmup + 衰减**：3%~10% 步数线性 warmup，随后 cosine 衰减到 0 或一个小值，是稳妥默认。
- **epoch**：1~3 遍。盯住验证集 loss——回答类任务 loss 触底回升即过拟合信号，宁可早停。多 epoch 反复看同一批数据极易让模型背诵而非泛化。
- **batch size**：用梯度累积凑到有效 batch 数十万 token 量级；过小 batch 噪声大、过大可能损害泛化。
- **序列长度与 packing**：开 [Packing](/sft/packing) 把短样本拼满上下文窗口，吞吐可显著提升，但务必用 attention mask 隔离不同样本，否则会跨样本"串味"。
- **数据质量 >> 超参**：在 SFT 阶段，把精力投在 [数据构造](/sft/data-construction) 上的回报远高于反复调超参。脏数据、格式不一致、答案错误带来的损害是任何超参都救不回来的。
- **NEFTune 等小技巧**：训练时给 embedding 加噪声有时能小幅提升对话质量，可作为低成本尝试，但收益因数据而异，不应作为主要手段。

## 参考文献

- Ouyang et al., 2022. *Training Language Models to Follow Instructions with Human Feedback.* arXiv:2203.02155（InstructGPT）
- Zhou et al., 2023. *LIMA: Less Is More for Alignment.* arXiv:2305.11206
- Rajbhandari et al., 2020. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* arXiv:1910.02054
- Jain et al., 2023. *NEFTune: Noisy Embeddings Improve Instruction Finetuning.* arXiv:2310.05914
