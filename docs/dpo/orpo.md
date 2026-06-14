---
title: ORPO
---

# ORPO（Odds Ratio Preference Optimization）

> **一句话**：把偏好优化并进 SFT——在最大化 chosen 似然的 NLL 损失上，叠加一个基于 odds ratio 的偏好惩罚项，单阶段同时完成「学会回答」和「对齐偏好」，无需 reference model。
>
> 论文：*ORPO: Monolithic Preference Optimization without Reference Model* (Hong et al., 2024) ·
>
> 提出年份：2024 · 机构/团队：KAIST AI · 会议/来源：EMNLP 2024 / arXiv:2403.07691
>
> 前置阅读：[DPO](/dpo/dpo)、[SFT 总览](/sft/)

## 直觉与动机

主流对齐流水线是 **SFT → 偏好优化** 两阶段：先用 SFT 让模型学会任务格式与基本能力，再用 [DPO](/dpo/dpo) 之类的方法在偏好数据上对齐。ORPO 指出这套流程有两处可以合并的冗余。

**第一，SFT 阶段会无意中抬高「坏回答」的概率。** SFT 的目标是最大化目标回答的似然，但它对「与目标风格相近、却属于不该输出的回答」没有任何抑制。作者通过实验观察到：随着 SFT 进行，模型在被拒绝风格的回答上的对数概率也在同步上升。换句话说，SFT 把整片相关的输出分布都抬高了，包括我们想压下去的部分——这正是后续需要一个独立偏好阶段来「纠偏」的原因之一。

**第二，两阶段意味着两份数据流程、两次训练、（DPO 还要）一份 reference model。** 如果能在 SFT 的同时就对 rejected 施加压制，就能省掉独立的偏好阶段和 reference model。

ORPO 的做法：在标准 SFT 的 NLL 损失之外，加一个**弱**的 odds ratio 惩罚项，让模型在抬高 chosen 似然的同时，相对压低 rejected 的 odds。整个训练只有一个阶段、一个模型。

## 方法与公式

ORPO 的损失由 SFT 项和 odds ratio 项相加构成：

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

其中 $\mathcal{L}_{\text{SFT}}$ 是对 chosen 回答 $y_w$ 的标准 token 级 NLL（交叉熵）损失。偏好项定义为：

$$
\mathcal{L}_{\text{OR}} = -\log \sigma\!\left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right),
\qquad
\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}
$$

这里 $P_\theta(y|x)$ 是长度归一化后的序列概率，即 $\exp\!\big(\frac{1}{|y|}\sum_t \log \pi_\theta(y_t|x,y_{<t})\big)$，避免长短回答间的量纲不可比。

**为什么用 odds ratio 而不是概率比？** 关键在梯度的「温和」程度。可以证明 $\mathcal{L}_{\text{OR}}$ 的梯度中带有因子 $\big(1+\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\big)^{-1}$：当 chosen 的 odds 已经远大于 rejected 时，该因子趋近 0，惩罚自动减弱。这使得 odds ratio 项不会过度压制 rejected——而概率比（如 log-likelihood ratio）在确定性偏好下会无界放大（这正是 [IPO](/dpo/ipo) 所诊断的 DPO 问题）。odds ratio 提供了一种自带「软上限」的对比信号，恰好适合与 SFT 项共存而不互相打架。

**SFT 项的作用**：保证模型持续学习 chosen 的内容与格式，是主信号；odds ratio 项只做温和的相对区分，所以 $\lambda$ 通常取得很小。

![ORPO 与 RLHF / DPO 等对齐范式的对比：ORPO 把偏好优化并入单阶段 SFT，无需 reference model](/papers/orpo/x2.png)

> 图源：Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | SFT + DPO | ORPO |
| --- | --- | --- |
| 训练阶段 | 两阶段 | 单阶段 |
| Reference model | 需要 | 不需要 |
| 数据 | SFT 数据 + 偏好数据 | 偏好数据（chosen 兼作 SFT 目标） |
| 起点 | 通常需先有 SFT 模型 | 可直接从基座开始 |
| 对比信号 | logistic / log-ratio | odds ratio（自带软上限） |
| 显存 / 计算 | 高（两轮 + ref） | 低（一轮，一个模型） |

## 实现要点

ORPO 在 TRL 中由 `ORPOTrainer` 提供。一个 batch 的核心计算：

```python
# logps_w, logps_l: 长度归一化的 (平均) logprob, 形状 [B]
# nll_loss:         对 chosen 回答 token 级交叉熵 (即 SFT 损失)

# log-odds = log p - log(1 - p) = logp - log(1 - exp(logp))
log_odds_w = logps_w - torch.log1p(-torch.exp(logps_w))
log_odds_l = logps_l - torch.log1p(-torch.exp(logps_l))
or_loss = -F.logsigmoid(log_odds_w - log_odds_l).mean()

loss = nll_loss + lam * or_loss
```

关键细节：

- **同一次前向同时算两件事**：chosen 的 NLL 直接复用其 logprob，无需第二个模型，这是 ORPO「monolithic（单体）」的含义。
- **数值稳定**：$\log(1-\exp(\text{logp}))$ 在 logp 接近 0 时不稳定，需用 `log1p` 与 `expm1` 等稳定写法（TRL 内部已处理）。
- **长度归一化的概率**用于 odds 计算，避免长回答因 token 多而概率天然偏低。

## 调参与实践经验

- **$\lambda$**：常见 0.1~1.0 量级，论文默认 0.1。它是 odds ratio 项相对 SFT 项的权重。$\lambda$ 太大，偏好惩罚盖过学习信号，可能损害基础能力；太小则几乎退化为纯 SFT，区分不出好坏。
- **可从基座直接训**：ORPO 的一大卖点是把 SFT 和对齐合一，因此可以跳过独立 SFT、直接在偏好数据（chosen 当作 SFT 目标）上从基座起训，适合数据量适中、想省流程的场景。但若任务格式与基座差异很大，先做一轮轻量 SFT 再 ORPO 往往更稳。
- **学习率与 epoch**：因为同时承担 SFT 职责，学习率可参考 SFT 设置；epoch 通常 1~3，过多易过拟合 chosen 风格。
- **何时选 ORPO**：流程简化、显存敏感、且偏好数据的 chosen 本身质量足以当 SFT 目标时，ORPO 是优雅之选。若已有高质量 SFT 模型、且追求对齐效果上限，分阶段 DPO/RL 仍有优势。

## 参考文献

- Hong et al., 2024. *ORPO: Monolithic Preference Optimization without Reference Model.* arXiv:2403.07691
- Rafailov et al., 2023. *Direct Preference Optimization.* arXiv:2305.18290
