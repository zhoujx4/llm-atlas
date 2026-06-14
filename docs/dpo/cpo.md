---
title: CPO
---

# CPO（Contrastive Preference Optimization）

> **一句话**：用「均匀分布先验」近似 reference model，得到 DPO 损失的一个上界从而去掉 $\pi_{\text{ref}}$，再补一个 SFT 正则项防止 chosen 概率塌缩；为机器翻译提出，方法本身通用。
>
> 论文：*Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation* (Xu et al., 2024) ·
>
> 提出年份：2024 · 机构/团队：JHU / Microsoft · 会议/来源：ICML 2024 / arXiv:2401.08417
>
> 前置阅读：[DPO](/dpo/dpo)、[SFT 总览](/sft/)

## 直觉与动机

CPO 诞生于机器翻译场景，但其动机对所有偏好优化都成立。

**第一，reference model 既费资源又设上限。** [DPO](/dpo/dpo) 用 $\pi_{\text{ref}}$ 作 KL 锚点，代价是多一份显存、多一次前向。更微妙的是，KL 项把 $\pi_\theta$ 往 $\pi_{\text{ref}}$ 拉，当 reference（通常是 SFT 模型）本身就接近任务上限时，这种「拉回」反而限制了模型超越 reference 的空间。在翻译这种「参考译文已经很强、但仍有提升余地」的任务里，这个上限尤其碍事。

**第二，翻译里的偏好是「好 vs. 更好」的细粒度对比。** 标准做法是用高质量参考译文做 SFT，但参考译文也可能有瑕疵；真正需要的是在多个都不错的候选里学出更精细的偏好。这要求一个能做细粒度对比、又不被 reference 束缚的目标。

CPO 的两步走：(1) 用一个均匀分布近似 $\pi_{\text{ref}}$，从数学上把 DPO 损失放缩成一个**不含 reference** 的上界，于是 reference model 直接消失；(2) 注意到去掉 KL 锚点后 chosen 的绝对概率可能塌缩，补一个 SFT（NLL）项把 chosen 似然顶住。

## 方法与公式

CPO 的损失由对比项和 SFT 项相加：

$$
\mathcal{L}_{\text{CPO}} = \underbrace{-\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\big(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x)\big)\right]}_{\mathcal{L}_{\text{prefer}}} \;\; \underbrace{-\;\mathbb{E}_{(x,y_w)}\!\left[\log \pi_\theta(y_w|x)\right]}_{\mathcal{L}_{\text{SFT}}}
$$

**从 DPO 推导上界（uniform reference 假设）**：在 DPO 的隐式 reward $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 中，若把 $\pi_{\text{ref}}$ 取为均匀分布（对所有 $y$ 是常数），则 reward 差里的 $-\beta\log\pi_{\text{ref}}(y_w)+\beta\log\pi_{\text{ref}}(y_l)$ 退化为常数，整条 reference 项消失，对比项化简为只含 $\pi_\theta$ 的形式。作者进一步论证这是真实 DPO 目标的一个上界（近似），因此最小化 $\mathcal{L}_{\text{prefer}}$ 是在优化 DPO 目标的代理。

**SFT 项防塌缩**：去掉 reference 后，对比项只关心 $y_w$ 与 $y_l$ 的**相对**对数概率——把两者一起压低也能减小损失，导致 chosen 绝对概率塌缩、生成质量下降。$\mathcal{L}_{\text{SFT}}=-\log\pi_\theta(y_w|x)$ 强制 chosen 的绝对似然保持高位，充当锚点，扮演了 DPO 里 KL 项的部分角色。

![CPO 的偏好三元组：由 reference-free 评估模型为候选译文打分，据此选出 preferred 与 dis-preferred 构造对比样本](/papers/cpo/x3.png)

> 图源：Xu et al., *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation*, arXiv:2401.08417（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | DPO | CPO | SimPO |
| --- | --- | --- | --- |
| Reference model | 需要 | 不需要 | 不需要 |
| 去 ref 的依据 | —— | 均匀先验 → 损失上界 | 直接换 reward 定义 |
| 防塌缩手段 | 隐式 KL | SFT 项 | margin $\gamma$ |
| 长度归一化 | 无 | 无 | 有 |
| reward 量纲 | 序列求和 logprob | 序列求和 logprob | per-token 平均 logprob |
| 显存 / 计算 | 高 | 低 | 低 |

## 实现要点

CPO 在 TRL 中由 `CPOTrainer` 提供（它同时承载 [SimPO](/dpo/simpo) 与 CPO-SimPO 混合）。核心计算：

```python
# logps_w, logps_l: 序列求和 logprob, 形状 [B]; nll_w: chosen 的 NLL
prefer = -F.logsigmoid(beta * (logps_w - logps_l)).mean()
sft    = nll_w.mean()                  # = -E[log pi(y_w|x)]
loss   = prefer + cpo_alpha * sft
```

关键细节：

- **无 reference 前向**：与 SimPO 一样省掉一份模型和一次前向，这是 CPO 相对 DPO 的算力收益来源。
- **`cpo_alpha`** 控制 SFT 项权重，TRL 默认 1.0；置 0 即退化为纯对比项（容易塌缩，不建议）。
- **`loss_type`**：`"sigmoid"` 为标准 CPO；切到 `"simpo"` 则把对比项换成长度归一化形式，可与 `cpo_alpha` 组合成 CPO-SimPO 混合目标。
- **logprob 按序列求和**，CPO 标准形式不做长度归一化（与 SimPO 的区别）。

## 调参与实践经验

- **$\beta$**：对比项里 reward 是序列求和量纲，$\beta$ 取值与 DPO 接近（0.05~0.5），而非 SimPO 那种较大的 $\beta$。
- **`cpo_alpha`（SFT 项权重）**：默认 1.0 多数情况可用。偏小会让 chosen 概率塌缩、生成退化；偏大则过强地拟合 chosen、削弱对比信号。可在 0.5~1.0 间扫。
- **与 SimPO 的实测取舍**：两者都去 reference，但 CPO 用 SFT 项锚定绝对似然、SimPO 用长度归一化 + margin 对齐生成度量。在易出现长度膨胀的任务上 SimPO 往往更稳；在希望 chosen 似然被牢牢顶住、对长度不敏感的任务（如翻译）上 CPO 更直接。两者的混合（CPO-SimPO）在部分公开评测中表现不俗，值得作为对比实验。
- **数据要求**：CPO 的对比项无 KL 缰绳，对偏好数据质量与分布契合度敏感；建议在与基座分布接近的高质量数据上训，并以较少 epoch 配合留出集监控。

## 参考文献

- Xu et al., 2024. *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation.* arXiv:2401.08417
- Rafailov et al., 2023. *Direct Preference Optimization.* arXiv:2305.18290
