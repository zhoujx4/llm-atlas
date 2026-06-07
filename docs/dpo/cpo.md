---
title: CPO
---

# CPO（Contrastive Preference Optimization）

> **一句话**：TODO —— 用均匀先验近似 reference model 得到 DPO 损失的上界，再加 SFT 正则项；为机器翻译场景提出，方法本身通用。
>
> 论文：*Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation* (2024) ·
> 前置阅读：[DPO](/dpo/dpo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] reference model 既费显存又限制模型超越 reference
- [ ] 翻译场景：好与更好之间的细粒度偏好

## 2. 方法与公式

$$
\mathcal{L}_{\text{CPO}} = \underbrace{-\mathbb{E}\left[\log \sigma\left(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x)\right)\right]}_{\mathcal{L}_{\text{prefer}}} \; \underbrace{- \;\mathbb{E}\left[\log \pi_\theta(y_w|x)\right]}_{\mathcal{L}_{\text{SFT}}}
$$

TODO：

- [ ] 从 DPO 损失推导上界（uniform reference 假设）
- [ ] SFT 项防止 chosen 概率塌缩

## 3. 与 baseline 对比

| 维度 | DPO | CPO | SimPO |
| --- | --- | --- | --- |
| Reference model | 需要 | 不需要 | 不需要 |
| 防塌缩手段 | 隐式 KL | SFT 项 | margin $\gamma$ |
| 长度归一化 | 无 | 无 | 有 |

## 4. 实现要点与伪代码

```python
# TODO: CPO loss 伪代码
```

## 5. 实验与调参经验

TODO：SFT 项权重；与 SimPO 的实测对比。

## 6. 参考文献

- [ ] Xu et al., 2024. *CPO.* arXiv:2401.08417
