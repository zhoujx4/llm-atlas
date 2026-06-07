---
title: SimPO
---

# SimPO（Simple Preference Optimization）

> **一句话**：TODO —— 去掉 reference model，用「长度归一化的平均 logprob」作隐式 reward 并加目标 margin，更简单且缓解长度偏置。
>
> 论文：*SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024) ·
> 前置阅读：[DPO](/dpo/dpo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] DPO 的隐式 reward 与生成时实际使用的度量（平均 logprob）不一致
- [ ] reference model 的显存与计算开销
- [ ] 长度偏置：序列求和的 logprob 天然偏向短回答 / 引发长度膨胀

## 2. 方法与公式

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]
$$

TODO：

- [ ] 长度归一化 $\frac{\beta}{|y|}$ 的作用
- [ ] 目标 margin $\gamma$ 的作用
- [ ] 与生成度量对齐的论证

## 3. 与 baseline 对比

| 维度 | DPO | SimPO |
| --- | --- | --- |
| Reference model | 需要 | 不需要 |
| 隐式 reward | 求和 logprob 比值 | 长度归一化平均 logprob |
| 显存开销 | 高（两个模型） | 低 |
| KL 约束 | 有（隐式） | 无（更易跑飞） |

## 4. 实现要点与伪代码

```python
# TODO: 一个 batch 的 SimPO loss 计算
```

- [ ] 长度用有效 token 数（去 padding / 去 prompt）
- [ ] 与 TRL `CPOTrainer(loss_type="simpo")` 的对应

## 5. 实验与调参经验

TODO：$\beta$（2~2.5）、$\gamma$（0.5~1.5）常见取值；无 KL 约束带来的稳定性风险。

## 6. 参考文献

- [ ] Meng et al., 2024. *SimPO.* arXiv:2405.14734
