---
title: rsLoRA
---

# rsLoRA（Rank-Stabilized LoRA）

> **一句话**：TODO —— 把 LoRA 的缩放因子从 $\alpha/r$ 改成 $\alpha/\sqrt{r}$，使高 rank 时梯度尺度稳定、收益不再饱和。
>
> 论文：*A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (2023) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 原版 $\alpha/r$ 缩放在大 rank 下导致更新量过小、学习停滞
- [ ] 从初始化理论推导合适的尺度

## 2. 方法与公式

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} B A x
$$

TODO：

- [ ] 推导：保证 $\Delta W x$ 的方差不随 $r$ 衰减
- [ ] 只改一行代码的变体

## 3. 与 baseline 对比

| 维度 | LoRA ($\alpha/r$) | rsLoRA ($\alpha/\sqrt{r}$) |
| --- | --- | --- |
| 小 rank (8~16) | 基本无差异 | 基本无差异 |
| 大 rank (≥64) | 收益饱和 | TODO：持续提升 |
| 实现成本 | — | 一行 |

## 4. 实现要点

- [ ] peft 中 `use_rslora=True`
- [ ] 换缩放后 $\alpha$ 与 lr 需要重新标定

## 5. 实验与调参经验

TODO：什么时候值得开大 rank + rsLoRA。

## 6. 参考文献

- [ ] Kalajdzievski, 2023. *rsLoRA.* arXiv:2312.03732
