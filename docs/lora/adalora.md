---
title: AdaLoRA
---

# AdaLoRA（Adaptive Budget Allocation）

> **一句话**：TODO —— 不同层/模块对秩的需求不同；AdaLoRA 用 SVD 形式参数化增量并按重要性动态分配秩预算。
>
> 论文：*Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning* (2023) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] LoRA 给所有模块同样的 rank 是浪费
- [ ] 重要性度量与剪枝式秩分配

## 2. 方法与公式

SVD 形式的参数化：

$$
\Delta W = P \Lambda Q, \quad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_r)
$$

TODO：

- [ ] 正交正则 $\lVert P^\top P - I \rVert_F^2 + \lVert Q Q^\top - I \rVert_F^2$
- [ ] 基于敏感度的重要性打分，训练中逐步裁剪 $\lambda_i$
- [ ] 秩预算调度（先宽后窄）

## 3. 与 baseline 对比

| 维度 | LoRA | AdaLoRA |
| --- | --- | --- |
| 秩分配 | 全局固定 | 按模块自适应 |
| 训练复杂度 | 低 | 较高（打分+调度） |
| 同预算效果 | TODO | 论文称更优 |

## 4. 实现要点与伪代码

```python
# TODO: 重要性打分与秩裁剪调度伪代码
```

## 5. 实验与调参经验

TODO：预算调度超参；现今使用度（工程上常被简单加大 rank 替代）。

## 6. 参考文献

- [ ] Zhang et al., 2023. *AdaLoRA.* arXiv:2303.10512
