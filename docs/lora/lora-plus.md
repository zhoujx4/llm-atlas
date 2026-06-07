---
title: LoRA+
---

# LoRA+

> **一句话**：TODO —— 给 $B$ 矩阵设置比 $A$ 大得多的学习率（比例 $\lambda \gg 1$），修正 LoRA 在宽网络下的次优学习动态。
>
> 论文：*LoRA+: Efficient Low Rank Adaptation of Large Models* (2024) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 无穷宽极限分析：$A$、$B$ 的梯度尺度天然不对称
- [ ] 同一学习率导致特征学习效率低

## 2. 方法与公式

$$
\eta_B = \lambda \cdot \eta_A, \quad \lambda \gg 1 \;(\text{论文建议 } \lambda \approx 16)
$$

TODO：

- [ ] 推导要点（μP 风格的尺度分析）
- [ ] 与 $B$ 零初始化的关系

## 3. 与 baseline 对比

| 维度 | LoRA | LoRA+ |
| --- | --- | --- |
| 改动 | — | 仅优化器分组学习率 |
| 收敛速度 | 基线 | 论文称约 2× |
| 最终效果 | 基线 | TODO：小幅提升 |

## 4. 实现要点与伪代码

```python
# TODO: optimizer param group 设置 B 的 lr = λ * lr 伪代码
```

## 5. 实验与调参经验

TODO：$\lambda$ 的敏感性；与 rsLoRA 是否可叠加。

## 6. 参考文献

- [ ] Hayou et al., 2024. *LoRA+.* arXiv:2402.12354
