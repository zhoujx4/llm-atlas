---
title: DoRA
---

# DoRA（Weight-Decomposed Low-Rank Adaptation）

> **一句话**：TODO —— 把权重分解为「幅值 × 方向」，幅值直接训练、方向用 LoRA 更新，行为更接近全量微调。
>
> 论文：*DoRA: Weight-Decomposed Low-Rank Adaptation* (2024) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 对 LoRA 与全量微调更新模式的分析：幅值与方向变化的相关性不同
- [ ] 解耦后让 LoRA 的学习行为更像全量微调

## 2. 方法与公式

$$
W = m \cdot \frac{W_0 + BA}{\left\lVert W_0 + BA \right\rVert_c}
$$

其中 $m$ 为可训练的幅值向量，$\lVert \cdot \rVert_c$ 表示按列的 L2 范数。

TODO：

- [ ] $m$ 初始化为 $\lVert W_0 \rVert_c$
- [ ] 梯度流与显存开销分析

## 3. 与 baseline 对比

| 维度 | LoRA | DoRA |
| --- | --- | --- |
| 额外可训练参数 | $BA$ | $BA + m$ |
| 训练开销 | 低 | 略高（范数计算） |
| 低 rank 下效果 | TODO | 论文称更优 |

## 4. 实现要点与伪代码

```python
# TODO: DoRA 前向伪代码
```

## 5. 实验与调参经验

TODO：何时优于 LoRA；推理合并注意事项。

## 6. 参考文献

- [ ] Liu et al., 2024. *DoRA.* arXiv:2402.09353
