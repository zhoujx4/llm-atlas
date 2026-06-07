---
title: RLOO
---

# RLOO（REINFORCE Leave-One-Out）

> **一句话**：TODO —— 回归 REINFORCE：对每个 prompt 采样 $k$ 个回答，每个回答用「其余 $k{-}1$ 个的平均 reward」做基线，无偏且无需 critic。
>
> 论文：*Back to Basics: Revisiting REINFORCE-Style Optimization for RLHF* (2024) ·
> 前置阅读：[PPO](/rlhf/ppo)、[GRPO](/rlhf/grpo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] LLM-RLHF 的初始策略已经很好，PPO 的许多机制（clip、GAE、critic）可能是不必要的复杂度
- [ ] 留一法基线：无偏方差缩减

## 2. 方法与公式

$$
\nabla_\theta \mathcal{J} = \frac{1}{k} \sum_{i=1}^{k} \left( r(x, y_i) - \frac{1}{k-1}\sum_{j \neq i} r(x, y_j) \right) \nabla_\theta \log \pi_\theta(y_i | x)
$$

TODO：

- [ ] 把整个回答当作单一 action（序列级，无 token 级 credit assignment）
- [ ] 与 GRPO 的差异：基线是留一均值而非全组均值/std；通常单步更新（on-policy）无 clip

## 3. 与 baseline 对比

| 维度 | PPO | GRPO | RLOO |
| --- | --- | --- | --- |
| Critic | 需要 | 不需要 | 不需要 |
| 基线 | value 网络 | 组均值（含自身）/std | 留一均值 |
| 无偏性 | — | std 缩放有偏 | 无偏 |
| clip / off-policy | 有 | 有 | 通常无 |

## 4. 实现要点与伪代码

```python
# TODO: RLOO 梯度估计伪代码
```

## 5. 实验与调参经验

TODO：$k$ 的选择（2~8）；与 PPO 的效果/成本实测对比。

## 6. 参考文献

- [ ] Ahmadian et al., 2024. *Back to Basics.* arXiv:2402.14740
