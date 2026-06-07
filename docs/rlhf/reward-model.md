---
title: Reward Model
---

# Reward Model

> **一句话**：TODO —— 在偏好数据上训练打分模型，作为 RL 阶段的奖励来源；质量直接决定 RLHF 上限。
>
> 前置阅读：[RLHF 总览](/rlhf/)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 为什么需要 RM：人类偏好不可在线查询
- [ ] RM 的结构：LM backbone + 标量 value head

## 2. 方法与公式

Bradley-Terry 损失：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

TODO：

- [ ] 取哪个位置的 hidden state（最后一个 token）
- [ ] margin 变体、多目标 RM（helpfulness/safety 分头）
- [ ] Process Reward Model（PRM）vs Outcome Reward Model（ORM）

## 3. 常见问题

TODO：

- [ ] Reward hacking：policy 找到 RM 的盲区
- [ ] 长度偏置：RM 偏好更长回答
- [ ] OOD：policy 分布漂移后 RM 打分失真

## 4. 实现要点与伪代码

```python
# TODO: BT loss 训练伪代码
```

## 5. 实验与调参经验

TODO：RM 准确率与下游 RLHF 效果的关系；数据量级经验。

## 6. 参考文献

- [ ] Ouyang et al., 2022. *InstructGPT*
- [ ] Lightman et al., 2023. *Let's Verify Step by Step*（PRM）
