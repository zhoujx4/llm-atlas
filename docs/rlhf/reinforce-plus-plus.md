---
title: REINFORCE++
---

# REINFORCE++

> **一句话**：TODO —— 在 REINFORCE 上叠加 PPO 的稳定化技巧（token 级 KL、clip、全局批次优势归一化），不需要 critic 也不需要组采样。
>
> 论文/报告：*REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models* (2025) ·
> 前置阅读：[PPO](/rlhf/ppo)、[RLOO](/rlhf/rloo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] GRPO 组采样的 prompt 内基线可能引入偏差 / 浪费采样预算
- [ ] 用全局 batch 归一化替代组内基线

## 2. 方法与公式

token 级 reward 整合 KL 惩罚后，优势做全局批次归一化：

$$
\hat{A}_t = \frac{A_t - \mu_{\text{batch}}}{\sigma_{\text{batch}}}
$$

TODO：

- [ ] token 级 KL 惩罚的塑形方式
- [ ] PPO-clip 的保留
- [ ] 与 GRPO 的稳定性对比论点

## 3. 与 baseline 对比

| 维度 | GRPO | RLOO | REINFORCE++ |
| --- | --- | --- | --- |
| 每 prompt 采样数 | $G$ | $k$ | 1 即可 |
| 基线 | 组内 | 留一 | 全局 batch |
| Critic | 否 | 否 | 否 |

## 4. 实现要点与伪代码

```python
# TODO: 伪代码
```

## 5. 实验与调参经验

TODO：OpenRLHF 中的实现与默认超参。

## 6. 参考文献

- [ ] Hu, 2025. *REINFORCE++.* arXiv:2501.03262
