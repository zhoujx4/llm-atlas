---
title: DPO
---

# DPO（Direct Preference Optimization）

> **一句话**：TODO —— 把 RLHF 的 KL 约束目标求出闭式解，将"训 RM + 跑 PPO"两步压缩成一个分类损失。
>
> 论文：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) ·
> 前置阅读：[RLHF 总览](/rlhf/)、[Reward Model](/rlhf/reward-model)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] RLHF 流程复杂、不稳定、显存吃紧（4 个模型）
- [ ] 关键洞察：最优策略与 reward 之间存在闭式映射，"语言模型本身就是隐式 reward model"

## 2. 方法与公式

从 KL 约束的 RLHF 目标出发，最优策略满足 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$，代入 Bradley-Terry 模型得：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

TODO：

- [ ] 完整推导（KL 约束目标 → 闭式解 → 代入 BT）
- [ ] $\beta$ 的含义：偏离 reference 的惩罚强度
- [ ] 隐式 reward $\hat{r} = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ 的解读

## 3. 与 baseline 对比

| 维度 | RLHF (PPO) | DPO |
| --- | --- | --- |
| 训练阶段 | RM + RL 两阶段 | 单阶段 |
| 同时驻留模型数 | 4 | 2 |
| 在线采样 | 需要 | 不需要（离线） |
| 效果上限 | TODO | TODO |

## 4. 实现要点与伪代码

```python
# TODO: 一个 batch 的 DPO loss 计算（policy/ref 两次前向, 按 token 求和 logprob）
```

- [ ] logprob 按序列求和（不要平均，这是与 SimPO 的关键差异）
- [ ] reference model 的处理：冻结副本 / 预计算 logprob
- [ ] 与 TRL `DPOTrainer` 的对应关系

## 5. 实验与调参经验

TODO：

- [ ] $\beta$ 常见取值 0.05~0.5
- [ ] 经典现象：chosen 与 rejected 的 logprob 同时下降
- [ ] 先 SFT 再 DPO 的必要性

## 6. 参考文献

- [ ] Rafailov et al., 2023. *DPO.* arXiv:2305.18290
