---
title: PPO
---

# PPO（Proximal Policy Optimization）

> **一句话**：TODO —— 用裁剪的重要性采样比值限制每步策略更新幅度；RLHF 的经典算法，稳定但工程复杂（需 4 个模型）。
>
> 论文：*Proximal Policy Optimization Algorithms* (2017)；用于 RLHF 见 InstructGPT (2022) ·
> 前置阅读：[RLHF 总览](/rlhf/)、[Reward Model](/rlhf/reward-model)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 策略梯度的步长问题：一步走太大就崩
- [ ] TRPO → PPO：用 clip 替代二阶约束

## 2. 方法与公式

裁剪目标（token 级）：

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t \left[ \min \left( \rho_t A_t, \; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

优势用 GAE 估计：

$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

TODO：

- [ ] LLM 场景的建模：state = prompt + 已生成 token，action = 下一 token
- [ ] reward 的构成：序列末端 RM 分数 + 每 token KL 惩罚
- [ ] value model（critic）的训练与 value clip

## 3. 与 baseline 对比

| 维度 | PPO | GRPO / RLOO |
| --- | --- | --- |
| Critic | 需要（再训一个模型） | 不需要 |
| 同时驻留模型 | 4（policy/ref/RM/critic） | 3 |
| 优势估计 | GAE，token 级 | 组内相对，序列级 |
| 稳定性 | TODO | TODO |

## 4. 实现要点与伪代码

```python
# TODO: rollout -> 计算 advantage -> 多轮 minibatch 更新 的主循环伪代码
```

- [ ] KL 惩罚加在 reward 里 vs 加在 loss 里
- [ ] 经验回放 epoch 数（通常 1~4）
- [ ] 各种 normalization（advantage whitening、reward scaling）

## 5. 实验与调参经验

TODO：clip $\epsilon$、$\gamma$、$\lambda$、KL 系数的常见取值；训练崩溃的征兆（KL 爆炸、熵塌缩）。

## 6. 参考文献

- [ ] Schulman et al., 2017. *PPO.* arXiv:1707.06347
- [ ] Ouyang et al., 2022. *InstructGPT.* arXiv:2203.02155
