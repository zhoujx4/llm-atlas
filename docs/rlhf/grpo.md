---
title: GRPO
---

# GRPO（Group Relative Policy Optimization）

> **一句话**：TODO —— 去掉 critic：同一 prompt 采样一组回答，用组内 reward 的标准化值作优势估计；DeepSeek-R1 使其成为推理 RL 的主流算法。
>
> 论文：*DeepSeekMath* (2024)、*DeepSeek-R1* (2025) ·
> 前置阅读：[PPO](/rlhf/ppo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] critic 训练难、显存贵；对序列级 reward 而言 value 估计本就粗糙
- [ ] 同 prompt 多采样天然提供 baseline

## 2. 方法与公式

对每个 prompt 采样 $G$ 个回答 $\{y_1, \dots, y_G\}$，组内标准化优势：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\})}
$$

目标函数（PPO 式 clip + 显式 KL 项）：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t} \min\left( \rho_{i,t} \hat{A}_i,\; \mathrm{clip}(\rho_{i,t}, 1\pm\epsilon) \hat{A}_i \right) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right]
$$

TODO：

- [ ] KL 的无偏估计器（k3）
- [ ] 序列级优势广播到每个 token
- [ ] 后续修正：DAPO 对长度归一化与 std 的批评

## 3. 与 baseline 对比

| 维度 | PPO | GRPO |
| --- | --- | --- |
| Critic | 需要 | 不需要 |
| 优势粒度 | token 级（GAE） | 序列级（组内相对） |
| 每 prompt 采样数 | 1 | $G$（典型 8~64） |
| 显存 | 高 | 中 |

## 4. 实现要点与伪代码

```python
# TODO: 组采样 -> 组内标准化 -> clip 更新 伪代码
```

- [ ] 全组同分（全对/全错）时优势为 0 的处理
- [ ] 与 verl / OpenRLHF 实现的对应

## 5. 实验与调参经验

TODO：组大小 $G$ 的选择；std 归一化引入的难度偏置（DAPO 的批评）。

## 6. 参考文献

- [ ] Shao et al., 2024. *DeepSeekMath.* arXiv:2402.03300
- [ ] DeepSeek-AI, 2025. *DeepSeek-R1.* arXiv:2501.12948
