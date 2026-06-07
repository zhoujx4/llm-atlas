---
title: ORPO
---

# ORPO（Odds Ratio Preference Optimization）

> **一句话**：TODO —— 把偏好优化并进 SFT：NLL 损失 + odds ratio 惩罚项，单阶段完成"学会回答 + 对齐偏好"，无需 reference model。
>
> 论文：*ORPO: Monolithic Preference Optimization without Reference Model* (2024) ·
> 前置阅读：[DPO](/dpo/dpo)、[SFT 总览](/sft/)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] SFT → DPO 两阶段的冗余；SFT 本身会同时抬高坏回答的概率
- [ ] 用 odds ratio 在 SFT 的同时压低 rejected

## 2. 方法与公式

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}, \quad
\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)
$$

其中 $\text{odds}(y|x) = \frac{P(y|x)}{1 - P(y|x)}$（$P$ 为长度归一化后的序列概率）。

TODO：

- [ ] 为什么用 odds ratio 而不是概率比（梯度性质）
- [ ] 长度归一化的处理

## 3. 与 baseline 对比

| 维度 | SFT + DPO | ORPO |
| --- | --- | --- |
| 训练阶段 | 两阶段 | 单阶段 |
| Reference model | 需要 | 不需要 |
| 数据 | SFT 数据 + 偏好数据 | 偏好数据（chosen 兼作 SFT） |

## 4. 实现要点与伪代码

```python
# TODO: ORPO loss 伪代码
```

## 5. 实验与调参经验

TODO：$\lambda$ 常见取值（0.1 量级）；从基座直接 ORPO 的可行性。

## 6. 参考文献

- [ ] Hong et al., 2024. *ORPO.* arXiv:2403.07691
