---
title: IPO
---

# IPO（Identity Preference Optimization）

> **一句话**：TODO —— 指出 DPO 在确定性偏好下会无限放大 logit 差导致过拟合，改用平方损失把隐式 reward 差拉向固定目标。
>
> 论文：*A General Theoretical Paradigm to Understand Learning from Human Preferences* (ΨPO, 2023) ·
> 前置阅读：[DPO](/dpo/dpo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] DPO 的问题：BT 假设下偏好概率趋近 1 时，最优解把 reward 差推向无穷 → 偏离 reference 无界
- [ ] ΨPO 统一框架视角

## 2. 方法与公式

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w|x)\,\pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_{\text{ref}}(y_w|x)} - \frac{1}{2\tau} \right)^2 \right]
$$

TODO：

- [ ] 与 DPO 的对照：logistic 损失 → 平方损失
- [ ] $\tau$ 的作用（目标 margin 的倒数）

## 3. 与 baseline 对比

| 维度 | DPO | IPO |
| --- | --- | --- |
| 损失形式 | $-\log\sigma(\cdot)$ | $(\cdot - \frac{1}{2\tau})^2$ |
| 确定性偏好下 | reward 差 → ∞ | 有界 |
| 实际效果 | TODO | TODO（实践中未必更好） |

## 4. 实现要点与伪代码

```python
# TODO: IPO loss（注意 logprob 差需除以... 与 TRL 实现核对）
```

## 5. 实验与调参经验

TODO：$\tau$ 的取值；实践中 IPO vs DPO 的真实差距。

## 6. 参考文献

- [ ] Azar et al., 2023. *ΨPO / IPO.* arXiv:2310.12036
