---
title: KTO
---

# KTO（Kahneman-Tversky Optimization）

> **一句话**：TODO —— 不需要成对偏好数据，只要每条样本一个「好/坏」标签；损失函数借鉴前景理论的损失厌恶设计。
>
> 论文：*KTO: Model Alignment as Prospect Theoretic Optimization* (2024) ·
> 前置阅读：[DPO](/dpo/dpo)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 成对偏好数据贵且难收集；二元标签（点赞/点踩）到处都是
- [ ] 前景理论：人对损失比对收益更敏感（损失厌恶）

## 2. 方法与公式

定义隐式 reward $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，参考点 $z_0$ 为批内 KL 估计：

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y)} \left[ \lambda_y - v(x, y) \right]
$$

$$
v(x,y) = \begin{cases} \lambda_D \,\sigma\!\left(\hat{r}_\theta(x,y) - z_0\right) & y \sim \text{desirable} \\ \lambda_U \,\sigma\!\left(z_0 - \hat{r}_\theta(x,y)\right) & y \sim \text{undesirable} \end{cases}
$$

TODO：

- [ ] $z_0$（参考点）的具体估计方式
- [ ] $\lambda_D / \lambda_U$ 处理好坏样本不均衡

## 3. 与 baseline 对比

| 维度 | DPO | KTO |
| --- | --- | --- |
| 数据形式 | 成对 $(y_w, y_l)$ | 单条 + 二元标签 |
| 数据获取成本 | 高 | 低 |
| 同数据量效果 | TODO | TODO |

## 4. 实现要点与伪代码

```python
# TODO: KTO loss 伪代码（注意 batch 内参考点共享）
```

## 5. 实验与调参经验

TODO：好坏样本比例失衡时 $\lambda$ 的设置。

## 6. 参考文献

- [ ] Ethayarajh et al., 2024. *KTO.* arXiv:2402.01306
