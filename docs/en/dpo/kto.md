---
title: KTO
translation: synced
---

# KTO (Kahneman-Tversky Optimization)

> **In one sentence**: TODO — no paired preference data needed, only a "good/bad" label per sample; the loss borrows the loss-aversion design of prospect theory.
>
> Paper: *KTO: Model Alignment as Prospect Theoretic Optimization* (2024) ·
> Prerequisites: [DPO](/en/dpo/dpo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Paired preference data is expensive and hard to collect; binary labels (thumbs up/down) are everywhere
- [ ] Prospect theory: humans are more sensitive to losses than gains (loss aversion)

## 2. Method and Formulas

Define the implicit reward $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, with the reference point $z_0$ as an in-batch KL estimate:

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y)} \left[ \lambda_y - v(x, y) \right]
$$

$$
v(x,y) = \begin{cases} \lambda_D \,\sigma\!\left(\hat{r}_\theta(x,y) - z_0\right) & y \sim \text{desirable} \\ \lambda_U \,\sigma\!\left(z_0 - \hat{r}_\theta(x,y)\right) & y \sim \text{undesirable} \end{cases}
$$

TODO:

- [ ] How exactly $z_0$ (the reference point) is estimated
- [ ] $\lambda_D / \lambda_U$ for handling good/bad sample imbalance

## 3. Comparison with Baselines

| Dimension | DPO | KTO |
| --- | --- | --- |
| Data format | Pairs $(y_w, y_l)$ | Single samples + binary labels |
| Data acquisition cost | High | Low |
| Quality at equal data size | TODO | TODO |

## 4. Implementation Notes and Pseudocode

```python
# TODO: KTO loss pseudocode (note the reference point is shared within the batch)
```

## 5. Experiments and Tuning Experience

TODO: setting $\lambda$ when the good/bad sample ratio is imbalanced.

## 6. References

- [ ] Ethayarajh et al., 2024. *KTO.* arXiv:2402.01306
