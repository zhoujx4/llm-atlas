---
title: CPO
translation: synced
---

# CPO (Contrastive Preference Optimization)

> **In one sentence**: TODO — approximates the reference model with a uniform prior to obtain an upper bound on the DPO loss, plus an SFT regularization term; proposed for machine translation but generally applicable.
>
> Paper: *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation* (2024) ·
> Prerequisites: [DPO](/en/dpo/dpo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The reference model both costs memory and caps the model at the reference's level
- [ ] The translation setting: fine-grained preferences between good and better

## 2. Method and Formulas

$$
\mathcal{L}_{\text{CPO}} = \underbrace{-\mathbb{E}\left[\log \sigma\left(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x)\right)\right]}_{\mathcal{L}_{\text{prefer}}} \; \underbrace{- \;\mathbb{E}\left[\log \pi_\theta(y_w|x)\right]}_{\mathcal{L}_{\text{SFT}}}
$$

TODO:

- [ ] Deriving the upper bound from the DPO loss (uniform reference assumption)
- [ ] The SFT term prevents the chosen probability from collapsing

## 3. Comparison with Baselines

| Dimension | DPO | CPO | SimPO |
| --- | --- | --- | --- |
| Reference model | Required | Not required | Not required |
| Anti-collapse mechanism | Implicit KL | SFT term | Margin $\gamma$ |
| Length normalization | None | None | Yes |

## 4. Implementation Notes and Pseudocode

```python
# TODO: CPO loss pseudocode
```

## 5. Experiments and Tuning Experience

TODO: weight of the SFT term; measured comparison with SimPO.

## 6. References

- [ ] Xu et al., 2024. *CPO.* arXiv:2401.08417
