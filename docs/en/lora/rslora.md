---
title: rsLoRA
translation: synced
---

# rsLoRA (Rank-Stabilized LoRA)

> **In one sentence**: TODO — change LoRA's scaling factor from $\alpha/r$ to $\alpha/\sqrt{r}$ so that gradient scales remain stable at high ranks and the gains no longer saturate.
>
> Paper: *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (2023) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The original $\alpha/r$ scaling makes updates too small at large ranks, stalling learning
- [ ] Deriving the appropriate scale from initialization theory

## 2. Method and Formulas

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} B A x
$$

TODO:

- [ ] Derivation: ensuring the variance of $\Delta W x$ does not decay with $r$
- [ ] A one-line-of-code change

## 3. Comparison with Baselines

| Dimension | LoRA ($\alpha/r$) | rsLoRA ($\alpha/\sqrt{r}$) |
| --- | --- | --- |
| Small rank (8~16) | Essentially no difference | Essentially no difference |
| Large rank (≥64) | Gains saturate | TODO: continued gains |
| Implementation cost | — | One line |

## 4. Implementation Notes

- [ ] `use_rslora=True` in peft
- [ ] After changing the scaling, $\alpha$ and lr need to be re-standardized

## 5. Experiments and Tuning Experience

TODO: when it is worth going to a large rank + rsLoRA.

## 6. References

- [ ] Kalajdzievski, 2023. *rsLoRA.* arXiv:2312.03732
