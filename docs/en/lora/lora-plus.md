---
title: LoRA+
translation: synced
---

# LoRA+

> **In one sentence**: TODO — give the $B$ matrix a much larger learning rate than $A$ (ratio $\lambda \gg 1$), fixing LoRA's suboptimal learning dynamics in wide networks.
>
> Paper: *LoRA+: Efficient Low Rank Adaptation of Large Models* (2024) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Infinite-width analysis: the gradient scales of $A$ and $B$ are inherently asymmetric
- [ ] Using the same learning rate makes feature learning inefficient

## 2. Method and Formulas

$$
\eta_B = \lambda \cdot \eta_A, \quad \lambda \gg 1 \;(\text{paper suggests } \lambda \approx 16)
$$

TODO:

- [ ] Key derivation steps (μP-style scaling analysis)
- [ ] Relationship to the zero initialization of $B$

## 3. Comparison with Baselines

| Dimension | LoRA | LoRA+ |
| --- | --- | --- |
| Change | — | Only per-group learning rates in the optimizer |
| Convergence speed | Baseline | Paper claims ~2× |
| Final quality | Baseline | TODO: small improvement |

## 4. Implementation Notes and Pseudocode

```python
# TODO: pseudocode for optimizer param groups setting B's lr = λ * lr
```

## 5. Experiments and Tuning Experience

TODO: sensitivity to $\lambda$; whether it stacks with rsLoRA.

## 6. References

- [ ] Hayou et al., 2024. *LoRA+.* arXiv:2402.12354
