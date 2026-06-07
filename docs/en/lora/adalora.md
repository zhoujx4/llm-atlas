---
title: AdaLoRA
translation: synced
---

# AdaLoRA (Adaptive Budget Allocation)

> **In one sentence**: TODO — different layers/modules need different ranks; AdaLoRA parameterizes the increment in SVD form and dynamically allocates the rank budget by importance.
>
> Paper: *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning* (2023) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Giving every module the same rank in LoRA is wasteful
- [ ] Importance metrics and pruning-style rank allocation

## 2. Method and Formulas

SVD-form parameterization:

$$
\Delta W = P \Lambda Q, \quad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_r)
$$

TODO:

- [ ] Orthogonality regularizer $\lVert P^\top P - I \rVert_F^2 + \lVert Q Q^\top - I \rVert_F^2$
- [ ] Sensitivity-based importance scoring, gradually pruning $\lambda_i$ during training
- [ ] Rank budget schedule (wide first, then narrow)

## 3. Comparison with Baselines

| Dimension | LoRA | AdaLoRA |
| --- | --- | --- |
| Rank allocation | Globally fixed | Adaptive per module |
| Training complexity | Low | Higher (scoring + scheduling) |
| Quality at equal budget | TODO | Paper claims better |

## 4. Implementation Notes and Pseudocode

```python
# TODO: importance scoring and rank-pruning schedule pseudocode
```

## 5. Experiments and Tuning Experience

TODO: budget-schedule hyperparameters; current usage (in practice often replaced by simply increasing rank).

## 6. References

- [ ] Zhang et al., 2023. *AdaLoRA.* arXiv:2303.10512
