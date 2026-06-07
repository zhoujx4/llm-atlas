---
title: LoRA
translation: synced
---

# LoRA (Low-Rank Adaptation)

> **In one sentence**: TODO — freeze the pretrained weights $W_0$ and train only a low-rank increment $\Delta W = BA$, reducing trainable parameters to the per-mille level.
>
> Paper: *LoRA: Low-Rank Adaptation of Large Language Models* (2021) ·
> Prerequisites: [Full Fine-Tuning](/en/sft/full-finetuning)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The hypothesis that weight updates have low "intrinsic rank"
- [ ] Motivation vs Adapters / Prefix-tuning (no inference latency)

## 2. Method and Formulas

Forward computation:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

TODO:

- [ ] Initialization: $A$ Gaussian, $B$ zero (so $\Delta W$ starts at 0)
- [ ] The role of the scaling factor $\alpha/r$
- [ ] Merging for deployment: $W = W_0 + \frac{\alpha}{r}BA$, zero inference overhead

## 3. Comparison with Baselines

| Dimension | Full fine-tuning | LoRA |
| --- | --- | --- |
| Trainable parameters | 100% | ~0.1%-1% |
| Optimizer-state memory | Large | Tiny |
| Inference latency | — | None after merging |
| Task switching | Full set of weights | Swap the adapter |

## 4. Implementation Notes and Pseudocode

```python
# TODO: LoRA Linear layer pseudocode
```

- [ ] Which modules to inject into: q/k/v/o or all Linear layers
- [ ] Common rank/α combinations (r=8~64, α=2r)

## 5. Experiments and Tuning Experience

TODO: diminishing-returns curve over rank; lr is usually an order of magnitude larger than full fine-tuning.

## 6. References

- [ ] Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
