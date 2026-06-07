---
title: DoRA
translation: synced
---

# DoRA (Weight-Decomposed Low-Rank Adaptation)

> **In one sentence**: TODO — decompose the weights into "magnitude × direction", train the magnitude directly and update the direction with LoRA, making the behavior closer to full fine-tuning.
>
> Paper: *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Analysis of LoRA vs full fine-tuning update patterns: different correlations between magnitude and direction changes
- [ ] Decoupling makes LoRA's learning behavior more like full fine-tuning

## 2. Method and Formulas

$$
W = m \cdot \frac{W_0 + BA}{\left\lVert W_0 + BA \right\rVert_c}
$$

where $m$ is a trainable magnitude vector and $\lVert \cdot \rVert_c$ denotes the column-wise L2 norm.

TODO:

- [ ] $m$ initialized to $\lVert W_0 \rVert_c$
- [ ] Analysis of gradient flow and memory overhead

## 3. Comparison with Baselines

| Dimension | LoRA | DoRA |
| --- | --- | --- |
| Extra trainable parameters | $BA$ | $BA + m$ |
| Training overhead | Low | Slightly higher (norm computation) |
| Quality at low rank | TODO | Paper claims better |

## 4. Implementation Notes and Pseudocode

```python
# TODO: DoRA forward-pass pseudocode
```

## 5. Experiments and Tuning Experience

TODO: when it beats LoRA; caveats for merging at inference time.

## 6. References

- [ ] Liu et al., 2024. *DoRA.* arXiv:2402.09353
