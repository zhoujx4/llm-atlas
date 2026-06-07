---
title: Loss Masking
translation: synced
---

# Loss Masking

> **In one sentence**: TODO — during SFT, compute the loss only on the assistant response portion; prompt and template tokens do not contribute gradients.
>
> Prerequisites: [SFT Overview](/en/sft/), [Chat Template](/en/sft/chat-template)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Why not compute loss on the prompt: avoid learning the distribution of "repeating the question"
- [ ] Some work does compute loss on the prompt (small-data settings); what is the trade-off

## 2. Method and Formulas

SFT loss with a mask:

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid y_{<t}), \quad m_t \in \{0, 1\}
$$

TODO:

- [ ] Rules for constructing $m_t$: set user turns, system, and template tokens to 0
- [ ] Multi-turn dialogue: compute loss on every assistant turn or only the last one

## 3. Implementation Notes and Pseudocode

```python
# TODO: the standard HuggingFace approach of setting labels to -100
```

- [ ] The `ignore_index=-100` convention
- [ ] Concatenating masks when used together with [Packing](/en/sft/packing)
- [ ] Normalization: average per token or per sample

## 4. Experiments and Tuning Experience

TODO: typical symptoms of misaligned masks.

## 5. References

- [ ] TODO
