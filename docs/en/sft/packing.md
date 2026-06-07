---
title: Sequence Packing
translation: synced
---

# Sequence Packing

> **In one sentence**: TODO — pack multiple short samples into a single training sequence to fill the context, significantly improving training throughput.
>
> Prerequisites: [SFT Overview](/en/sft/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] How much padding wastes: the fraction of effective tokens with short samples
- [ ] The magnitude of throughput gains from packing

## 2. Method

TODO:

- [ ] Naive concatenation vs first-fit binning
- [ ] The cross-contamination problem: whether attention is isolated between samples
- [ ] Block-diagonal attention mask / resetting position_ids
- [ ] FlashAttention's varlen interface (`cu_seqlens`)

## 3. Comparison with Baselines

| Dimension | Padding | Naive packing | Isolated packing |
| --- | --- | --- | --- |
| Throughput | Low | High | High |
| Cross-sample leakage | None | Yes | None |
| Implementation complexity | Low | Low | Medium |

## 4. Implementation Notes and Pseudocode

```python
# TODO: first-fit packing + cu_seqlens construction pseudocode
```

## 5. Experiments and Tuning Experience

TODO: whether packing's impact on quality is negligible; when attention isolation is mandatory.

## 6. References

- [ ] Krell et al., 2021. *Efficient Sequence Packing*
