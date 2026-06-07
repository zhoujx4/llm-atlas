---
title: Full Fine-Tuning
translation: synced
---

# Full Fine-Tuning

> **In one sentence**: TODO — the standard SFT approach that updates all model parameters; highest performance ceiling, but also the largest memory footprint.
>
> Prerequisites: [SFT Overview](/en/sft/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Why a pretrained model cannot be used as an assistant out of the box
- [ ] A framework for trading off full fine-tuning vs PEFT (LoRA, etc.)
- [ ] At what scale / in which scenarios full fine-tuning is worth it

## 2. Method and Formulas

The training objective is the standard NLL:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

TODO:

- [ ] Similarities and differences with the pretraining objective (data distribution, loss mask, number of epochs)
- [ ] Memory breakdown: parameters + gradients + optimizer states (Adam is roughly 4x the parameter count)

## 3. Comparison with Baselines

| Dimension | Full fine-tuning | LoRA |
| --- | --- | --- |
| Trainable parameters | 100% | Usually < 1% |
| Memory | TODO | TODO |
| Performance ceiling | TODO | TODO |
| Multi-task switching | Requires a full set of weights | Just swap the adapter |

## 4. Implementation Notes and Pseudocode

```python
# TODO: minimal SFT training loop pseudocode (with loss masking)
```

- [ ] Learning rate magnitude (typically 1e-5 ~ 2e-5, much smaller than pretraining)
- [ ] Number of epochs and early stopping
- [ ] Choosing a distributed strategy: ZeRO / FSDP

## 5. Experiments and Tuning Experience

TODO: common values and pitfalls for lr, warmup, epochs, batch size.

## 6. References

- [ ] Ouyang et al., 2022. *Training language models to follow instructions with human feedback* (InstructGPT)
- [ ] Zhou et al., 2023. *LIMA: Less Is More for Alignment*
