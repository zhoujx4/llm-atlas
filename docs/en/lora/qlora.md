---
title: QLoRA
translation: synced
---

# QLoRA

> **In one sentence**: TODO — quantize the frozen base weights to 4-bit (NF4) while training the LoRA adapters in bf16, enabling fine-tuning of large models on a single GPU.
>
> Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The bulk of LoRA's memory is still the base weights themselves
- [ ] Quantization is only for "storage"; weights are dequantized for computation

## 2. Method and Formulas

Forward computation:

$$
h = \mathrm{dequant}(W_0^{\text{NF4}})\, x + \frac{\alpha}{r} B A x
$$

Three core components, TODO to expand:

- [ ] **NF4**: an information-theoretically optimal 4-bit data type for normally distributed weights
- [ ] **Double Quantization**: quantizing the quantization constants again
- [ ] **Paged Optimizer**: spilling memory peaks to the CPU

## 3. Comparison with Baselines

| Dimension | LoRA (bf16 base) | QLoRA (NF4 base) |
| --- | --- | --- |
| Base-model memory | 2 bytes/param | ~0.5 bytes/param |
| Training speed | Fast | Slower (dequantization overhead) |
| Quality | TODO | Paper claims parity with 16-bit |

## 4. Implementation Notes and Pseudocode

```python
# TODO: minimal bitsandbytes load_in_4bit + peft example
```

## 5. Experiments and Tuning Experience

TODO: suitable scenarios (extremely memory-constrained); measured speed/quality differences vs LoRA.

## 6. References

- [ ] Dettmers et al., 2023. *QLoRA.* arXiv:2305.14314
