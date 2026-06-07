---
title: PiSSA
translation: synced
---

# PiSSA (Principal Singular Values and Singular Vectors Adaptation)

> **In one sentence**: TODO — initialize $A$ and $B$ from the principal singular components of $W_0$ and freeze the residual; train "the most important directions" instead of an increment from zero.
>
> Paper: *PiSSA: Principal Singular Values and Singular Vectors Adaptation* (2024) ·
> Prerequisites: [LoRA](/en/lora/lora)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] LoRA's $B=0$ initialization makes updates slow early in training
- [ ] Directly fine-tuning the principal-component directions converges faster

## 2. Method and Formulas

Take the SVD of $W_0$: $W_0 = U S V^\top$, keep the top $r$ singular values:

$$
W_0 = \underbrace{U_{[:,:r]} S_{[:r,:r]} V_{[:,:r]}^\top}_{\text{init } BA \text{ (trainable)}} + \underbrace{W^{\text{res}}}_{\text{frozen}}
$$

TODO:

- [ ] The concrete construction $A = \sqrt{S_{[:r]}}\,V^\top_{[:r]}$, $B = U_{[:r]}\sqrt{S_{[:r]}}$
- [ ] Fast SVD (randomized SVD) to reduce initialization cost
- [ ] Combination with QLoRA (quantize the residual, reducing quantization error)

## 3. Comparison with Baselines

| Dimension | LoRA | PiSSA |
| --- | --- | --- |
| Initialization | A Gaussian / B zero | Principal singular components of $W_0$ |
| Early convergence | Slow | TODO: faster |
| Initialization cost | Zero | One SVD |

## 4. Implementation Notes and Pseudocode

```python
# TODO: SVD initialization pseudocode
```

## 5. Experiments and Tuning Experience

TODO: head-to-head comparison with LoRA/DoRA; caveats when merging back into the base model (the residual is not $W_0$).

## 6. References

- [ ] Meng et al., 2024. *PiSSA.* arXiv:2404.02948
