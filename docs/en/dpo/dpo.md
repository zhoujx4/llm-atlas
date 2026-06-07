---
title: DPO
translation: synced
---

# DPO (Direct Preference Optimization)

> **In one sentence**: TODO — solve the KL-constrained RLHF objective in closed form, compressing the two steps "train an RM + run PPO" into a single classification loss.
>
> Paper: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) ·
> Prerequisites: [RLHF Overview](/en/rlhf/), [Reward Model](/en/rlhf/reward-model)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The RLHF pipeline is complex, unstable, and memory-hungry (4 models)
- [ ] Key insight: a closed-form mapping exists between the optimal policy and the reward — "the language model is itself an implicit reward model"

## 2. Method and Formulas

Starting from the KL-constrained RLHF objective, the optimal policy satisfies $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$; substituting into the Bradley-Terry model gives:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

TODO:

- [ ] Full derivation (KL-constrained objective → closed-form solution → substitute into BT)
- [ ] The meaning of $\beta$: the penalty strength for deviating from the reference
- [ ] Interpreting the implicit reward $\hat{r} = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$

## 3. Comparison with Baselines

| Dimension | RLHF (PPO) | DPO |
| --- | --- | --- |
| Training stages | Two stages: RM + RL | Single stage |
| Models resident simultaneously | 4 | 2 |
| Online sampling | Required | Not required (offline) |
| Performance ceiling | TODO | TODO |

## 4. Implementation Notes and Pseudocode

```python
# TODO: DPO loss for one batch (two forward passes for policy/ref, sum logprobs over tokens)
```

- [ ] Sum logprobs over the sequence (do not average — this is the key difference from SimPO)
- [ ] Handling the reference model: frozen copy / precomputed logprobs
- [ ] Correspondence with TRL's `DPOTrainer`

## 5. Experiments and Tuning Experience

TODO:

- [ ] Common values of $\beta$: 0.05~0.5
- [ ] Classic phenomenon: logprobs of both chosen and rejected dropping together
- [ ] Why SFT before DPO is necessary

## 6. References

- [ ] Rafailov et al., 2023. *DPO.* arXiv:2305.18290
