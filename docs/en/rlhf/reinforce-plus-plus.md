---
title: REINFORCE++
translation: synced
---

# REINFORCE++

> **In one sentence**: TODO — layers PPO's stabilization tricks (token-level KL, clipping, global-batch advantage normalization) on top of REINFORCE, needing neither a critic nor group sampling.
>
> Paper/report: *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models* (2025) ·
> Prerequisites: [PPO](/en/rlhf/ppo), [RLOO](/en/rlhf/rloo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] GRPO's per-prompt in-group baseline may introduce bias / waste sampling budget
- [ ] Replacing the in-group baseline with global batch normalization

## 2. Method and Formulas

After folding the KL penalty into token-level rewards, the advantage is normalized over the global batch:

$$
\hat{A}_t = \frac{A_t - \mu_{\text{batch}}}{\sigma_{\text{batch}}}
$$

TODO:

- [ ] How the token-level KL penalty shapes the reward
- [ ] Retaining PPO-clip
- [ ] The stability argument vs GRPO

## 3. Comparison with Baselines

| Dimension | GRPO | RLOO | REINFORCE++ |
| --- | --- | --- | --- |
| Samples per prompt | $G$ | $k$ | 1 is enough |
| Baseline | In-group | Leave-one-out | Global batch |
| Critic | No | No | No |

## 4. Implementation Notes and Pseudocode

```python
# TODO: pseudocode
```

## 5. Experiments and Tuning Experience

TODO: the implementation and default hyperparameters in OpenRLHF.

## 6. References

- [ ] Hu, 2025. *REINFORCE++.* arXiv:2501.03262
