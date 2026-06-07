---
title: RLOO
translation: synced
---

# RLOO (REINFORCE Leave-One-Out)

> **In one sentence**: TODO — back to REINFORCE: sample $k$ responses per prompt and use "the average reward of the other $k{-}1$" as each response's baseline; unbiased and critic-free.
>
> Paper: *Back to Basics: Revisiting REINFORCE-Style Optimization for RLHF* (2024) ·
> Prerequisites: [PPO](/en/rlhf/ppo), [GRPO](/en/rlhf/grpo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] In LLM-RLHF the initial policy is already strong; many PPO mechanisms (clip, GAE, critic) may be unnecessary complexity
- [ ] The leave-one-out baseline: unbiased variance reduction

## 2. Method and Formulas

$$
\nabla_\theta \mathcal{J} = \frac{1}{k} \sum_{i=1}^{k} \left( r(x, y_i) - \frac{1}{k-1}\sum_{j \neq i} r(x, y_j) \right) \nabla_\theta \log \pi_\theta(y_i | x)
$$

TODO:

- [ ] Treating the whole response as a single action (sequence-level, no token-level credit assignment)
- [ ] Differences from GRPO: the baseline is the leave-one-out mean rather than the full-group mean/std; usually single-step on-policy updates without clipping

## 3. Comparison with Baselines

| Dimension | PPO | GRPO | RLOO |
| --- | --- | --- | --- |
| Critic | Required | Not required | Not required |
| Baseline | Value network | Group mean (incl. self) / std | Leave-one-out mean |
| Unbiasedness | — | std scaling is biased | Unbiased |
| Clip / off-policy | Yes | Yes | Usually none |

## 4. Implementation Notes and Pseudocode

```python
# TODO: RLOO gradient-estimation pseudocode
```

## 5. Experiments and Tuning Experience

TODO: choosing $k$ (2~8); measured quality/cost comparison with PPO.

## 6. References

- [ ] Ahmadian et al., 2024. *Back to Basics.* arXiv:2402.14740
