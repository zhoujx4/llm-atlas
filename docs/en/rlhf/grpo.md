---
title: GRPO
translation: synced
---

# GRPO (Group Relative Policy Optimization)

> **In one sentence**: TODO — drops the critic: sample a group of responses for the same prompt and use the standardized in-group rewards as the advantage estimate; DeepSeek-R1 made it the mainstream algorithm for reasoning RL.
>
> Papers: *DeepSeekMath* (2024), *DeepSeek-R1* (2025) ·
> Prerequisites: [PPO](/en/rlhf/ppo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Critic training is hard and memory-expensive; value estimation is inherently coarse for sequence-level rewards
- [ ] Multiple samples per prompt naturally provide a baseline

## 2. Method and Formulas

Sample $G$ responses $\{y_1, \dots, y_G\}$ for each prompt; the in-group standardized advantage is:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\})}
$$

Objective (PPO-style clipping + explicit KL term):

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t} \min\left( \rho_{i,t} \hat{A}_i,\; \mathrm{clip}(\rho_{i,t}, 1\pm\epsilon) \hat{A}_i \right) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right]
$$

TODO:

- [ ] The unbiased KL estimator (k3)
- [ ] Broadcasting the sequence-level advantage to every token
- [ ] Later corrections: DAPO's critique of length normalization and the std term

## 3. Comparison with Baselines

| Dimension | PPO | GRPO |
| --- | --- | --- |
| Critic | Required | Not required |
| Advantage granularity | Token-level (GAE) | Sequence-level (group-relative) |
| Samples per prompt | 1 | $G$ (typically 8~64) |
| Memory | High | Medium |

## 4. Implementation Notes and Pseudocode

```python
# TODO: pseudocode: group sampling -> in-group standardization -> clipped update
```

- [ ] Handling the all-same-score case (all correct / all wrong) where the advantage is 0
- [ ] Correspondence with the verl / OpenRLHF implementations

## 5. Experiments and Tuning Experience

TODO: choosing the group size $G$; the difficulty bias introduced by std normalization (DAPO's critique).

## 6. References

- [ ] Shao et al., 2024. *DeepSeekMath.* arXiv:2402.03300
- [ ] DeepSeek-AI, 2025. *DeepSeek-R1.* arXiv:2501.12948
