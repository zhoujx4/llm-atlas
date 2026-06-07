---
title: SimPO
translation: synced
---

# SimPO (Simple Preference Optimization)

> **In one sentence**: TODO — removes the reference model, uses the "length-normalized average logprob" as the implicit reward, and adds a target margin; simpler and mitigates length bias.
>
> Paper: *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024) ·
> Prerequisites: [DPO](/en/dpo/dpo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] DPO's implicit reward is inconsistent with the metric actually used at generation time (average logprob)
- [ ] Memory and compute cost of the reference model
- [ ] Length bias: summed logprobs naturally favor short responses / cause length inflation

## 2. Method and Formulas

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]
$$

TODO:

- [ ] The role of length normalization $\frac{\beta}{|y|}$
- [ ] The role of the target margin $\gamma$
- [ ] The argument for aligning with the generation metric

## 3. Comparison with Baselines

| Dimension | DPO | SimPO |
| --- | --- | --- |
| Reference model | Required | Not required |
| Implicit reward | Ratio of summed logprobs | Length-normalized average logprob |
| Memory cost | High (two models) | Low |
| KL constraint | Yes (implicit) | None (easier to diverge) |

## 4. Implementation Notes and Pseudocode

```python
# TODO: SimPO loss for one batch
```

- [ ] Use the effective token count for length (excluding padding / prompt)
- [ ] Correspondence with TRL's `CPOTrainer(loss_type="simpo")`

## 5. Experiments and Tuning Experience

TODO: common values of $\beta$ (2~2.5) and $\gamma$ (0.5~1.5); stability risks from the lack of a KL constraint.

## 6. References

- [ ] Meng et al., 2024. *SimPO.* arXiv:2405.14734
