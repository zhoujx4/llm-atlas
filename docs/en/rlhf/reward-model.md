---
title: Reward Model
translation: synced
---

# Reward Model

> **In one sentence**: TODO — train a scoring model on preference data to serve as the reward source for the RL stage; its quality directly determines the ceiling of RLHF.
>
> Prerequisites: [RLHF Overview](/en/rlhf/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Why an RM is needed: human preferences cannot be queried online
- [ ] RM architecture: LM backbone + scalar value head

## 2. Method and Formulas

Bradley-Terry loss:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

TODO:

- [ ] Which position's hidden state to use (the last token)
- [ ] Margin variants, multi-objective RMs (separate heads for helpfulness/safety)
- [ ] Process Reward Model (PRM) vs Outcome Reward Model (ORM)

## 3. Common Problems

TODO:

- [ ] Reward hacking: the policy finds blind spots in the RM
- [ ] Length bias: the RM prefers longer responses
- [ ] OOD: RM scores become distorted after the policy distribution drifts

## 4. Implementation Notes and Pseudocode

```python
# TODO: BT loss training pseudocode
```

## 5. Experiments and Tuning Experience

TODO: relationship between RM accuracy and downstream RLHF performance; empirical data-scale guidance.

## 6. References

- [ ] Ouyang et al., 2022. *InstructGPT*
- [ ] Lightman et al., 2023. *Let's Verify Step by Step* (PRM)
