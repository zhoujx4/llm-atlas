---
title: PPO
translation: synced
---

# PPO (Proximal Policy Optimization)

> **In one sentence**: TODO — uses a clipped importance-sampling ratio to bound each policy update step; the classic RLHF algorithm — stable, but engineering-heavy (requires 4 models).
>
> Papers: *Proximal Policy Optimization Algorithms* (2017); for RLHF see InstructGPT (2022) ·
> Prerequisites: [RLHF Overview](/en/rlhf/), [Reward Model](/en/rlhf/reward-model)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] The step-size problem of policy gradients: one overly large step and training collapses
- [ ] TRPO → PPO: replacing the second-order constraint with clipping

## 2. Method and Formulas

Clipped objective (token-level):

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t \left[ \min \left( \rho_t A_t, \; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

Advantage estimated with GAE:

$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

TODO:

- [ ] Modeling for the LLM setting: state = prompt + generated tokens so far, action = next token
- [ ] Reward composition: RM score at the sequence end + per-token KL penalty
- [ ] Training the value model (critic) and value clipping

## 3. Comparison with Baselines

| Dimension | PPO | GRPO / RLOO |
| --- | --- | --- |
| Critic | Required (an extra model to train) | Not required |
| Models resident simultaneously | 4 (policy/ref/RM/critic) | 3 |
| Advantage estimation | GAE, token-level | Group-relative, sequence-level |
| Stability | TODO | TODO |

## 4. Implementation Notes and Pseudocode

```python
# TODO: main-loop pseudocode: rollout -> compute advantage -> multiple minibatch update epochs
```

- [ ] KL penalty in the reward vs in the loss
- [ ] Number of replay epochs (typically 1~4)
- [ ] Various normalizations (advantage whitening, reward scaling)

## 5. Experiments and Tuning Experience

TODO: common values for clip $\epsilon$, $\gamma$, $\lambda$, KL coefficient; signs of training collapse (KL explosion, entropy collapse).

## 6. References

- [ ] Schulman et al., 2017. *PPO.* arXiv:1707.06347
- [ ] Ouyang et al., 2022. *InstructGPT.* arXiv:2203.02155
