---
title: IPO
translation: synced
---

# IPO (Identity Preference Optimization)

> **In one sentence**: TODO — points out that under deterministic preferences DPO amplifies the logit gap without bound, causing overfitting; uses a squared loss instead to pull the implicit reward gap toward a fixed target.
>
> Paper: *A General Theoretical Paradigm to Understand Learning from Human Preferences* (ΨPO, 2023) ·
> Prerequisites: [DPO](/en/dpo/dpo)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] DPO's problem: under the BT assumption, as the preference probability approaches 1, the optimal solution pushes the reward gap toward infinity → unbounded deviation from the reference
- [ ] The ΨPO unified-framework perspective

## 2. Method and Formulas

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w|x)\,\pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_{\text{ref}}(y_w|x)} - \frac{1}{2\tau} \right)^2 \right]
$$

TODO:

- [ ] Contrast with DPO: logistic loss → squared loss
- [ ] The role of $\tau$ (inverse of the target margin)

## 3. Comparison with Baselines

| Dimension | DPO | IPO |
| --- | --- | --- |
| Loss form | $-\log\sigma(\cdot)$ | $(\cdot - \frac{1}{2\tau})^2$ |
| Under deterministic preferences | Reward gap → ∞ | Bounded |
| Practical performance | TODO | TODO (not necessarily better in practice) |

## 4. Implementation Notes and Pseudocode

```python
# TODO: IPO loss (note the logprob difference needs dividing by... verify against the TRL implementation)
```

## 5. Experiments and Tuning Experience

TODO: choices of $\tau$; the real-world gap between IPO and DPO in practice.

## 6. References

- [ ] Azar et al., 2023. *ΨPO / IPO.* arXiv:2310.12036
