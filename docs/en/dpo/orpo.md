---
title: ORPO
translation: synced
---

# ORPO (Odds Ratio Preference Optimization)

> **In one sentence**: TODO — folds preference optimization into SFT: NLL loss + an odds ratio penalty, completing "learn to answer + align preferences" in a single stage, with no reference model.
>
> Paper: *ORPO: Monolithic Preference Optimization without Reference Model* (2024) ·
> Prerequisites: [DPO](/en/dpo/dpo), [SFT Overview](/en/sft/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Redundancy of the SFT → DPO two-stage pipeline; SFT itself also raises the probability of bad responses
- [ ] Using the odds ratio to suppress rejected responses while doing SFT

## 2. Method and Formulas

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}, \quad
\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)
$$

where $\text{odds}(y|x) = \frac{P(y|x)}{1 - P(y|x)}$ ($P$ is the length-normalized sequence probability).

TODO:

- [ ] Why use the odds ratio instead of the probability ratio (gradient properties)
- [ ] How length normalization is handled

## 3. Comparison with Baselines

| Dimension | SFT + DPO | ORPO |
| --- | --- | --- |
| Training stages | Two stages | Single stage |
| Reference model | Required | Not required |
| Data | SFT data + preference data | Preference data (chosen doubles as SFT) |

## 4. Implementation Notes and Pseudocode

```python
# TODO: ORPO loss pseudocode
```

## 5. Experiments and Tuning Experience

TODO: common values of $\lambda$ (on the order of 0.1); feasibility of running ORPO directly from the base model.

## 6. References

- [ ] Hong et al., 2024. *ORPO.* arXiv:2403.07691
