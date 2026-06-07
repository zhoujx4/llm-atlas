---
title: Notation
translation: synced
---

# Notation

> All formulas on this site use the following notation to avoid ambiguity from mixing the symbol conventions of different papers.

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## Basic Notation

| Symbol | Meaning |
| --- | --- |
| $x$ | Input prompt |
| $y$ | Model response; $y_t$ denotes the $t$-th token |
| $\pi_\theta$ | Policy model being trained (parameters $\theta$) |
| $\pi_{\text{ref}}$ | Reference model (usually the SFT model) |
| $r_\phi(x, y)$ | Reward model (parameters $\phi$) |
| $(y_w, y_l)$ | Chosen / rejected responses in a preference pair |
| $\sigma(\cdot)$ | Sigmoid function |
| $\mathbb{D}$ | Training dataset |
| $\beta$ | KL constraint / temperature coefficient |

## Example: Autoregressive Factorization of a Language Model

Inline formula example: the policy's probability of a full response is $\pi_\theta(y|x) = \prod_{t} \pi_\theta(y_t | x, y_{<t})$.

Block formula example (the negative log-likelihood loss of SFT):

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

## TODO

- [ ] RL notation (states, actions, advantage function $A_t$, GAE parameters $\gamma, \lambda$)
- [ ] LoRA notation ($W_0, B, A, r, \alpha$)
- [ ] Superscript / subscript usage rules
