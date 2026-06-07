---
title: Agentic RL
translation: synced
---

# Agentic RL (RL Training for Multi-Turn Tool Calling)

> **In one sentence**: TODO — extends RL from "single-turn generation" to "multi-turn interaction": the model generates → the environment executes → results feed back → generation continues, optimizing the policy over the whole trajectory.
>
> Prerequisites: [RLHF Overview](/en/rlhf/), [GRPO](/en/rlhf/grpo), [Tool-Use Training](/en/agent/tool-use)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] SFT trajectory data cannot cover the combinatorial explosion of environments; RL directly optimizes for task success
- [ ] Essential difference from single-turn RLHF: episodes include environment steps, rewards are sparse and delayed

## 2. Problem Setup

TODO:

- [ ] Trajectory definition: $\tau = (x, a_1, o_1, a_2, o_2, \dots, a_T)$, where $o_t$ is the tool result
- [ ] Whether environment-observation tokens get a loss (mask them out, analogous to [Loss Masking](/en/sft/loss-masking))
- [ ] Reward design: outcome verification (unit tests passing, correct answers) + process shaping

## 3. Training Methods

TODO:

- [ ] Adapting GRPO/PPO to multi-turn trajectories (advantage broadcasting, turn boundaries)
- [ ] Asynchronous rollouts and environment sandbox engineering
- [ ] Curriculum: short chains first, long chains later

## 4. Comparison with Baselines

| Dimension | Single-turn RLHF | Agentic RL |
| --- | --- | --- |
| Episode | One generation | Multi-turn interaction |
| Reward | RM score | Mainly task-outcome verification |
| Engineering complexity | Medium | High (environments/sandboxes/async) |

## 5. Experiments and Tuning Experience

TODO: how reward hacking manifests in the agent setting (deleting tests, bypassing verification).

## 6. References

- [ ] TODO: representative work such as SWE-RL, WebRL, AgentRL
