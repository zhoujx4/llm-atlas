---
title: Tool-Use Training
translation: synced
---

# Tool Use / Function Calling Training

> **In one sentence**: TODO — teach the model to issue tool calls at the right time and with the correct schema, and to digest the tool's returned results.
>
> Prerequisites: [SFT Overview](/en/sft/), [Chat Template](/en/sft/chat-template)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Knowledge inside model weights is limited: computation, retrieval, and acting on the external world require tools
- [ ] A tool call is essentially constrained structured generation

## 2. Data and Format

TODO:

- [ ] How tool schemas are injected into the prompt (JSON schema in the system section)
- [ ] Role design for call turns / tool-result turns in the chat template
- [ ] Data synthesis pipeline: API library → scenario generation → trajectory generation → validation and filtering (the ToolLLM / APIGen approach)

## 3. Training Methods

TODO:

- [ ] SFT: compute loss on tool-call turns, mask out tool-result turns
- [ ] Preference optimization: construct DPO pairs from correct vs incorrect calls
- [ ] When to decline a call / answer directly (negative samples)

## 4. Evaluation

TODO: BFCL, τ-bench, etc.

## 5. Experiments and Tuning Experience

TODO: common failures: hallucinated arguments, missed calls, excessive calls.

## 6. References

- [ ] Qin et al., 2023. *ToolLLM*
- [ ] Schick et al., 2023. *Toolformer*
