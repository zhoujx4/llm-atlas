---
title: Chat Template
translation: synced
---

# Chat Template

> **In one sentence**: TODO — the format convention for serializing multi-turn conversations into model input; training and inference must use the same template.
>
> Prerequisites: [SFT Overview](/en/sft/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Why templates are needed: distinguishing roles, delimiting turn boundaries
- [ ] Training/inference template mismatch is the most common SFT accident

## 2. Mainstream Template Formats

TODO:

- [ ] ChatML (`<|im_start|>` / `<|im_end|>`)
- [ ] Llama family (`[INST]` / header tokens)
- [ ] How the system prompt is handled

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
TODO: example<|im_end|>
<|im_start|>assistant
TODO: example<|im_end|>
```

## 3. Implementation Notes

TODO:

- [ ] Whether special tokens are added to the vocabulary and how their embeddings are initialized
- [ ] HuggingFace `apply_chat_template` and Jinja templates
- [ ] Multi-turn concatenation in combination with [Loss Masking](/en/sft/loss-masking)
- [ ] Generation prompt (at inference the template must append the assistant start marker)

## 4. Experiments and Tuning Experience

TODO: typical symptoms of template misalignment (emitting role markers, failing to stop).

## 5. References

- [ ] HuggingFace chat templating documentation
