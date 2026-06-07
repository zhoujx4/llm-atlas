---
title: Chat Template
---

# Chat Template

> **一句话**：TODO —— 把多轮对话序列化成模型输入的格式约定；训练和推理必须使用同一套模板。
>
> 前置阅读：[SFT 总览](/sft/)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 为什么需要模板：区分角色、界定回合边界
- [ ] 训练/推理模板不一致是最常见的 SFT 事故

## 2. 主流模板格式

TODO：

- [ ] ChatML（`<|im_start|>` / `<|im_end|>`）
- [ ] Llama 系（`[INST]` / header token）
- [ ] system prompt 的处理方式

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
TODO: 示例<|im_end|>
<|im_start|>assistant
TODO: 示例<|im_end|>
```

## 3. 实现要点

TODO：

- [ ] special token 是否加入词表、embedding 如何初始化
- [ ] HuggingFace `apply_chat_template` 与 jinja 模板
- [ ] 多轮对话拼接与 [Loss Masking](/sft/loss-masking) 的配合
- [ ] generation prompt（推理时模板要补 assistant 起始符）

## 4. 实验与调参经验

TODO：模板错位的典型症状（输出角色标记、停不下来）。

## 5. 参考文献

- [ ] HuggingFace chat templating 文档
