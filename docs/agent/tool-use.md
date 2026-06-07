---
title: Tool Use 训练
---

# Tool Use / Function Calling 训练

> **一句话**：TODO —— 教模型在合适的时机、以正确的 schema 发起工具调用，并消化工具返回结果。
>
> 前置阅读：[SFT 总览](/sft/)、[Chat Template](/sft/chat-template)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 模型权重内的知识有限：计算、检索、操作外部世界需要工具
- [ ] tool call 本质是受约束的结构化生成

## 2. 数据与格式

TODO：

- [ ] tool schema 在 prompt 中的注入方式（system 段 JSON schema）
- [ ] 调用轮 / 工具返回轮在 chat template 中的角色设计
- [ ] 数据合成 pipeline：API 库 → 场景生成 → 轨迹生成 → 校验过滤（ToolLLM、APIGen 思路）

## 3. 训练方法

TODO：

- [ ] SFT：对 tool call 轮算 loss，工具返回轮 mask 掉
- [ ] 偏好优化：正确调用 vs 错误调用构造 DPO 对
- [ ] 何时该拒绝调用 / 直接回答（negative 样本）

## 4. 评测

TODO：BFCL、τ-bench 等。

## 5. 实验与调参经验

TODO：常见 failure：幻觉参数、漏调用、过度调用。

## 6. 参考文献

- [ ] Qin et al., 2023. *ToolLLM*
- [ ] Schick et al., 2023. *Toolformer*
