---
title: QLoRA
---

# QLoRA

> **一句话**：TODO —— 把冻结的基座权重量化到 4-bit（NF4），LoRA 适配器仍用 bf16 训练，单卡即可微调大模型。
>
> 论文：*QLoRA: Efficient Finetuning of Quantized LLMs* (2023) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] LoRA 的显存大头仍是基座权重本身
- [ ] 量化只用于"存储"，反量化后参与计算

## 2. 方法与公式

前向计算：

$$
h = \mathrm{dequant}(W_0^{\text{NF4}})\, x + \frac{\alpha}{r} B A x
$$

三个核心组件，TODO 展开：

- [ ] **NF4**：面向正态分布权重的信息论最优 4-bit 数据类型
- [ ] **Double Quantization**：对量化常数再量化
- [ ] **Paged Optimizer**：显存峰值溢出到 CPU

## 3. 与 baseline 对比

| 维度 | LoRA (bf16 基座) | QLoRA (NF4 基座) |
| --- | --- | --- |
| 基座显存 | 2 bytes/param | ~0.5 bytes/param |
| 训练速度 | 快 | 慢（反量化开销） |
| 效果 | TODO | 论文称与 16-bit 持平 |

## 4. 实现要点与伪代码

```python
# TODO: bitsandbytes load_in_4bit + peft 最小示例
```

## 5. 实验与调参经验

TODO：适合的场景（显存极度受限）；与 LoRA 的速度/效果实测差异。

## 6. 参考文献

- [ ] Dettmers et al., 2023. *QLoRA.* arXiv:2305.14314
