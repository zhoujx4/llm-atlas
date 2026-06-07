---
title: 全量微调
---

# 全量微调（Full Fine-Tuning）

> **一句话**：TODO —— 更新模型全部参数的标准 SFT 做法，效果上限高，显存开销也最大。
>
> 前置阅读：[SFT 总览](/sft/)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 为什么预训练模型不能直接当助手用
- [ ] 全量微调 vs PEFT（LoRA 等）的取舍框架
- [ ] 什么规模/场景下值得全量微调

## 2. 方法与公式

训练目标即标准 NLL：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

TODO：

- [ ] 与预训练目标的异同（数据分布、loss mask、epoch 数）
- [ ] 显存构成：参数 + 梯度 + 优化器状态（Adam 为参数量的约 4 倍）

## 3. 与 baseline 对比

| 维度 | 全量微调 | LoRA |
| --- | --- | --- |
| 可训练参数 | 100% | 通常 < 1% |
| 显存 | TODO | TODO |
| 效果上限 | TODO | TODO |
| 多任务切换 | 需整套权重 | 换 adapter 即可 |

## 4. 实现要点与伪代码

```python
# TODO: 最小 SFT 训练循环伪代码（含 loss masking）
```

- [ ] 学习率量级（通常 1e-5 ~ 2e-5，远小于预训练）
- [ ] epoch 数与早停
- [ ] ZeRO / FSDP 分布式策略选择

## 5. 实验与调参经验

TODO：lr、warmup、epoch、batch size 的常见取值与坑。

## 6. 参考文献

- [ ] Ouyang et al., 2022. *Training language models to follow instructions with human feedback* (InstructGPT)
- [ ] Zhou et al., 2023. *LIMA: Less Is More for Alignment*
