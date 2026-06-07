---
title: 序列 Packing
---

# 序列 Packing

> **一句话**：TODO —— 把多条短样本拼进同一条训练序列以填满 context，可显著提升训练吞吐。
>
> 前置阅读：[SFT 总览](/sft/)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] padding 浪费有多大：短样本场景下有效 token 占比
- [ ] packing 的吞吐收益量级

## 2. 方法

TODO：

- [ ] 朴素拼接 vs first-fit binning
- [ ] cross-contamination 问题：样本间 attention 是否隔离
- [ ] block-diagonal attention mask / position_ids 重置
- [ ] FlashAttention 的 varlen 接口（`cu_seqlens`）

## 3. 与 baseline 对比

| 维度 | Padding | 朴素 Packing | 隔离 Packing |
| --- | --- | --- | --- |
| 吞吐 | 低 | 高 | 高 |
| 样本间泄漏 | 无 | 有 | 无 |
| 实现复杂度 | 低 | 低 | 中 |

## 4. 实现要点与伪代码

```python
# TODO: first-fit packing + cu_seqlens 构造伪代码
```

## 5. 实验与调参经验

TODO：packing 对效果的影响是否可忽略；何时必须隔离 attention。

## 6. 参考文献

- [ ] Krell et al., 2021. *Efficient Sequence Packing*
