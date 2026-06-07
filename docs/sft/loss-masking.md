---
title: Loss Masking
---

# Loss Masking

> **一句话**：TODO —— SFT 时只对 assistant 回答部分计算 loss，prompt 与模板 token 不参与梯度。
>
> 前置阅读：[SFT 总览](/sft/)、[Chat Template](/sft/chat-template)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 为什么不对 prompt 算 loss：避免学习"复述问题"的分布
- [ ] 也有工作对 prompt 算 loss（小数据场景），trade-off 是什么

## 2. 方法与公式

带 mask 的 SFT 损失：

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid y_{<t}), \quad m_t \in \{0, 1\}
$$

TODO：

- [ ] $m_t$ 的构造规则：user 轮、system、模板 token 置 0
- [ ] 多轮对话：每一轮 assistant 都算还是只算最后一轮

## 3. 实现要点与伪代码

```python
# TODO: labels 置 -100 的标准 HuggingFace 写法
```

- [ ] `ignore_index=-100` 约定
- [ ] 与 [Packing](/sft/packing) 同时使用时 mask 的拼接
- [ ] 归一化口径：按 token 平均还是按样本平均

## 4. 实验与调参经验

TODO：mask 错位的典型症状。

## 5. 参考文献

- [ ] TODO
