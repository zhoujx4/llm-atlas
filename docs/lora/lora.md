---
title: LoRA
---

# LoRA（Low-Rank Adaptation）

> **一句话**：TODO —— 冻结预训练权重 $W_0$，只训练低秩增量 $\Delta W = BA$，可训练参数降到千分之一量级。
>
> 论文：*LoRA: Low-Rank Adaptation of Large Language Models* (2021) ·
> 前置阅读：[全量微调](/sft/full-finetuning)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] 权重更新的"内在秩"很低这一假设
- [ ] 与 Adapter / Prefix-tuning 的对比动机（无推理延迟）

## 2. 方法与公式

前向计算：

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

TODO：

- [ ] 初始化：$A$ 高斯、$B$ 零初始化（保证 $\Delta W$ 起点为 0）
- [ ] 缩放因子 $\alpha/r$ 的作用
- [ ] 部署时合并：$W = W_0 + \frac{\alpha}{r}BA$，零推理开销

## 3. 与 baseline 对比

| 维度 | 全量微调 | LoRA |
| --- | --- | --- |
| 可训练参数 | 100% | ~0.1%-1% |
| 优化器状态显存 | 大 | 极小 |
| 推理延迟 | — | 合并后无 |
| 任务切换 | 整套权重 | 换 adapter |

## 4. 实现要点与伪代码

```python
# TODO: LoRA Linear 层伪代码
```

- [ ] 注入哪些模块：q/k/v/o 还是全部 Linear
- [ ] rank 与 α 常见组合（r=8~64, α=2r）

## 5. 实验与调参经验

TODO：rank 收益递减曲线；lr 通常比全量大一个量级。

## 6. 参考文献

- [ ] Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
