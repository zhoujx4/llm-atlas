---
title: PiSSA
---

# PiSSA（Principal Singular Values and Singular Vectors Adaptation）

> **一句话**：TODO —— 用 $W_0$ 的主奇异成分初始化 $A$、$B$，残差部分冻结；训练"最重要的方向"而不是从零增量。
>
> 论文：*PiSSA: Principal Singular Values and Singular Vectors Adaptation* (2024) ·
> 前置阅读：[LoRA](/lora/lora)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] LoRA 的 $B=0$ 初始化使训练初期更新缓慢
- [ ] 直接微调主成分方向收敛更快

## 2. 方法与公式

对 $W_0$ 做 SVD：$W_0 = U S V^\top$，取前 $r$ 个奇异值：

$$
W_0 = \underbrace{U_{[:,:r]} S_{[:r,:r]} V_{[:,:r]}^\top}_{\text{init } BA\text{（可训练）}} + \underbrace{W^{\text{res}}}_{\text{冻结}}
$$

TODO：

- [ ] $A = \sqrt{S_{[:r]}}\,V^\top_{[:r]}$、$B = U_{[:r]}\sqrt{S_{[:r]}}$ 的具体构造
- [ ] 快速 SVD（随机化 SVD）降低初始化成本
- [ ] 与 QLoRA 结合（量化残差，减少量化误差）

## 3. 与 baseline 对比

| 维度 | LoRA | PiSSA |
| --- | --- | --- |
| 初始化 | A 高斯 / B 零 | $W_0$ 主奇异成分 |
| 初期收敛 | 慢 | TODO：更快 |
| 初始化成本 | 零 | 一次 SVD |

## 4. 实现要点与伪代码

```python
# TODO: SVD 初始化伪代码
```

## 5. 实验与调参经验

TODO：与 LoRA/DoRA 的横向对比；合并回基座时的注意点（残差不是 $W_0$）。

## 6. 参考文献

- [ ] Meng et al., 2024. *PiSSA.* arXiv:2404.02948
