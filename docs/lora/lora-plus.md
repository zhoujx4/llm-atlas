---
title: LoRA+
---

# LoRA+

> **一句话**：给 $B$ 矩阵设置比 $A$ 大得多的学习率（比例 $\lambda \gg 1$），修正 LoRA 在宽网络下 $A$、$B$ 更新尺度天然不对称导致的次优学习动态。（*LoRA+: Efficient Low Rank Adaptation of Large Models*, 2024）
>
> 提出年份：2024（arXiv 2024-02） · 机构/团队：UC Berkeley / Simons Institute · 会议/来源：ICML 2024 / arXiv:2402.12354
>
> 前置阅读：[LoRA](/lora/lora)

## 直觉与动机

[LoRA](/lora/lora) 把增量写成 $\Delta W = \frac{\alpha}{r} BA$，默认用同一个学习率同时更新 $A$ 和 $B$。LoRA+ 指出：在大模型这种"宽网络"（特征维度 $d$ 很大）的极限下，对 $A$、$B$ 用相同学习率在数学上是**次优**的。

直觉来自两点不对称：

1. **初始化不对称**：$A$ 用随机高斯初始化（量级 $O(1)$ 经标准缩放），$B$ 零初始化。训练初期 $A$ 已经携带信息而 $B$ 还是零，二者所处的状态完全不同。
2. **梯度尺度不对称**：前向是 $\frac{\alpha}{r}BAx$。$B$ 的梯度依赖于 $Ax$（已被激活），$A$ 的梯度依赖于 $B^\top$（初始为零、量级很小）。在宽度 $d \to \infty$ 的极限里，用同一学习率会使得其中一个矩阵的特征更新被"卡住"，无法达到使每一层输出都发生 $\Theta(1)$ 量级有效变化的理想学习状态。

LoRA+ 用类似 μP（最大更新参数化）的无穷宽尺度分析证明：要让 $A$、$B$ 的特征学习都处在高效区间，二者的学习率必须按宽度成比例地拉开，且 $B$ 应当远大于 $A$。

## 方法与公式

LoRA+ 把 $A$、$B$ 放进优化器的两个参数组，分别用学习率 $\eta_A$ 与 $\eta_B$：

$$
\eta_B = \lambda \cdot \eta_A, \qquad \lambda \gg 1
$$

只调一个基准学习率 $\eta_A$ 和一个固定比例 $\lambda$。理论分析给出的**最优阶数是 $\lambda \sim \Theta(d)$**（与特征维度同阶）——也就是说网络越宽，$B$ 的学习率就应当相对 $A$ 拉得越开。落到实践，论文建议把 $\lambda$ 当作一个可调常数，在多数语言/NLP 任务上 $\lambda \approx 16$ 是一个稳健的起点；具体最优值仍与任务、模型相关，需小范围扫描。

**与 $B$ 零初始化的联系**：正因为 $B$ 从零起步、且其梯度路径让它的更新天然偏小，给它一个更大的学习率恰好补偿了这一不对称，让 $B$ 能尽快"追上"已经携带信息的 $A$，把整体学习动态拉到高效区间。这也是为什么是 $B$ 的学习率更大，而不是反过来。

整个方法不增加任何参数、不改前向、不改初始化，**唯一改动是优化器的参数分组学习率**，因此推理时与 LoRA 完全一致，可正常合并回基座。

![LoRA 与 LoRA+ 的关键区别：LoRA+ 把 $B$ 的学习率设为 $A$ 的 $\lambda$ 倍（$\lambda \gg 1$）](/papers/lora-plus/loraplus-lr.png)

> 图源：Hayou et al., *LoRA+: Efficient Low Rank Adaptation of Large Models*, arXiv:2402.12354（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA | LoRA+ |
| --- | --- | --- |
| 改动范围 | — | 仅优化器分组学习率（$\eta_B = \lambda\eta_A$） |
| 新增参数 | — | 无 |
| 前向/初始化 | — | 不变 |
| 收敛速度 | 基线 | 论文报告约 1.5–2× |
| 最终效果 | 基线 | 论文报告小幅但稳定提升 |
| 关键超参 | $\eta$ | $\eta_A$ 与比例 $\lambda$ |
| 推理开销 | 合并后无 | 合并后无 |

## 实现要点

```python
# 把 A、B 拆成两个 param group，B 的 lr = lambda * base_lr
lambda_ratio = 16
base_lr = 1e-4

param_groups = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "lora_B" in name:
        param_groups.append({"params": p, "lr": base_lr * lambda_ratio})
    else:  # lora_A 及其它可训练参数
        param_groups.append({"params": p, "lr": base_lr})

optimizer = torch.optim.AdamW(param_groups)
```

注意点：分组时要确保 bias、归一化等可训练项归到基准组而不是 $B$ 组；若用带 weight decay 的优化器，$B$ 组学习率被放大后通常无需同步放大 weight decay。HuggingFace `peft` / 部分训练框架已内置 LoRA+ 选项，可直接传 `loraplus_lr_ratio`。

## 调参与实践经验

- **$\lambda$ 的敏感性。** $\lambda$ 偏小（接近 1）就退化回普通 LoRA；偏大（如几百）则 $B$ 学习率过激、训练不稳。实践中 $4\sim32$ 是常用区间，$16$ 是不错的默认；对极宽的大模型可以往更大试，因为理论最优随 $d$ 增长。
- **先定基准 lr 再定比例。** 建议先按普通 LoRA 找到一个可用的基准学习率，再固定它去扫 $\lambda$，比同时调两个参数高效。
- **收益场景。** LoRA+ 的主要卖点是**更快收敛**，在训练步数受限、或需要快速跑通多个配置时收益最明显；最终效果提升通常是小幅的。
- **与其他变体叠加。** LoRA+（学习率比）与 [rsLoRA](/lora/rslora)（缩放因子）、[AdaLoRA](/lora/adalora)（秩分配）关注点正交，理论可叠加。但 LoRA+ 与 rsLoRA 都改变有效尺度，叠加时务必把两个旋钮分开标定——先固定 rsLoRA 的 $\sqrt{r}$ 缩放，再单独调 LoRA+ 的 $\lambda$，否则容易互相掩盖效果。
- **与 [PiSSA](/lora/pissa) 的区别。** 两者都想解决 LoRA "初期学得慢"的问题，但路径不同：LoRA+ 从优化器学习率入手，PiSSA 从初始化（主成分）入手，可视情况二选一或组合。

## 参考文献

- Hayou et al., 2024. *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv:2402.12354
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
