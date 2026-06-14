---
title: PiSSA
---

# PiSSA（Principal Singular Values and Singular Vectors Adaptation）

> **一句话**：用 $W_0$ 的主奇异成分来初始化可训练的 $A$、$B$，把剩下的次要成分冻结成残差矩阵；于是你从一开始就在微调"权重里最重要的方向"，而不是从零增量慢慢学起。（*PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models*, 2024）
>
> 提出年份：2024（arXiv 2024-04） · 机构/团队：Peking University · 会议/来源：NeurIPS 2024（Spotlight） / arXiv:2404.02948
>
> 前置阅读：[LoRA](/lora/lora)

## 直觉与动机

[LoRA](/lora/lora) 默认 $A$ 高斯、$B$ 零初始化，保证训练起点 $\Delta W = BA = 0$。这个设计安全（不破坏基座），但代价是：训练最初的若干步里，增量几乎全靠 $B$ 从零慢慢长出来，梯度信号小、收敛慢，相当于把一部分训练预算花在"启动"上。

更重要的是，LoRA 的初始增量与 $W_0$ 的结构完全无关——它在一个随机的低秩子空间里盲目摸索，要花时间才能对齐到对任务真正有用的方向。

PiSSA 换了个角度：既然 $W_0$ 本身已经把信息按重要性编码在它的奇异谱里，那**最值得微调的方向，就是 $W_0$ 的主奇异方向**。把这部分主成分拿出来作为可训练的低秩部分初始化，剩下的次要成分原封不动冻结。这样训练一开始就站在"高信息量、与基座对齐"的子空间里，收敛更快、最终效果也更好。

## 方法与公式

对每个目标权重 $W_0$ 做 SVD：

$$
W_0 = U S V^\top
$$

取前 $r$ 个奇异值/向量构成主成分部分，剩余部分作为冻结残差：

$$
W_0 = \underbrace{U_{[:,:r]}\, S_{[:r,:r]}\, V_{[:,:r]}^\top}_{\text{初始化 } BA\ (\text{可训练})} \;+\; \underbrace{W^{\text{res}}}_{\text{冻结}}
$$

可训练的 $A$、$B$ 按"奇异值开方均分到两侧"的方式构造，使其乘积恰好等于主成分部分：

$$
B = U_{[:,:r]}\, S_{[:r,:r]}^{1/2}, \qquad A = S_{[:r,:r]}^{1/2}\, V_{[:,:r]}^\top
$$

于是 $BA = U_{[:,:r]} S_{[:r,:r]} V_{[:,:r]}^\top$ 正是 $W_0$ 的秩-$r$ 主成分近似。残差矩阵则是被减掉主成分后的剩余：

$$
W^{\text{res}} = W_0 - BA = U_{[:,r:]}\, S_{[r:,r:]}\, V_{[:,r:]}^\top
$$

前向计算与 LoRA 同形 $h = W^{\text{res}} x + BA x$，但注意此时 $BA \ne 0$，且被冻结的是 $W^{\text{res}}$ 而非完整的 $W_0$。

**关键工程细节——残差不是 $W_0$。** 训练结束要把 adapter 合并回去时，正确的权重是 $W = W^{\text{res}} + B'A'$（$B'A'$ 是训练后的值），而**不是** $W_0 + B'A'$。如果框架里残差仍存的是 $W_0$，必须额外减去初始主成分 $B_0A_0$，否则会把主成分算两遍。

**快速初始化。** 对大模型每一层都做完整 SVD 代价不小，PiSSA 用随机化 SVD（randomized SVD）只求前 $r$ 个奇异成分，几秒内即可完成整模型初始化，相对训练成本可忽略。

![全量微调、LoRA 与 PiSSA 的对比：蓝色为冻结部分，橙色为可训练部分；PiSSA 训练主成分而冻结残差](/papers/pissa/pissa-arch.png)

> 图源：Meng et al., *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models*, arXiv:2404.02948（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA | PiSSA |
| --- | --- | --- |
| $A$/$B$ 初始化 | $A$ 高斯、$B$ 零 | $W_0$ 的主奇异成分 |
| 冻结部分 | 完整 $W_0$ | 残差 $W^{\text{res}}=W_0-BA$ |
| 初始增量 | $BA = 0$ | $BA = W_0$ 的秩-$r$ 主成分 |
| 初期收敛 | 慢（从零启动） | 更快 |
| 最终效果 | 基线 | 论文报告优于 LoRA |
| 初始化成本 | 零 | 一次（随机化）SVD，秒级 |
| 合并回基座 | $W_0 + BA$ | $W^{\text{res}} + BA$（注意不是 $W_0+BA$） |

## 实现要点

```python
import torch

def pissa_init(W0, r):
    # 随机化/截断 SVD 取前 r 个主成分
    U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
    Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]

    sqrt_S = torch.diag(Sr.sqrt())
    B = Ur @ sqrt_S            # d x r
    A = sqrt_S @ Vhr           # r x k

    W_res = W0 - B @ A         # 冻结残差（= 次要奇异成分之和）
    return A, B, W_res

# 前向：注意冻结的是 W_res，不是 W0
# h = x @ W_res.T + (x @ A.T) @ B.T
```

实现注意：SVD 要在 fp32 下做以保证数值精度；得到的 $W^{\text{res}}$ 替换原权重存储；HuggingFace `peft` 通过 `LoraConfig(init_lora_weights="pissa")` 直接支持，并提供把 PiSSA 残差转回标准 LoRA 格式的转换工具，方便复用 LoRA 的部署链路。

## 调参与实践经验

- **与 LoRA/DoRA 横向对比。** PiSSA 的主要增益在"更快收敛 + 略好的最终质量"，本质来自更好的初始化；它和 [DoRA](/lora/dora)（幅值/方向解耦）、[LoRA+](/lora/lora-plus)（学习率比）解决的是不同子问题，必要时可组合。若只追求实现简单，PiSSA 是性价比很高的"换初始化即用"改进。
- **与 [QLoRA](/lora/qlora) 结合（QPiSSA）。** PiSSA 把主成分（数值大、对量化误差最敏感）放进**不量化**的可训练 $BA$，只对残差 $W^{\text{res}}$ 做 4-bit 量化。由于残差去掉了主成分、奇异谱更平、动态范围更小，量化误差显著低于直接量化整个 $W_0$。这使 PiSSA + 量化的精度损失明显优于朴素 QLoRA，是 PiSSA 一个很实用的卖点。
- **合并陷阱再强调。** 上线前务必确认合并用的是 $W^{\text{res}} + B'A'$。把残差误当成 $W_0$ 是最常见的复现错误，会导致主成分被叠加两次、模型行为异常。
- **rank 选择。** 与 LoRA 类似，$r$ 越大主成分覆盖越全、初始增量越接近 $W_0$ 的低秩近似；但 PiSSA 的优势主要体现在中小 rank 下"初始化质量"带来的差距，rank 极大时与 LoRA 的差距会收窄。

## 参考文献

- Meng et al., 2024. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* arXiv:2404.02948
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
