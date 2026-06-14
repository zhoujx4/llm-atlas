---
title: rsLoRA
---

# rsLoRA（Rank-Stabilized LoRA）

> **一句话**：把 LoRA 的缩放因子从 $\alpha/r$ 改成 $\alpha/\sqrt{r}$，使得增量在高秩下的尺度保持稳定，从而真正享受到大 rank 带来的容量，而不是因尺度衰减提前饱和。（*A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, 2023）
>
> 提出年份：2023（arXiv 2023-12） · 机构/团队：Damien Kalajdzievski（Telus / Mila） · 会议/来源：arXiv:2312.03732
>
> 前置阅读：[LoRA](/lora/lora)

## 直觉与动机

[LoRA](/lora/lora) 的前向是 $h = W_0 x + \frac{\alpha}{r} BA x$，缩放因子取 $\alpha/r$。这个 $1/r$ 在最初的论文里只是一个"让你换 rank 时不必重调学习率"的工程便利项，并没有从尺度稳定的角度严格推导过。

问题在于：当你把 $r$ 调大想换取更多容量时，$1/r$ 这个除数也线性变大，把每个秩-1 分量的贡献压得越来越小。直觉上，$BA$ 这个矩阵积的输出尺度本来就随着 $r$ 的增大而增大（更多项相加），而 $1/r$ 的压制过强——两者叠加的净效果是：**增量 $\Delta W x$ 的尺度随 $r$ 增大而衰减**，相当于隐式地把有效学习率随 rank 越调越小。结果就是大家普遍观察到的"LoRA 的 rank 收益很快饱和"：从 $r=8$ 加到 $r=64$，效果几乎不动，因为你加的那些维度被缩放因子掐死了。

rsLoRA 的洞察是：要让"加 rank 真的有用"，缩放因子必须选得让 $\Delta W$ 的尺度对 $r$ 不变。

## 方法与公式

从初始化角度做一次方差分析。设 $A$ 的元素独立同分布、方差为常数，$B$ 零初始化（或对称地分析其前向尺度）。$BA x$ 中每个输出元素是 $r$ 个独立项之和，其方差按 $\Theta(r)$ 增长，标准差则按 $\Theta(\sqrt{r})$ 增长。要让缩放后的增量 $\gamma_r \cdot BA x$ 的尺度（标准差量级）不随 $r$ 变化，缩放因子 $\gamma_r$ 必须满足 $\gamma_r \propto 1/\sqrt{r}$。

于是 rsLoRA 把缩放因子改为：

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}}\, BA x
$$

论文给出了更一般的结论：在 $r \to \infty$ 的极限下，唯一能使前向与梯度都既不爆炸也不消失（rank-stabilized）的缩放阶数就是 $\Theta(1/\sqrt{r})$；$\alpha/r$ 属于"过度衰减"，会导致大 rank 下学习停滞。换句话说，$1/\sqrt{r}$ 不是一个经验技巧，而是保持训练动态尺度不变的唯一正确阶数。

**实现上只改一行**：把 $\alpha/r$ 换成 $\alpha/\sqrt{r}$。其余初始化、合并方式（部署时仍可把 $W_0 + \frac{\alpha}{\sqrt{r}}BA$ 合并回基座，零推理开销）都与 LoRA 完全一致。

![不同 rank 下的微调困惑度：标准 LoRA（铜色）随 rank 增大基本停滞，rsLoRA（蓝绿色）则随 rank 增大持续提升](/papers/rslora/rslora-rank.png)

> 图源：Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, arXiv:2312.03732（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA（$\alpha/r$） | rsLoRA（$\alpha/\sqrt{r}$） |
| --- | --- | --- |
| 缩放因子阶数 | $\Theta(1/r)$（过度衰减） | $\Theta(1/\sqrt{r})$（尺度稳定） |
| 小 rank（8–16） | 与 rsLoRA 基本无差异 | 基本无差异 |
| 大 rank（≥64） | 收益快速饱和 | 持续受益，效果随 rank 继续提升 |
| 实现成本 | — | 一行改动 |
| 推理开销 | 合并后无 | 合并后无 |
| 需重标定的超参 | — | $\alpha$、学习率 |

注意：在小 rank 时两者差距小，因为 $r$ 和 $\sqrt{r}$ 此时数值接近（$r=4$ 时 $\sqrt{r}=2$，差一倍但可被 $\alpha$/lr 吸收）；差异是在你想"用大 rank 换效果"时才决定性地显现。

## 实现要点

```python
# LoRA 前向，唯一区别是 scaling 的分母
import math

scaling = alpha / math.sqrt(r)   # rsLoRA；标准 LoRA 为 alpha / r
h = x @ W0.T + scaling * (x @ A.T) @ B.T
```

在 HuggingFace `peft` 中开启极其简单——`LoraConfig(use_rslora=True)` 即可，库会自动把缩放分母改成 $\sqrt{r}$。

## 调参与实践经验

- **换缩放后必须重标定 $\alpha$ 和学习率。** 因为有效尺度变了，直接沿用 LoRA 的 $\alpha$/lr 组合可能偏大或偏小。一个实用做法是固定 $\alpha$（不再用 $\alpha=2r$ 这类与 $r$ 绑定的经验），把学习率当作主要调节旋钮重新扫一遍。
- **什么时候值得开。** 只有当你**确实想用大 rank**（如 $r \ge 64$，做较重的领域适配或需要更大增量容量）时，rsLoRA 才有明显价值。如果你本来就只用 $r=8\sim16$ 的轻量适配，开不开几乎无差别，没必要额外引入调参负担。
- **与 QLoRA 叠加。** rsLoRA 只动缩放因子，与 [QLoRA](/lora/qlora) 的量化基座完全正交，常被一起使用：QLoRA 省显存让你能上更大 rank，rsLoRA 让这些 rank 真正发挥作用。
- **与其他变体的关系。** rsLoRA（缩放）、[LoRA+](/lora/lora-plus)（$A/B$ 学习率比）、[AdaLoRA](/lora/adalora)（秩分配）各管一件事，可组合；其中 rsLoRA 与 LoRA+ 都涉及"尺度"，叠加时建议先固定 rsLoRA 的缩放，再单独标定 LoRA+ 的学习率比，避免两个尺度旋钮互相干扰。

## 参考文献

- Kalajdzievski, 2023. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA).* arXiv:2312.03732
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
