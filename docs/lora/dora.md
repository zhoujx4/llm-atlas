---
title: DoRA：幅值-方向解耦的低秩适配
---

# DoRA（Weight-Decomposed Low-Rank Adaptation）

> **一句话**：把权重分解为「幅值 × 方向」，幅值用一个可训练向量直接学、方向交给 LoRA 更新，使学习模式更接近全量微调，在低秩下显著缩小与全量微调的差距且推理零额外开销。论文 *DoRA: Weight-Decomposed Low-Rank Adaptation* (Liu et al., ICML 2024)。
>
> 提出年份：2024（arXiv 2024-02） · 机构/团队：NVIDIA & HKUST · 会议/来源：ICML 2024（Oral） / arXiv:2402.09353
>
> 前置阅读：[LoRA](/lora/lora) · [记号表](/guide/notation)

## 直觉与动机

DoRA 的出发点是一项对「更新模式」的分析。作者把任意权重矩阵的列分解成两个量：**幅值**（列的 L2 范数，标量）和**方向**（单位化后的列向量），然后观察微调过程中这两者各自变化了多少。

结论很有意思：
- **全量微调**在幅值变化和方向变化之间呈现一种**负相关**——可以做到只微调方向而几乎不动幅值，或反之，表现出灵活、解耦的更新。
- **LoRA**则倾向于让幅值和方向**同向、成比例地一起变**，缺乏「只做细微方向调整」的能力。

这种结构性差异被认为是 LoRA 在低秩下逊于全量微调的一个原因。DoRA 的方案直截了当：**把幅值和方向显式拆开训练**——给幅值一个独立的可训练参数，让它不再受低秩分支牵制；方向仍由 LoRA 负责。这样 LoRA 只需专注于学方向的低秩增量，幅值则获得了全量微调那样的自由度，整体更新模式更贴近全量微调。

## 方法与公式

对一个权重矩阵 $W_0$，先按列做幅值-方向分解，再让 LoRA 只作用在方向分量上：

$$
W = m \cdot \frac{W_0 + \frac{\alpha}{r} B A}{\left\lVert W_0 + \frac{\alpha}{r} B A \right\rVert_c}
$$

其中：
- $m \in \mathbb{R}^{1 \times k}$ 是**可训练的幅值向量**，每列一个标量；
- $\lVert \cdot \rVert_c$ 表示**按列**的 L2 范数（vector-wise norm across columns）；分母把括号内矩阵的每一列归一化为单位向量，即得到纯「方向」；
- $B A$ 仍是 LoRA 的低秩增量，负责更新方向；
- $m$ **初始化为 $\lVert W_0 \rVert_c$**，使训练起点 $W = W_0$，与原模型一致（和 LoRA 的零初始化精神一致）。

直观理解：分子 $W_0 + \Delta W$ 给出更新后的「未归一化方向」，除以列范数把幅值信息剥离掉，只留方向；再乘上独立学习的 $m$ 重新赋予幅值。于是方向由低秩的 $BA$ 控制、幅值由 $m$ 单独控制，两者解耦。

**梯度与开销**：$m$ 引入的参数极少——按列只有一个标量，相对 LoRA 约 +0.01% 的可训练参数。主要的额外开销来自训练时要计算列范数及其梯度（一个标准化操作）。

**推理合并**：训练完后可把整个表达式折叠回一个 dense 权重 $W$，与 LoRA 一样**推理零额外延迟**——这点优于 Adapter 类方法。

![DoRA 概览：将预训练权重分解为幅度（magnitude）与方向（direction）两部分分别微调](/papers/dora/dora-arch.png)

> 图源：Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, arXiv:2402.09353（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA | DoRA |
| --- | --- | --- |
| 额外可训练参数 | $BA$ | $BA + m$（幅值向量，约 +0.01%） |
| 训练开销 | 低 | 略高（列范数标准化及其梯度） |
| 更新模式 | 幅值与方向成比例同变 | 幅值、方向解耦，近似全量微调 |
| 低秩（$r$ 小）下效果 | 基线 | 论文报告多任务上稳定更优 |
| 推理延迟 | 合并后无 | 合并后无 |

## 实现要点

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base                       # 冻结 W0
        for p in self.base.parameters():
            p.requires_grad = False
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * (1 / r ** 0.5))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r
        # m 初始化为 W0 的按列范数
        self.m = nn.Parameter(base.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        W = self.base.weight + self.scaling * (self.B @ self.A)   # W0 + ΔW
        col_norm = W.norm(p=2, dim=0, keepdim=True)               # 按列范数
        W_dora = self.m * (W / col_norm)                          # 重新赋幅值
        return F.linear(x, W_dora)
```

要点：
- **范数沿输入维（列）计算**：对 `[d_out, d_in]` 的权重按 `dim=0`（PEFT 实现里以输出列为单位）求范数并归一化。
- **梯度截断的小技巧**：原论文为降训练开销，对分母列范数做了 `detach`（不回传范数项的二阶梯度），实测对效果影响可忽略而能省显存/算力，PEFT 实现可开启。
- **幅值与方向用同一学习率即可**：DoRA 不强制对 $m$ 单独设 lr，但可与 [LoRA+](/lora/lora-plus) 思想结合微调。
- **可量化**：DoRA 可叠加在 [QLoRA](/lora/qlora) 的 4-bit 基座上（QDoRA），兼顾显存与效果。

## 调参与实践经验

- **何时优于 LoRA**：参数预算紧、秩偏小（$r=4\sim16$）时收益最明显；此时 LoRA 受限于「同变」更新模式，DoRA 的解耦能补回不少效果。秩很大时两者差距收敛。
- **超参沿用 LoRA**：$r$、$\alpha$、目标模块、学习率的设法与 LoRA 基本一致，迁移成本低；直接把现有 LoRA 配置换成 DoRA 通常即可获得提升。
- **训练略慢**：列范数的归一化与反传带来一定开销，但通常远小于 QLoRA 反量化的代价；追效果时是划算的。
- **合并部署**：确认在推理前把 $m \cdot (W_0+\Delta W)/\lVert\cdot\rVert_c$ 折叠成单一权重，避免在线计算范数拖慢推理。

与其它初始化/缩放改进的关系见 [LoRA 家族总览](/lora/)；DoRA 与 PiSSA 都瞄准「逼近全量微调」，前者改结构、后者改初始化，可按任务分别尝试。

## 参考文献

- Liu et al., 2024. *DoRA: Weight-Decomposed Low-Rank Adaptation.* arXiv:2402.09353
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
