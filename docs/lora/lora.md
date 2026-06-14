---
title: LoRA：低秩适配
---

# LoRA（Low-Rank Adaptation）

> **一句话**：冻结预训练权重 $W_0$，只训练一对低秩矩阵 $\Delta W = BA$，把可训练参数压到千分之一量级，且训练完可合并回原权重、推理零延迟。论文 *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)。
>
> 提出年份：2021（arXiv） · 机构/团队：Microsoft · 会议/来源：ICLR 2022 / arXiv:2106.09685
>
> 前置阅读：[全量微调](/sft/full-finetuning) · [记号表](/guide/notation)

## 直觉与动机

全量微调 70B 模型，光是 AdamW 的一阶、二阶动量就要占去权重本身 2 倍的显存，再加上梯度，单任务训练动辄上百 GB；每个下游任务还要存一份完整 checkpoint。问题是：适配一个具体任务，真的需要更新全部参数吗？

LoRA 的核心假设是**权重更新具有低「内在秩」**。预训练模型已经学到了通用表示，适配下游任务所需的改动 $\Delta W$ 落在一个低维子空间里——也就是说 $\Delta W$ 这个满秩矩阵可以用一个秩远小于其维度的矩阵很好地近似。既然如此，就不必把 $\Delta W$ 当作 $d \times k$ 个自由参数来学，而是把它分解成两个瘦长矩阵的乘积。

相比之前的 PEFT 方案，LoRA 的关键优势是**推理零开销**：
- Adapter 在每层之间串联额外的瓶颈 MLP，必然增加前向的串行深度和延迟；
- Prefix/Prompt-tuning 把可学习向量拼到序列前面，挤占了宝贵的上下文长度；
- LoRA 的增量是对原权重的线性叠加，训练完直接做 $W = W_0 + \Delta W$ 合并，得到的就是一个普通的 dense 权重，推理时与原模型结构、速度完全一致。

## 方法与公式

对任意一个线性层 $h = W_0 x$，LoRA 把权重更新约束为低秩分解，前向变为：

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 冻结不动，$B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$ 是仅有的可训练参数，秩 $r \ll \min(d, k)$。可训练参数量从 $d \times k$ 降到 $r \times (d + k)$，当 $r=8$、$d=k=4096$ 时压缩比约 256 倍。

**初始化**是 LoRA 能稳定起步的关键：$A$ 用随机高斯（如 Kaiming）初始化，$B$ 用全零初始化。这样训练开始的瞬间 $\Delta W = BA = 0$，模型行为与原始预训练模型完全一致，从而把 LoRA 视为对预训练点的一个「无伤起步」的扰动，避免初始的随机增量破坏已学到的能力。

**缩放因子 $\alpha/r$** 用来调节低秩分支的影响幅度。$\alpha$ 是一个与 $r$ 解耦的常数：固定 $\alpha$ 改变 $r$ 时，缩放因子会反向变化，使不同秩之间无需重新搜超参。直观上 $\alpha/r$ 的作用近似于给 LoRA 分支单独设了一个学习率倍率——增大 $\alpha$ 等价于让 $\Delta W$ 走得更远。

**部署合并**：训练结束后计算
$$
W = W_0 + \frac{\alpha}{r} B A
$$
把结果写回权重矩阵，得到一个标准 dense 模型。多任务场景下也可不合并，保留多个 $(B, A)$ 适配器，按请求热切换——切换成本只是几 MB 的矩阵而非整套权重。

![LoRA 的低秩重参数化：冻结预训练权重 $W$，只训练旁路的低秩矩阵 $A$、$B$](/papers/lora/lora-arch.png)

> 图源：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | 全量微调 | LoRA |
| --- | --- | --- |
| 可训练参数 | 100% | ~0.1%–1% |
| 优化器状态显存 | 巨大（动量随参数量） | 极小（只覆盖 $B,A$） |
| checkpoint 体积 | 整套权重（GB 级） | 单个 adapter（MB 级） |
| 推理延迟 | — | 合并后无额外延迟 |
| 任务切换 | 换整套权重 | 换 adapter，热插拔 |
| 新知识容纳量 | 高 | 受秩 $r$ 限制 |

## 实现要点

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base                      # 冻结的 W0
        for p in self.base.parameters():
            p.requires_grad = False
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * (1 / r ** 0.5))  # 高斯
        self.B = nn.Parameter(torch.zeros(d_out, r))                  # 零初始化
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scaling * (self.dropout(x) @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):                          # 部署时合并回基座
        self.base.weight += self.scaling * (self.B @ self.A)
```

实现上要注意：
- **注入哪些模块**：最小集是注意力的 $q,v$ 投影（论文的主推配置）；要逼近全量微调，应把 $k,o$ 乃至 MLP 的所有 Linear 都加上。覆盖越全，效果上限越高、参数也越多。
- **dropout 加在输入侧**：作用于 $x$ 进入 LoRA 分支之前，不影响主干。
- **混合精度**：$B,A$ 一般保持 bf16/fp32，与基座 dtype 解耦。
- **不要对 LayerNorm、embedding、lm_head 盲目加 LoRA**：embedding 维度大、收益有限，需要时单独处理。

## 调参与实践经验

- **秩 $r$**：从 8 起步，效果不够再翻到 16/32/64。收益随 $r$ 递减——任务越接近预训练分布，越小的 $r$ 就够；要灌入大量新知识时小秩会成为瓶颈。
- **$\alpha$**：经验默认 $\alpha = 2r$。调大 $\alpha$ 相当于放大 LoRA 分支的步长，过大易不稳。注意若改用 [rsLoRA](/lora/rslora) 的 $\alpha/\sqrt{r}$ 缩放，高秩下行为更稳定。
- **学习率**：通常比全量微调大一个数量级，$1\text{e-}4 \sim 5\text{e-}4$ 是常见区间，因为只更新极少量参数、梯度信号需要更大步长。可进一步参考 [LoRA+](/lora/lora-plus) 给 $B$ 设更大 lr。
- **目标模块优先级**：显存紧→只加 $q,v$；追效果→全 Linear。MLP 模块参数量占大头，加上它对复杂推理任务帮助明显。
- **过拟合**：小数据集（几千条以内）配 LoRA dropout 0.05~0.1，并适当减小 $r$。

进一步的初始化、缩放、结构改进见 [LoRA 家族总览](/lora/)；显存受限场景见 [QLoRA](/lora/qlora)；想更贴近全量微调见 [DoRA](/lora/dora)。

## 参考文献

- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Aghajanyan et al., 2020. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv:2012.13255
