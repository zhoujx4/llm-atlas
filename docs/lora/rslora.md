---
title: rsLoRA（Tenyx）：把 LoRA 缩放因子从 α/r 改成 α/√r，解锁大 rank
---

# rsLoRA（Tenyx）：把 LoRA 缩放因子从 α/r 改成 α/√r，解锁大 rank

**📄 [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)**

2023-12 · Tenyx（Damjan Kalajdzievski）

**一句话**：从无穷宽极限的学习动态出发，证明 LoRA 适配器的正确缩放阶数是 $\alpha/\sqrt{r}$ 而非 $\alpha/r$——只改这一行，就能让"加 rank 真的有用"，把更多训练算力换成更好的微调效果，且推理零额外开销。

::: details 📖 论文原文 Abstract（英文）
As large language models (LLMs) have become increasingly compute and memory intensive, parameter-efficient fine-tuning (PEFT) methods are now a common strategy to fine-tune LLMs. A popular PEFT method is Low-Rank Adapters (LoRA), which adds trainable low-rank "adapters" to selected layers. Each adapter consists of a low-rank matrix product, multiplicatively scaled by a rank-dependent factor. This scaling factor, which divides adapters by a factor of the rank, results in slowed learning and stunted performance for LoRA with higher-rank adapters. Consequently, the use of LoRA in practice has generally been limited to very low ranks. In this work, we study the impact of the scaling factor on the learning process and prove that LoRA adapters should be divided by a factor of the square root of the rank. Modifying LoRA with the appropriate scaling factor, which we call the rank-stabilized LoRA (rsLoRA) method, easily provides for a fine-tuning compute/performance trade-off, where larger ranks can be used to trade off increased computational resources during training for better fine-tuning performance, with no change in inference computing cost.
:::

**相关**：[LoRA 总览](/lora/) · [LoRA](/lora/lora) · [QLoRA](/lora/qlora) · [LoRA+](/lora/lora-plus) · [AdaLoRA](/lora/adalora) · [DoRA](/lora/dora) · [PiSSA](/lora/pissa)

![LoRA 适配器结构：冻结的预训练权重 $W$ 与可训练的低秩支路 $\gamma_r BA$ 并联，$B$ 零初始化、$A$ 高斯初始化，输出相加得到 $h$](/papers/rslora/arch.png)

> 图源：Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*（arXiv:2312.03732）Figure 1（引自 Hu et al., 2022）——LoRA 适配器示意（用于学习注解，版权归原作者）。

## 动机与创新点：α/r 把大 rank 掐死了，rsLoRA 用 α/√r 把它救回来

[LoRA](/lora/lora) 的前向是 $x_{\text{out}} = (W + \gamma_r BA)\,x_{\text{in}} + b$，其中缩放因子在原始论文里被设成 $\gamma_r = \alpha/r$。这个 $1/r$ 当初只是"换 rank 时不必重调学习率"的工程便利项，**从未从尺度稳定的角度严格推导过**。

问题在于：$BA$ 这个矩阵积的输出尺度本来就随 $r$ 增大而增大（更多秩-1 项相加），而 $1/r$ 的压制**过强**——两者叠加的净效果是增量 $\gamma_r BA\,x$ 的尺度随 $r$ 增大而衰减，相当于隐式地把有效学习率随 rank 越调越小。论文把这个现象称为 **gradient collapse**（梯度坍缩）：

> the setting of the scaling factor $\gamma_r$ in LoRA is overly aggressive and causes gradient collapse as the rank increases, which slows the learning such that LoRA fine-tuning using larger ranks performs no different than that with very small ranks.

这正是社区普遍观察到的"LoRA 的 rank 收益很快饱和"：从 $r=8$ 加到 $r=64$ 效果几乎不动，加上去的维度被缩放因子掐死了。作者甚至指出，这可能误导了原始 LoRA 论文得出"很低的 rank（4/8/16）就够用（suffice）"的结论——因为在 $\alpha/r$ 下 $r=64$ 本来就训不出提升。

rsLoRA 的洞察是：要让"加 rank 真的有用"，缩放因子必须选得让增量 $\gamma_r BA\,x$ 的尺度对 $r$ 不变。论文从 Yang & Hu (2022) 的无穷宽（infinite-width，这里取 $r\to\infty$）学习动态分析出发，严格证明唯一能让前向与梯度都既不爆炸也不消失的缩放阶数是 $\Theta(1/\sqrt{r})$。

**关键创新**：

- **理论定阶**：把 LoRA 的"缩放-初始化-更新"放进 Yang & Hu 的无穷宽框架分析，给出 **Definition 3.1（rank-stabilized）+ Theorem 3.2**，证明 rank 稳定当且仅当 $\gamma_r\in\Theta(1/\sqrt{r})$。
- **一行改动**：把 $\gamma_r=\alpha/r$ 换成 $\gamma_r=\alpha/\sqrt{r}$，其余初始化、合并方式都与 LoRA 完全一致——这就是 **rsLoRA**。
- **解锁算力/性能权衡**：因为 rsLoRA 不再坍缩，"加 rank → 更好"重新成立，可以把更大的训练算力换成更好的微调效果，而**推理零额外开销**（合并回基座）。
- **充分实验验证**：在 Llama 2、GPT-J 上，跨 AdamW/SGD/Adafactor、跨困惑度/梯度范数，一致复现"LoRA 坍缩、rsLoRA 稳定"，并用消融排除"只是变相调大学习率"的解释。

## 方法：从无穷宽学习动态推出唯一正确的缩放阶数

### LoRA 适配器的形式与缩放因子的角色

先把记号摆清楚。一个线性子模块 $x_{\text{out}} = W x_{\text{in}} + b$（$W\in\mathbb{R}^{d_2\times d_1}$）被加上一个低秩适配器，得到

$$x_{\text{out}} = (W + \gamma_r BA)\,x_{\text{in}} + b$$

其中 $A\in\mathbb{R}^{r\times d_1}$、$B\in\mathbb{R}^{d_2\times r}$，初始化为 $B = 0_{d_2\times r}$、$A$ 的元素 iid、均值 0、方差 $\sigma_A$ **不依赖于 $r$**；$\gamma_r\in\mathbb{R}^+$ 是依赖 rank 的缩放因子（结构见页首 Figure 1）。微调后把 $(W + \gamma_r BA)$ 合并成单个矩阵替换 $W$，所以"推理时零额外开销"。适配器的秩至多 $r$，通常 $r\ll d_1,d_2$。

关键问题就是 $\gamma_r$ 该怎么随 $r$ 取。直觉上的方差分析已经能给出答案：$BAx$ 中每个输出元素是 $r$ 个独立项之和，方差按 $\Theta(r)$ 增长、标准差按 $\Theta(\sqrt{r})$ 增长；要让缩放后的增量尺度不随 $r$ 变化，必须 $\gamma_r\propto 1/\sqrt{r}$。LoRA 用的 $\alpha/r$ 比这更狠，属于**过度衰减**；而下面的定理把这个直觉升级成在整个学习轨迹上都成立的充要条件。

> 举例：把 $r$ 从 4 加到 2048（512 倍），$\sqrt{r}$ 只放大约 22.6 倍，但 $r$ 放大了 512 倍。LoRA 的 $\alpha/r$ 等于在大 rank 上额外乘了一个约 $1/22.6$ 的衰减，把新增维度的贡献压没了；rsLoRA 的 $\alpha/\sqrt{r}$ 恰好抵掉这块多出来的衰减。

### Definition 3.1 + Theorem 3.2：rank 稳定 ⟺ Θ(1/√r)

论文先定义什么叫"对 rank 稳定"。直觉是：**前向**（激活的矩）和**反向**（梯度的量级）都不应随 $r$ 爆炸或消失。

> **Definition 3.1.** An adapter $\gamma_r BA$ is **rank-stabilized** if the following two conditions hold:
> 1. If the inputs to the adapter are iid such that the $m$'th moment is $\Theta_r(1)$ in each entry, then the $m$'th moment of the outputs of the adapter is also $\Theta_r(1)$ in each entry.
> 2. If the gradient of the loss with respect to the adapter outputs are $\Theta_r(1)$ in each entry, then the loss gradients into the input of the adapter are also $\Theta_r(1)$ in each entry.

这里 $\Theta_r(1)$ 表示"关于 $r$ 是常数阶"（既不随 $r$ 趋 0 也不趋 ∞）。基于此，主定理给出唯一可行的缩放阶数：

> **Theorem 3.2.** Consider LoRA adapters of the form $\gamma_r BA$ … In expectation over initialization, all adapters are rank-stabilized **if and only if**
> $$\gamma_r \in \Theta_r\!\left(\tfrac{1}{\sqrt{r}}\right).$$
> In particular, the above holds at any point in the learning trajectory, and unless $\gamma_r \in \Theta_r(\tfrac{1}{\sqrt{r}})$, there is unstable or collapsing learning for sufficiently large values of $r$.

证明思路（附录 A）：先对一个适配器做归纳——$B$ 从 0 出发、经 $n$ 步 SGD 后，对初始化 $A_0$ 取期望，利用 $\mathbb{E}_{A_0}[A_0^\top A_0] = r\sigma_A\, I$ 这一项把 $r$ 显式拽出来，得到前向输出与反向梯度都带一个 $\Theta_r(\gamma_r^2 r)$ 因子。要让它在 $r\to\infty$ 下既不爆炸也不消失，就需要 $\gamma_r^2 r = \Theta_r(1)$，即 $\gamma_r\in\Theta_r(1/\sqrt{r})$。再把单层结论沿前向（激活）与反向（梯度）在层间归纳传播，得到整网的稳定性。

于是 rsLoRA 取最简单的代表元：

$$\gamma_r = \frac{\alpha}{\sqrt{r}}, \qquad h = W_0 x + \frac{\alpha}{\sqrt{r}}\, BA\,x$$

注意 LoRA 的 $\alpha/r = \Theta_r(1/r)$ **不属于** $\Theta_r(1/\sqrt{r})$，因此落在"过度衰减、大 rank 必坍缩"那一侧；这不是经验技巧，而是保持训练动态尺度不变的**唯一正确阶数**。论文也提醒：定理只刻画了 $r\to\infty$ 极限下的稳定/坍缩，并未断言"稳定时不同 rank 学到的特征质量"，后者要靠实验验证——这恰好引出第 4 节。

### 实现：只改 scaling 的分母

实现上 rsLoRA 与 LoRA 的唯一区别就是把缩放分母从 $r$ 换成 $\sqrt{r}$：

```python
import math

# 标准 LoRA: scaling = alpha / r
scaling = alpha / math.sqrt(r)          # rsLoRA：唯一改动
h = x @ W0.T + scaling * (x @ A.T) @ B.T
```

在 HuggingFace `peft` 中开启极其简单——`LoraConfig(use_rslora=True)`，库会自动把缩放分母改成 $\sqrt{r}$。初始化（$B=0$、$A$ 高斯）、合并部署（$W_0 + \frac{\alpha}{\sqrt{r}}BA$ 合回基座、推理零开销）都与 LoRA 完全一致。

## 实验结果：困惑度随 rank 持续下降、梯度不再坍缩

### 设置

主实验微调 **Llama 2**，数据用 **OpenOrca** 指令微调集的 20,000 条样本，**AdamW**、HuggingFace 默认学习率 0.0005、常数调度，适配器加在所有线性（非 LayerNorm）注意力与前馈 MLP 子模块上。扫 $r\in\{4,8,32,128,512,2048\}$，同时跟踪微调困惑度与平均参数梯度范数。

### Benchmark 表现（以原文为准）

**困惑度（Figure 2）**：LoRA（铜色梯度）的曲线无论 rank 大小几乎挤成一团、收敛到相近的损失，"larger ranks even performing slightly worse"（更大 rank 甚至略差）；rsLoRA（蓝绿色梯度）则 rank 越大困惑度越低，**解锁了大 rank 的容量收益**。

![不同 rank 下的微调困惑度：标准 LoRA（铜色梯度）随 rank 增大几乎停滞、曲线挤在一起，rsLoRA（蓝绿色梯度）则随 rank 越大困惑度越低、持续受益](/papers/rslora/rslora-rank.png)

> 图源：Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*（arXiv:2312.03732）Figure 2——LoRA 与 rsLoRA 在 $r\in\{4,8,32,128,512,2048\}$ 下的微调困惑度（用于学习注解，版权归原作者）。

**梯度范数（Figure 3）**：直接验证坍缩机制——

![梯度范数对比：左 LoRA，随 rank 增大梯度范数逐级坍缩（铜色越亮=rank 越大、范数越低，跨约两个数量级）；右 rsLoRA，各 rank 的梯度范数在训练全程保持在同一数量级](/papers/rslora/gradnorm.png)

> 图源：Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*（arXiv:2312.03732）Figure 3——LoRA 梯度随 rank 坍缩、rsLoRA 保持稳定（用于学习注解，版权归原作者）。

左图 LoRA 的梯度范数随 rank 增大逐级下降、跨越约两个数量级，印证"大 rank 学不动"；右图 rsLoRA 各 rank 在训练起点就保持相同梯度范数，且全程维持在同一数量级。

### 消融：排除"只是变相调大学习率"

论文跑了一组消融（附录 B），逐一堵住替代解释：

- **换模型/优化器/数据集**：把 **GPT-J（6B）在 GSM8k 上用 Adafactor** 微调（Figure 5），同样出现"LoRA 各 rank 曲线几乎重叠、rsLoRA 随 rank 解锁更优性能"，证明结论可迁移。
- **换 SGD**：去掉 AdamW 后梯度范数稳定性模式不变（Figure 4），排除"是自适应优化器带来的稳定"。
- **学习率扫描**：对 LoRA 的 $r=4$ 扫学习率，**怎么调都追不上**默认学习率下 rsLoRA 高 rank 的表现——说明 rsLoRA 的增益来自"更大容量真的被用上了"，而非 $\gamma_r$ 变相充当 learning-rate boost。
- **只加注意力模块**：与全模块结果相似。
- **只校正初始化、不改缩放**：仅把 $A$ 的初始化按 $1/\sqrt{r}$ 缩放、缩放因子仍用 $1/r$，大 rank 下训练变得不稳定/次优——证明**必须靠重参数化的缩放因子**纠正，初始化补不了。

> 看榜须知：本文是机制性论文而非刷榜，报告的是同一套实验里 LoRA vs rsLoRA 的**相对趋势**（困惑度、梯度范数），并非跨系统绝对分数对比；关注"rsLoRA 让大 rank 重新有收益"这一定性结论即可。

## 在 LoRA 谱系里的位置

rsLoRA 在 LoRA 家族里只动**缩放因子**这一个旋钮，因此与几乎所有其他变体正交、可叠加：

| 维度 | LoRA（$\alpha/r$） | rsLoRA（$\alpha/\sqrt{r}$） |
| --- | --- | --- |
| 缩放因子阶数 | $\Theta(1/r)$（过度衰减、大 rank 坍缩） | $\Theta(1/\sqrt{r})$（rank 稳定） |
| 小 rank（4–16） | 与 rsLoRA 基本无差异 | 基本无差异 |
| 大 rank（≥64） | 收益快速饱和 | 持续受益，效果随 rank 继续提升 |
| 实现成本 | — | 一行改动 |
| 推理开销 | 合并后无 | 合并后无 |
| 需重标定的超参 | — | $\alpha$、学习率 |

小 rank 时两者差距小，因为 $r$ 和 $\sqrt{r}$ 数值接近（$r=4$ 时 $\sqrt{r}=2$，差一倍可被 $\alpha$/lr 吸收）；差异在"想用大 rank 换效果"时才决定性显现。

- **vs [LoRA](/lora/lora)**：rsLoRA 是 LoRA 的"缩放因子修正版"——同样的结构与合并方式，只把 $r$ 改成 $\sqrt{r}$，把 LoRA 论文里被掩盖的"大 rank 收益"重新打开。
- **vs [QLoRA](/lora/qlora)**：rsLoRA 只动缩放、与 QLoRA 的量化基座完全正交，常被一起用——QLoRA 省显存让你能上更大 rank，rsLoRA 让这些 rank 真正发挥作用。
- **vs [AdaLoRA](/lora/adalora)**：AdaLoRA 动态分配各层 rank、但仍沿用 LoRA 的 $\gamma_r=\alpha/r$；本文明确指出 AdaLoRA 在不同 rank 间切换时正受同一个过度衰减缩放之累，"optimizing the selection of $\gamma_r$ … can improve upon AdaLoRA"，把 AdaLoRA 建在 rsLoRA 之上有望进一步提升（尤其高 rank 预算）。
- **vs [LoRA+](/lora/lora-plus)**：[LoRA+](/lora/lora-plus) 调的是 $A/B$ 的学习率比，rsLoRA 调的是整体缩放尺度——两者都涉及"尺度"，叠加时建议先固定 rsLoRA 的缩放、再单独标定 LoRA+ 的学习率比，避免两个尺度旋钮互相干扰。
- **vs [DoRA](/lora/dora) / [PiSSA](/lora/pissa)**：DoRA（幅度-方向分解）、PiSSA（用主奇异成分初始化）改的是适配器的参数化/初始化，与 rsLoRA 的缩放修正各管一件事，原则上可组合。

**实践经验**：

- **换缩放后必须重标定 $\alpha$ 与学习率**：有效尺度变了，别再用 $\alpha=2r$ 这类与 $r$ 绑定的经验值；一个实用做法是固定 $\alpha$、把学习率当主旋钮重扫一遍。
- **什么时候值得开**：只有当你**确实想用大 rank**（如 $r\ge 64$，做较重领域适配或要更大增量容量）时 rsLoRA 才有明显价值；若本来只用 $r=8\sim16$ 的轻量适配，开不开几乎无差别，没必要额外引入调参负担。
