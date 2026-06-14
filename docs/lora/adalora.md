---
title: AdaLoRA
---

# AdaLoRA（Adaptive Budget Allocation）

> **一句话**：不同层、不同模块对秩的需求并不相同；AdaLoRA 把增量用 SVD 形式参数化，并在训练中按"重要性"动态地把有限的秩预算分配到最需要的地方。（*Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, 2023）
>
> 提出年份：2023（arXiv） · 机构/团队：Georgia Tech & Microsoft · 会议/来源：ICLR 2023 / arXiv:2303.10512
>
> 前置阅读：[LoRA](/lora/lora)

## 直觉与动机

标准 [LoRA](/lora/lora) 在所有被注入的模块上使用同一个固定的秩 $r$。这隐含了一个并不成立的假设：每一层、每一种投影矩阵（$q/k/v/o$、FFN 的 up/down）对增量容量的需求是均等的。实际上经验观察恰恰相反——FFN 层往往比 attention 层需要更多秩，靠近输出端的层比靠近输入端的层更"吃"容量。给所有模块发同样的额度，等价于一部分模块严重过剩、另一部分却被饿着。

一个朴素的想法是：先用大 $r$ 训练，再事后剪枝。但 LoRA 的增量 $\Delta W = BA$ 是两个矩阵相乘，没有天然的"哪一维更重要"的结构——你无法直接把 $BA$ 的某些秩单独丢掉而不破坏其余部分。AdaLoRA 的核心贡献就是：把增量改写成显式的 SVD 形式，让每一个"奇异方向"成为可以独立度量重要性、可以独立剪掉的最小单元，从而把"分配秩预算"变成一个连续、可微、可调度的过程。

## 方法与公式

AdaLoRA 把每个模块的增量参数化为类 SVD 的三元组：

$$
\Delta W = P \Lambda Q, \qquad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_r)
$$

其中 $P \in \mathbb{R}^{d \times r}$ 近似左奇异向量，$Q \in \mathbb{R}^{r \times k}$ 近似右奇异向量，$\Lambda$ 是对角的奇异值矩阵。$P$ 的每一列、$Q$ 的每一行、$\Lambda$ 的每一个对角元一起构成一个"秩-1 三元组"，它就是可以被增删的基本单位。

为了让 $P$、$Q$ 真正具有奇异向量的正交性（否则奇异值的"重要性"含义会失真），AdaLoRA 加一个正交正则项：

$$
R(P, Q) = \lVert P^\top P - I \rVert_F^2 + \lVert Q Q^\top - I \rVert_F^2
$$

**重要性打分。** 关键在于决定剪哪些三元组。AdaLoRA 不直接用奇异值 $\lambda_i$ 的大小（数值大不等于对损失贡献大），而是用基于敏感度的打分：对三元组中每个参数 $w$，其敏感度近似为 $|w \cdot \nabla_w \mathcal{L}|$，即"参数 × 梯度"的绝对值——这正是损失对该参数置零的一阶泰勒估计。由于单步梯度噪声很大，AdaLoRA 再对敏感度做指数滑动平均（平滑项 $\bar{I}$）并叠加一个不确定性项 $\bar{U}$，得到平滑后的重要性 $s_i$。一个三元组的重要性由它的 $\lambda_i$、$P$ 的对应列、$Q$ 的对应行三部分的得分汇总而成。

**预算调度。** 训练分三段，遵循"先宽后窄"：

1. 预热阶段保持初始的较高总预算 $b^{(0)}$，让所有三元组都先学一会儿，避免过早误杀；
2. 中间阶段按一个三次衰减（cubic schedule）逐步把总预算从 $b^{(0)}$ 降到目标值 $b^{(T)}$，每一步把全局重要性最低的若干三元组的 $\lambda_i$ 置零（并停止其更新）；
3. 末段固定在目标预算上继续训练到收敛。

最终目标函数是任务损失加正交正则：$\mathcal{L} + \gamma R(P, Q)$。注意被剪掉的是奇异值置零，$P$、$Q$ 的对应行列仍保留参数槽位，使得后续若重要性回升还有恢复余地。

![AdaLoRA 在 MNLI 上微调 DeBERTaV3-base 得到的各增量矩阵秩分布：更多预算被分配给 FFN 与高层](/papers/adalora/adalora-rank.png)

> 图源：Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, arXiv:2303.10512（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA | AdaLoRA |
| --- | --- | --- |
| 增量参数化 | $\Delta W = BA$ | $\Delta W = P\Lambda Q$（SVD 形式） |
| 秩分配 | 全局固定 $r$ | 按模块/层自适应，动态裁剪 |
| 重要性度量 | 无 | 敏感度（参数×梯度）+ 滑动平均 |
| 额外正则 | 无 | 正交正则 $R(P,Q)$ |
| 训练复杂度 | 低 | 较高（打分、调度、正则） |
| 同等参数预算下效果 | 基线 | 论文报告更优，尤其低预算时差距明显 |
| 推理 | 可合并，零开销 | 同样可合并为 $W_0 + P\Lambda Q$ |

## 实现要点

```python
# 训练步内的伪代码（每个 AdaLoRA 模块）
h = x @ W0.T + (x @ Q.T) @ Lambda @ P.T          # 前向：x P Λ Q
loss = task_loss(h, target) + gamma * ortho_reg(P, Q)
loss.backward()

# 敏感度 = |参数 * 梯度|，再做 EMA 平滑
for w in (P_col_i, Lambda_ii, Q_row_i):
    I_bar = beta1 * I_bar + (1 - beta1) * (w.detach() * w.grad).abs()
    U_bar = beta2 * U_bar + (1 - beta2) * (I_bar - I_inst).abs()
score_i = aggregate(I_bar * U_bar over the i-th triplet)

# 按 cubic schedule 计算当前总预算 b(t)，剪掉得分最低的三元组
budget = cubic_schedule(t, b0, bT, warmup, final)
mask = topk_by_score(scores, k=budget)           # 选出保留的三元组
Lambda.data[~mask] = 0.0                          # 奇异值置零 = 剪枝
```

实现层面注意：正交正则只对 $P$、$Q$ 计算，对 $\Lambda$ 不约束；调度的步数粒度通常按若干训练步剪一次，而非每步；初始总预算建议设为目标预算的 1.3–1.5 倍，给剪枝留出空间。

## 调参与实践经验

- **关键超参**：初始预算 $b^{(0)}$、目标预算 $b^{(T)}$、预热步数 $t_i$、最终固定步数 $t_f$、正交正则系数 $\gamma$（常用 $0.1\sim1$）。预算调度的预热/收尾步数若设得太短，会因敏感度估计还没稳定就开始剪枝而误杀。
- **何时值得用**：在**总参数预算被严格卡死**的场景（极端显存受限、或要把 adapter 控制到很小）下，AdaLoRA 比固定秩 LoRA 更划算。如果预算不紧，工程上往往直接把 LoRA 的 $r$ 调大就能达到接近的效果，而省去了打分、调度、正则的全部复杂度——这也是 AdaLoRA 在实际工程中使用度不如 LoRA/[QLoRA](/lora/qlora) 的主要原因。
- **与其他变体的关系**：AdaLoRA 解决的是"秩怎么分配"，[rsLoRA](/lora/rslora) 解决的是"缩放因子怎么随秩稳定"，[LoRA+](/lora/lora-plus) 解决的是"$A/B$ 学习率怎么配"，三者关注点正交，理论上可叠加，但叠加后调参空间变大，需谨慎。
- **训练成本**：因为多了正交正则的 $P^\top P$、$QQ^\top$ 矩阵乘和逐三元组打分，单步开销比 LoRA 高，但相对全量微调仍然很小。

## 参考文献

- Zhang et al., 2023. *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (AdaLoRA).* arXiv:2303.10512
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
