---
title: IPO
---

# IPO（Identity Preference Optimization）

> **一句话**：指出 DPO 在「近乎确定性」的偏好下会把隐式 reward 差推向无穷、从而过拟合并无限偏离 reference；IPO 改用平方损失，把 reward 差拉向一个固定目标值而非越大越好。
>
> 论文：*A General Theoretical Paradigm to Understand Learning from Human Preferences*（ΨPO，Azar et al., 2023）·
>
> 提出年份：2023 · 机构/团队：Google DeepMind · 会议/来源：AISTATS 2024 / arXiv:2310.12036
>
> 前置阅读：[DPO](/dpo/dpo)、[符号约定](/guide/notation)

## 直觉与动机

IPO 来自一篇统一框架论文 ΨPO，它把「从偏好中学习」抽象成最大化一个关于偏好概率的非线性变换 $\Psi$ 的期望。在这个框架里，RLHF 和 [DPO](/dpo/dpo) 都是取 $\Psi=\text{logit}$（即 $\Psi(p)=\log\frac{p}{1-p}$）的特例。这一视角揭示了 DPO 的一个结构性弱点。

**问题出在 Bradley-Terry 假设与确定性偏好的冲突。** DPO 隐含地假设偏好服从 BT 模型，把成对偏好转成一个 reward 差。但真实的偏好标注常常是**确定性**的：标注者几乎总是把 $y_w$ 排在 $y_l$ 前面，对应的经验偏好概率 $p(y_w\succ y_l)\to 1$。在 logit 变换下，$p\to 1$ 意味着目标 reward 差 $\Psi(p)\to+\infty$。于是 DPO 的最优解要求 $\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\to\infty$——模型被驱使无限增大这个差，唯一的刹车（KL 正则的强度 $\beta$）在确定性偏好面前几乎失效。

**后果是过拟合与失控偏离。** 模型会不计代价地拉高 chosen、压低 rejected 的相对概率，哪怕已经远离 $\pi_{\text{ref}}$、哪怕代价是把概率质量挪到分布外的劣质区域。这与实践中常见的「DPO 训久了 chosen 和 rejected 概率一起塌、通用能力退化」现象一致。

IPO 的修正思路很直接：不要让 reward 差「越大越好」，而是让它**等于一个固定的有限目标**。这样即便偏好是确定性的，最优解也是有界的，KL 约束始终在场。

## 方法与公式

IPO 在 ΨPO 框架中取 $\Psi=\text{Identity}$（恒等映射，这也是「Identity Preference Optimization」名字的来源），由此推导出一个**平方损失**：

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x,y_w,y_l)}\left[\left( \log \frac{\pi_\theta(y_w|x)\,\pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_{\text{ref}}(y_w|x)} - \frac{1}{2\tau} \right)^2\right]
$$

括号里的第一项正是 DPO 的隐式 reward 差（去掉 $\beta$ 因子）：

$$
h_\theta(x,y_w,y_l) = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

**与 DPO 的核心对照**：DPO 是 $-\log\sigma(\beta h_\theta)$，logistic 损失对 $h_\theta$ 单调递减，永远奖励「更大的差」；IPO 是 $(h_\theta - \frac{1}{2\tau})^2$，平方损失在 $h_\theta=\frac{1}{2\tau}$ 处取最小，把差**钉**在一个固定目标 margin 上。一旦达到目标，继续增大差反而被惩罚——这就从根本上消除了「reward 差 → ∞」的失控。

**$\tau$ 的作用**：$\tau$ 是正则强度，$\frac{1}{2\tau}$ 是目标 margin。$\tau$ 越大，目标 margin 越小，意味着对偏离 reference 越保守（更强的 KL 约束）；$\tau$ 越小，允许 chosen/rejected 差距越大，行为越接近激进的 DPO。

![IPO 与 DPO 的动作概率学习曲线对比：DPO 在各正则强度下都塌向确定性策略，IPO 则随正则增强保持靠近 reference](/papers/ipo/x1.png)

> 图源：Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences (ΨPO / IPO)*, arXiv:2310.12036（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | DPO | IPO |
| --- | --- | --- |
| ΨPO 中的变换 | $\Psi=\text{logit}$ | $\Psi=\text{Identity}$ |
| 损失形式 | $-\log\sigma(\beta\, h_\theta)$ | $\big(h_\theta - \frac{1}{2\tau}\big)^2$ |
| 对 reward 差的偏好 | 越大越好（单调） | 钉在固定目标 $\frac{1}{2\tau}$ |
| 确定性偏好下 | reward 差 → ∞，KL 约束失效 | 有界，KL 约束始终生效 |
| 关键超参 | $\beta$ | $\tau$ |

## 实现要点

IPO 在 TRL 中通过 `DPOTrainer(loss_type="ipo")` 选择，复用 DPO 的双前向与 reference 处理：

```python
# pi_logratios = (logp_w - logp_l) on policy
# ref_logratios = (logp_w - logp_l) on reference
h = (pi_w - pi_l) - (ref_w - ref_l)   # = h_theta, 隐式 reward 差（TRL 中 logp 已按 completion 长度归一化）
loss = ((h - 1.0 / (2 * tau)) ** 2).mean()
```

关键细节：

- **目标 margin 是 $\frac{1}{2\tau}$**，TRL 中超参名为 `beta`，但其语义对应上式的 $\tau$（即 `beta` 越大、目标 margin 越小、约束越强），与 DPO 的 `beta` 含义相反，迁移时务必核对。
- **仍需 reference model**：IPO 没有去掉 $\pi_{\text{ref}}$，双前向开销与 DPO 相同；它解决的是损失形状问题，不是显存问题。
- **TRL 实现对 logprob 做长度归一化**：在 `loss_type="ipo"` 下，TRL 会把 completion 的 logprob 除以其 token 数（per-token 平均）再算 $h_\theta$，因此上式的 `pi_w/pi_l/ref_w/ref_l` 都是长度归一化后的均值。TRL 维护者说明此选择是与 IPO 作者确认过的，论文报告结果对应的就是归一化形式。需要与 [SimPO](/dpo/simpo) 对比时，二者都做了长度归一化，区别在于：IPO 的归一化是 TRL 实现层面引入（原论文未显式讨论），且仍保留 reference model；SimPO 的长度归一化是方法本身的定义，并彻底去掉了 reference。

## 调参与实践经验

- **$\tau$**：常见取值使目标 margin $\frac{1}{2\tau}$ 落在 0.1~1.0 量级；以 TRL 的 `beta` 表示约为 0.1~1.0。从较强约束（小 margin）起调更安全。
- **理论漂亮 ≠ 实测更好**：IPO 在理论上修正了 DPO 的过拟合，但多个公开评测显示，在常规标注质量与中等数据量下，IPO 相比调好的 DPO 并无稳定优势，有时还略逊。它的价值更多体现在**偏好高度确定、且观察到 DPO 明显过拟合/塌缩**的场景。
- **诊断信号**：如果你的 DPO 训练出现 chosen 与 rejected logprob 一起急剧下降、留出集质量回退，可以把 IPO 当作一个「带刹车」的替代实验，对比目标 margin 是否带来更稳的曲线。
- **与其它变体的关系**：IPO 用「固定目标 margin」防失控，与 SimPO 用 $\gamma$、[CPO](/dpo/cpo)/[ORPO](/dpo/orpo) 用 SFT/odds 项，都是对「DPO 无界放大」这一共性问题的不同解法。

## 参考文献

- Azar et al., 2023. *A General Theoretical Paradigm to Understand Learning from Human Preferences (ΨPO / IPO).* arXiv:2310.12036
- Rafailov et al., 2023. *Direct Preference Optimization.* arXiv:2305.18290
