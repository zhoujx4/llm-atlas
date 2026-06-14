---
title: KTO（Kahneman-Tversky Optimization）
---

# KTO（Kahneman-Tversky Optimization）

> **一句话**：不需要成对偏好数据，每条样本只要一个「好/坏」二元标签；损失函数借鉴前景理论的「损失厌恶」设计，让模型在好样本上抬高隐式 reward、在坏样本上压低，对正负样本不均衡天然鲁棒。论文 *KTO: Model Alignment as Prospect Theoretic Optimization*（Ethayarajh et al., 2024）。
>
> 提出年份：2024 · 机构/团队：Stanford / Contextual AI · 会议/来源：ICML 2024 / arXiv:2402.01306
>
> 前置阅读：[DPO](/dpo/dpo)、[Reward Model](/rlhf/reward-model)

## 直觉与动机

[DPO](/dpo/dpo) 要求成对偏好数据：同一 prompt 下两个回答 $(y_w, y_l)$ 并标注哪个更好。这种数据贵且慢——标注者得读两段文本再排序，难任务上一致性还低。但现实世界里**二元反馈到处都是**：用户对一条回复点赞/点踩、客服对话是否被解决、生成的代码是否通过单测、模型回答是否被人工 reject。这些都是单条样本 + 一个「好/坏」标签，不成对。KTO 的目标就是直接吃这种数据。

第二个动机来自行为经济学。Kahneman & Tversky 的前景理论指出：人对**收益和损失的感受是不对称的**——同样幅度的损失带来的痛苦，大于等量收益带来的快乐（损失厌恶）。KTO 论文进一步论证：包括 DPO 在内的一类成功对齐损失，之所以比朴素的交叉熵好，部分原因正是它们隐式地编码了这类人类感知偏置，作者把这类损失统称为 HALO（Human-Aware Loss）。KTO 则把损失厌恶**显式**写进损失函数，并用两个权重 $\lambda_D,\lambda_U$ 显式控制好/坏样本的相对权重。

## 方法与公式

沿用 DPO 的隐式 reward 定义：

$$
\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

前景理论的价值函数是相对于一个**参考点**来度量收益/损失的。KTO 把参考点 $z_0$ 取为当前策略相对 reference 的平均偏移，用 batch 内的 KL 估计：

$$
z_0 = \mathbb{E}_{y'\sim\pi_\theta}\!\left[\mathrm{KL}\big(\pi_\theta(y'|x)\,\|\,\pi_{\text{ref}}(y'|x)\big)\right]
$$

实现上 $z_0$ 用**当前 batch（错位配对的样本）估计**，并且**不回传梯度**（detach），它只起「基准线」作用。单样本的价值函数 $v$ 按好/坏分两支，套 sigmoid 实现损失厌恶式的饱和：

$$
v(x,y) =
\begin{cases}
\lambda_D\,\sigma\!\big(\beta(\hat r_\theta(x,y) - z_0)\big) & y \text{ 为 desirable（好）}\\[4pt]
\lambda_U\,\sigma\!\big(\beta(z_0 - \hat r_\theta(x,y))\big) & y \text{ 为 undesirable（坏）}
\end{cases}
$$

KTO 损失就是最大化价值（即最小化负价值）：

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y)}\big[\,\lambda_y - v(x,y)\,\big]
$$

读法：好样本要让 $\hat r_\theta(x,y)$ 超过参考点 $z_0$（价值升高），坏样本要让 $\hat r_\theta(x,y)$ 低于参考点。sigmoid 让单个样本的价值有上界，这正对应前景理论里「收益的边际效用递减」，也防止个别样本主导梯度。

![不同 human-aware 损失（HALO）隐含的效用函数，呈现 Kahneman-Tversky 价值函数式的「损失厌恶」非对称形状](/papers/kto/utility.png)

> 图源：Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, arXiv:2402.01306（用于学习注解，版权归原作者）

**$\lambda_D$ 与 $\lambda_U$ 处理不均衡**。$\lambda_D,\lambda_U$ 分别是好/坏样本的权重。当好坏样本数量悬殊时，按数量对它们做反向加权，使两类对总损失的贡献平衡。论文建议保持 $\frac{\lambda_D\,n_D}{\lambda_U\,n_U}$ 在一个合理范围内（$n_D,n_U$ 为两类样本数），避免少数类被淹没。损失厌恶通常意味着给坏样本略高的权重。

## 与 baseline 对比

| 维度 | DPO | KTO |
| --- | --- | --- |
| 数据形式 | 成对 $(y_w, y_l)$ | 单条 $y$ + 二元好/坏标签 |
| 数据获取成本 | 高（需排序） | 低（点赞/点踩、单测通过即可） |
| 是否需要 reference | 需要 | 需要（隐式 reward 与参考点都用到） |
| 损失结构 | 成对 sigmoid 排序 | 单样本前景理论价值函数 |
| 正负不均衡 | 需配对，天然平衡 | 用 $\lambda_D/\lambda_U$ 显式调权 |
| batch 内耦合 | 无（每对独立） | 有（参考点 $z_0$ 跨样本估计） |
| 同数据量效果 | 强基线 | 论文中在 1B~30B 规模可匹配或超过 DPO |

核心权衡：KTO 用「batch 内共享参考点」这点耦合，换来了「不需要成对数据」的巨大数据优势。当你手上**本来就是成对数据**时，DPO 通常是更直接的选择；当数据天生是二元、或正负严重不均衡时，KTO 才显出价值。

## 实现要点

```python
# KTO loss：注意参考点 z0 在 batch 内共享、且 detach
def kto_loss(policy, ref, x, y, label, beta, lam_D, lam_U):
    # label: 1=desirable, 0=undesirable
    pi  = policy.seq_logprob(x, y)
    with torch.no_grad():
        rf = ref.seq_logprob(x, y)
    r_hat = beta * (pi - rf)                       # 隐式 reward

    # 参考点 z0: 用错位配对样本估计 KL, 不回传梯度
    z0 = compute_kl_reference(policy, ref, x).detach()

    v_D = lam_D * torch.sigmoid(r_hat - z0)        # 好样本: 越超过 z0 越好
    v_U = lam_U * torch.sigmoid(z0 - r_hat)        # 坏样本: 越低于 z0 越好
    v = torch.where(label == 1, v_D, v_U)

    loss = (torch.where(label == 1, lam_D, lam_U) - v).mean()
    return loss
```

- **参考点 $z_0$ 必须 detach**，它是「当前策略平均偏移多少」的基准估计，不是优化对象；若让它带梯度会破坏前景理论的语义并使训练不稳。
- **batch 内样本不独立**。因为 $z_0$ 跨样本估计，KTO 的有效性对 batch 内好/坏样本的混合比例敏感——尽量保证每个 batch 同时含好样本和坏样本，否则参考点估计偏。
- 与 DPO 一样：logprob 对 response token 求和、mask 掉 prompt 与 padding；reference 可冻结或预计算缓存。
- 现成实现：HF TRL 的 `KTOTrainer`，数据集每行给 `prompt / completion / label`（布尔），并暴露 `desirable_weight`、`undesirable_weight` 对应 $\lambda_D,\lambda_U$。

## 调参与实践经验

- **$\beta$** 含义同 [DPO](/dpo/dpo)（KL 偏移强度），常用量级 $0.1$ 附近；可先沿用 DPO 的经验值再微调。
- **不均衡时调 $\lambda$ 是第一旋钮**。正负样本比例失衡（典型如线上日志里点踩远少于点赞，或反之）时，按数量反向设置 $\lambda_D,\lambda_U$，目标是让两类对梯度的总贡献大致相当。论文给出的实用区间：$\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,\,4/3]$ 左右，可据实际类别比扫一扫。
- **batch 要足够大且混合**，以保证参考点 $z_0$ 估计稳定；过小的 batch 会让 $z_0$ 噪声大。
- 同样需要先 [SFT](/sft/) 再 KTO：reference 必须是个能用的指令模型。
- **数据来源优势要用好**：KTO 最大的价值在于能直接消费成本极低的二元反馈（线上点赞点踩、自动可验证信号如单测通过/失败）。当反馈天然二元时，与其硬凑成对数据做 DPO，不如直接 KTO。
- 监控指标：分别看好样本与坏样本上的隐式 reward 均值是否朝预期方向分离；若坏样本 reward 不降，多半是 $\lambda_U$ 太小或 batch 内坏样本太少。

## 参考文献

- Ethayarajh et al., 2024. *KTO: Model Alignment as Prospect Theoretic Optimization.* arXiv:2402.01306
- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290
- Kahneman & Tversky, 1979 / 1992. *Prospect Theory* 及 *Advances in Prospect Theory: Cumulative Representation of Uncertainty.*（损失厌恶与价值函数）
