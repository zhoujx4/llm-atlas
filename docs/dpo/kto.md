---
title: KTO（Stanford / Contextual AI）：用前景理论把对齐做成「单条二元反馈」的效用最大化
---

# KTO（Stanford / Contextual AI）：用前景理论把对齐做成「单条二元反馈」的效用最大化

**📄 [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)**

2024-02 · Stanford University · Contextual AI（ICML 2024）· [代码](https://github.com/ContextualAI/HALOs)

**一句话**：把「成对偏好」这个硬约束拆掉——每条样本只要一个「好/坏」二元标签，借 Kahneman-Tversky 前景理论的「损失厌恶」价值函数直接最大化生成的人类效用，而不是最大化偏好似然。它把 DPO 一类成功损失抽象成一族「人感知损失」（HALO），KTO 是其中显式编码前景理论偏置的一个实例。

::: details 📖 论文原文 Abstract（英文）
Kahneman & Tversky's *prospect theory* tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases—the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them belonging to a family of loss functions that we call **human-aware losses (HALOs)**. However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach **KTO**, and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B, despite only learning from a binary signal of whether an output is desirable. More broadly, our work suggests that there is no one HALO that is universally superior; the best loss depends on the inductive biases most appropriate for a given setting, an oft-overlooked consideration.
:::

**相关**：[DPO](/dpo/dpo) · [Reward Model](/rlhf/reward-model) · [SFT](/sft/)

![不同 human-aware 损失（HALO）隐含的「人类效用」曲线：Kahneman-Tversky（红）、PPO-Clip（黄）、DPO（蓝）都在参考点附近呈现「损失一侧更陡」的损失厌恶（loss aversion）与增益一侧的凹性（concavity），形状与前景理论的经典价值函数一致；DPO 的参考点正是「被拒答案 $y_l$ 的 reward」](/papers/kto/utility.png)

> 图源：Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*（arXiv:2402.01306）Figure 1——三种 HALO 隐含的人类效用函数都共享损失厌恶、围绕一个参考点度量增益/损失（用于学习注解，版权归原作者）。

## 动机与创新点：偏好似然 ≠ 人类效用，二元反馈更便宜也更鲁棒

[DPO](/dpo/dpo) 要求**成对偏好数据**：同一 prompt 下两个回答 $(y_w, y_l)$ 并标注哪个更好。这种数据"相对稀缺、采集贵"（*preferences … are relatively scarce and expensive to collect in practice*），标注者得读两段文本再排序，难任务上一致性还低。但现实里**二元反馈到处都是**：点赞/点踩、客服对话是否被解决、生成代码是否过单测、回答是否被人工 reject——都是单条样本 + 一个「好/坏」标签，不成对。KTO 的第一目标就是直接吃这种数据。

第二个动机来自行为经济学。作者用**前景理论（prospect theory）**重新审视对齐：它解释了"人为何会做出不最大化期望值的决策"——相对某个**参考点**，人对损失比对等量增益更敏感（loss aversion）。论文进一步论证：包括 DPO、PPO-Clip 在内的一批成功对齐损失，之所以比朴素交叉熵好，部分原因正是它们**隐式编码了这类人类感知偏置**。作者把这类损失抽象成一个统一的函数族——**HALO（human-aware loss）**，并证明 DPO、PPO-Clip 都是 HALO（Theorem 3.5）。KTO 则用 Kahneman-Tversky 的价值函数**显式**写出损失厌恶，直接最大化生成的效用而非偏好似然。

更宏观的论点是：**"there is no one HALO that is universally superior"**——最好的损失取决于具体场景该用什么归纳偏置，这是常被忽视的设计自由度。

**关键创新**：

- **HALO 框架**：把"人感知损失"形式化为一个函数族（带参考点的、凹于增益的、有损失厌恶的价值函数作用在隐式 reward 上），并证明 DPO / PPO-Clip 都落在其中，为"为什么这些损失好用"给了一个统一解释。
- **KTO 损失**：用 Kahneman-Tversky 价值函数构造一个新 HALO，**只需单条样本 + 二元标签**，直接最大化效用；在 1B–30B 全程匹配或超过 DPO。
- **两个权重 $\lambda_D,\lambda_U$ 显式控不均衡**：好/坏样本天然不均衡（线上点踩往往远少于点赞）时，按数量反向加权即可，无需配对；可承受到 1:10 量级的极端不均衡。
- **可跳过成对、甚至跳过 SFT**：足够规模下 KTO 单独（不先 SFT）就能匹配 SFT+DPO，且不像无 SFT 的 DPO 那样把回复越写越长、产生幻觉。
- **更好的最坏情况保证**：理论上 KTO 会"忽略噪声且非传递（intransitive）的反馈"，对真实世界普遍存在的噪声偏好更鲁棒（Prop 4.1 / Theorem 4.3）。

## 方法：把前景理论价值函数套到隐式 reward 上

### HALO：把成功对齐损失抽象成「人感知损失」族

先给隐式 reward 一个通用定义——$r_\theta(x,y) = l(x,y)\log[\pi_\theta(y|x)/\pi_{\text{ref}}(y|x)]$，即对齐模型相对 reference 的对数似然比（乘一个归一化因子 $l$）。一个损失 $f(\pi_\theta,\pi_{\text{ref}})$ 被称为 **HALO**，当它能写成

$$
f(\pi_\theta,\pi_{\text{ref}}) = \mathbb{E}_{x,y\sim\mathcal D}\Big[a_{x,y}\,v\big(r_\theta(x,y) - \mathbb{E}_{Q}[r_\theta(x,y')]\big)\Big] + C_\mathcal{D}
$$

其中 $v$ 是**非减且凹**的价值函数、$Q(Y'|x)$ 给出度量增益/损失所用的**参考点分布**、$a_{x,y}\in\{-1,+1\}$。直觉是：人不是看 reward 的绝对值，而是看它**相对参考点**高了还是低了，再经一个"增益递减、损失更痛"的价值函数打分。论文证明：

> **Theorem 3.5.** *DPO and PPO-Clip are human-aware losses.*

DPO 之所以是 HALO，关键在它**把"被拒答案 $y_l$ 的 reward"当成参考点**——这也是上面 Figure 1 里 DPO 曲线参考点的含义。相对地，CSFT、SLiC 不满足 HALO 定义（参考点固定、或价值函数非对称性不对），论文实验里它们也确实更弱（见下文 Figure 2 一类对比）。

### KTO 损失：前景理论价值函数的单样本形式

KTO 直接采用 Kahneman-Tversky 的价值函数（原文 eq 4，相对参考点 $z_0$）：

$$
v(z;\lambda,\alpha,z_0)=\begin{cases}(z-z_0)^{\alpha} & z\ge z_0\\[2pt]-\lambda\,(z_0-z)^{\alpha} & z<z_0\end{cases}
$$

经验上 $\alpha\approx0.88$（增益侧凹、损失侧凸）、$\lambda\approx2.25$（损失比增益陡约 2.25 倍，即损失厌恶）。但指数 $\alpha$ 在优化时数值不稳，KTO 把它**换成 logistic $\sigma$**（同样增益凹、损失凸），并做两处改造：

- 引入 $\beta\in\mathbb{R}^+$ 控**风险厌恶强度**——"$\beta$ 越大，增益侧越风险厌恶、损失侧越风险偏好"。它和 DPO 里的 $\beta$ 作用类似（控 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 的强度），只是这里显式写进价值函数。
- 把损失厌恶系数 $\lambda$ 拆成 $\{\lambda_D,\lambda_U\}$——分别对应 desirable / undesirable 两类输出的权重。

最终 KTO 损失（原文 eq 8）：

$$
\mathcal{L}_{\text{KTO}}(\pi_\theta,\pi_{\text{ref}}) = \mathbb{E}_{x,y\sim\mathcal D}\big[\lambda_y - v(x,y)\big]
$$

其中

$$
r_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)},\qquad
z_0=\mathrm{KL}\big(\pi_\theta(y'|x)\,\|\,\pi_{\text{ref}}(y'|x)\big)
$$

$$
v(x,y)=\begin{cases}\lambda_D\,\sigma\!\big(\beta\,(r_\theta(x,y)-z_0)\big) & y\sim y_{\text{desirable}}|x\\[4pt]\lambda_U\,\sigma\!\big(\beta\,(z_0-r_\theta(x,y))\big) & y\sim y_{\text{undesirable}}|x\end{cases}
$$

读法：好样本要让 $r_\theta$ **超过参考点 $z_0$**（价值升高），坏样本要让 $r_\theta$ **低于 $z_0$**。sigmoid 给单样本价值上界，对应"增益的边际效用递减"，也防个别样本主导梯度。注意 KTO 把参考点取成**整个策略相对 reference 的平均 KL 偏移**（一个标量基准线），而非 DPO 那样把"另一个具体答案"当参考点——这正是它能丢掉成对结构的关键：每条样本独立地跟"群体平均偏移"比，不再需要配对的另一半。

> 举例：把同一 prompt 下用户**点赞**的回答标 desirable、**点踩**的标 undesirable，各自独立喂进损失。点赞样本只需把自己的 $r_\theta$ 顶到平均基线 $z_0$ 之上；点踩样本只需被压到 $z_0$ 之下——全程不需要"这条比那条好"的成对标注。

### 为什么 KTO 学得动：KL 一鼓就停的自约束

论文给的直觉很关键：

> if the model increases the reward of a desirable example in a *blunt* manner, then the KL penalty also rises and no progress is made.

也就是说，如果模型只是粗暴地抬高某个好样本的似然，参考点 $z_0$（平均 KL）会同步上抬、价值不增，等于白干。这**逼模型去学"到底什么让这条输出变好"**，从而能在抬 reward 的同时把 KL 项压平（甚至下降）。坏样本方向同理，只是 KL 的非负性让损失侧能更快饱和。这套"自带刹车"的机制，是 KTO 在二元弱信号下还能稳的核心。

### $z_0$ 的有偏估计：错位配对 + detach + 截断

理论上 $z_0$ 要从 $\pi_\theta$ 采样估计 KL，太慢。KTO 改用一个**有偏但便宜**的估计：在同一 microbatch 内做**错位配对**（把第 $i$ 个输入配第 $j=(i+1)\bmod m$ 个输出），共享一个参考点

$$
\hat z_0 = \max\!\Big(0,\ \tfrac{1}{m}\textstyle\sum_{1\le i<m}\log\frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\Big)
$$

三个要点：

- **用错位的 $y_j$ 而非对应的 $y_i$**：因为 $y_i$ 是被人刻意选为"典型好/坏"的输出、reward 量级不具代表性，错位的 $y_j$ 更接近"随机输出"的基线。
- **截断到 $\ge 0$**：带来正偏差但**方差更低**——作者认为人感知的参考点本就有偏（不会用完整分布做平均，而是"可得性启发式"），所以有偏估计反而合理。
- **必须 detach（不回传梯度）**：$z_0$ 只作基准线、"do not backpropagate through $z_0$; it exists purely to control the loss saturation"。让它带梯度会破坏前景理论语义并使训练不稳。

一个实用捷径：若 KTO 紧接在**用同一份数据做的 SFT** 之后（SFT 模型即 $\pi_{\text{ref}}$），KL 估计会很快趋近 0，此时可直接令 $\hat z_0=0$；但若没先 SFT、或 SFT 数据与 KTO 数据不同源，则必须老实估 $z_0$。

### $\lambda_D / \lambda_U$ 处理不均衡

$\lambda_D,\lambda_U$ 默认都为 1。当好/坏样本数 $n_D,n_U$ 悬殊时，按数量反向加权，使两类对总损失贡献平衡。论文给的经验区间（eq 9）：

$$
\frac{\lambda_D\,n_D}{\lambda_U\,n_U}\in\Big[1,\ \tfrac{3}{2}\Big]
$$

> 举例：若 desirable : undesirable = 1 : 10，则设 $\lambda_U=1,\ \lambda_D\in[10,15]$。区间偏向"增益敏感"（gain sensitivity）——经验上"产出好输出比避免坏输出更重要"；但在毒性防控这类**最坏情况更要紧**的任务，可反过来设 $\lambda_D n_D<\lambda_U n_U$。

### 实现要点

```python
# KTO loss：注意参考点 z0 在 batch 内共享、且 detach
def kto_loss(policy, ref, x, y, label, beta, lam_D, lam_U):
    # label: 1=desirable, 0=undesirable
    pi  = policy.seq_logprob(x, y)
    with torch.no_grad():
        rf = ref.seq_logprob(x, y)
    r_hat = pi - rf                                # 隐式 reward（log 比）

    # 参考点 z0: 用错位配对样本估计 KL, 截断到 >=0, 不回传梯度
    z0 = compute_kl_reference(policy, ref, x).clamp_min(0).detach()

    v_D = lam_D * torch.sigmoid(beta * (r_hat - z0))   # 好样本: 越超过 z0 越好
    v_U = lam_U * torch.sigmoid(beta * (z0 - r_hat))   # 坏样本: 越低于 z0 越好
    v = torch.where(label == 1, v_D, v_U)

    loss = (torch.where(label == 1, lam_D, lam_U) - v).mean()
    return loss
```

- **参考点 $z_0$ 必须 detach 且截断到 $\ge 0$**；它是"当前策略平均偏移多少"的基准估计，不是优化对象。
- **batch 内样本不独立**：$z_0$ 跨样本估计，KTO 有效性对 batch 内好/坏混合比例敏感——每个 batch 尽量同时含好样本和坏样本，否则参考点估偏。microbatch 至少为 2，论文建议 batch 8–128（实验用有效 batch 32）。
- 与 DPO 一样：logprob 对 response token 求和、mask 掉 prompt 与 padding；reference 可冻结或预计算缓存。
- 现成实现：HF TRL 的 `KTOTrainer`，数据集每行给 `prompt / completion / label`（布尔），并暴露 `desirable_weight` / `undesirable_weight` 对应 $\lambda_D,\lambda_U$。

### 调参与实践经验

- **学习率要比 DPO 大很多**：因 reference 调整后的 reward 量级更小，KTO 推荐默认 **5e-6**（AdamW），约为 DPO 5e-7 的 **2×–10×**。
- **$\beta$ 控风险厌恶**：大模型（已做过 SFT）用低 $\beta\in[0.01,0.10]$；小模型直接 KTO（无 SFT）用高 $\beta\in[0.10,1.00]$。可先沿用 DPO 经验值再扫。
- **不均衡时调 $\lambda$ 是第一旋钮**：按上面 $[1,3/2]$ 区间反向设 $\lambda_D,\lambda_U$，目标让两类对梯度总贡献相当。
- **先 SFT 再 KTO 仍是默认更稳的路径**（reference 是个能用的指令模型）；但足够规模下可省 SFT 直接 KTO。
- **监控指标**：分别看好/坏样本上的隐式 reward 均值是否朝预期方向分离；若坏样本 reward 不降，多半是 $\lambda_U$ 太小或 batch 内坏样本太少。

## 实验结果：1B–30B 匹配或超过 DPO，且抗极端不均衡

### KTO ≥ DPO，全程 1B–30B

把 §3.3 的 winrate 评测（**GPT-4-0613 当裁判**，比对齐模型 vs SFT 目标输出）重跑到各家：

![六种对齐方法在 Pythia-{1.4B,2.8B,6.9B,12.0B}（红）与 Llama-{7B,13B,30B}（蓝）上「相对 SFT 目标的 winrate − 50%」柱状图：offline PPO / SFT+offline PPO / DPO / SFT+DPO / KTO / SFT+KTO；KTO 与 SFT+KTO 这两组的 Llama 蓝柱在 13B、30B 处冲到 0% 以上（超过 SFT 目标），KTO 单独一组已追平甚至超过 SFT+DPO](/papers/kto/kto-vs-dpo.png)

> 图源：Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*（arXiv:2402.01306）Figure 3——KTO 在 1B–30B 全程不输 DPO；Llama 上 KTO 单独即追平 SFT+DPO、显著优于 DPO 单独（用于学习注解，版权归原作者）。

- **SFT+KTO 在 1B–30B 全程与 SFT+DPO 持平**，尽管只学二元弱信号。
- **Llama-{7B,13B,30B} 上 KTO 单独 > DPO 单独**，且在 7B、30B 显著（$p<0.01$，经多重比较校正）。Pythia 上两者无显著差异，作者推测"需要最小模型容量这些差异才显现"。
- 更早一组对照（Figure 2）显示 **HALO（DPO、offline PPO）整体优于非 HALO（SLiC、CSFT）**，且只有 HALO 对齐的 Llama-{13B,30B} 能把 winrate 顶到 50% 以上——印证"是 HALO 这个归纳偏置在起作用"。

### 不靠成对、抗不均衡、甚至能省掉 SFT

![五种设置下 Llama-7B 输出长度箱线图：SFT 最短、**DPO（无 SFT）中位数和离群点都爆炸式拉长**、SFT+DPO / KTO / SFT+KTO 都保持紧凑——说明无 SFT 的 DPO 会"长篇大论、整段幻觉"，而 KTO 不会](/papers/kto/length.png)

> 图源：Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*（arXiv:2402.01306）Figure 4——不先做 SFT 时，DPO 对齐的模型倾向 ramble/hallucinate（输出超长），KTO 不受此困扰（用于学习注解，版权归原作者）。

- **可跳过 SFT**：足够规模（Llama-13B/30B）下，KTO 单独就能匹配 SFT+KTO，是唯一表现出这一行为的方法——因为 KTO 把平均回复长度保持得差不多，而无 SFT 的 DPO 会让回复长度暴涨（上图）。
- **KTO 数据不必来自偏好**：随机丢弃 desirable 数据做不均衡测试，**丢掉 90% 的 desirable 数据**（desirable:undesirable 从 1:1 变 1:10）后，按 $\lambda$ 区间补偿（如 $\lambda_U=1,\lambda_D=13.33$），KTO 仍胜过 DPO（Figure 5）。
- **天然非成对数据**：在 OpenAssistant 上对齐 Mistral-7B，**每个 $x$ 只用一个 $y$（one-$y$-per-$x$）**、彻底抹掉成对结构、训练数据少 **72%**，KTO 仍同时超过 DPO 与官方 Mistral-7B-Instruct（下表）。

| 方法（Mistral-7B / OpenAssistant） | Winrate vs SFT 目标 |
| --- | --- |
| Mistral-7B（未对齐） | 0.525 |
| Mistral-7B + DPO（$n$ 对） | 0.600 |
| **Mistral-7B + KTO（全部 $2n$ 输出）** | **0.652** |
| Mistral-7B + KTO（one-$y$-per-$x$，数据 −72%） | 0.631 |
| Mistral-7B-Instruct（官方） | 0.621 |

### 生成基准与设计消融（Zephyr-β-SFT / UltraFeedback，1 epoch）

| 方法 | MMLU | GSM8K | HumanEval | BBH |
| --- | --- | --- | --- | --- |
| SFT | 57.2 | 39.0 | 30.1 | 46.3 |
| DPO | 58.2 | 40.0 | 30.1 | 44.1 |
| ORPO（$\lambda{=}0.1$） | 57.1 | 36.5 | 29.5 | 47.5 |
| **KTO（$\beta{=}0.1,\lambda_D{=}1$）** | **58.6** | **53.5** | **30.9** | **52.6** |
| KTO（one-$y$-per-$x$） | 58.0 | 50.0 | 30.7 | 49.9 |

- **GSM8K 提升最猛**：在 UltraFeedback 上仅把 DPO 换成 KTO，数学推理 **+13.5 分**（40.0→53.5）。
- **每个设计都要紧**（消融）：去掉参考点 $z_0$（不再是 HALO）→ BBH −3.6 / GSM8K −4.0；把价值函数改成处处凹的 $v=\log\sigma$（像 DPO）→ BBH −9.4 / GSM8K −11.0；改成风险中性的恒等函数 → BBH **直接崩盘**。这从反面验证了"前景理论形状（带 $z_0$、损失厌恶、非对称）"是 KTO 有效的根因。
- **推荐超参（Table 1，AdamW / 有效 batch 32 / $\lambda_D{=}\lambda_U{=}1$）**：Llama-3 8B 用 LR 5e-6、$\beta$ 取 0.05（SFT+KTO）或 0.10（KTO 直训）；Qwen2.5-3B-Instruct 直训 KTO 时 $\beta$ 可大到 0.50。

> 看榜须知：这些分数的口径、底座、SFT 数据、采样设置各异，**跨设置直接比绝对值意义有限**；当作"同数据量下 KTO 与 DPO/ORPO 同档可比、二元信号不吃亏"的量级参照即可。

## 在偏好对齐谱系里的位置

| 维度 | DPO | KTO |
| --- | --- | --- |
| 数据形式 | 成对 $(y_w, y_l)$ | 单条 $y$ + 二元好/坏标签 |
| 数据获取成本 | 高（需排序） | 低（点赞/点踩、单测通过即可） |
| 参考点 $z_0$ | 「被拒答案 $y_l$ 的 reward」（成对内定） | 「策略相对 ref 的平均 KL 偏移」（batch 内估、detach） |
| 是否需要 reference | 需要 | 需要（标准版）；论文给了去 reference 的内存高效变体（略弱） |
| 损失结构 | 成对 sigmoid 排序 | 单样本前景理论价值函数 |
| 正负不均衡 | 需配对，天然平衡 | 用 $\lambda_D/\lambda_U$ 显式调权，可抗 1:10 |
| batch 内耦合 | 无（每对独立） | 有（参考点 $z_0$ 跨样本估计，batch 需混合好坏） |
| 噪声/非传递偏好 | 可能拟合到少数派偏好 | 理论上确定性产出多数派偏好，最坏情况更好 |
| 同数据量效果 | 强基线 | 1B–30B 匹配或超过 DPO |

- **vs [DPO](/dpo/dpo)**：核心权衡是"用 batch 内共享参考点这点耦合，换不需要成对数据的巨大数据优势"。手上**本来就是干净成对数据**时，DPO 通常更直接；论文还从理论上指出——当偏好数据**噪声小、非传递性低**时 DPO 更优（KTO 有欠拟合复杂分布的风险，可用更低 $\beta$ + 更多 epoch 缓解）；但当数据**天生二元、严重不均衡、或噪声大/自相矛盾**（SHP、OpenAssistant、UltraFeedback 这类公开集都不同程度如此），KTO 才显出价值，也解释了它为何能匹配甚至超过 DPO。
- **vs RLHF / [Reward Model](/rlhf/reward-model)**：HALO 框架把 RLHF 的 PPO-Clip 也纳入同一族（Theorem 3.5），说明"在线 RL + 显式 reward model"和"离线闭式损失"共享同一套人感知归纳偏置；KTO 站在离线一侧，省掉了单独训 reward model 与在线采样的开销。
- **vs ORPO 等 reference-free 方法**：KTO 也有"假设 $\pi_{\text{ref}}$ 为均匀分布"的去 reference 变体（$r_\theta - z_0$ 退化成 $\log\pi_\theta - H(\pi_\theta)$），内存更省、在部分任务上仍胜 DPO，但整体不如标准 KTO，且对损失厌恶超参更敏感。
- **理论亮点**：Prop 4.1 指出"当某样本的隐式 reward 趋于 $\pm\infty$（太难或太易学）时，KTO 对该样本的梯度趋于 0"——等于**自动忽略噪声与离群反馈**，这对充满噪声的真实反馈是「因祸得福」；Theorem 4.3 进一步证明在**自相矛盾的偏好**下，DPO 最优策略可能输出少数派偏好答案，而损失中性（$\lambda_D=\lambda_U$）的 KTO 会**确定性地输出多数派**，最坏情况保证更好。这是"二元信号反而更鲁棒"的根因。
