---
title: RLOO
---

# RLOO（REINFORCE Leave-One-Out）

> **一句话**：回归 REINFORCE——每个 prompt 采样 $k$ 个回答，用其余 $k-1$ 个的平均奖励做基线，得到无偏、低方差的优势估计；不需要 critic，也不需要 clip。论文 *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*（Cohere，ACL 2024）。
> 提出年份：2024 · 机构/团队：Cohere For AI · 会议/来源：ACL 2024 / arXiv:2402.14740（留一基线引自 Kool et al., 2019）
>
> 前置阅读：[PPO](/rlhf/ppo)；建议对照 [GRPO](/rlhf/grpo)

## 直觉与动机

PPO 的全套机制——critic + GAE、比值裁剪、多 epoch 复用——是为经典 deep RL 设计的：随机初始化的策略、长 horizon、密集但噪声大的奖励。RLOO 论文的核心论点是，**RLHF 根本不是那种场景**：

- 初始策略是 SFT 模型，已经相当好，单步更新不至于灾难性偏移，对"防走崩"机制的需求远小于从零训练的 agent；
- 奖励只在序列末端给一次（[Reward Model](/rlhf/reward-model) 打分或规则判分），没有中间奖励，token 级 bootstrapping 的意义存疑——critic 对部分序列的 value 估计本就粗糙，还要再训一个与 policy 同尺寸的模型；
- prompt 即 episode 起点，单步 bandit 视角足够。

于是退回最朴素的方案：把**整条回答 $y$ 当作一个 action**，用 REINFORCE 估计梯度 $\nabla_\theta\mathcal{J} = \mathbb{E}\big[(R(x,y)-b)\,\nabla_\theta \log \pi_\theta(y|x)\big]$。剩下唯一的问题是基线 $b$ 怎么取——它不能依赖当前样本（否则有偏），又要尽量贴近 $R$ 的期望（否则方差大）。留一法（leave-one-out）给出免费答案：同一 prompt 反正要多采样，就让每个样本用**其余样本的均值**当基线。该估计量引自 Kool et al. 2019（*Buy 4 REINFORCE Samples, Get a Baseline for Free!*），RLOO 论文将其系统性地引入 RLHF 并验证有效。

## 方法与公式

对每个 prompt $x$ 从 $\pi_\theta$ 采样 $k$ 个回答 $y_1,\dots,y_k$，梯度估计为：

$$
\nabla_\theta \mathcal{J} = \mathbb{E}\left[\frac{1}{k} \sum_{i=1}^{k} \Big( R(x, y_i) - \underbrace{\frac{1}{k-1}\sum_{j \neq i} R(x, y_j)}_{\text{留一基线 } b_i} \Big)\, \nabla_\theta \log \pi_\theta(y_i \mid x)\right]
$$

性质：

- **无偏**：$b_i$ 与 $y_i$ 独立，$\mathbb{E}[b_i\nabla_\theta\log\pi_\theta(y_i|x)] = 0$，不改变梯度期望；
- **方差缩减**：$b_i$ 是 $\mathbb{E}[R]$ 的蒙特卡洛估计，$k$ 越大越准；
- **序列级 credit assignment**：整条回答共享一个标量优势，不做 token 级分配（这点与 GRPO 相同，与 PPO+GAE 不同）；
- **on-policy**：每个 batch 采样后只更新一次，没有重要性比值 $\rho_t$，也就不需要 clip。

KL 约束的处理与 PPO-RLHF 相同：把 $-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 并入 $R$（或作为独立 loss 项），防止策略漂离 $\pi_{\text{ref}}$。

**与 GRPO 的精确关系**。一个常被忽视的恒等式：

$$
r_i - \frac{1}{k}\sum_{j=1}^{k} r_j \;=\; \frac{k-1}{k}\Big( r_i - \frac{1}{k-1}\sum_{j\neq i} r_j \Big)
$$

即"组均值（含自身）做基线"与留一基线只差一个常数缩放 $\frac{k-1}{k}$，可被学习率吸收。所以 **GRPO 与 RLOO 的实质差异不在均值怎么算**，而在于：GRPO 还要除以组内 std（引入偏差和难度加权——奖励方差小的过易/过难 prompt 的优势被放大），并套 PPO 式 clip 做多步 off-policy 更新；RLOO 保持无偏估计 + 单步 on-policy，是两者中理论上更"干净"的那个。

![RLOO 与 PPO、RAFT 等基线在训练过程中的测试奖励对比](/papers/rloo/rloo-vs-baselines.png)

> 图源：Ahmadian et al., *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*, arXiv:2402.14740（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | PPO | GRPO | RLOO |
| --- | --- | --- | --- |
| Critic | 需要（同尺寸模型） | 不需要 | 不需要 |
| 基线 | value 网络 | 组均值（含自身），再除以组内 std | 留一均值 |
| 优势无偏性 | 取决于 value 质量 | std 缩放有偏 | 无偏 |
| credit assignment | token 级（GAE） | 序列级广播 | 序列级 |
| clip / off-policy 复用 | 有 | 有 | 通常无（on-policy 单步） |
| 每 prompt 采样数 | 1 | $G$ | $k$（论文用 2/4） |
| 驻留模型数 | 4（policy/ref/RM/critic） | 3 | 3 |

论文实验结论：在 RLHF 偏好对齐任务上，REINFORCE 风格的优化（含 RLOO）一致优于 PPO，也优于 DPO、RAFT 等"RL-free"方法；RLOO 相比只取最优样本做 SFT 的 RAFT，能利用全部 $k$ 个样本的梯度信息（包括负优势样本"往哪儿压"的信息），对奖励噪声更稳健。

## 实现要点

```python
# RLOO：每个 prompt 采 k 个回答
ys      = policy.generate(x, n=k)
rewards = reward_fn(x, ys) - beta * kl_to_ref(ys)        # [k]，KL 并入奖励

baseline = (rewards.sum() - rewards) / (k - 1)           # 留一均值，向量化
adv      = rewards - baseline                            # 无偏优势

logp = policy.logp(ys).sum(dim=-1)                       # 整条序列 log prob
loss = -(adv.detach() * logp).mean()
```

- 留一基线可向量化为 `(sum - r_i) / (k-1)`，无需双重循环。
- 优势是序列级标量，乘的是**整条回答的 $\log\pi_\theta$ 之和**；实现成 token 级时即把同一优势广播到每个 response token（注意 mask 掉 prompt 与 padding）。
- on-policy 意味着 rollout 与更新交替进行、经验不复用。如果工程上想复用 minibatch 多更新几步，就需要补上重要性比值与 clip——那基本就走回 GRPO 了。
- 现成实现：HF TRL 提供 `RLOOTrainer`，OpenRLHF、verl 等框架也内置了 RLOO 优势估计器。

## 调参与实践经验

- **$k$ 的选择**：论文用 $k=2/4$。$k$ 越大基线方差越小、单步梯度越稳，但采样成本线性增长；$k=2$ 即可显著优于单样本 REINFORCE（带 EMA 基线），性价比拐点通常在 4 附近。
- **KL 系数 $\beta$** 仍是最重要的稳定旋钮：RLOO 没有 clip 兜底，策略偏移完全靠 KL 项约束，$\beta$ 过小会先于 PPO 出现 reward hacking。
- **奖励方差监控**：全组奖励相同的 prompt 优势为 0、无梯度（与 GRPO 同样的问题），可借鉴 [DAPO](/rlhf/dapo) 的动态采样把它们过滤掉。
- 适用判断：奖励来自 RM 的经典对齐场景、希望最小化工程复杂度时，RLOO 是 critic-free 家族里最简洁的选择；长 CoT 推理 + 大规模 off-policy 训练场景，社区实践更多落在 GRPO/[DAPO](/rlhf/dapo)/[GSPO](/rlhf/gspo) 一系，因为它们容许 minibatch 复用与更激进的吞吐优化。

## 参考文献

- Ahmadian et al., 2024. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* arXiv:2402.14740（ACL 2024）
- Kool et al., 2019. *Buy 4 REINFORCE Samples, Get a Baseline for Free!*
- Williams, 1992. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.*（REINFORCE 原始论文）
