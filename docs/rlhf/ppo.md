---
title: PPO（Proximal Policy Optimization）
---

# PPO（Proximal Policy Optimization）

> **一句话**：用裁剪的重要性采样比值限制每步策略更新幅度，配合 critic 估计的优势函数做策略梯度更新；它是 RLHF 的经典完整方案，稳定但工程复杂（需 4 个模型同时驻留）。出自 *Proximal Policy Optimization Algorithms*（Schulman et al., 2017），用于 RLHF 见 *InstructGPT*（2022）。
> 提出年份：2017 · 机构/团队：OpenAI · 会议/来源：arXiv:1707.06347（用于 RLHF 见 InstructGPT, 2022, arXiv:2203.02155）
>
> 前置阅读：[RLHF 总览](/rlhf/)、[Reward Model](/rlhf/reward-model)

## 直觉与动机

普通策略梯度（如 REINFORCE）的核心问题是**步长不可控**：用同一批采样数据做多步更新时，策略一旦走得太远，旧数据的重要性权重就严重失真，更新方向变得不可信，训练随之崩溃。TRPO 用一个二阶 KL 信赖域约束来限制每步走多远，但二阶优化在大模型上代价高昂、实现繁琐。

PPO 的思路是：把"别走太远"这个约束**塞进损失函数本身**，用一阶优化器就能实现。具体做法是裁剪重要性比值——当某个动作的概率比相对旧策略变化超过 $\epsilon$ 时，就把它对应的优势项"夹住"，使其不再贡献梯度。这样既能在一批数据上安全地做多轮更新（sample efficiency），又避免了二阶计算。

## 方法与公式

**LLM 场景的 MDP 建模**：自回归生成被建模为序列决策——状态 $s_t$ 是 prompt $x$ 加上已生成的前 $t-1$ 个 token，动作 $a_t$ 是第 $t$ 个 token $y_t$，策略 $\pi_\theta(a_t|s_t)$ 即语言模型在该位置的 token 分布。一条回答就是一条 trajectory。

**奖励构成**：奖励模型 $r_\phi$ 只在序列末端给一个标量分；但 KL 约束需要落到每个 token。标准做法是把奖励写成逐 token 形式——中间 token 只有 KL 惩罚，末端 token 额外加 RM 分：

$$
r_t = -\beta\,\log\frac{\pi_\theta(y_t|s_t)}{\pi_{\text{ref}}(y_t|s_t)} + \underbrace{r_\phi(x,y)\cdot \mathbb{1}[t = T]}_{\text{仅末端 token}}
$$

把 KL 写进 reward 而非显式加在 loss 上，是 InstructGPT 的经典做法（也称 per-token KL penalty）。

**优势估计（GAE）**：用 critic $V(s_t)$ 配合 TD 残差做指数加权，平衡偏差与方差（Schulman et al., 2015, arXiv:1506.02438）：

$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \,\delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$\lambda=1$ 退化为蒙特卡洛回报（高方差低偏差），$\lambda=0$ 退化为单步 TD（低方差高偏差）。

**裁剪策略目标（token 级）**：

$$
\mathcal{L}_{\text{PPO}}^{\text{clip}} = -\,\mathbb{E}_t \Big[ \min \big( \rho_t A_t, \; \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, A_t \big) \Big], \qquad \rho_t = \frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_{\text{old}}}(y_t|s_t)}
$$

$\min$ 与 clip 的组合保证：当 $A_t>0$（这个 token 是好动作）时增大其概率，但比值涨到 $1+\epsilon$ 就封顶；当 $A_t<0$ 时减小其概率，但跌到 $1-\epsilon$ 也封底。即"好的别奖太多，坏的别罚太狠"。

**Critic 损失**：value head 回归到 GAE 目标 $V_{\text{targ}} = A_t + V_{\text{old}}(s_t)$，并常对 value 也做裁剪以稳定：

$$
\mathcal{L}_{\text{value}} = \mathbb{E}_t\Big[\max\big((V_\theta(s_t)-V_{\text{targ}})^2,\;(\mathrm{clip}(V_\theta,V_{\text{old}}\pm\epsilon_v)-V_{\text{targ}})^2\big)\Big]
$$

四个模型同时存在：**policy**（训练）、**critic**（训练）、**reference**（冻结，算 KL）、**reward model**（冻结，打分）。这是 PPO 显存开销大的根源。

![InstructGPT 的三阶段 RLHF 流程：SFT、训练奖励模型、用 PPO 优化策略](/papers/ppo/instructgpt-rlhf.png)

> 图源：Ouyang et al., *Training language models to follow instructions with human feedback (InstructGPT)*, arXiv:2203.02155（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | PPO | GRPO / RLOO |
| --- | --- | --- |
| Critic | 需要（再训一个同规模模型） | 不需要 |
| 同时驻留模型 | 4（policy/ref/RM/critic） | 3 |
| 优势估计 | GAE，token 级、有 value 基线 | 组内相对，序列级 |
| 每 prompt 采样数 | 1（可多，但非必需） | $G$ 个（组采样） |
| 显存与工程复杂度 | 高 | 中 |
| 稳定性 | 成熟、可控，但调参敏感 | 实现简单，长序列下需额外修正 |

## 实现要点

```python
# PPO-for-RLHF 主循环（伪代码）
for iteration in range(N):
    # 1) Rollout：用 π_old 采样回答，并缓存 logprob、value、reward
    prompts = sample_prompts(batch)
    responses, old_logprobs, values = policy.generate(prompts)      # π_old
    ref_logprobs = ref_model.logprobs(prompts, responses)          # 冻结
    rm_scores   = reward_model(prompts, responses)                 # 冻结，末端分
    rewards = per_token_kl_penalty(old_logprobs, ref_logprobs, beta)
    rewards[:, -1] += rm_scores                                    # 末端加 RM 分
    # 2) 计算优势与回报
    adv, returns = gae(rewards, values, gamma, lam)
    adv = whiten(adv)                                              # advantage 归一化
    # 3) 在同一批数据上做多个 epoch 的 minibatch 更新
    for epoch in range(ppo_epochs):          # 通常 1~4
        for mb in minibatches(...):
            ratio = exp(policy.logprobs(mb) - mb.old_logprobs)
            pg_loss = -min(ratio*mb.adv, clip(ratio,1-eps,1+eps)*mb.adv)
            v_loss  = value_clip_loss(critic(mb), mb.returns, mb.values)
            loss = pg_loss.mean() + c_v*v_loss - c_ent*entropy
            loss.backward(); optimizer.step()
```

关键工程细节：
- **KL 加在 reward 还是 loss**：加在 reward（per-token penalty）是 InstructGPT 主流；也有实现把 KL 作为显式 loss 项。两者数学上不等价（前者影响优势估计，后者只影响梯度），实践中前者更常用。
- **重放 epoch 数**：同批数据更新 1~4 个 epoch；越多越省采样但越易让 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$，clip 触发频繁。
- **归一化**：advantage whitening（减均值除标准差）几乎是必做；reward scaling/clipping 也常见。
- **adaptive KL**：用目标 KL 自动调 $\beta$——实测 KL 高于目标就增大 $\beta$，反之减小。

## 调参与实践经验

- **clip $\epsilon$**：常用 $0.2$；偏大更激进、偏小更保守。
- **$\gamma$、$\lambda$**：语言任务奖励集中在末端，$\gamma$ 常取 $1.0$（不折扣），$\lambda$ 取 $0.95$ 左右。
- **KL 系数 $\beta$**：最关键的稳定旋钮。太小则 reward hacking、模型跑飞；太大则学不动、退化回 SFT。常配合 adaptive KL，目标 KL 设在个位数（按实现的归一化方式而定）。
- **崩溃征兆**：
  - **KL 爆炸**：$\pi_\theta$ 急速偏离 $\pi_{\text{ref}}$，输出开始出现乱码或重复——立即加大 $\beta$ 或降学习率。
  - **熵塌缩**：策略熵骤降、输出高度雷同、探索消失——往往是学习率过大或 reward 过尖。
  - **reward 与 KL 同时飙升**：典型 reward hacking，RM 被钻空子，需回看 RM 质量。
- **critic 预热**：开训前先让 critic 单独拟合几步 value，可减少早期优势估计噪声。
- **batch 与采样规模**：rollout batch 越大优势估计越稳，但生成是瓶颈；通常用 vLLM 等推理引擎加速采样阶段。

## 参考文献

- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Schulman et al., 2015. *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE).* arXiv:1506.02438
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback (InstructGPT).* arXiv:2203.02155
