---
title: GRPO（Group Relative Policy Optimization）
---

# GRPO（Group Relative Policy Optimization）

> **一句话**：去掉 critic——对同一 prompt 采样一组回答，用组内 reward 的标准化值直接当优势估计，从而省掉一个同规模的 value 模型；DeepSeek-R1 使它成为推理 RL 的主流算法。出自 *DeepSeekMath*（Shao et al., 2024）、*DeepSeek-R1*（DeepSeek-AI, 2025）。
> 提出年份：2024 · 机构/团队：DeepSeek · 会议/来源：arXiv:2402.03300（DeepSeekMath；经 DeepSeek-R1, 2025, arXiv:2501.12948 发扬光大）
>
> 前置阅读：[PPO](/rlhf/ppo)、[RLHF 总览](/rlhf/)

## 直觉与动机

[PPO](/rlhf/ppo) 的最大负担是 critic：它和策略同规模，既占显存又难训。更关键的是——在 LLM 的 RLHF 里，奖励本就只在序列末端给一个标量，让 critic 去逐 token 估计 value 本身就是个高方差、回报稀疏的难题，估出来的优势往往很粗糙。

GRPO 的洞察是：critic 存在的唯一目的是给优势估计提供一个 baseline（"这个回答相对平均水平好多少"）。既然如此，**为什么不直接对同一个 prompt 多采样几条回答，用这组回答 reward 的均值当 baseline？** 同组样本面对的是完全相同的 prompt，天然可比，组均值就是一个无需学习、无偏的基线。这样 value 模型可以彻底删掉，从 4 模型降到 3 模型。

## 方法与公式

对每个 prompt $x$ 采样 $G$ 个回答 $\{y_1,\dots,y_G\}$，各自得到 reward $\{r_1,\dots,r_G\}$（来自 [RM](/rlhf/reward-model) 或规则可验证奖励）。**组内标准化优势**：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}
$$

这是一个序列级标量——同一回答 $y_i$ 内的每个 token 共享同一个 $\hat{A}_i$（即把序列级优势广播到每个 token）。

**目标函数**（PPO 式 clip + 显式 KL 项）：

$$
\mathcal{L}_{\text{GRPO}} = -\,\mathbb{E}\Bigg[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\!\big( \rho_{i,t}\hat{A}_i,\; \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i \big) \;-\; \beta\,\mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] \Bigg]
$$

其中 $\rho_{i,t} = \dfrac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}$ 是 token 级重要性比值。

与 PPO 的两个关键差异：
1. **优势来自组内标准化**，不用 GAE、不用 critic。
2. **KL 是显式正则项**（直接加在 loss 上），而非像 PPO 那样塞进 per-token reward。

**KL 的无偏估计器（k3）**：GRPO 用一个低方差、恒非负的估计器：

$$
\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}] \approx \frac{\pi_{\text{ref}}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - \log\frac{\pi_{\text{ref}}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - 1
$$

它逐 token 计算、期望等于真实 KL 且始终 $\geq 0$，比朴素的 $\log$ 比值估计方差更小。

![PPO 与 GRPO 流程对比：GRPO 去掉 value 模型，改用组内采样的奖励均值作为基线](/papers/grpo/ppo-vs-grpo.png)

> 图源：Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300（用于学习注解，版权归原作者）

**后续修正**：GRPO 的两个细节被后来工作批评——其一，外层对 token 取 $\frac{1}{|y_i|}$ 平均引入**长度归一化偏置**，短回答里每个 token 权重更大；其二，除以组内 $\mathrm{std}$ 会放大那些"恰好难度适中"（std 小）的样本梯度，引入**难度偏置**。[DAPO](/rlhf/dapo) 改用 token 级全局归一化并去掉 std 除法，[GSPO](/rlhf/gspo) 则改用序列级重要性比值来降噪，二者都是针对长思维链大规模 RL 的稳定性修正。

## 与 baseline 对比

| 维度 | PPO | GRPO |
| --- | --- | --- |
| Critic / value | 需要 | 不需要（组均值当基线） |
| 优势粒度 | token 级（GAE） | 序列级（组内相对，广播到 token） |
| 每 prompt 采样数 | 1 | $G$（典型 8~64） |
| KL 处理 | 塞进 per-token reward | 显式 loss 正则（k3 估计） |
| 同时驻留模型 | 4 | 3 |
| 适配场景 | 通用对齐、稠密 RM | 数学/代码等可验证、推理 RL |

## 实现要点

```python
# GRPO 主循环（伪代码）
for step in range(N):
    prompts = sample_prompts(batch)
    # 1) 组采样：每个 prompt 采 G 条回答
    groups = policy.generate(prompts, num_samples=G)      # π_old
    rewards = reward_fn(groups)                           # RM 或规则可验证奖励
    # 2) 组内标准化优势（序列级标量）
    adv = (rewards - rewards.mean(dim="group")) / (rewards.std(dim="group") + 1e-6)
    adv = adv.broadcast_to_tokens()                      # 广播到每个 token
    # 3) clip 更新 + 显式 KL
    for epoch in range(grpo_epochs):
        ratio = exp(policy.logprobs(groups) - old_logprobs)
        pg = -min(ratio*adv, clip(ratio,1-eps,1+eps)*adv)
        kl = k3_kl(policy.logprobs, ref.logprobs)        # 无偏、非负
        loss = (pg + beta*kl).mean()                     # 注意 token 平均方式
        loss.backward(); optimizer.step()
```

工程要点：
- **全组同分的处理**：若一组回答全对或全错，$\mathrm{std}=0$、优势全为 0，该 prompt 贡献零梯度（白采样）。可在采样前/后做难度筛选，或如 DAPO 那样动态过滤掉全对/全错的 prompt，把算力留给"有信息量"的样本。
- **采样与训练分离**：组采样阶段用 vLLM/SGLang 等推理引擎批量生成，再切回训练引擎做更新；GRPO 因每 prompt 采 $G$ 条，采样占比更高，推理引擎效率是吞吐瓶颈。
- **框架对应**：[verl](/harness/systems)、OpenRLHF、TRL 均原生支持 GRPO；注意不同实现对 token 平均、std 归一化、KL 估计的处理细节不同，迁移时务必核对。
- **off-policy 程度**：同批多 epoch 会让 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$，clip 触发增多；推理 RL 中常用 1 个 epoch 接近 on-policy。

## 调参与实践经验

- **组大小 $G$**：典型 8~64。$G$ 越大组内基线越稳、优势估计方差越小，但采样成本线性上升。数学/代码任务里 $G=8\sim16$ 已常见好用；难题、长思维链可适当加大。
- **std 归一化要不要保留**：除以 std 会引入难度偏置（让中等难度样本梯度被放大）。若发现训练偏向某类难度，可参考 DAPO 去掉 std、改用更简单的减均值。
- **KL 系数 $\beta$**：推理 RL 场景常把 $\beta$ 设得很小甚至为 0——因为可验证奖励本身较难被 hack，且过强 KL 会压制模型探索更长推理链的能力。DeepSeek-R1 路线偏向弱 KL、强探索。但用学习型 RM 时仍需足够 $\beta$ 防 hacking。
- **奖励设计**：可验证奖励常组合"答案正确性 + 格式合规"（如是否用指定标签包裹推理）。奖励要稀疏但明确，避免可被钻的中间奖励。
- **长度爆炸监控**：随着训练，模型倾向写更长的思维链。关注响应长度曲线，配合长度归一化修正（DAPO）或长度惩罚，防止无意义灌水。

## 参考文献

- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300
- DeepSeek-AI, 2025. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948
- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
