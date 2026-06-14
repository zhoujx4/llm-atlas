---
title: GSPO
---

# GSPO（Group Sequence Policy Optimization）

> **一句话**：把重要性采样从 token 级提升到序列级——用长度归一化的整句似然比作为裁剪与优化的基本单元，根治 GRPO 在长序列与 MoE 模型上的累积噪声和训练崩溃；Qwen3 系列 RL 训练所用算法。论文 *Group Sequence Policy Optimization*（Qwen Team, Alibaba，2025）。
> 提出年份：2025 · 机构/团队：阿里巴巴 Qwen Team · 会议/来源：arXiv:2507.18071
>
> 前置阅读：[GRPO](/rlhf/grpo)、[PPO](/rlhf/ppo)

## 直觉与动机

GSPO 的出发点是一个理论洁癖式的观察：**重要性采样权重要起到分布校正作用，前提是在同一分布上对多个样本求平均**。[GRPO](/rlhf/grpo) 在每个 token 位置 $t$ 上使用比值 $\rho_{i,t} = \pi_\theta(y_{i,t}|x,y_{i,<t})\,/\,\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})$，但每个位置只有一个样本——这个权重根本起不到校正 $\pi_{\theta_{\text{old}}}$ 与 $\pi_\theta$ 之间分布差异的作用，它只是往梯度里注入高方差噪声。更糟的是：

1. **噪声随长度累积**。序列越长，逐 token 的乘性噪声越积越多；裁剪机制不仅消除不了这种噪声，反而因为有偏截断进一步加剧问题。长 CoT 训练中模型可能出现**不可逆崩溃**——一旦坏更新破坏了策略结构，靠后续训练拉不回来。
2. **粒度错配**。奖励是序列级的（整道题对/错、整条回答的 RM 分），优势也是序列级的，优化和裁剪的单元却是 token 级——"奖励的单位"与"优化的单位"不一致。GSPO 的主张就是把二者对齐。
3. **MoE 的专家激活波动**。论文测量发现：MoE 模型经过一步 RL 梯度更新后，约 **10% 被激活的专家会发生改变**（expert-activation volatility）。新旧策略走的根本不是同一组专家，token 级比值因此剧烈波动。此前 Qwen 团队需要 **Routing Replay**——缓存 $\pi_{\theta_{\text{old}}}$ 的专家路由模式、计算 $\pi_\theta$ 时强制回放——才能让 GRPO 在 MoE 上收敛，代价是额外的内存与通信开销，且限制了模型用上真实路由的容量。

序列级比值对单 token 的概率波动天然平滑（做了长度归一化的几何平均），对专家路由抖动**免疫**，GSPO 训练 MoE 不再需要 Routing Replay。

## 方法与公式

定义**序列级重要性比值**（对整条序列似然比做 $1/|y_i|$ 次方的长度归一化，即逐 token 对数比的算术平均再取指数）：

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\right)^{1/|y_i|} = \exp\!\left( \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}\mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x, y_{i,<t})} \right)
$$

长度归一化有两个作用：消除长短序列间比值量级的系统差异（否则同一裁剪范围对不同长度的序列含义完全不同），并把少数 token 的似然剧变压平。优势沿用组内标准化 $\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}$，目标函数是组内**序列级**裁剪代理目标：

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x\sim\mathcal{D},\ \{y_i\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}}\left[ \frac{1}{G}\sum_{i=1}^{G} \min\Big( s_i(\theta)\,\hat{A}_i,\ \mathrm{clip}\big(s_i(\theta),\ 1-\epsilon,\ 1+\epsilon\big)\,\hat{A}_i \Big) \right]
$$

裁剪、奖励、优化全部发生在序列级。从梯度看：GSPO 对一条序列内的所有 token 施加**相同**的权重 $s_i(\theta)\hat{A}_i$，相当于"组内序列级 REINFORCE + 裁剪"；GRPO 则给每个 token 不同的噪声权重 $\rho_{i,t}\hat{A}_i$——这正是二者稳定性差异的来源。

**裁剪阈值的量级完全不同**：$s_i$ 经过几何平均后紧贴 1，论文实验取 $\epsilon$ 为 **left = 3e-4、right = 4e-4**（对比 GRPO 基线的 0.2/0.27）。论文还观察到 GSPO 裁剪掉的 token 比例比 GRPO 高约两个数量级，但训练效率反而更高——侧面说明 GRPO 保留的那些 token 级梯度大多是噪声。

论文另给出 **GSPO-token** 变体：通过 stop-gradient 改写使数值上仍等于 $s_i$、但允许给每个 token 配不同优势 $\hat{A}_{i,t}$，为多轮对话、[Agentic RL](/agent/agentic-rl/) 等需要 token 级 credit assignment 的场景留接口。

![GSPO 与 GRPO 的训练曲线对比：序列级比值带来更稳定、更高效的训练](/papers/gspo/gspo-training-curve.png)

> 图源：Zheng et al., *Group Sequence Policy Optimization*, arXiv:2507.18071（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | GRPO | GSPO |
| --- | --- | --- |
| 重要性比值 | token 级 $\rho_{i,t}$ | 序列级 $s_i$（长度归一化几何平均） |
| 裁剪单元 | 单个 token | 整条序列 |
| 裁剪阈值量级 | 0.2 / 0.27 | 3e-4 / 4e-4 |
| 噪声随序列长度 | 累积放大 | 被 $1/\lvert y_i\rvert$ 平均压制 |
| MoE 训练 | 需 Routing Replay 等补丁 | 直接收敛，无需补丁 |
| 优势估计 | 组内标准化（序列级，广播到 token） | 组内标准化（序列级，用在序列上） |
| 对训练/推理引擎数值差异 | 敏感（逐 token 比值放大误差） | 容忍度高（只需序列级似然） |

## 实现要点

```python
# GSPO 核心：log 域计算序列级比值
log_ratio = (logp_new - logp_old) * resp_mask          # [B, T] 逐 token 对数比
s = torch.exp(log_ratio.sum(-1) / resp_mask.sum(-1))   # [B] 长度归一化 + 指数

adv  = group_normalize(rewards)                         # 同 GRPO：(r - mean) / std
loss = -torch.min(s * adv,
                  s.clamp(1 - eps_low, 1 + eps_high) * adv).mean()
```

- **一切在 log 域进行**：先逐 token 求 $\log\pi_\theta - \log\pi_{\theta_{\text{old}}}$，按有效 token 数（注意是 response mask 后的长度，不含 prompt 和 padding）取平均再 `exp`，避免数值上溢/下溢。
- 序列级裁剪意味着一条序列要么整体保留、要么整体丢弃梯度，监控"被裁剪序列占比"而非 token 占比。
- 因为只依赖序列级似然，GSPO 对 rollout 引擎（vLLM/SGLang）与训练引擎之间的精度差异更鲁棒，工程上甚至可以直接用推理引擎返回的 logprob 计算比值，省一次训练框架的重计算前向；GRPO 的逐 token 比值则会把这种数值差异逐位放大。
- MoE 上无需缓存/回放路由；这是 GSPO 相对 GRPO 最大的基础设施简化。

## 调参与实践经验

- **裁剪范围必须重调**：3e-4/4e-4 是与 $s_i$ 的量级配套的，照搬 GRPO 的 0.2 等于不裁剪。可从论文值出发，按"被裁剪序列比例"微调——比例长期为 0 说明范围太宽，大面积裁剪正优势样本说明太窄。
- 长度归一化用 **token 数**，且新旧策略要用同一套 tokenization 的长度，否则比值定义本身就漂了。
- 组大小 $G$、组内标准化、动态过滤全对/全错组等 [DAPO](/rlhf/dapo) 式技巧与 GSPO 正交，可以叠加。
- 论文报告 GSPO 在同等算力下训练效率与最终性能均优于 GRPO 基线，并支撑了 [Qwen3](/base-models/qwen) 系列（含大规模 MoE）的 RL 训练；若你在 dense 小模型 + 短序列场景，GRPO 的噪声问题不明显，收益会相对有限——GSPO 的优势随序列长度和模型稀疏度增长。

## 参考文献

- Zheng et al., 2025. *Group Sequence Policy Optimization.* arXiv:2507.18071
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300（GRPO 原始提出）
- Qwen Team, 2025. *Qwen3 Technical Report.*
