---
title: DPO（Direct Preference Optimization）
---

# DPO（Direct Preference Optimization）

> **一句话**：把 RLHF 的 KL 约束目标求出闭式解，发现「语言模型本身就是隐式 reward model」，从而把「训 reward model + 跑 PPO」两步压缩成一个在偏好对上的 sigmoid 分类损失。论文 *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（Rafailov et al., 2023）。
>
> 提出年份：2023 · 机构/团队：Stanford · 会议/来源：NeurIPS 2023 / arXiv:2305.18290
>
> 前置阅读：[RLHF 总览](/rlhf/)、[Reward Model](/rlhf/reward-model)、[PPO](/rlhf/ppo)

## 直觉与动机

经典 [RLHF](/rlhf/) 的痛点几乎都来自它的两阶段结构。先用人类偏好对训一个 [reward model](/rlhf/reward-model) $r_\phi$，再用 [PPO](/rlhf/ppo) 让策略去最大化 $r_\phi$ 的打分。这意味着：训练时显存里要同时驻留 policy、reference、reward、critic 四个模型；reward model 的误差会被 PPO 放大成 reward hacking；PPO 本身超参敏感、容易训崩。

DPO 的关键洞察是：RLHF 那个被优化的目标——「最大化 reward，同时用 KL 约束策略不要偏离 reference 太远」——其最优解是有**解析形式**的。既然有解析形式，就能把 reward 反解成策略自身的函数，于是 reward model 这个中间变量根本不必显式训练出来。把反解代回偏好的概率模型，整个对齐就变成一个直接在偏好对 $(y_w, y_l)$ 上的监督分类任务。不再有在线采样，不再有 critic，不再有 RL 的不稳定。

![左：经典 RLHF 先训 reward model 再用 RL 优化；右：DPO 直接用一个分类损失在偏好对上优化策略](/papers/dpo/teaser.png)

> 图源：Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290（用于学习注解，版权归原作者）

## 方法与公式

RLHF 优化的 KL 约束目标是：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big] - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)\right]
$$

这个目标对 $\pi_\theta$ 有闭式最优解（标准的 KL 正则化奖励最大化结论）：

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\tfrac{1}{\beta}\,r(x,y)\right)
$$

其中 $Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp(\tfrac{1}{\beta}r(x,y))$ 是难以计算的配分函数。两边取对数整理，把 reward **反解**成策略的函数：

$$
r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)
$$

关键一步：DPO 用 Bradley-Terry 偏好模型，$P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$。注意这里 reward 是**作差**出现的，而 $\beta\log Z(x)$ 只依赖 $x$、不依赖 $y$，在 $y_w$ 与 $y_l$ 上完全相同，作差后**直接消掉**。这正是 DPO 绕开配分函数的核心技巧。代入并对偏好数据做最大似然，得到 DPO 损失：

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x, y_w, y_l)}\left[ \log\sigma\!\left( \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right]
$$

**隐式 reward**。定义 $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，损失就是 $-\log\sigma(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l))$——一个标准的成对排序（pairwise ranking）分类损失。训练做的事就是：**拉高 chosen 的隐式 reward、压低 rejected 的隐式 reward**。

**$\beta$ 的含义**。$\beta$ 是 KL 惩罚强度，控制策略可以偏离 $\pi_{\text{ref}}$ 多远。$\beta$ 越大约束越强、改动越保守；$\beta\to 0$ 则约束消失，模型可以为了满足偏好任意偏移。它同时出现在隐式 reward 的尺度里，因此也影响损失梯度的标度。

**梯度直觉**。对损失求梯度可得（记 $\hat r_w,\hat r_l$ 为 chosen/rejected 隐式 reward）：

$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l-\hat r_w)}_{\text{权重}}\big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\big)\Big]
$$

当模型把偏好排错（$\hat r_l > \hat r_w$）时权重 $\sigma(\hat r_l-\hat r_w)$ 大、更新强；已经排对的样本权重趋近 0、几乎不更新。这是一种天然的难样本聚焦。

## 与 baseline 对比

| 维度 | RLHF（PPO） | DPO |
| --- | --- | --- |
| 训练阶段 | RM + RL 两阶段 | 单阶段 |
| 同时驻留模型数 | 4（policy/ref/RM/critic） | 2（policy/ref） |
| 在线采样 | 需要 | 不需要（离线偏好对） |
| 训练稳定性 | 超参敏感、易崩 | 类似监督训练，稳 |
| 显式 reward model | 需要 | 无（隐式） |
| 优化目标 | KL 约束奖励最大化 | 同一目标的闭式解 |
| 效果上限 | 在难任务上更高（可用新采样数据） | 受离线数据分布限制 |

DPO 与 PPO 解的是同一个 KL 约束目标，区别只在求解方式。DPO 的根本局限也由此而来：它只能从**给定的离线偏好对**学习，无法利用策略训练中新产生的分布；在数学、代码、长 CoT 等需要策略不断探索的任务上，在线 RL（[GRPO](/rlhf/grpo)、[PPO](/rlhf/ppo)）通常仍有更高天花板。

## 实现要点

```python
# 一个 batch 的 DPO loss（policy 与 ref 各一次前向）
def dpo_loss(policy, ref, x, y_w, y_l, beta):
    # 每条序列的 token logprob 之和（注意：sum 不是 mean）
    pi_w = policy.seq_logprob(x, y_w)     # 对 response token 求和, mask 掉 prompt/pad
    pi_l = policy.seq_logprob(x, y_l)
    with torch.no_grad():                 # ref 冻结, 也可预计算缓存
        ref_w = ref.seq_logprob(x, y_w)
        ref_l = ref.seq_logprob(x, y_l)

    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))   # β·(r̂_w − r̂_l)
    loss = -F.logsigmoid(logits).mean()
    # 监控指标: reward = β(π−ref); margin = logits; acc = (logits>0)
    return loss
```

- **logprob 按序列求和，不要平均**。这是与 [SimPO](/dpo/simpo) 的关键差异：DPO 的隐式 reward 定义里就是 token logprob 之和，平均会改变目标。求和也是 DPO 长度偏置的来源。
- **reference model 的两种工程处理**：① 显存够就放一份冻结副本，每步对 $(y_w,y_l)$ 各前向一次；② 显存紧就**预计算** $\pi_{\text{ref}}$ 在全部偏好对上的 logprob 并缓存到磁盘，训练时不再加载 ref——这能省下一份模型显存。
- **mask**：只对 response 部分的 token 累加 logprob，prompt 和 padding 必须 mask 掉，否则梯度被 prompt 污染。
- 现成实现：HF TRL 的 `DPOTrainer`，传入偏好数据集和 `beta` 即可；`loss_type` 参数可切到 IPO、其它变体。

## 调参与实践经验

- **$\beta$**：常见取值 $0.05\sim0.5$，最常用 $0.1$。偏好数据质量高、希望改动大就调小（如 0.05）；担心模型退化就调大。$\beta$ 几乎是 DPO 第一个要扫的超参。
- **必须先 SFT**。DPO 假设 $\pi_{\text{ref}}$ 已经是个像样的指令模型，偏好对也应接近 $\pi_{\text{ref}}$ 的分布。直接对 base model 做 DPO 效果差。标准流程是先 [SFT](/sft/) 得到 $\pi_{\text{ref}}$，再以它为起点 DPO。
- **chosen logprob 一起下降的现象**。训练中常看到 $\log\pi_\theta(y_w|x)$ 和 $\log\pi_\theta(y_l|x)$ **同时下降**，只是 rejected 降得更快、margin 在扩大。这是因为 BT 损失只约束两者之差、不约束绝对水平。轻微下降正常；但若 chosen logprob 暴跌，说明模型在「靠压低 rejected」而非「抬高 chosen」来满足偏好，可能损害生成质量——可加一个 SFT/NLL 辅助项（在 chosen 上的负对数似然）来锚住 chosen 的绝对概率。
- **学习率**要比 SFT 小，常见 $5\text{e-}7\sim5\text{e-}6$ 量级；DPO 对学习率敏感，过大会快速退化。
- **长度偏置**：上线前检查输出长度分布，必要时在数据侧做长度配对，或改用 [SimPO](/dpo/simpo) 的长度归一损失。
- **训练轮数**：通常 1~3 个 epoch，DPO 容易过拟合，多看验证集上的偏好准确率而非只看 loss。

## 参考文献

- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290（NeurIPS 2023）
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback.* arXiv:2203.02155（InstructGPT，RLHF 三段式流程）
- Bradley & Terry, 1952. *Rank Analysis of Incomplete Block Designs.*（BT 偏好模型）
