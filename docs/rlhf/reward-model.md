---
title: Reward Model（奖励模型）
---

# Reward Model（奖励模型）

> **一句话**：在人类偏好对上训练一个标量打分模型 $r_\phi(x,y)$，作为 RL 阶段的奖励来源；它把"人类喜欢什么"压缩进一个可在线查询的函数，其质量直接决定 RLHF 的上限。出自 *InstructGPT*（Ouyang et al., 2022）。
> 提出年份：2022 · 机构/团队：OpenAI · 会议/来源：NeurIPS 2022 / arXiv:2203.02155（Bradley-Terry RM 思路可上溯至 Stiennon et al., 2020, arXiv:2009.01325）
>
> 前置阅读：[RLHF 总览](/rlhf/)、[记号约定](/guide/notation)

## 直觉与动机

强化学习每一步都需要对当前采样的回答打分，但人类标注无法在线、实时、按 token 提供。Reward Model（RM）的作用就是把离线收集的人类偏好"蒸馏"成一个可微、可高频调用的代理函数：在 RL 阶段，策略 $\pi_\theta$ 每采样一条回答 $y$，RM 立刻给出标量奖励 $r_\phi(x,y)$，供 [PPO](/rlhf/ppo)、[GRPO](/rlhf/grpo) 等算法使用。

为什么用"偏好对"而不是直接打绝对分？因为人类对"这个回答值 7 分还是 8 分"的判断噪声极大、标注者之间难以校准，但对"A 和 B 哪个更好"的相对判断要稳定得多。RM 训练正是建立在这种相对比较上的。

## 方法与公式

**结构**：RM 通常复用 SFT 模型的 backbone，去掉原本的 language modeling head，换上一个输出标量的 value head（一个线性层）。给定 $(x,y)$，取序列**最后一个 token** 位置的 hidden state，经线性层映射为标量分数 $r_\phi(x,y)$。之所以取最后一个 token，是因为只有读完整条回答后模型才能给出整体评价。

**Bradley-Terry 损失**：给定偏好对 $(x, y_w, y_l)$，其中 $y_w$ 是人类更偏好的回答（chosen），$y_l$ 是较差的（rejected），BT 模型假设"$y_w$ 胜过 $y_l$"的概率为 $\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$，于是最小化负对数似然：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \big[ \log \sigma \big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \big]
$$

其中 $\sigma$ 是 sigmoid。注意损失只依赖**分数差**，所以 RM 的绝对数值没有语义——它只在同一 prompt 内可比，这也是后续算法都对 reward 做归一化的原因。

**常见变体**：
- **Margin 损失**：当标注带有"好多少"的等级时，引入间隔 $m$：$\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)-m)$，强迫差距更大的对拉开更大分差。
- **多目标 RM**：把 helpfulness、safety、honesty 等拆成多个 value head 分别打分，再加权融合，便于按场景调权重。
- **一次比较多个回答**：当一个 prompt 下有 $K$ 个排序回答时，可在一个 batch 内组成 $\binom{K}{2}$ 个对，提高数据利用率（InstructGPT 即如此）。

**ORM vs PRM**：
- **Outcome Reward Model（ORM）**：只对最终结果打一个分，是上面描述的默认形态。
- **Process Reward Model（PRM）**：对推理过程的每一步打分（如每步是否正确），出自 *Let's Verify Step by Step*（Lightman et al., 2023）。PRM 的奖励更稠密、对长思维链监督更细，但需要昂贵的步骤级标注。注意当前推理 RL（如 DeepSeek-R1）更多直接用规则可验证奖励（RLVR）替代了 PRM/ORM——详见 [RLHF 总览](/rlhf/)。

## 与 baseline 对比

| 维度 | Reward Model | 规则/可验证奖励（RLVR） | DPO 隐式奖励 |
| --- | --- | --- | --- |
| 奖励来源 | 学习得到的 $r_\phi$ | 规则判定（答案/单测） | $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ |
| 适用任务 | 开放式、主观 | 数学/代码等可验证 | 有离线偏好对即可 |
| 是否需训练额外模型 | 需要 | 不需要 | 不需要 |
| reward hacking 风险 | 高 | 低 | 中（隐式奖励仍可被钻） |
| 在线打分能力 | 有 | 有 | 不在线（监督式） |

## 实现要点

```python
# Bradley-Terry RM 训练核心（伪代码）
def rm_loss(model, batch):
    # batch: chosen/rejected 成对，已拼好 prompt+response
    r_w = model(batch.chosen_ids).last_token_score   # r_φ(x, y_w)
    r_l = model(batch.rejected_ids).last_token_score  # r_φ(x, y_l)
    loss = -F.logsigmoid(r_w - r_l).mean()
    # 可选：把 reward 均值拉向 0，稳定后续 RL（中心化正则）
    loss = loss + 1e-3 * (r_w + r_l).mean().pow(2)
    return loss
```

工程上需注意：
- **取分位置**：用 attention mask 找到每条序列真正的最后一个非 padding token，再取该位置打分；padding 在右还是在左会改变索引方式。
- **chosen/rejected 同 batch**：保证同一对在同一 forward，避免梯度不一致；可把成对样本展平后用 reshape 还原。
- **freeze backbone 与否**：数据少时可只训 value head + 顶部几层；数据充足时全参微调效果更好。
- **value head 初始化**：用小方差初始化，避免训练初期 reward 数值爆炸。

## 调参与实践经验

- **数据质量 > 数据量**：标注者一致性低的偏好对会直接污染 RM。先做标注校准、剔除"平局"或矛盾对，往往比单纯加量更有效。
- **RM 准确率不是越高越好**：RM 在留出偏好对上的二分类准确率（pairwise accuracy）是常用监控指标，但准确率过拟合的 RM 在 RL 中反而更易被 hacking——它对分布外回答的打分更脆。关注的应是**与下游 RL 效果的相关性**，而非 RM 单点指标。
- **长度偏置**：人类标注天然偏好更长、更详尽的回答，RM 会学到"越长分越高"。这会诱导 RL 阶段策略疯狂灌水。缓解手段：训练时做长度去偏（如对长度做回归后取残差）、RL 阶段对长度加惩罚、或使用对长度归一化更鲁棒的算法（见 [DAPO](/rlhf/dapo)）。
- **Reward hacking**：策略会找到 RM 的盲区，产出 RM 高分但人类不喜欢的回答（重复套话、特定格式、谄媚等）。监控手段：RL 过程中定期人评抽样、观察 KL 与 reward 是否"reward 飞涨但 KL 也飞涨"——这通常是 hacking 的信号。KL 系数 $\beta$ 是第一道防线。
- **OOD 漂移**：RL 进行中策略分布持续偏离 RM 的训练分布，RM 打分逐渐失真。应对方式是**迭代式 RLHF**：训练一段后重新采样、重新标注、重训 RM，让 RM 跟上策略分布（InstructGPT 即多轮迭代）。

## 参考文献

- Ouyang et al., 2022. *Training language models to follow instructions with human feedback (InstructGPT).* arXiv:2203.02155
- Lightman et al., 2023. *Let's Verify Step by Step.* arXiv:2305.20050
- Stiennon et al., 2020. *Learning to summarize from human feedback.* arXiv:2009.01325
