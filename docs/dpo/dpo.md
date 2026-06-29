---
title: DPO（Stanford）：把 RLHF 求出闭式解，语言模型本身就是隐式奖励模型
---

# DPO（Stanford）：把 RLHF 求出闭式解，语言模型本身就是隐式奖励模型

**📄 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)**

2023-05 · Stanford University / CZ Biohub · [代码](https://github.com/eric-mitchell/direct-preference-optimization)

**一句话**：把 RLHF「KL 约束下最大化奖励」的目标求出**闭式解**，发现奖励可以反解成策略自身的函数——于是「训 reward model + 跑 PPO」两阶段被压缩成一个直接在偏好对上做的 sigmoid 分类损失，不再需要在线采样、不再需要 RL。

::: details 📖 论文原文 Abstract（英文）
While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call **Direct Preference Optimization (DPO)**, is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.
:::

**相关**：[偏好优化（DPO 家族）总览](/dpo/) · [RLHF 总览](/rlhf/) · [Reward Model](/rlhf/reward-model) · [PPO](/rlhf/ppo) · [SimPO](/dpo/simpo) · [IPO](/dpo/ipo) · [KTO](/dpo/kto)

![左：经典 RLHF 先用偏好数据训一个 reward model，再用强化学习（采样补全→打分→更新 LM policy）这一闭环优化策略；右：DPO 跳过 reward model 与 RL，直接用一个最大似然分类目标把偏好数据训成 final LM](/papers/dpo/teaser.png)

> 图源：Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（arXiv:2305.18290）Figure 1——左 RLHF 两阶段、右 DPO 单阶段（用于学习注解，版权归原作者）。

## 动机与创新点：RLHF 的两阶段是不是必要的？

经典 [RLHF](/rlhf/) 的痛点几乎都来自它的**两阶段结构**。论文开篇就点名 RLHF 是 *"a complex and often unstable procedure"*：先用人类偏好对训一个 [reward model](/rlhf/reward-model) $r_\phi$，再用 [PPO](/rlhf/ppo) 让策略去最大化 $r_\phi$ 的打分、同时不要偏离初始模型太远。这条流水线的代价是实打实的——

> the RLHF pipeline is considerably more complex than supervised learning, involving training multiple LMs and sampling from the LM policy in the loop of training, incurring significant computational costs.

具体落到工程上：训练时显存里要同时驻留 policy / reference / reward / critic **四个模型**；reward model 的误差会被 PPO 放大成 reward hacking；PPO 本身超参敏感、容易训崩。

DPO 的关键洞察是：RLHF 那个被优化的目标——「最大化奖励，同时用 KL 约束策略不要偏离 reference 太远」——其最优解是有**解析形式**的。既然策略和奖励之间存在一个解析映射，就可以**反过来把奖励写成策略自身的函数**，于是 reward model 这个中间变量根本不必显式训练出来。论文把这一步称作 *"a change of variables"*：

> DPO uses a change of variables to define the preference loss as a function of the policy directly.

把反解代回 Bradley-Terry 偏好概率模型，整个对齐就退化成一个直接在偏好对 $(y_w, y_l)$ 上的监督分类任务。不再有在线采样，不再有 critic，不再有 RL 的不稳定。论文还强调 DPO 并非靠简化牺牲了理论——它优化的是**和 RLHF 完全相同的目标**，*"an algorithm that implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL-divergence constraint)"*，只是换了求解方式。

**关键创新**：

- **reward 的新参数化**：用 $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 把奖励直接写成「策略相对 reference 的对数概率比」，让对应最优策略可被闭式抽取。
- **配分函数消去技巧**：难算的配分函数 $Z(x)$ 只依赖 $x$，在 Bradley-Terry 的「奖励作差」里被自动消掉——这是 DPO 能绕开归一化常数的核心。
- **单阶段、离线、无 RL**：把「训 RM + 跑 PPO」压成一个 sigmoid 分类损失，训练稳定得像监督学习，无需在线采样、无需 critic、几乎不用调超参。
- **隐式 reward = 语言模型本身**：训练后的策略对数概率比就是一个隐式 reward model（标题的「secretly a reward model」），还可反过来用作打分器。
- **动态难样本加权**：损失梯度自带一个 per-example 重要性权重，对「排错的偏好对」更新更强，*"prevents the model degeneration that we find occurs with a naive probability ratio objective"*。

把 DPO 与 PPO 版 RLHF 放在一张表里对照，差异一目了然：

| 维度 | RLHF（PPO） | DPO |
| --- | --- | --- |
| 训练阶段 | RM + RL 两阶段 | 单阶段 |
| 同时驻留模型数 | 4（policy / ref / RM / critic） | 2（policy / ref） |
| 在线采样 | 需要（训练循环里采样补全） | 不需要（离线偏好对） |
| 训练稳定性 | 超参敏感、易崩 | 类似监督训练，稳 |
| 显式 reward model | 需要 | 无（隐式） |
| 优化目标 | KL 约束奖励最大化 | **同一目标的闭式解** |
| 效果上限 | 难任务上更高（可用新采样数据） | 受离线偏好分布限制 |

## 方法：从 KL 约束目标到一个分类损失

### 起点：RLHF 的 Bradley-Terry 偏好与 KL 约束目标

RLHF 的标准流程分三步：SFT → 偏好采样 + reward 建模 → RL 微调。偏好建模这一步通常假设人类偏好服从 **Bradley-Terry（BT）模型**——一个补全好不好，由它和对手的奖励之差决定：

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x,y_1))}{\exp(r^*(x,y_1)) + \exp(r^*(x,y_2))} = \sigma\big(r^*(x,y_1) - r^*(x,y_2)\big)
$$

于是 reward model 用一个二分类负对数似然来拟合：$\mathcal{L}_R = -\mathbb{E}\big[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big]$。拿到 $r_\phi$ 后，RL 阶段优化的是带 KL 约束的奖励最大化目标：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)\right]
$$

论文特别说明这个 KL 约束**不是可有可无的**：*"it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers."* 但由于语言生成是离散的、这个目标不可微，传统做法只能上 PPO。DPO 要做的，就是把这一步从 RL 里解放出来。

### 核心代数：求闭式解 → 反解 reward → 消去配分函数

这一节是 DPO 全部的「魔法」，四步走：

```mermaid
flowchart LR
    A["KL 约束<br/>奖励最大化目标"] -->|"标准结论"| B["闭式最优解 π*<br/>= (1/Z)·π_ref·exp(r/β)"]
    B -->|"两边取对数<br/>移项"| C["反解 reward<br/>r = β·log(π*/π_ref) + β·logZ(x)"]
    C -->|"代入 BT 模型<br/>奖励作差"| D["β·logZ(x) 抵消<br/>偏好只依赖 π 与 π_ref"]
    D -->|"对偏好数据<br/>做最大似然"| E["DPO 损失<br/>一个 sigmoid 分类目标"]
```

**第一步——写出闭式最优解。** 上面那个 KL 约束目标对 $\pi_\theta$ 有标准的解析最优解（KL 正则化奖励最大化的经典结论）：

$$
\pi_r(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\tfrac{1}{\beta}\,r(x,y)\right)
$$

其中 $Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp(\tfrac{1}{\beta}r(x,y))$ 是配分函数。论文直言它没法用：*"it is still expensive to estimate the partition function $Z(x)$, which makes this representation hard to utilize in practice."*

**第二步——反解 reward。** 既然策略写成了奖励的函数，就两边取对数、移项，把 reward 反过来表达成策略的函数：

$$
r(x,y) = \beta\log\frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)
$$

这正是标题那句话的数学化身——**reward 就藏在语言模型（相对 reference 的对数概率比）里**。

**第三步——配分函数自动消去。** 把上式代入 BT 模型。注意 BT 只依赖**两个补全的奖励之差**，而 $\beta\log Z(x)$ 只跟 $x$ 有关、在 $y_w$ 与 $y_l$ 上完全相同，作差时**直接抵消**：

$$
p^*(y_w \succ y_l \mid x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

> 举例：就像比两家店谁更便宜时，两边同时加的「停车费 $Z(x)$」会在相减时消掉——你只需要比商品本身的差价。DPO 正是靠「奖励作差」躲开了那个谁也算不动的归一化常数。

**第四步——最大似然得到 DPO 损失。** 把策略参数化为 $\pi_\theta$、对偏好数据集做最大似然，就得到最终目标：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\,\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[ \log\sigma\!\left( \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right]
$$

论文还从理论上证明（Theorem 1）这个重参数化**没有损失任何泛化性**——在 Plackett-Luce/BT 框架下，任何奖励等价类都能用 $r(x,y)=\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 表示，所以「不显式训 reward model」并不缩小可表达的奖励空间。

### 隐式 reward 与梯度：这个更新到底在做什么

定义**隐式 reward** $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，损失就是 $-\log\sigma(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l))$——一个标准的成对排序（pairwise ranking）分类损失。训练做的事就是：**拉高 chosen 的隐式 reward、压低 rejected 的隐式 reward**。

对损失求梯度，机制更清楚（$\hat r_w,\hat r_l$ 为 chosen/rejected 隐式 reward）：

$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l-\hat r_w)}_{\text{排错时权重大}}\big(\underbrace{\nabla_\theta\log\pi_\theta(y_w|x)}_{\text{抬高 chosen}}-\underbrace{\nabla_\theta\log\pi_\theta(y_l|x)}_{\text{压低 rejected}}\big)\Big]
$$

论文强调那个权重系数不是装饰，而是防止退化的关键：

> the DPO update increases the relative log probability of preferred to dispreferred responses, but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective.

> 举例：当模型把一对偏好**排错**了（$\hat r_l > \hat r_w$，即它觉得 rejected 反而更好），$\sigma(\hat r_l-\hat r_w)$ 接近 1、梯度很大、使劲纠正；对已经排对的「简单样本」，权重趋近 0、几乎不更新。这是一种天然的难样本聚焦，也是为什么不能直接用裸的概率比目标。

### β 与 reference model：两个绕不开的工程量

**$\beta$ 的含义**。$\beta$ 是 KL 惩罚强度，控制策略可以偏离 $\pi_{\text{ref}}$ 多远：$\beta$ 越大约束越强、改动越保守；$\beta\to 0$ 约束消失，模型可以为满足偏好任意偏移。它同时出现在隐式 reward 的尺度里，因此也是损失梯度的标度。

**reference model 从哪来**。论文规定 $\pi_{\text{ref}}$ 应当就是 SFT 模型：*"we initialize $\pi_{\text{ref}}=\pi^{\text{SFT}}$ whenever available."* 若拿不到对应的 SFT 模型（如对话任务无现成 SFT），则退而求其次——在 chosen 补全上做最大似然得到一个近似 reference，以缓解 $\pi_{\text{ref}}$ 与真实参考分布之间的分布漂移。

### 落到代码：一个 batch 的 DPO loss

```python
# policy 与 ref 各前向一次, 注意 logprob 按序列「求和」, 不是平均
def dpo_loss(policy, ref, x, y_w, y_l, beta):
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

- **logprob 按序列求和，不要平均**。隐式 reward 定义里就是 token logprob 之和，平均会改变目标。这是与 [SimPO](/dpo/simpo) 的关键差异，也是 DPO 长度偏置的来源。
- **reference model 两种工程处理**：① 显存够就放一份冻结副本，每步对 $(y_w,y_l)$ 各前向一次；② 显存紧就**预计算** $\pi_{\text{ref}}$ 在全部偏好对上的 logprob 缓存到磁盘，训练时不再加载 ref，省下一份模型显存。
- **mask**：只对 response 部分累加 logprob，prompt 与 padding 必须 mask，否则梯度被 prompt 污染。
- 现成实现：HF TRL 的 `DPOTrainer`，传偏好数据集与 `beta` 即可；`loss_type` 可切到 [IPO](/dpo/ipo) 等变体。

### 调参与实践经验

- **$\beta$**：常见 $0.05\sim0.5$，最常用 $0.1$。偏好数据质量高、希望改动大就调小；担心退化就调大。几乎是 DPO 第一个要扫的超参。
- **必须先 SFT**。DPO 假设 $\pi_{\text{ref}}$ 已是个像样的指令模型、偏好对也接近其分布。直接对 base model 做 DPO 效果差，标准流程是先 [SFT](/sft/) 再以它为 $\pi_{\text{ref}}$ 起点。
- **chosen logprob 一起下降的现象**。训练中常见 $\log\pi_\theta(y_w|x)$ 和 $\log\pi_\theta(y_l|x)$ **同时下降**，只是 rejected 降得更快、margin 在扩大——因为 BT 损失只约束两者之差、不约束绝对水平。轻微下降正常；但若 chosen logprob 暴跌，说明模型在「靠压低 rejected」而非「抬高 chosen」来满足偏好，可能损害生成质量，可加一个 SFT/NLL 辅助项锚住 chosen 的绝对概率。
- **学习率**比 SFT 小，常见 $5\text{e-}7\sim5\text{e-}6$；DPO 对学习率敏感，过大会快速退化。
- **长度偏置**：上线前检查输出长度分布，必要时做长度配对，或改用 [SimPO](/dpo/simpo) 的长度归一损失。
- **训练轮数**：通常 1~3 个 epoch，DPO 容易过拟合，多看验证集上的偏好准确率而非只看 loss。

## 实验结果：情感/摘要/对话三任务上追平或超过 PPO

### 评测设置：三个开放式生成任务

论文在三个任务上对比 DPO 与一系列基线（PPO、PPO-GT 用真值奖励的 oracle、Preferred-FT、Unlikelihood、Best-of-128、GPT-J/Pythia 等），模型规模到 6B：

- **可控情感生成（IMDb）**：用预训练情感分类器当**真值奖励**，可直接画出「奖励 vs KL」前沿，纯净地考察优化效率。
- **摘要（Reddit TL;DR）**：GPT-2-large/GPT-J，用 GPT-4 当评委算对参考摘要的胜率。
- **单轮对话（Anthropic HH）**：Pythia-2.8B，无现成 SFT，先在 chosen 上微调当 reference，再 DPO。

### Benchmark 表现（以原文为准）

![左：IMDb 情感任务的「期望奖励 vs KL」前沿——DPO（黄）在所有 KL 取值下都拿到最高奖励，前沿严格压住 PPO，甚至压过能访问真值奖励的 PPO-GT；右：TL;DR 摘要对参考摘要的 GPT-4 胜率随采样温度变化，DPO（黄）在温度 0 约 0.61、且对温度变化最稳健，PPO（粉）在高温迅速崩塌](/papers/dpo/frontier.png)

> 图源：Rafailov et al., *Direct Preference Optimization*（arXiv:2305.18290）Figure 2——左 IMDb 奖励-KL 前沿、右 TL;DR 胜率 vs 采样温度（用于学习注解，版权归原作者）。

- **可控情感（左图）**：DPO 的奖励/KL 前沿**严格压住 PPO**，并且 *"even when PPO can access ground truth rewards (PPO-GT)"* 仍然更优——说明同一目标下 DPO 的优化效率更高。
- **摘要（右图）**：DPO 在温度 0 的胜率约 **61%**，超过 PPO 在其最优温度下的约 **57%**；且 DPO 对采样温度**远更稳健**，而 PPO 的胜率在高温会退化到接近基座水平。

![左：Anthropic-HH 单轮对话对 chosen 回答的 GPT-4 胜率——DPO（黄）是唯一稳定越过 0.5、即超过数据集里人类偏好回答的方法，并追平甚至超过昂贵的 Best-of-128（绿）；右：DPO 胜率随微调步数的演化，约 300 步即收敛到最佳并保持稳定](/papers/dpo/dialogue.png)

> 图源：Rafailov et al., *Direct Preference Optimization*（arXiv:2305.18290）Figure 3——左 Anthropic-HH 对话胜率、右 训练步数 vs 胜率（用于学习注解，版权归原作者）。

- **对话（左图）**：DPO 是**唯一**在 Anthropic-HH 上稳定超过 chosen 回答的方法，并与需要在测试时采样 128 次的 **Best-of-128** 打平或更好——后者计算代价高得多。
- **收敛快（右图）**：DPO 约 **300 步**即达到最佳胜率并保持稳定，对不同采样温度都稳。
- **泛化到新分布**：把 TL;DR 上训的策略直接评到 CNN/DailyMail 新闻摘要（分布外），DPO 仍明显优于 PPO：

| 算法 | 胜率 @ 温度 0 | 胜率 @ 温度 0.25 |
| --- | --- | --- |
| **DPO** | **0.36** | **0.31** |
| PPO | 0.26 | 0.23 |

- **评测可信度**：论文做了人评对照（Table 2），发现 *"GPT-4 judgments correlate strongly with humans, with human agreement with GPT-4 typically similar or higher than inter-human annotator agreement"*——GPT-4 当评委与人的一致性不低于人和人之间。

> 看榜须知：这些胜率口径（评委 prompt、采样温度、基线选择）各异，且规模仅到 6B、任务偏窄（情感/摘要/单轮对话）。跨设置直接比绝对值意义有限，当作「DPO 与 PPO 在同口径下孰优」的相对参照即可；DPO 的核心卖点是**用更简单的训练拿到追平或更优的效果**，而非刷某个绝对分。

## 在偏好优化（DPO 家族）谱系里的位置

DPO 是这一整支「不训 reward model、不跑 RL 的偏好对齐」谱系的**起点与公共基座**，后续变体大多是在它的损失上做加减法，沿三条线演化（详见 [DPO 家族总览](/dpo/)）：

- **DPO 与 PPO 解的是同一个目标**。两者优化的都是「KL 约束奖励最大化」，区别只在求解方式——PPO 在线 RL、DPO 离线闭式解。DPO 的根本局限也由此而来：它**只能从给定的离线偏好对学习**，无法利用策略训练中新产生的分布。在数学、代码、长 CoT 等需要持续探索的任务上，在线 RL（[PPO](/rlhf/ppo)、[GRPO](/rlhf/grpo)）通常仍有更高天花板。
- **去掉 reference model 这条线**：[SimPO](/dpo/simpo) 用「长度归一化平均对数概率 + 目标 margin」替掉隐式 reward，连 $\pi_{\text{ref}}$ 一起省掉；[CPO](/dpo/cpo) 用均匀先验近似 reference 得到 DPO 损失的上界从而去掉 $\pi_{\text{ref}}$；[ORPO](/dpo/orpo) 干脆把偏好惩罚并进 SFT 的 NLL 损失，单阶段同时学会回答与对齐。三者都在攻 DPO「要多驻留一份 ref 模型」这个成本。
- **修损失形状这条线**：[IPO](/dpo/ipo) 指出 DPO 在近乎确定性的偏好下会把隐式 reward 差推向无穷而过拟合，改用平方损失把 reward 差拉向固定目标值；这是对 DPO「margin 越大越好」倾向的直接修正。
- **放宽数据形态这条线**：[KTO](/dpo/kto) 不要成对偏好，每条样本只需一个「好/坏」二元标签，用前景理论的损失厌恶设计奖惩，对正负样本不均衡更鲁棒——把 DPO 对「成对数据」的硬要求松绑。

一句话定位：**DPO 把 RLHF 的可行下界拉到了「监督训练」的复杂度**，让偏好对齐第一次变得人人可训；它之后的工作，基本都是在「更省（去 ref）、更稳（修损失）、更宽（松数据）」三个方向上继续打磨这同一个 sigmoid 分类目标。
