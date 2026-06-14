---
title: SimPO
---

# SimPO（Simple Preference Optimization）

> **一句话**：去掉 reference model，把隐式 reward 换成「长度归一化的平均对数概率」，并引入一个目标 margin $\gamma$，让训练目标与生成时实际使用的度量对齐，同时缓解长度偏置。
>
> 论文：*SimPO: Simple Preference Optimization with a Reference-Free Reward* (Meng et al., 2024) ·
>
> 提出年份：2024 · 机构/团队：Princeton / UVA · 会议/来源：NeurIPS 2024 / arXiv:2405.14734
>
> 前置阅读：[DPO](/dpo/dpo)、[符号约定](/guide/notation)

## 直觉与动机

SimPO 的出发点是审视 [DPO](/dpo/dpo) 隐式 reward 的两个「不优雅」之处。

**第一，DPO 的隐式 reward 与生成度量不一致。** DPO 用 $\hat{r}(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 作为隐式 reward，其中 $\log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|x,y_{<t})$ 是**序列求和**的对数概率。但模型在解码时，无论是 greedy、beam search 还是采样，排序候选用的实际度量更接近**平均**对数概率（长度归一化后的得分）。训练目标和推理目标之间存在系统性错位：训练时把一个回答排在前面，生成时不一定真的更倾向它。

**第二，序列求和的对数概率引入长度偏置。** 由于每个 token 的对数概率都是负数，求和会让长回答的总分天然偏低、短回答偏高。DPO 试图通过比值消掉这一点，但 $y_w$ 与 $y_l$ 长度往往不同，残余的长度信号会被优化器利用——常见后果是 DPO 训完后回答显著变长（reward hacking 的一种）。

**第三，reference model 是纯负担。** 它占用一份额外显存、每步要多一次前向，而它存在的唯一意义是提供 KL 锚点。SimPO 想问：能不能把锚点的作用换一种更便宜的方式实现？

SimPO 的答案是：用长度归一化的平均对数概率直接当 reward，去掉 $\pi_{\text{ref}}$；再用一个固定的目标 margin $\gamma$ 来替代「锚点」的角色，要求 chosen 不只是比 rejected 高，而要高出 $\gamma$。

## 方法与公式

SimPO 定义的 reward 是长度归一化的平均对数概率：

$$
r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|}\log \pi_\theta(y|x) = \frac{\beta}{|y|}\sum_{t=1}^{|y|}\log \pi_\theta(y_t \mid x, y_{<t})
$$

其中 $|y|$ 是回答的有效 token 数。代入 Bradley-Terry 偏好模型，并加入目标 margin $\gamma>0$：

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]
$$

**长度归一化 $\frac{1}{|y|}$ 的作用**：把 reward 变成 per-token 量纲，直接对齐生成时的打分方式，并消除「序列越长总分越低」的偏置，从根上抑制长度膨胀。

**目标 margin $\gamma$ 的作用**：没有了 $\pi_{\text{ref}}$，损失只看 $\pi_\theta$ 自身在 chosen/rejected 上的差。若只要求差大于 0，模型可以靠把两者都压低、只保留微弱差距来「偷懒」。$\gamma$ 强制差距至少为 $\gamma$，把决策边界往里推，相当于给优化施加了一个最小置信间隔，是 SimPO 区分质量的关键超参。

**与生成度量对齐的论证**：作者在分析中指出，SimPO 的 reward 形式（平均 logprob）与解码时的 length-normalized log-likelihood 完全一致，因此「训练时偏好的回答」和「生成时更可能产出的回答」方向一致，减少了 DPO 那种训练-推理目标错位。

![SimPO 与 DPO 的核心差异在于 reward 形式：SimPO 用长度归一化平均 logprob 并加目标 margin γ，且去掉 reference model](/papers/simpo/x1.png)

> 图源：Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, arXiv:2405.14734（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | DPO | SimPO |
| --- | --- | --- |
| Reference model | 需要 | 不需要 |
| 隐式 reward | 序列求和 logprob 的比值 | 长度归一化平均 logprob |
| 防塌缩 / 锚点 | 隐式 KL（靠 $\pi_{\text{ref}}$） | 目标 margin $\gamma$ |
| 长度偏置 | 残留，易导致回答变长 | 显式归一化缓解 |
| 显存 / 计算 | 高（两份模型，两次前向） | 低（一份模型） |
| 跑飞风险 | 较低（有 KL 约束） | 较高（无显式约束，依赖 margin） |

## 实现要点

SimPO 在 TRL 中通过 `CPOTrainer(loss_type="simpo")` 提供，与 [CPO](/dpo/cpo) 共用一套去 reference 的代码路径。核心 loss 计算：

```python
# logps_w, logps_l: 序列求和的 logprob, 形状 [B]
# len_w, len_l:     回答有效 token 数 (不含 prompt 与 padding)
r_w = logps_w / len_w          # 长度归一化, 平均 logprob
r_l = logps_l / len_l
logits = beta * (r_w - r_l) - gamma
loss = -F.logsigmoid(logits).mean()
```

关键细节：

- **长度必须用回答的有效 token 数**：去掉 prompt 部分、去掉 padding，否则归一化失真。实践中等价于「对 response 区域、非 pad 位置的 per-token logprob 求平均」。
- **没有任何 reference 前向**：因此 SimPO 比 DPO 省掉接近一半的前向计算与一份模型显存，这是它「Simple」的直接收益。
- TRL 实现里 $\gamma$ 以 `cpo_alpha` 之外的 `simpo_gamma` 暴露；注意它是绝对量纲（per-token reward 的差），需与 $\beta$ 配合调。

## 调参与实践经验

- **$\beta$**：常见 2.0~2.5，明显高于 DPO 的 0.05~0.5。因为 reward 已被长度归一化到 per-token 量纲，数值变小，需要更大的 $\beta$ 把 logits 拉回有效区间。
- **$\gamma$**：常见 0.5~1.6。$\gamma$ 太小退化为「只要 chosen 略高即可」，区分度不足；太大则大量样本无法满足 margin，梯度饱和、训练停滞。实践中常以 $\gamma/\beta$ 的比值来感受相对强度。
- **稳定性**：没有 $\pi_{\text{ref}}$ 这条 KL 缰绳，SimPO 更容易在偏好数据分布外「跑飞」，表现为通用能力回退或重复退化。建议在质量较高、与基座分布接近的数据上用，并配合较小学习率、较少 epoch（1~2）以及在留出集上盯紧通用指标。
- **与 DPO/CPO 的取舍**：算力紧张、且已观察到 DPO 训出的回答异常变长时，SimPO 是首选；但它对超参更敏感，调参成本高于 DPO。生产中常见做法是先用 DPO 拿到稳定 baseline，再尝试 SimPO 看能否在更低开销下持平或超越。

## 参考文献

- Meng et al., 2024. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* arXiv:2405.14734
- Rafailov et al., 2023. *Direct Preference Optimization.* arXiv:2305.18290
