---
title: 符号约定
---

# 符号约定

> **一句话**：全站公式统一使用本页定义的记号，避免不同论文符号体系混用造成歧义。遇到任何公式不确定符号含义，回到本页查表即可。

不同论文对同一概念常用不同符号（比如 KL 系数有人写 $\beta$、有人写 $\lambda$；优势函数有人写 $A$、有人写 $\hat{A}$）。本站做了统一，正文一律以下表为准。如果你在阅读原始论文时发现符号对不上，以本站记号为参照换算即可。

## 基本记号

| 记号 | 含义 |
| --- | --- |
| $x$ | 输入 prompt（问题 / 指令 / 上下文） |
| $y$ | 模型输出 response（完整回答序列） |
| $y_t$ | 回答的第 $t$ 个 token |
| $y_{<t}$ | 第 $t$ 个 token 之前的所有 token，即 $y_1, \dots, y_{t-1}$ |
| $\lvert y \rvert$ | 回答的 token 长度 |
| $\pi_\theta$ | 待训练的策略模型，参数为 $\theta$ |
| $\pi_{\text{ref}}$ | 参考模型，通常是 SFT 后冻结的模型 |
| $\pi_{\text{old}}$ | 采样时刻的旧策略（PPO/GRPO 等需要重要性采样的算法用） |
| $r_\phi(x, y)$ | 奖励模型（reward model），参数为 $\phi$ |
| $r(x, y)$ | 标量奖励（可来自 RM、规则或环境） |
| $(y_w, y_l)$ | 偏好对：$y_w$ 为被选中（win）的回答，$y_l$ 为被拒绝（lose）的回答 |
| $\sigma(\cdot)$ | sigmoid 函数，$\sigma(z) = 1/(1+e^{-z})$ |
| $\beta$ | KL 约束强度 / 偏好优化温度系数 |
| $\mathbb{D}$ | 训练数据集 |
| $\mathbb{E}_{(\cdot) \sim \mathbb{D}}[\cdot]$ | 在数据分布上取期望 |
| $\mathrm{KL}(p \,\|\, q)$ | $p$ 相对 $q$ 的 KL 散度 |

策略对完整回答的概率按自回归分解为各 token 条件概率之积：行内写作 $\pi_\theta(y \mid x) = \prod_{t=1}^{\lvert y \rvert} \pi_\theta(y_t \mid x, y_{<t})$。

## 强化学习记号

RLHF / RL 章节把语言生成建模为序列决策过程：每生成一个 token 是一个动作，已生成的前缀构成状态。

| 记号 | 含义 |
| --- | --- |
| $s_t$ | 时刻 $t$ 的状态，即 $(x, y_{<t})$（prompt 加已生成前缀） |
| $a_t$ | 时刻 $t$ 的动作，即生成的 token $y_t$ |
| $R$ | 一条轨迹（一次完整生成）的累计回报 |
| $\gamma$ | 折扣因子，$\gamma \in [0,1]$；LLM RLHF 中常取 $1$ |
| $\lambda$ | GAE 的偏差-方差权衡参数，$\lambda \in [0,1]$ |
| $V_\psi(s_t)$ | 价值函数 / Critic，参数为 $\psi$ |
| $A_t$ | 时刻 $t$ 的优势（advantage） |
| $\hat{A}_t$ | 优势的估计值 |
| $\delta_t$ | TD 残差，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| $\rho_t$ | 重要性采样比值，$\rho_t = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$ |
| $\epsilon$ | PPO 裁剪范围参数（clip $\rho_t$ 到 $[1-\epsilon, 1+\epsilon]$） |

广义优势估计（GAE）由 TD 残差按 $\gamma\lambda$ 几何加权得到：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t).
$$

PPO 的裁剪式目标（单 token 项）为：

$$
\mathcal{L}^{\text{clip}}_t(\theta) = \min\!\Big( \rho_t \hat{A}_t,\; \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, \hat{A}_t \Big).
$$

## LoRA 记号

参数高效微调章节使用以下记号描述低秩适配。

| 记号 | 含义 |
| --- | --- |
| $W_0$ | 预训练得到的原始权重矩阵，$W_0 \in \mathbb{R}^{d \times k}$，训练中冻结 |
| $\Delta W$ | 权重更新量，LoRA 将其约束为低秩 |
| $A$ | 降维矩阵，$A \in \mathbb{R}^{r \times k}$ |
| $B$ | 升维矩阵，$B \in \mathbb{R}^{d \times r}$ |
| $r$ | 低秩秩数，$r \ll \min(d, k)$ |
| $\alpha$ | 缩放系数，控制 LoRA 增量的幅度 |

LoRA 把权重更新约束为两个低秩矩阵之积，前向传播改写为：

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x,
$$

其中 $A$ 通常用高斯随机初始化、$B$ 初始化为零，保证训练起点处 $\Delta W = 0$，模型行为与原始一致。缩放因子 $\tfrac{\alpha}{r}$ 让秩 $r$ 与有效学习幅度解耦（注意 [rsLoRA](/lora/rslora) 对此缩放方式提出了修正）。

## 公式书写约定

为保证全站一致，撰写公式时遵循以下规则：

- **行内公式**用单美元符号包裹，例如 $\pi_\theta(y \mid x)$；**块级公式**用双美元符号单独成段。
- **条件分隔**统一用 `\mid`（$y_t \mid x, y_{<t}$）而非裸竖线，避免与绝对值或集合记号混淆。
- **期望**写作 $\mathbb{E}_{(x,y)\sim\mathbb{D}}[\cdot]$，下标标注采样分布。
- **估计量**加 hat，如优势估计写 $\hat{A}_t$、奖励估计写 $\hat{r}$，与真值区分。
- **时间步下标**用 $t$，token 维度与时间步在 LLM 序列决策中等价；**样本/批次下标**用 $i$。
- **参数**：策略用 $\theta$、Critic 用 $\psi$、RM 用 $\phi$，保持全站一致。
- 损失函数统一记作 $\mathcal{L}$，目标函数（需最大化）记作 $\mathcal{J}$；正文会明确说明是最小化损失还是最大化目标。

作为完整示例，SFT 的负对数似然损失（块级）写作：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{\lvert y \rvert} \log \pi_\theta\big(y_t \mid x, y_{<t}\big) \right].
$$

DPO 的偏好损失（块级，可对照上表逐项核对符号）写作：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathbb{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].
$$

掌握以上记号后，即可顺畅阅读 [DPO](/dpo/dpo)、[PPO](/rlhf/ppo)、[GRPO](/rlhf/grpo)、[LoRA](/lora/lora) 等正文页。
