---
title: 符号约定
---

# 符号约定

> 全站公式统一使用以下记号，避免不同论文符号体系混用造成歧义。

::: warning 状态
🚧 本页为占位大纲，记号表会随正文撰写逐步补全。
:::

## 基本记号

| 记号 | 含义 |
| --- | --- |
| $x$ | 输入 prompt |
| $y$ | 模型输出 response，$y_t$ 表示第 $t$ 个 token |
| $\pi_\theta$ | 待训练的策略模型（参数 $\theta$） |
| $\pi_{\text{ref}}$ | 参考模型（通常为 SFT 模型） |
| $r_\phi(x, y)$ | 奖励模型（参数 $\phi$） |
| $(y_w, y_l)$ | 偏好对中被选中 / 被拒绝的回答 |
| $\sigma(\cdot)$ | sigmoid 函数 |
| $\mathbb{D}$ | 训练数据集 |
| $\beta$ | KL 约束 / 温度系数 |

## 示例：语言模型的自回归分解

行内公式示例：策略对完整回答的概率为 $\pi_\theta(y|x) = \prod_{t} \pi_\theta(y_t | x, y_{<t})$。

块级公式示例（SFT 的负对数似然损失）：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathbb{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

## TODO

- [ ] RL 部分记号（状态、动作、优势函数 $A_t$、GAE 参数 $\gamma, \lambda$）
- [ ] LoRA 部分记号（$W_0, B, A, r, \alpha$）
- [ ] 上标 / 下标使用规则
