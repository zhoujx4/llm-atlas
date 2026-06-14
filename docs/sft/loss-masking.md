---
title: Loss Masking
---

# Loss Masking

> **一句话**：SFT 时只对 assistant 回答部分计算 loss，prompt、system 与模板 token 不参与梯度——让模型学"怎么回答"而不是"怎么复述问题"。
> 代表工作年份：Loss Masking 为 SFT 通用工程实践（自 2022 年 InstructGPT 起即为标准做法），无单一论文年份；相关探讨见 Shi et al., 2024（*Instruction Tuning With Loss Over Instructions*，arXiv:2405.14394）
>
> 前置阅读：[SFT 总览](/sft/)、[Chat Template](/sft/chat-template)

## 直觉与动机

一条 SFT 样本拼接后是 prompt（system + user + 模板标记）加 assistant 回答。如果对整条序列都算 language modeling loss，模型会同时被训练去"生成 prompt"——而 prompt 在推理时是用户给定的、模型根本不需要生成它。把梯度浪费在拟合 user 提问的分布上，不仅没用，还会：

- **稀释有效信号**：prompt 往往比回答长，loss 被 prompt token 主导，真正想学的回答部分权重被压低；
- **学到复述倾向**：模型偏向复读、改写问题，而非作答；
- **过拟合输入分布**：在小数据上尤其容易把训练集的 prompt 风格背下来。

因此标准做法是 **loss masking**：只在 assistant 内容段反传梯度，其余位置屏蔽。

也有例外。在**数据极少**（如 LIMA 量级）或希望模型同时建模输入分布的场景，有工作选择对 prompt 也算（部分）loss，论点是更充分利用每个 token、缓解过拟合。trade-off 是：prompt-loss 会把容量分给"不需要生成的内容"，在数据充足时通常是净损失。默认建议：**只对回答算 loss**；除非有明确动机，不要打开 prompt loss。

## 方法与公式

带 mask 的 SFT 损失在标准 NLL 上乘一个 0/1 掩码 $m_t$：

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid y_{<t}), \qquad m_t \in \{0, 1\}
$$

掩码构造规则：

$$
m_t = \begin{cases}
1 & y_t \ \text{属于 assistant 回答内容（含其回合结束符）} \\
0 & y_t \ \text{属于 system / user / 模板前缀}
\end{cases}
$$

要点：

- **错位对齐**：causal LM 预测的是"下一个 token"，第 $t$ 位的预测对应标签 $y_{t+1}$。HuggingFace 内部会把 logits 与 labels 各自 shift 一位，所以你只需把 labels 在 prompt 位置置 `-100`，对齐由框架处理——但要确认 mask 的边界是在"内容 token"上而非偏移一位。
- **回合结束符是否计入**：让模型学会输出 assistant 的结束符（如 `<|im_end|>` / EOS）通常是**有益的**——否则推理时它不知道何时停。常见做法是把结束符纳入 $m_t=1$。角色前缀（`<|im_start|>assistant`）一般置 0。
- **多轮对话**：一条多轮样本里有多个 assistant 段。两种策略——
  - **全轮计入**（推荐）：每一轮 assistant 都算 loss，等价于把多轮当成多个训练信号，样本利用率高；
  - **只算最后一轮**：把前面轮次当作纯上下文。实现简单、与"单轮指令"分布更接近，但浪费了中间轮的监督信号。
  默认推荐全轮计入，除非数据中早期轮次质量不可靠。

## 实现要点

HuggingFace 约定：`CrossEntropyLoss(ignore_index=-100)`，把不算 loss 的位置 label 置 `-100` 即可。

```python
IGNORE = -100

def build_labels(messages, tokenizer):
    input_ids, labels = [], []
    for msg in messages:
        # 逐段 tokenize：角色前缀 + 内容 + 结束符
        prefix = tokenizer.encode(role_prefix(msg["role"]), add_special_tokens=False)
        body   = tokenizer.encode(msg["content"],          add_special_tokens=False)
        suffix = tokenizer.encode(turn_end_token,          add_special_tokens=False)

        input_ids += prefix + body + suffix
        if msg["role"] == "assistant":
            labels += [IGNORE]*len(prefix) + body + suffix   # 内容+结束符算 loss
        else:
            labels += [IGNORE]*(len(prefix)+len(body)+len(suffix))
    return input_ids, labels
```

要点清单：

- **优先复用 chat template**：能拿到 assistant 段偏移就别手搓字符串。部分模板支持 `return_assistant_tokens_mask`，可直接得到 $m_t$；否则逐段 tokenize 并记录偏移（如上）。务必保证 token 边界与 [Chat Template](/sft/chat-template) 完全一致。
- **与 Packing 同用**：packed 序列里多条样本的 labels 直接顺序拼接，pad 段全置 `-100`；隔离 attention 与 position 重置见 [序列 Packing](/sft/packing)。
- **归一化口径**（容易踩坑）：
  - **按 token 平均**（对整个 batch 的有效 token 求和再除以总有效 token 数）——长回答样本贡献更多 token，权重更高；
  - **按样本平均**（每条样本各自归一化后再平均）——每条样本等权，短回答不被淹没。
  两者会改变长短样本的相对权重，进而影响行为。`transformers` 默认是按 token 平均；做梯度累积 / DDP 时尤其要核对分母是否为"全局有效 token 数"，否则有效学习率会随 batch 内有效 token 数波动。

## 调参与实践经验

- **mask 错位的典型症状**：① loss 异常低且训练异常快——很可能 mask 反了或大部分被置 0，模型几乎没在学；② 模型推理时复读用户问题——prompt 被算进了 loss；③ 停不下来——结束符没被纳入 $m_t=1$，模型没学会停。出问题时第一步永远是**打印一条样本的 (token, label) 对照表**肉眼核对边界。
- **prompt loss 权重**：若确有动机打开 prompt loss，可用较小权重 $\lambda$（如 0.1）折中：$m_t$ 在 prompt 位置取 $\lambda$ 而非 0，避免它主导。
- **结束符监督**：确认结束符既被算 loss、又与解码时的 `eos_token_id` / stop token 一致，三处对齐才能正确停。
- **长尾长回答**：若数据里有极长回答，按 token 平均会让它们主导梯度；可考虑按样本平均或对超长样本截断。
- **与评测对齐**：训练 mask 的边界（尤其结束符）应与线上部署的模板、停止条件一致，避免"训练时学到的停止行为推理时用不上"。

## 参考文献

- Ouyang et al., 2022. *Training language models to follow instructions with human feedback* (InstructGPT). arXiv:2203.02155.
- Zhou et al., 2023. *LIMA: Less Is More for Alignment*. arXiv:2305.11206.
- Shi et al., 2024. *Instruction Tuning With Loss Over Instructions*. arXiv:2405.14394.
