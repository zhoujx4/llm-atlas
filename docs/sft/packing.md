---
title: 序列 Packing
---

# 序列 Packing

> **一句话**：序列 Packing 把多条长短不一的样本拼进同一条定长训练序列以填满 context，省掉 padding 的浪费，在短样本场景下能显著提升训练吞吐。（Krell et al., 2021. *Efficient Sequence Packing without Cross-contamination*）
> 代表工作年份：2021（Krell et al.，无跨样本污染的高效 Packing）· 机构/团队：Graphcore · 会议/来源：arXiv:2107.02027
>
> 前置阅读：[SFT 总览](/sft/)、[Chat Template](/sft/chat-template)

## 直觉与动机

SFT 数据的长度分布通常长尾且方差大：有的样本几十个 token，有的几千个。训练时 batch 内要对齐到统一长度，常规做法是把每条样本 **pad** 到 batch 最长（或固定 `max_len`）。pad token 不携带信息、要被 mask 掉、却照样消耗算力。

衡量浪费的指标是**有效 token 占比**：

$$
\eta = \frac{\sum_i L_i}{B \cdot L_{\max}}
$$

其中 $L_i$ 是第 $i$ 条样本的真实长度，$B$ 是 batch size，$L_{\max}$ 是对齐长度。当长度分布很散时，$\eta$ 可能低到 50% 甚至更差——意味着接近一半的前向/反向算力花在 pad 上。

Packing 的思路是：把多条短样本**首尾相接**拼成一条接近 $L_{\max}$ 的序列，让 $\eta$ 逼近 1。由于 Transformer 的计算量随序列长度近似平方增长，但在固定总 token 预算下，填满 context 比"装一半 pad"能在同样的 step 内吃进更多真实 token，从而把每个 epoch 的 wall-clock 时间显著压低。收益大小取决于原始长度分布——分布越散、平均长度越短于 `max_len`，收益越大。

## 方法与公式

### 装箱：怎么拼

把"长度 $L_i$ 的样本装进容量 $L_{\max}$ 的箱子"本质是 **bin packing** 问题。常见两档：

- **朴素流式拼接**：顺序遍历样本，往当前序列里塞，塞不下就开新序列。实现极简，但若数据没打乱、长度有序，碎片会较多。
- **first-fit / first-fit-decreasing binning**：先（按长度降序）排序，再用 first-fit 策略选箱，碎片更少、$\eta$ 更高。对静态数据集可离线预计算 packing 方案。

### Cross-contamination：必须隔离 attention

把多条样本拼成一条后，如果直接走标准 causal attention，序列后段的 token 会 attend 到前段**属于别的样本**的 token——这就是 cross-contamination（样本间泄漏）。它会让模型学到不该有的跨样本依赖，污染训练信号。

解决办法是 **block-diagonal attention mask**：只允许同一原始样本内部的 token 互相注意，跨样本一律屏蔽。设拼接序列被切成若干段 $s_1, \dots, s_k$，注意力可见性为：

$$
\text{mask}(i, j) = \begin{cases} 1 & \text{seg}(i) = \text{seg}(j) \ \text{且}\ j \le i \\ 0 & \text{otherwise} \end{cases}
$$

同时 **position_ids 要在每段开头重置为 0**，否则后段样本会拿到偏大的位置编码（尤其 RoPE 会因此偏移），等价于把它当成"一条超长序列的后半段"，与训练分布不符。

### FlashAttention varlen 接口

显式构造 $L \times L$ 的 block-diagonal mask 内存开销是平方级，不可取。工程上用 FlashAttention 的 **变长（varlen）接口**：传入累积序列长度 `cu_seqlens`（cumulative sequence lengths），kernel 内部就只在每段内部做 attention，既隔离了样本又零额外内存、且保持 FlashAttention 的速度。这是当前隔离 Packing 的主流实现路径。

$$
\texttt{cu\_seqlens} = [0,\ L_1,\ L_1{+}L_2,\ \dots,\ \textstyle\sum_i L_i]
$$

## 与 baseline 对比

| 维度 | Padding | 朴素 Packing | 隔离 Packing（varlen） |
| --- | --- | --- | --- |
| 有效 token 占比 $\eta$ | 低（可能 50% 量级） | 高（接近 1） | 高（接近 1） |
| 训练吞吐 | 低 | 高 | 高 |
| 样本间泄漏 | 无 | **有** | 无 |
| position_ids 正确性 | 正确 | 需重置 | 需重置 |
| 实现复杂度 | 低 | 低 | 中（需 cu_seqlens / mask） |
| 推荐度 | 短样本场景偏低 | 不推荐（有泄漏） | **推荐** |

## 实现要点

核心是离线/在线构造 packed batch，并把 `cu_seqlens` 与 position_ids 一并传给模型。

```python
def first_fit_pack(samples, max_len):
    # samples: List[List[int]] 已 tokenize 的样本（含各自模板/EOS）
    bins = []                       # 每个 bin 是 List[sample]
    for s in sorted(samples, key=len, reverse=True):
        if len(s) > max_len:        # 超长样本单独成箱或截断
            bins.append([s[:max_len]]); continue
        for b in bins:
            if sum(len(x) for x in b) + len(s) <= max_len:
                b.append(s); break
        else:
            bins.append([s])
    return bins

def build_packed(bin_samples, max_len, pad_id):
    input_ids, labels, pos, seqlens = [], [], [], [0]
    for s in bin_samples:
        input_ids += s
        labels    += loss_mask_labels(s)   # 见 /sft/loss-masking
        pos       += list(range(len(s)))   # 每段 position 从 0 重置
        seqlens.append(len(input_ids))
    # 尾部 pad 到 max_len（pad 段的 label 全 -100）
    pad = max_len - len(input_ids)
    input_ids += [pad_id] * pad
    labels    += [-100]   * pad
    pos       += [0]      * pad
    cu_seqlens = seqlens + ([len(input_ids)] if pad else [])
    return dict(input_ids=input_ids, labels=labels,
                position_ids=pos, cu_seqlens=cu_seqlens)
```

要点清单：

- **每条样本自带完整模板与 EOS**，拼接后段间天然有结束符，不要因为 packing 省掉它。
- **position_ids 每段从 0 重置**，对 RoPE 模型尤其关键。
- **labels 与 input_ids 同步拼接**，pad 段全置 `-100`；与 [Loss Masking](/sft/loss-masking) 协同。
- **cu_seqlens 传给 varlen attention**（FlashAttention `flash_attn_varlen_func` 或框架封装的 `_get_unpad_data` 路径），不要手搓 $L\times L$ mask。
- **loss 归一化口径**要想清楚：packed batch 把多条样本混在一起，按 token 平均 vs 按样本平均会改变长短样本的相对权重，详见 [Loss Masking](/sft/loss-masking) 的归一化讨论。

## 调参与实践经验

- **是否影响效果**：在正确做了 attention 隔离与 position 重置的前提下，Packing 对最终效果的影响通常可忽略——它只改吞吐，不改训练信号。**朴素 Packing（不隔离）则可能掉点**，因为引入了跨样本泄漏。
- **什么时候必须隔离**：只要一条序列里装了多于一条独立样本，就应隔离。唯一例外是"本来就是同一篇长文档/同一段连续上下文"的拼接（continued pretraining 场景），那种情况跨段注意力是合理的。
- **收益评估**：先统计数据集的 $\eta$。若平均长度接近 `max_len`，padding 浪费本就不大，Packing 收益有限，没必要为它增加实现复杂度；若 $\eta$ 偏低（短样本为主），Packing 收益明显。
- **超长样本**：超过 `max_len` 的样本要么截断、要么单独成箱，避免破坏装箱逻辑。
- **梯度等价性**：Packing 后单条序列里有效样本数不固定，配合梯度累积时要注意分母口径，确保与逐样本训练在期望上一致，否则学习率的有效尺度会漂移。
- **吞吐验证**：上线 Packing 后实测 tokens/sec 与显存占用，确认 varlen kernel 真的生效（没退化回稠密 mask）。

## 参考文献

- Krell et al., 2021. *Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance*. arXiv:2107.02027.
- Dao et al., 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135.
- Dao, 2023. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691.
