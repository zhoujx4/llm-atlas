---
title: KV Cache
---

# KV Cache（键值缓存）

> **一句话**：KV cache 用显存换计算——缓存历史 token 的 Key/Value，使每步 decode 免于重算整个前缀；但它随 batch × 序列长度线性膨胀，成为推理吞吐的头号瓶颈，由此催生架构压缩（GQA/MLA）、分页管理（PagedAttention）、跨请求复用（prefix caching）三条优化路线。
>
> 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023)、*DeepSeek-V2* (2024) ·
> 关键年份：MQA 2019（Shazeer）· GQA 2023（Google，arXiv:2305.13245）· PagedAttention 2023（UC Berkeley，SOSP 2023）· MLA 2024（DeepSeek-V2，arXiv:2405.04434）· RadixAttention 2023（SGLang，arXiv:2312.07104）
> 前置阅读：[推理优化总览](/inference/)

## 1. 直觉与动机

Causal attention 中，第 $t$ 步的输出要对前缀所有 token 的 $K, V$ 做注意力。而每个 token 的 $k_i = W_K h_i$、$v_i = W_V h_i$ 只依赖该 token 自身的 hidden state，在整个生成过程中不变——算一次、缓存、之后只读不写。若不缓存，每步 decode 都要对整个前缀重做投影与前向，生成 $n$ 个 token 的总计算量多出一个数量级；缓存后每步只需计算当前 token 的 Q/K/V，再与缓存做一次注意力。

代价是显存。标准（MHA）模型的 KV cache 大小：

$$
\text{每 token 字节数} = \underbrace{2}_{K \text{和} V} \times L \times \underbrace{n_h d_h}_{=\,\text{hidden\_size}} \times p,
\qquad
\text{总显存} = b \times s \times \text{每 token 字节数}
$$

其中 $L$ 为层数、$n_h$ 头数、$d_h$ 每头维度、$p$ 为精度字节数（FP16 为 2）、$b$ 为 batch size、$s$ 为序列长度。两个例子：

- **Llama 2 7B**（$L=32$，hidden 4096，FP16）：每 token $2 \times 32 \times 4096 \times 2 = 512\text{KB}$，一条 4096 token 序列约 **2GB**——7 条这样的并发序列就抵上整份 FP16 权重（约 14GB）；
- **OPT-13B**（$L=40$，hidden 5120，FP16）：每 token 约 **800KB**（vLLM 论文中的示例）。

两个直接后果：decode 的 batch 上限被 KV cache 显存卡死（吞吐上不去）；每步 decode 都要把该序列的全部 cache 从 HBM 读一遍（延迟随上下文线性增长）。

## 2. 方法与公式

### 2.1 基本机制：prefill 写入，decode 追加

Prefill 阶段对 prompt $x$ 的所有 token 并行计算 K/V 并整批写入 cache；decode 阶段每步只计算 $y_t$ 一个 token 的 K/V 并追加。两阶段计算特性迥异（compute-bound vs memory-bound），现代引擎用不同 kernel 分别处理。

### 2.2 路线一：架构压缩——少存几个头（MQA / GQA / MLA）

每 token 每层的缓存元素数由注意力变体决定：MHA 为 $2 n_h d_h$；MQA（所有 query 头共享一组 K/V）为 $2 d_h$；GQA（$n_g$ 组 KV 头，每组被 $n_h / n_g$ 个 query 头共享）为 $2 n_g d_h$，介于两者之间。MQA/GQA 用质量换显存，且必须在预训练阶段就确定（GQA 论文给出了从 MHA checkpoint uptraining 转换的方法）。

**MLA**（Multi-head Latent Attention，DeepSeek-V2，2024）走得更远：对 K、V 做低秩联合压缩，只缓存一个共享的低维 latent 向量：

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad
k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}, \qquad d_c \ll n_h d_h
$$

推理时上投影矩阵 $W^{UK}$、$W^{UV}$ 可被吸收进 query/输出投影，无需显式重建 K/V。一个细节：标准 RoPE 对 key 的逐位置旋转与这一矩阵吸收技巧不兼容，因此 MLA 采用**解耦 RoPE**——单独用一小部分维度（$d_R$）携带旋转位置信息，与 latent 部分拼接缓存。DeepSeek-V2 的配置：$d_c = 512 = 4 d_h$、$d_R = 64$，每 token 每层只缓存 $d_c + d_R = 576$ 个元素，而其 MHA 等价配置（128 头 × $d_h$=128）需要 $2 \times 128 \times 128 = 32768$ 个——压缩到约 1/57，论文称其 KV cache 体积仅相当于 2.25 组的 GQA。整机效果：DeepSeek-V2 相比 DeepSeek 67B，KV cache 减少 93.3%，最大生成吞吐提升至 5.76 倍；且论文报告 MLA 性能**优于** MHA，并非以质量换显存。

### 2.3 路线二：系统管理——PagedAttention 分页

压缩之外的另一半问题是**怎么放**。早期系统（FasterTransformer、Orca）按请求的最大可能长度连续预分配 KV 显存，内部碎片（预留了没用到）、外部碎片（空隙放不下新请求）叠加，导致 60%–80% 的 KV 显存被浪费（实际仅 20.4%–38.2% 存了真实 token 状态）。

PagedAttention（vLLM, SOSP 2023）把操作系统的虚拟内存分页搬进来：将每个序列的 cache 切成固定 token 数的块（block，默认 16 个 token），用 **block table** 把逻辑块映射到任意位置的物理块——物理显存无需连续、按需分配，浪费降到 **4% 以下**（仅最后一个未填满的块）。block table 还天然支持**共享**：parallel sampling、beam search 中多条候选共享同一 prompt 的物理块（引用计数 + copy-on-write），显存最多省 55%，对应吞吐最多提升 2.2 倍。端到端，vLLM 在同等延迟下吞吐为 FasterTransformer/Orca 的 2–4 倍，序列越长、模型越大、解码算法越复杂提升越明显。

![PagedAttention 的 block table：逻辑 KV 块映射到非连续的物理块](/papers/kv-cache/paged-attention-block-table.png)

> 图源：Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv:2309.06180（用于学习注解，版权归原作者）

### 2.4 路线三：跨请求复用——prefix caching

不同请求经常共享相同前缀（系统提示、few-shot 样例、多轮对话历史、同一篇长文档）。**Prefix caching** 把历史请求的 KV cache 留在显存里，新请求命中相同前缀时直接复用、跳过这部分 prefill：

- **vLLM APC**（Automatic Prefix Caching）：`enable_prefix_caching=True` 启用，按 block 哈希匹配；
- **SGLang RadixAttention**（Zheng et al., 2023）：用基数树（radix tree）在运行时自动管理跨调用的前缀复用，论文报告在 agent、few-shot 推理、RAG 等任务上比 SOTA 系统吞吐最高提升 6.4 倍。

关键限制：prefix caching 只加速 **prefill**（降低 TTFT），不加速 decode——输出很长的场景收益有限。

## 3. 与 baseline 对比

注意力变体（每 token 每层缓存元素数）：

| 变体 | 缓存元素数 | 相对 MHA | 质量 | 代表模型 |
| --- | --- | --- | --- | --- |
| MHA | $2 n_h d_h$ | 1× | 基线 | Llama 2 7B/13B |
| MQA | $2 d_h$ | $1/n_h$ | 有损 | PaLM |
| GQA | $2 n_g d_h$ | $n_g / n_h$ | 轻微有损 | Llama 2 70B、Llama 3、Qwen2 |
| MLA | $d_c + d_R$ | DeepSeek-V2 中约 1/57 | 论文报告优于 MHA | [DeepSeek](/base-models/deepseek)-V2/V3 |

显存管理方式：

| 维度 | 连续预分配（FasterTransformer/Orca） | PagedAttention（vLLM） |
| --- | --- | --- |
| 分配粒度 | 按 max_len 整段预留 | 16-token 物理块按需分配 |
| 显存浪费 | 60%–80% | < 4% |
| 前缀共享 | 不支持 | block 级共享 + copy-on-write |
| 同等延迟吞吐 | 1× | 2–4× |

## 4. 实现要点

```python
# decode 一步（带 KV cache，示意）
def decode_step(h_t, kv_cache):
    for l, layer in enumerate(layers):
        q = layer.W_q(h_t)                       # 只算当前 token
        kv_cache[l].append(layer.W_k(h_t), layer.W_v(h_t))
        h_t = layer.attn(q, kv_cache[l].K, kv_cache[l].V)  # 读全部历史
        h_t = layer.mlp(h_t)
    return lm_head(h_t)
```

- **正交性**：FlashAttention 省的是 $N \times N$ 注意力分数矩阵的物化与 HBM 读写，**不减少 KV cache 本身**——K/V 仍需完整保存。FlashAttention、PagedAttention、GQA/MLA、KV 量化四者正交，可全部叠加。
- **KV cache 量化**：对 cache 用 FP8/INT8 存储可再省一半显存并减少 decode 读带宽，详见[量化](/inference/quantization)。
- **框架对应**：HuggingFace transformers 的 `DynamicCache`/`StaticCache`（`generate` 默认 `use_cache=True`）；vLLM/SGLang 自动管理 paged cache，用户无需干预。
- **MLA 的工程含义**：缓存的是 latent 向量而非 K/V 本身，注意力 kernel 需要配套改写（吸收上投影），不能直接套用为 MHA 写的 kernel。

## 5. 调参与实践经验

- **并发容量估算**：可并发序列数 ≈ (总显存 − 权重 − 激活预留) / (每 token KV 字节数 × 平均序列长)。这是部署前必须做的算术，决定了实例数和 batch 上限。
- **vLLM 关键参数**：`gpu_memory_utilization`（权重 + KV 池占显存比例，默认 0.9，越大 KV 池越大但越容易 OOM）；`block_size` 默认 16（论文消融显示 16–32 在多数负载上最优，过小增加 block table 开销、过大恢复内部碎片）；`max_num_seqs` / `max_num_batched_tokens` 控制调度上限。
- **何时开 prefix caching**：长系统提示、长文档多次问答、多轮对话、共享 few-shot 模板的批量评测——这些场景下 TTFT 改善显著；纯短 prompt 长输出场景收益接近零。
- **显存不足时的抢占**：vLLM 在 KV 池耗尽时会抢占低优先级序列（重计算或 swap 到 CPU），表现为长尾延迟突增；监控 preemption 指标，必要时调低 `max_num_seqs`。
- **长上下文选型**：上下文 32K 以上时 KV cache 往往超过权重本身，优先选 GQA/MLA 架构的模型；注意力变体是预训练决定的，推理侧无法更改。

## 6. 参考文献

- Kwon et al., 2023. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180（SOSP 2023）
- DeepSeek-AI, 2024. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434
- Zheng et al., 2023. *SGLang: Efficient Execution of Structured Language Model Programs.* arXiv:2312.07104
- Yu et al., 2022. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022
- Shazeer, 2019. *Fast Transformer Decoding: One Write-Head is All You Need.* arXiv:1911.02150（MQA）
- Ainslie et al., 2023. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245
- Dao et al., 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135
