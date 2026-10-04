---
title: 动手学 LLM 全栈：各种 nano 项目
---

# 动手学 LLM 全栈：各种 nano 项目

> **一句话**：vLLM、verl、Megatron 这类生产框架为了适配各种情况做了大量封装，读起来门槛很高；本页按技术栈分层，列出每一层对应的「教学版最小实现」——nano、mini、tiny 系列，几百到几千行就能把同一件事讲明白。
>
> **收录标准**：约 500 星以上、且确实以「可读」为目标的项目。星数与最近提交时间为 2026-09-29 查询 GitHub API 所得，会随时间变化，页面里的相对关系比绝对数字更有参考价值。

## 怎么用这一页

学习路径推荐分三步：

1. **先跑通一条龙**，建立全局感觉——[nanochat](https://github.com/karpathy/nanochat) 或中文的 [MiniMind](https://github.com/jingyaogong/minimind)，一个仓库覆盖分词、预训练、SFT、RL、推理。
2. **再按自己的方向挖深**，每一层挑一个最小实现读透，对照本站相应章节的原理页。
3. **最后补底层**——GPU kernel 这层不急，但决定了你能不能看懂 vLLM 和 verl 里真正花时间的地方。

| 技术栈层 | 生产框架 | 最小实现 | 本站章节 |
| --- | --- | --- | --- |
| 预训练与架构 | Megatron-LM | nanoGPT · nanochat · llm.c | [模型架构](/architecture/)、[基础模型](/base-models/) |
| 分布式训练 | Megatron / DeepSpeed | picotron · nanotron | [训练系统 / 分布式](/training-systems/) |
| 后训练与 RL | verl / OpenRLHF / TRL | rlhf-book/code · TinyZero · simple_GRPO · nano-aha-moment | [后训练总览](/post-training/)、[PPO / GRPO 系列](/rlhf/) |
| 推理引擎 | vLLM / SGLang | nano-vllm · mini-sglang · tiny-llm | [推理与解码](/inference/) |
| GPU kernel | FlashAttention / Triton | GPU-Puzzles · Triton-Puzzles · flash-attention-minimal | [显存与吞吐优化](/training-systems/efficiency) |
| 架构变体 | — | makeMoE · nanoVLM | [MoE 混合专家](/architecture/moe)、[VLM 多模态结构](/architecture/vlm) |
| Agent | LangGraph / Claude Code | mini-swe-agent · tiny-universe | [Agent 总览](/agent/)、[Harness 工程](/harness/) |

## 0. 打底：Karpathy 那条线

这是唯一一条从反向传播一路铺到 ChatGPT 的完整链路，建议先走完再分叉。

| 项目 | ⭐ | 最近提交 | 学什么 |
| --- | --- | --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 63.4k | 2025-11 | GPT 训练主循环，约 600 行 |
| [nanochat](https://github.com/karpathy/nanochat) | 58.3k | 2026-09 | **一条龙**：分词 → 预训练 → SFT → RL → 推理 → Web 服务，约 8000 行 |
| [llm.c](https://github.com/karpathy/llm.c) | 31.1k | 2025-06 | 不依赖 PyTorch，纯 C/CUDA 复现训练 |
| [micrograd](https://github.com/karpathy/micrograd) | 17.7k | 2026-08 | 自动微分，百行级 |
| [minbpe](https://github.com/karpathy/minbpe) | 10.7k | 2024-07 | BPE 分词器，单文件，对照 [Chat Template](/sft/chat-template) 看 |
| [makemore](https://github.com/karpathy/makemore) | 4.3k | 2024-06 | 语言模型的起点 |

nanochat 最值得精读：8×H100 训 4 小时约 100 美元就能走完全流程。2026 年 1 月作者加了 miniseries，其中 d12 档（GPT-1 大小）约 6 分钟训完，适合反复改着做消融。

课程配套是 [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)，作业就是手写 tokenizer、Transformer、优化器、kernel、并行与对齐，作业仓库 [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)（2.9k⭐，2026-04）。

## 1. 分布式训练：对标 Megatron / DeepSpeed

对应本站 [训练系统 / 分布式](/training-systems/)：[数据并行](/training-systems/data-parallel)、[模型并行](/training-systems/model-parallel)。

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [nanotron](https://github.com/huggingface/nanotron) | 2.8k | 2026-09 | HF 的精简训练框架，picotron 的「生产版」 |
| [picotron](https://github.com/huggingface/picotron) | 2.3k | 2025-08 | 教学版 4D 并行（DP/TP/PP/CP），`train.py`、`model.py` 与各并行模块都在 300 行以内 |

理论配套读 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)——HF 的免费在线书，讲 5D 并行、ZeRO 与 CUDA kernel，图可交互。picotron 还有官方分步教程仓库 `picotron_tutorial`（262⭐，星数偏少没单列，但内容对得上）。

## 2. 后训练与 RL：对标 verl / OpenRLHF / TRL

对应本站 [后训练总览](/post-training/) 与 [PPO / GRPO 系列](/rlhf/)。verl 这类框架为了同时支持多种算法、多种并行和多种 rollout 后端，封装层数最多，也最值得先看最小实现。

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | 13.2k | 2026-02 | [DeepSeek-R1-Zero](/reasoning/rlvr) 的最小复现，3B base 在 Countdown 任务上自发涌现出验证与搜索行为 |
| [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) | 1.9k | 2025-04 | 从零实现 [GRPO](/rlhf/grpo) |
| [simple_GRPO](https://github.com/lsdefine/simple_GRPO) | 1.7k | 2025-11 | 约 200 行、2 个文件，只依赖 deepspeed + torch，不用 ray |
| [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment) | 632 | 2025-10 | 单文件 notebook + 多卡脚本，每行都可读，比 TinyZero 更干净 |

读这几个仓库时，对照 [训练循环机制](/rlhf/training-loop) 一页看 rollout 与 learning 两阶段怎么切分，会快很多。

### 想横向比较不同算法

上面几个都只实现 GRPO 一种。要把本站 [PPO / GRPO 系列](/rlhf/) 里的算法挨个对照着看，用下面两个：

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [rlhf-book/code](https://github.com/natolambert/rlhf-book/tree/main/code) | 2.4k | 2026-09 | Nathan Lambert《RLHF Book》的配套实现，**算法清单和本站后训练章节几乎一一对应** |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 10.5k | 2026-04 | 单文件实现的鼻祖（PPO/DQN/SAC 等），`ppo_atari.py` 只有 340 行。不是 LLM 场景，但想先把 PPO 本身吃透，它比任何 RLHF 仓库都干净 |

rlhf-book 的 `code/` 按主题分目录，每个目录是一个 `loss.py` 加每种算法一份 config：

| 目录 | 覆盖的算法 | 对应本站 |
| --- | --- | --- |
| `policy_gradients/` | REINFORCE、[RLOO](/rlhf/rloo)、[PPO](/rlhf/ppo)、[GRPO](/rlhf/grpo)、Dr.GRPO、[GSPO](/rlhf/gspo)、[CISPO](/rlhf/cispo)、[DAPO](/rlhf/dapo) | [PPO / GRPO 系列](/rlhf/) |
| `direct_alignment/` | [DPO](/dpo/dpo)、[IPO](/dpo/ipo)、[KTO](/dpo/kto)、[ORPO](/dpo/orpo)、[SimPO](/dpo/simpo) | [DPO 家族](/dpo/) |
| `reward_models/` | 偏好 RM、ORM、PRM | [Reward Model](/rlhf/reward-model)、[过程奖励 vs 结果奖励](/reasoning/reward-models) |
| `rejection_sampling/` | 拒绝采样微调 | [数据构造](/sft/data-construction) |
| `instruction_tuning/` · `distillation/` | SFT、蒸馏 | [SFT](/sft/)、[蒸馏](/distillation/) |

同一套 rollout 和训练循环下只换 `loss.py` 里的一个函数，这正是看懂「这些算法到底差在哪一行」最省力的方式——和本站 [后训练总览](/post-training/) 里「统一梯度视角」那一节是同一个思路。

至于 DAPO、GSPO、CISPO 的生产级实现，看 [verl](https://github.com/verl-project/verl)（23.7k⭐）的 `recipe/` 目录，各算法一个配方，比读主干代码容易。

**一个空白**：

- **多轮 [Agentic RL](/agent/agentic-rl/)** 目前没有 nano 版，开源方案（VerlTool、SkyRL 等）基本都是在 verl 上加东西。比较现实的路径是拿 nano-aha-moment 自己改成多轮。

## 3. 推理引擎：对标 vLLM / SGLang

对应本站 [推理与解码](/inference/)：[KV Cache 与 PagedAttention](/inference/kv-cache)、[推理框架与服务引擎](/inference/frameworks)。

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [llama2.c](https://github.com/karpathy/llama2.c) | 20.1k | 2024-08 | 纯 C 推理，最朴素的解码循环 |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 15.7k | 2026-04 | **约 1200 行 Python**，速度接近 vLLM，带前缀缓存、张量并行、torch.compile 与 CUDA graph |
| [mini-sglang](https://github.com/sgl-project/mini-sglang) | 5.2k | 2026-05 | **SGLang 官方出的教学版**，约 5000 行 Python，保留 Radix Cache、chunked prefill、overlap scheduling、张量并行和 OpenAI 兼容 API，开箱跑 Llama-3 / Qwen-3。配套有 [LMSYS 博客](https://www.lmsys.org/blog/2025-12-17-minisgl/) |
| [tiny-llm](https://github.com/skyzh/tiny-llm) | 4.7k | 2026-09 | 三周课程：先用 mlx 数组手搓 Qwen3，再加 KV cache 与 Metal kernel，最后做连续批处理与 paged KV。[在线书](https://skyzh.github.io/tiny-llm/) |

三者分工很清楚：**nano-vllm**（1200 行）最小，看懂 PagedAttention 和连续批处理的骨架；**mini-sglang**（5000 行）多出 Radix Cache 这条 SGLang 独有的前缀树缓存，以及调度器和服务层，是「完整引擎长什么样」的参照；**tiny-llm** 则是从矩阵乘开始自己搭一遍的课程。

**空白**：[投机解码](/inference/speculative-decoding) 的最小实现都只有几十到一百多星，没有列入；这块建议直接读 vLLM 的 spec decode 模块。

## 4. GPU kernel：对标 FlashAttention / Triton

对应本站 [显存与吞吐优化](/training-systems/efficiency) 与 [注意力变体](/architecture/attention)。

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | 12.5k | 2024-09 | 做题学 CUDA，交互式 |
| [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | 2.6k | 2026-04 | 同作者，从零学 Triton，一路做到 Flash Attention 与量化 kernel |
| [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) | 1.2k | 2024-12 | 约 100 行 CUDA（仅前向），看懂 tiling 与 online softmax |

## 5. 架构变体与多模态

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [nanoVLM](https://github.com/huggingface/nanoVLM) | 5.0k | 2025-10 | 约 750 行纯 PyTorch，SigLIP + SmolLM2 拼成 222M 的 VLM，单卡 H100 训 6 小时；对照 [VLM 多模态结构](/architecture/vlm) |
| [makeMoE](https://github.com/AviSoori1x/makeMoE) | 818 | 2024-10 | 在 makemore 上换成稀疏 MoE，含 top-k 与 noisy top-k 门控；对照 [MoE 混合专家](/architecture/moe) |

## 6. Agent 与 RAG

对应本站 [Agent 总览](/agent/) 与 [Harness 工程](/harness/)。

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 8.1k | 2026-09 | **100 行**的 agent 类，只有 bash 一个工具，连 tool calling 接口都不用，SWE-bench Verified 过 74%；理解 [执行循环](/harness/agent-loop) 的最佳材料 |
| [tiny-universe](https://github.com/datawhalechina/tiny-universe) | 5.1k | 2026-02 | Datawhale 出品，全手搓 TinyRAG、TinyAgent、TinyGraphRAG、TinyDiffusion |

## 7. 量化

这一层缺好的教学版。[llm-awq](https://github.com/mit-han-lab/llm-awq)（3.6k⭐，2025-07）本身代码量不大、可读性尚可，建议直接读它加论文，而不是去读支持一堆硬件后端的工具包。原理见 [量化（GPTQ/AWQ/FP8）](/inference/quantization)。

## 中文整套项目

| 项目 | ⭐ | 最近提交 | 说明 |
| --- | --- | --- | --- |
| [MiniMind](https://github.com/jingyaogong/minimind) | 62.9k | 2026-09 | 2 小时从零训一个 64M 模型，约 3 美元；覆盖 MoE、数据清洗、预训练、[SFT](/sft/)、[LoRA](/lora/)、[DPO](/dpo/dpo)，全部 PyTorch 手写 |
| [happy-llm](https://github.com/datawhalechina/happy-llm) | 34.1k | 2026-08 | Datawhale 的从零构建大模型教程，从 Transformer 一路讲到 Agentic RL |

## 一句话总结各层该读谁

- 想建立全局感觉：**nanochat**（或 MiniMind）
- 想做训练基建：**picotron** + Ultra-Scale Playbook
- 想做推理引擎：**nano-vllm** + tiny-llm
- 想做 RL：**simple_GRPO** → nano-aha-moment
- 想往下钻到 kernel：**GPU-Puzzles** → Triton-Puzzles → flash-attention-minimal
