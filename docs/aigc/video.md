---
title: 视频与多模态生成
---

# 视频与多模态生成

> **一句话**：视频生成是把图像扩散从「二维空间」推广到「时空联合」的过程，核心难点是时间一致性与算力，主流答案是把视频切成时空 patch、用 DiT 在隐空间里去噪。
> 关键年份：Video Diffusion Models（Ho et al. 2022, arXiv:2204.03458）、Make-A-Video（2022, arXiv:2209.14792）、Stable Video Diffusion（Blattmann et al. 2023, arXiv:2311.15127）、Sora（OpenAI Technical Report, 2024）。
> 前置阅读：[架构演进：U-Net→DiT 与 Flow Matching](/aigc/dit-flow)、[Latent Diffusion 与 Stable Diffusion](/aigc/latent-diffusion)、[扩散模型基础](/aigc/diffusion-basics)

视频生成是 2024 年以来生成式模型最受关注的方向。它在数学上与图像扩散同源——仍然是「学习数据分布、从噪声里逐步采样」——但工程上几乎是另一个量级的问题。本页梳理从图像扩到视频的难点、主流技术路线（视频扩散、时空建模、Sora 式 DiT）、以及当前的产品代表，并一句带过音频生成。

## 一、从图像扩到视频，难在哪里

把一张图变成一段视频，本质是在原有的空间维度（高 × 宽）之外，再加一个时间维度（帧）。多出来的这一维带来两类根本困难。

**时间一致性（temporal consistency）**。逐帧独立生成会出现闪烁、物体形变、身份漂移：同一个人前后两帧脸不一样，背景纹理在抖动。模型必须显式建模帧与帧之间的关联，让运动连贯、内容稳定。这要求在网络里引入**时间维度上的信息交互**（时间注意力 / 3D 卷积 / 时空注意力），而不只是把图像模型套在每帧上。

**算力与序列长度**。一段 5 秒、24fps、720p 的视频，像素量是单张图的成百上千倍。若直接在像素空间建模，显存与计算都不可承受。因此视频生成几乎都建立在两个前提上：① 在**隐空间**里建模（沿用 [Latent Diffusion](/aigc/latent-diffusion) 的思路，用一个时空压缩的 VAE 把视频压成低维 latent）；② 用**时空因子化**的注意力（把全 3D 注意力拆成「空间注意力 + 时间注意力」）来降低复杂度——这正是 Ho et al. 2022 与后续工作的关键工程选择。

此外还有数据问题：高质量、有良好文本标注的视频数据远比图像稀缺，训练范式往往要先用海量图像预训练、再注入视频学习运动。

## 二、关键技术路线

### 视频扩散与时空建模

**Video Diffusion Models（Ho et al. 2022）** 是把 DDPM 直接推广到视频的奠基工作。它用 **3D U-Net** 处理 `帧 × 高 × 宽 × 通道` 的四维张量，并采用**时空因子化**的注意力以兼顾效率；同时支持图像与视频联合训练，并提出条件采样技术做时间与空间上的视频外扩。

**Make-A-Video（2022）** 走的是另一条路：复用强大的文本到图像（T2I）模型，从配对的图文数据里学「世界长什么样」，再从无标注视频里学「世界怎么动」，因此**不需要配对的文本-视频数据**即可做文本到视频。这一「图像先验 + 运动学习」的范式后来被广泛沿用。

**Stable Video Diffusion（Blattmann et al. 2023）** 把上述思路系统化为「**文本到图像预训练 → 视频预训练 → 高质量视频微调**」三阶段流程，强调数据策展（data curation）对视频 LDM 的决定性作用，并开源了图像到视频（I2V）能力，是开源社区的重要基线。

### Sora：spacetime patches + DiT

OpenAI 的 **Sora（2024 技术报告 *Video generation models as world simulators*）** 把视频生成推进到「可用」级别，其设计要点可定性概括为：

- **统一的时空 patch 表示**：先用一个视频压缩网络把视频压到低维隐空间，再把隐表示分解成一串 **spacetime latent patches**（时空 patch）。这相当于视频版的「token」，把不同分辨率、不同时长的视频统一成变长的 patch 序列。
- **DiT 主干**：在 patch 序列上跑 [Diffusion Transformer](/aigc/dit-flow)，以文本等条件信息预测「干净」patch（去噪）。Transformer 对序列长度天然灵活，因此能在数据与算力上良好扩展。
- **可变分辨率 / 时长 / 宽高比**：因为输入被统一成 patch 序列且去噪器不限制输入尺寸，Sora 可以联合训练并生成不同分辨率、宽高比与时长的视频。
- **训练增强**：沿用 DALL·E 3 的 recaptioning（为训练视频生成高描述性字幕），提升文本跟随能力。

> 注意：Sora 报告以定性描述与展示为主，具体参数与训练细节未完全公开。以下涉及具体能力的说法以官方为准，本页不臆造数字。

```mermaid
flowchart LR
  A[输入视频 / 噪声] --> B[时空压缩 VAE<br/>压到隐空间]
  B --> C[切成 spacetime patches<br/>变长 patch 序列]
  C --> D[DiT 去噪<br/>文本/条件注入]
  D --> E[去噪后的隐 patch]
  E --> F[VAE 解码<br/>还原像素视频]
```

时空 patch + DiT 这条路线的意义在于：它把图像 DiT 的可扩展性直接搬到了视频，使「加数据、加算力 → 质量提升」成为可预期的工程曲线，也因此成为 2024 年后视频模型的主流架构。

## 三、产品代表（以官方为准）

视频生成已进入产品竞争阶段，下表做定性对比，**具体能力、分辨率、时长以各家官方发布为准**：

| 产品 / 模型 | 团队 | 定位（定性） |
| --- | --- | --- |
| Sora | OpenAI | 长时长、高保真文本到视频，spacetime patch + DiT |
| Veo | Google DeepMind | 高质量文本/图像到视频，强调画质与可控性 |
| Kling / 可灵 | 快手 | 中文生态主流文本/图像到视频产品 |
| Wan / 通义万相 | 阿里 | 2025 年开源的视频生成模型，含 T2V / I2V，提供多档参数规模 |
| Stable Video Diffusion | Stability AI | 开源 I2V 基线（2023） |

**Wan（通义万相，阿里）** 于 2025 年开源，覆盖文本到视频与图像到视频等任务、提供不同参数规模的版本，是开源视频生成的代表性工作之一（具体版本与指标以官方为准）。开源模型的价值在于：研究者可在其上做 [LoRA](/lora/lora) 微调、ControlNet 式条件控制（见 [条件控制与定制](/aigc/control)）以及 [采样加速与蒸馏](/aigc/acceleration)。

闭源产品（Sora / Veo / Kling 等）则在画质、运镜、物理一致性与可控性上持续迭代——但具体参数普遍不公开，引用时应避免编造数字。

## 四、音频与多模态生成（一句带过）

扩散模型同样适用于音频：**扩散式 TTS**（如把声学特征生成建模为扩散过程）与**音乐 / 通用音频生成**已成为主流方案之一；视频生成也正与音频联合，走向「带声画的多模态生成」。更广义的多模态理解与生成统一，可参考 [全模态模型 Omni](/architecture/omni) 与 [StepFun 阶跃星辰](/base-models/stepfun) 等工作。

## 小结

- 视频 = 图像扩散 + 时间维度，难点是时间一致性与算力，标准解法是**隐空间建模 + 时空因子化注意力**。
- 技术路线从 **3D U-Net 视频扩散**（Ho 2022）→ **图像先验 + 运动学习**（Make-A-Video）→ **三阶段 LDM**（SVD）→ **spacetime patch + DiT**（Sora）。
- DiT 路线让视频生成具备可扩展性，是当前主流；产品端 Sora / Veo / Kling / Wan 各有侧重，细节以官方为准。
- 音频生成同样被扩散统一，多模态联合生成是下一步方向。

## 参考文献

- Ho et al. *Video Diffusion Models*. 2022. [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)
- Singer et al. *Make-A-Video: Text-to-Video Generation without Text-Video Data*. 2022. [arXiv:2209.14792](https://arxiv.org/abs/2209.14792)
- Blattmann et al. *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets*. 2023. [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
- OpenAI. *Video generation models as world simulators (Sora Technical Report)*. 2024. <https://openai.com/index/video-generation-models-as-world-simulators/>
- Alibaba. *Wan / 通义万相 开源视频生成模型*. 2025.（以官方发布为准）
