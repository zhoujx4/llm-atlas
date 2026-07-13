---
title: 全景速览
---

# LLM Atlas <Badge type="tip" text="全景速览" />

**LLM 训练算法知识图谱** —— 这张图是整站的地图：一眼看清各章节怎么串起来、该往哪个方向深入。具体算法的推导、公式、伪代码和调参经验都在各章节页里，点进侧边栏对应条目即可；本站只收**讨论度高、用得最多的出名算法**。

```mermaid
flowchart LR
    BM[基础模型] -.架构.-> ARCH[模型架构]
    BM -.另一条赛道·生成式.-> AIGC[AIGC / 扩散]
    BM --> SFT[SFT]
    SFT -.PEFT.-> LORA[LoRA 及变体]
    SFT --> PO[DPO 系列]
    SFT --> RL[PPO/GRPO 系列]
    SFT --> DST[蒸馏]
    PO --> A[对齐模型]
    RL --> A
    RL -.可验证奖励.-> RSN[推理模型]
    A --> INF[推理与解码]
    A -.度量.-> EVAL[评测]
    A --> AG[Agent]
    AG -.脚手架.-> HN[Harness]
    AG -.能力扩展.-> SK[Skills]
    HN -.自我改进.-> RSI[RSI]
```

[如何阅读本知识库 →](/guide/) · [符号约定 →](/guide/notation)
