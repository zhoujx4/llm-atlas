---
title: Agentic RL
---

# Agentic RL（多轮工具调用的 RL 训练）

> **一句话**：TODO —— 把 RL 从"单轮生成"扩展到"多轮交互"：模型生成 → 环境执行 → 结果反馈 → 继续生成，对整条轨迹做策略优化。
>
> 前置阅读：[RLHF 总览](/rlhf/)、[GRPO](/rlhf/grpo)、[Tool Use 训练](/agent/tool-use)

::: warning 状态
🚧 本页为占位大纲，正文尚未撰写。
:::

## 1. 直觉与动机

TODO：

- [ ] SFT 轨迹数据覆盖不了环境的组合爆炸；RL 直接以任务成功为目标
- [ ] 与单轮 RLHF 的本质区别：episode 含环境步，reward 稀疏且延迟

## 2. 问题设定

TODO：

- [ ] 轨迹定义：$\tau = (x, a_1, o_1, a_2, o_2, \dots, a_T)$，$o_t$ 为工具返回
- [ ] 环境观测 token 是否算 loss（mask 掉，类比 [Loss Masking](/sft/loss-masking)）
- [ ] reward 设计：结果验证（单测通过、答案正确）+ 过程塑形

## 3. 训练方法

TODO：

- [ ] GRPO/PPO 在多轮轨迹上的适配（advantage 广播、turn 边界）
- [ ] 异步 rollout 与环境沙箱工程
- [ ] 课程：先短链后长链

## 4. 与 baseline 对比

| 维度 | 单轮 RLHF | Agentic RL |
| --- | --- | --- |
| Episode | 一次生成 | 多轮交互 |
| Reward | RM 打分 | 任务结果验证为主 |
| 工程复杂度 | 中 | 高（环境/沙箱/异步） |

## 5. 实验与调参经验

TODO：reward hacking 在 agent 场景的表现（删测试、绕过验证）。

## 6. 参考文献

- [ ] TODO：SWE-RL、WebRL、AgentRL 等代表工作
