---
title: 代表系统对比
---

# 代表系统对比（Representative Agent Harnesses）

> **一句话**：把 [Harness 总览](/harness/) 的三件套（工具集 / 上下文 / 执行环境）落到四个真实系统——SWE-agent 证明「接口即性能」、OpenHands 给出可复现的开放平台、Claude Code 押注「单循环 + 极简」、GitHub Copilot 把 agent 包进 CI 流水线——看清同一组设计选择在不同约束下如何分叉。
>
> 论文 / 文档：*SWE-agent* (NeurIPS 2024) · *OpenHands* (2024) · Claude Code docs · GitHub Copilot docs
> 前置阅读：[Harness 总览](/harness/)、[执行循环与上下文管理](/harness/agent-loop)、[沙箱与工具执行](/harness/sandbox)
>
> 本页只从 **harness 设计范式**角度横向对比；各产品/框架的定位、安装与日常用法见 [代表性 Agent 框架](/agent/frameworks/)。

## 1. 为什么对比这几个

[Harness 总览](/harness/) 给出的判别标准——agent 是「LLM 在循环中基于环境反馈使用工具」——在这四个系统里有四种代表性答法，恰好覆盖了设计空间的几个关键维度：

- **SWE-agent**（Princeton，NeurIPS 2024）：学术原型，回答「接口设计能带来多大性能」；
- **OpenHands**（开源平台）：回答「一个可复现、可扩展的开放 agent 平台长什么样」；
- **Claude Code**（Anthropic）：回答「生产级 coding agent 该多简单」；
- **GitHub Copilot coding agent**：回答「agent 如何嵌入既有工程流程（PR + CI）」。

它们共享 [agent loop](/harness/agent-loop) 的同一骨架 `a_t \sim \pi_\theta(\cdot\mid\mathcal{C}(s_t))`，分歧集中在三处：**行动空间（工具集）怎么设计**、**上下文 $\mathcal{C}$ 怎么管**、**执行环境怎么隔离**。

## 2. 四个系统逐一拆解

### 2.1 SWE-agent：接口即性能

SWE-agent 的核心贡献是 **Agent-Computer Interface（ACI）**：不要让 LM 直接用为人类设计的 Linux shell，而要为它专门设计一套命令——创建/编辑文件、导航仓库、跑测试。每个 LM 步是一个原子的 `{thought, command}` 对，ACI 把它翻译成环境动作、再把结果摘要回传，从而**把自然语言推理与底层 shell/文件操作解耦**。

它最有力的证据是控制变量实验：同一个 GPT-4 Turbo，接定制 ACI 在 SWE-bench（full，2294 个 issue）解决 **12.5% pass@1**，而此前最好的非交互式 RAG 系统只有 **3.8%**。模型没变，接口换来三倍多的提升。ACI 的四条设计原则（动作简单、动作紧凑、反馈充分但简洁、用 guardrail 如 linter 拦截错误编辑）成了后续所有 harness 的母题。上下文侧它用折叠规则（仅最近若干 observation 保留原文，其余折一行），细节见 [agent loop](/harness/agent-loop)。

### 2.2 OpenHands：事件流开放平台

OpenHands 用 **event-stream 架构**：交互以类型化事件流动——User Message → Agent → LLM → Action → Runtime/sandbox → Observation → Agent。其 SDK 由四部分组成：无状态的 **Agent**（发出 Action）、运行循环并维护 append-only **EventLog** 的 **Conversation**、执行 Action 并返回 Observation 的 **Workspace**、以及被 LiteLLM 包装以支持多家供应商的 **LLM**。

![OpenHands 三大组件：Agent（agenthub）+ Event Stream（动作/观察历史）+ Runtime（把动作执行成观察）](/papers/systems/openhands-arch.png)

> 图源：Wang et al., *OpenHands: An Open Platform for AI Software Developers as Generalist Agents*, [arXiv:2407.16741](https://arxiv.org/abs/2407.16741)（用于学习注解，版权归原作者）

执行环境上，每个任务会话拉起一个隔离的 Docker 容器跑 **action-execution server**（bash、浏览器、Jupyter、VSCode 等），后端经 RESTful API 发 action、收 observation，容器边界阻止 agent 访问/修改宿主，支持 overlay 写时复制挂载做受控只读共享（详见 [沙箱](/harness/sandbox)）。性能上它在 SWE-bench Verified 上有公开的强成绩——官方榜单某次快照（约 2025 年）下 OpenHands + CodeAct（claude-3-5-sonnet）是领先的开源项之一，处于 50%+ 量级；具体百分比随榜单快照与 harness 版本变化，应视为版本相关而非定值。它的价值在于**全栈开源 + 可扩展**，是研究和自托管的首选底座。

### 2.3 Claude Code：单循环 + 极简

Claude Code 把自己定位为模型外围的 agentic harness，工程哲学是**刻意保持简单**：据第三方逆向分析，它只留一个主循环加扁平消息历史，至多一层 subagent 分支（subagent 不能再派生），理由是可调试性远比复杂的多 agent 编排重要。loop 描述为 gather context → take action → verify。

它的特色集中在**权限与上下文工程**：默认只读 + fail-closed 的权限模型（只读命令免审批，改系统的命令需批准，allow/deny 可按用户/库/组织配；未匹配默认转人工审批）；`/sandbox` 提供文件系统与网络隔离的本地沙箱（写权限锁工作目录）；prompt injection 防护（`curl`/`wget` 不自动批准、web fetch 用隔离上下文窗口、MCP server 需信任校验）；上下文用 auto-compact（先清旧工具输出、再按需总结）。云端版本每会话跑在 Anthropic 托管的隔离 VM 里、网络默认受限、凭据走安全代理、`git push` 限工作分支。这些机制详见 [沙箱与工具执行](/harness/sandbox)。

### 2.4 GitHub Copilot coding agent：嵌入 CI 的 agent

Copilot 的云端 coding agent 走的是「**把 agent 包进既有工程流程**」的路线：每个任务跑在一个**由 GitHub Actions 驱动的安全、临时（ephemeral）开发环境**里，**联网受限、仓库权限受限**，且在 reviewer 合并它开的 PR 之前**无法影响生产仓库**。环境也可跑在自托管 Actions runner（经 Actions Runner Controller）上。

这条路线的关键不是 loop 多聪明，而是**把人类 code review 这一层作为强制的安全/质量闸门**——agent 的产出必须以 PR 形式经人审后合并，天然契合 [沙箱](/harness/sandbox) 里讲的「高风险动作需人工审批」。GitHub 还在公开预览中推出了 Copilot 的云/本地沙箱：`copilot --cloud` 拉起 GitHub 托管的完全隔离临时 Linux 沙箱；本地沙箱则在用户机器上以受限文件系统、网络与系统能力运行 Copilot。

## 3. 横向对比

| 维度 | SWE-agent | OpenHands | Claude Code | Copilot coding agent |
| --- | --- | --- | --- | --- |
| 定位 | 学术原型 | 开源平台 | 生产 CLI / agent | 平台内置云 agent |
| 行动空间 | 定制 ACI 命令（带 linter） | CodeAct：可执行 Python/bash + 浏览器 | 内置工具 + Bash + subagent | Actions 环境内的工具 |
| 状态 / 上下文 | 历史 + 折叠规则 | event stream + LLM condenser | 扁平历史 + auto-compact | 任务级，绑定 PR |
| 执行环境 | shell 环境 | 隔离 Docker + action server | 本地 `/sandbox` / 云端隔离 VM | GitHub Actions 临时环境 |
| 隔离强度 | 进程级 | 容器级（写时复制挂载） | OS 原生沙箱 / 托管 VM | CI 容器 + 权限收窄 |
| 安全闸门 | guardrail（linter） | 容器边界 | 权限 allow/deny + 审批 | **PR 必须人审合并** |
| 多 agent | 单 agent | 支持委派（AgentDelegate） | 至多一层 subagent | 单 agent + 人审 |
| 主要价值 | 证明接口杠杆 | 可复现 / 可扩展 | 简单 / 可调试 / 安全默认 | 嵌入既有 PR 流程 |

> 关于 SWE-bench 数字的纪律：仅 paper 报告的 **SWE-agent 12.5%（full）**、以及 OpenHands 在 Verified 上**约 50%+ 量级的快照成绩**可作可靠参照。网络上出现过无法在官方榜单核实的、未来日期或畸高（如「95%」）的条目，本页不予引用。榜单成绩高度依赖快照与 harness 版本，跨系统直接比绝对值意义有限。

## 4. 共同的设计取舍

把四个系统叠在一起看，浮现出几条反复出现的取舍轴：

- **接口专用化 vs 通用 shell**：SWE-agent 用定制 ACI、OpenHands 用 CodeAct（直接可执行代码），代表两种极端。专用接口约束强、对模型友好、易加 guardrail；可执行代码表达力最强但更难约束。**接口设计本身就是与权重同量级的性能杠杆**，且迭代成本远低于重训。
- **单循环 vs 多 agent 编排**：Claude Code 押注极简单循环换可调试性；OpenHands 支持委派；多数生产系统选择「主循环 + 至多一层子 agent」。多 agent 的收益要和成本一起算（见 [多 Agent](/agent/multi-agent)）。
- **隔离强度 vs 开销**：从进程级到容器到托管 VM，越强隔离越贵；选型应按代码可信度而非一刀切（见 [沙箱](/harness/sandbox) 的隔离谱系）。
- **自动化 vs 人工闸门**：Copilot 把 PR 人审作为硬性闸门，Claude Code 用 allow/deny + 审批分档。在 prompt injection 无法被任何单层彻底缓解的前提下（OWASP LLM01），高风险动作保留人类把关是务实选择。

## 5. 选型建议

- **做研究 / 要复现消融**：OpenHands（开源、event stream 清晰、condenser 可调）或 SWE-agent（ACI 概念最干净，适合研究接口设计）。
- **要在本地仓库里高频开发**：Claude Code 一类生产 CLI——单循环可调试、安全默认（只读 + fail-closed + 本地沙箱）省心。
- **要把 agent 接进团队工程流程**：Copilot coding agent 一类与 PR/CI 绑定的方案，靠人审合并作为天然质量与安全闸门。
- **要大规模并行跑不可信代码**：在上面任意一种之上换用云端 microVM 执行后端（见 [沙箱](/harness/sandbox)），把隔离做到内核级。

无论选哪个，三件套的检查清单不变：**工具描述够不够详细、上下文 $\mathcal{C}$ 有没有治理 context rot、执行环境的隔离与审批分档是否匹配代码可信度**。

## 6. 参考文献

- Yang et al., 2024. *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* arXiv:2405.15793
- Wang et al., 2024. *OpenHands: An Open Platform for AI Software Developers as Generalist Agents.* arXiv:2407.16741
- Wang et al., 2025. *The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents.* arXiv:2511.03690（对应 2.2 节描述的 SDK 架构）
- Wang et al., 2024. *Executable Code Actions Elicit Better LLM Agents (CodeAct).* arXiv:2402.01030
- Claude Code 官方文档. *How Claude Code Works* / *Security.*（code.claude.com/docs）
- GitHub 官方文档. *About Copilot coding agent* / *Cloud and local sandboxes (public preview).*（docs.github.com, github.blog）
- SWE-bench. *Official Leaderboard.*（swebench.com）
