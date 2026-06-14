---
title: 技能设计与评测
---

# 技能设计与评测

> **一句话**：好 skill 的三条铁律——description 决定触发、正文控制预算、脚本固化确定性；开发流程上「评测先行、双实例迭代」。出处：Anthropic 工程博客 *Equipping agents for the real world with Agent Skills*（2025）与官方 best practices 文档。
> 提出年份：2025（随 Agent Skills 于 2025-10-16 发布）· 机构/团队：Anthropic · 来源：anthropic.com/engineering 工程博客 + 官方 best practices 文档
>
> 前置阅读：[Agent Skills 体系](/skills/)、[Agent Loop](/harness/agent-loop)

## 直觉与动机

Skill 本质是一段会被注入上下文的 prompt 工程产物，但相比普通 system prompt 有两个独特约束，决定了它的全部设计准则：

1. **触发是模型的自主决策**。skill 不会被强制加载——Claude 扫一眼 description 决定读不读 SKILL.md。官方 skill-creator 直白承认现状："currently Claude has a tendency to undertrigger skills"（Claude 目前倾向于少触发）。description 写得含糊，skill 等于不存在。
2. **上下文是公共资源**。官方原话 "context window is a public good"：skill 正文每多一行，留给任务本身的预算就少一行；而且元数据是所有已装 skill 共同分摊的常驻成本。

所以设计目标可以概括为最大化「触发准确率 × 信息密度」：该触发时必须触发，触发后每个 token 都要有用。

## 方法与公式

### SKILL.md 解剖

开放标准（agentskills.io）的 frontmatter 定义：

| 字段 | 必填 | 约束 |
| --- | --- | --- |
| `name` | 是 | 1–64 字符；仅小写字母、数字、连字符；不以连字符开头/结尾、无连续 `--`；须与父目录名一致；不能含 `anthropic`/`claude` 保留词 |
| `description` | 是 | 1–1024 字符，非空；应同时说明「做什么」和「何时用」 |
| `license` / `compatibility` / `metadata` | 否 | 许可证、环境要求（≤500 字符，多数 skill 不需要）、任意键值对 |
| `allowed-tools` | 否 | 预批准工具列表，标注 Experimental，各实现支持不一 |

Claude Code 在此之上扩展了大量可选字段：`when_to_use`、`argument-hint`、`disable-model-invocation`（仅用户经 `/skill-name` 可调）、`user-invocable: false`（仅模型可调）、`allowed-tools`/`disallowed-tools`、`model`、`context: fork`（在隔离 subagent 中运行）、`paths`（glob 限定仅操作匹配文件时激活）、`hooks` 等；正文支持 `$ARGUMENTS`、`${CLAUDE_SKILL_DIR}` 等占位符和 `` !`command` `` 动态注入（命令在模型看到内容前执行，输出替换占位符）。

### token 预算账

设已装 $N$ 个 skill，静息开销为

$$
C_{\text{idle}} = \sum_{i=1}^{N} c_i^{\text{meta}}, \qquad c_i^{\text{meta}} \approx 100 \text{ tokens}
$$

触发后的增量成本是 $c^{\text{body}}$（官方建议 SKILL.md 正文 <5k tokens、500 行以内）加按需读取的第三层文件。Claude Code 对第一层有硬预算：skill 列表默认占模型上下文窗口的 **1%**，溢出时最少使用的 skill 的 description 先被丢弃；单条目 description + when_to_use 合并上限 1,536 字符。可经 `skillListingBudgetFraction` 调整（如 0.02 = 2%），`/doctor` 可检查是否溢出。装几十个 skill 不是免费的——预算视角下，每个平庸 skill 都在挤占好 skill 的展示位。

### 触发设计：description 是第一公民

- **第三人称**。description 被注入 system prompt，第一/二人称（"I can help you..."）会导致发现问题。写成 "Processes PDF files and extracts form fields. Use when the user mentions PDFs, forms, or document extraction."
- **具体且含触发关键词**。模糊词（helper、utils）是反模式；命名推荐动名词形式（gerund），如 `processing-pdfs`、`analyzing-spreadsheets`。
- **适度 pushy**。针对 undertrigger 的现实，skill-creator 建议罗列应触发的场景："Make sure to use this skill whenever the user mentions...even if they don't explicitly ask"。
- 触发不可靠时还有结构手段：Claude Code 的 `paths` 字段按文件路径激活，用户侧 `/skill-name` 强制调用。

### 正文与文件组织

- 正文 **500 行以内**；超出就拆分为多文件，SKILL.md 只留路标。
- 引用文件保持**距 SKILL.md 一层深度**——嵌套引用（A 引 B 引 C）会导致 Claude 用 `head -100` 只读部分内容。
- 超过 100 行的参考文件开头加目录（TOC），便于模型选择性阅读。
- 默认假设模型已经很聪明，删掉它已知的常识解释——只写它不知道的：内部约定、领域陷阱、精确流程。

### 自由度设计与脚本

官方提出「自由度」（degrees of freedom）标尺：多种做法都有效的任务，给高自由度的文字指引，让模型自己发挥；操作脆弱、要求严格一致的任务，给低自由度的精确脚本（"Run exactly this script... Do not modify the command"）。带脚本时的四条纪律：

- **Solve, don't punt**：脚本自行处理错误分支，不要把异常甩回给模型即兴处理；
- 不留无解释的魔法常量（voodoo constants）；
- 明确每个脚本是「执行」还是「作为参考阅读」——两者的 token 成本截然不同；
- 复杂产出用 plan-validate-execute 模式：先产出可校验的中间产物（计划/中间文件），验证后再执行，失败可定点重试。引用 MCP 工具时写全限定名（`ServerName:tool_name`）。

### 评测驱动开发

官方反直觉的建议是 "Create evaluations BEFORE writing extensive documentation"：

```mermaid
flowchart LR
    A[无 skill 状态<br/>跑代表性任务] --> B[记录失败模式<br/>≥3 个评估场景]
    B --> C[建基线分数]
    C --> D[写最小指令<br/>只补差距]
    D --> E[Claude B 实测<br/>加载 skill 的新实例]
    E -->|观察行为| F[Claude A 修改<br/>共同设计的实例]
    F --> D
```

「Claude A/Claude B」双实例迭代是关键技巧：与 Claude A 共同设计 skill，用干净的 Claude B 加载实测——避免设计者实例的对话历史污染评估。另建议用 Haiku/Sonnet/Opus 全系测试：对 Opus 够用的指令对 Haiku 可能不够。skill-creator 本身内置了 eval 工作流（evals.json + 迭代式 workspace），可直接借用。

## 与 baseline 对比

| 维度 | 全量塞 system prompt / CLAUDE.md | MCP 工具 | Skill |
| --- | --- | --- | --- |
| 静息 token | 全文常驻 | 工具定义常驻（大型 MCP 可达数万 token） | 每个约 100 tokens 元数据 |
| 加载时机 | 永远在场 | 连接时全量注入 | 任务匹配时按层加载 |
| 内容规模上限 | 受上下文窗口硬限 | 工具 schema 为主 | 第三层「事实上无上限」 |
| 承载内容 | 通用约定 | 外部系统连接与动作 | 流程知识 + 脚本 + 资源 |
| 跨产品移植 | 各家格式不一 | 协议级标准 | 开放标准，Markdown 即格式 |
| 失败模式 | 稀释注意力、长文遗忘 | token 爆炸、工具选择困难 | undertrigger、description 内卷 |

三者互补而非互斥：MCP 管「连接什么」，skill 管「怎么用好这些连接」，CLAUDE.md 管「项目恒真的少量约定」。

## 实现要点

官方 pdf 技能的实际布局印证了标准结构：

```text
pdf/
├── SKILL.md        # frontmatter + 概览 + 路标
├── forms.md        # 第三层：表单处理细节，按需读取
├── reference.md    # 第三层：API 参考
└── scripts/        # 第三层：直接执行，不进上下文
```

```yaml
# SKILL.md frontmatter 模板
---
name: processing-pdfs        # 与目录名一致，gerund 命名
description: >-              # 第三人称；做什么 + 何时用 + 触发关键词
  Extracts text and form fields from PDF files. Use when the user
  mentions PDFs, fillable forms, or document data extraction.
---
```

Claude Code 的存放位置与优先级：Enterprise（managed settings）> Personal（`~/.claude/skills/`）> Project（`.claude/skills/`），同名时高层级覆盖低层级；插件 skill 走 `plugin-name:skill-name` 命名空间。原 `.claude/commands/` 自定义命令已并入 skills 体系。

生命周期细节容易踩坑：skill 被调用后其渲染内容作为单条消息**驻留整个会话**，后续轮次不重读文件（会话内改 SKILL.md 对已加载内容无效，但支持热更新供下次调用）；自动压缩（auto-compaction）时每个已调用 skill 只保留最近一次调用的前 5,000 tokens，所有重附 skill 共享 25,000 tokens 总预算，从最近调用者开始填充——超长 skill 在长会话后半段可能只剩残篇，这是「500 行以内」的另一重理由。

## 调参与实践经验

- **先治 undertrigger**：上线后第一个观察指标是该触发未触发率。对策按序：description 加触发场景枚举 → `when_to_use` 字段 → `paths` 自动激活 → 在 CLAUDE.md 里写强制合规规则。社区 Superpowers 框架的做法是显式写入 "If you have a skill to do something, you must use it"，并设计压力测试场景（借 Cialdini 说服原理构造诱导模型绕开 skill 的情境）验证 skill 真被遵守。
- **从失败轨迹里挖 skill**：工程博客建议让 Claude 把成功经验和常见错误自己沉淀为 skill；Superpowers 则从书籍和历史会话中提炼方法论。skill 是「把一次性 prompt 调试成果资产化」的容器。
- **删比写难**：迭代几轮后 SKILL.md 必然膨胀。定期用「Claude 不看这段会做错吗？」逐段拷问，答案是否就删。
- **安全审计**：使用第三方 skill 前 thoroughly audit——重点是脚本依赖、捆绑资源、指示连接不可信外部网络源的指令；官方将其类比为安装软件。
- 平台差异要心里有数：API 容器无网络、不能运行时装包；自定义 skill 在 claude.ai、API、Claude Code 间**不**自动同步，各自独立管理。

## 参考文献

- Anthropic, 2025. *Equipping agents for the real world with Agent Skills.* anthropic.com/engineering
- Anthropic. *Agent Skills — Best practices.* platform.claude.com/docs
- Anthropic. *Claude Code — Skills.* code.claude.com/docs
- Agent Skills 开放标准规范：agentskills.io/specification
- anthropics/skills 仓库（skill-creator 等 17 个官方技能）：github.com/anthropics/skills
- Jesse Vincent, 2025. *Superpowers: How I'm using coding agents in October 2025.* blog.fsck.com
