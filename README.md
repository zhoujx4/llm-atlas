# LLM Atlas

> LLM 训练算法知识图谱：基础模型 / SFT / LoRA / DPO / RLHF / 蒸馏 / 推理 / Agent —— 用 Markdown 写作，自动构建为在线网站。

**在线阅读**：https://zhoujx4.github.io/llm-atlas/ （中文 | [English](https://zhoujx4.github.io/llm-atlas/en/)）

收录原则：**只收讨论度高、用得最多的出名算法**，不堆砌冷门变体，也不堆砌易过期的 benchmark 数字。

## 知识体系

```
导读          如何阅读 · 符号约定
基础模型       Qwen · DeepSeek · GLM · Llama · Kimi · MiniMax · StepFun · Gemini · Claude · OpenAI
SFT           全量微调 · 数据构造 · Chat Template · Packing · Loss Masking
LoRA 及变体    LoRA · QLoRA · DoRA · AdaLoRA · rsLoRA · LoRA+ · PiSSA
DPO 系列       DPO · IPO · KTO · ORPO · SimPO · CPO
PPO/GRPO 系列  Reward Model · PPO · GRPO · DAPO · GSPO · RLOO · REINFORCE++
蒸馏           黑盒（数据/CoT） · 白盒（logits KL）
推理与解码     KV Cache & PagedAttention · 量化（GPTQ/AWQ/FP8） · 投机解码
Harness       执行循环与上下文 · 沙箱与工具执行 · 代表系统对比
              └ 自主科研与自动化 Agent：AI Scientist · Agent Laboratory · AIDE · AI co-scientist
Agent         Tool Use 训练 · 多智能体
              ├ Agentic RL：检索/工具 RL · 软件工程 RL · Web 导航 RL · 训练稳定性
              ├ 代表性 Agent 框架：LangChain · LangGraph · LlamaIndex · AutoGen · CrewAI · MetaGPT · Claude Agent SDK · Claude Code · Codex · OpenClaw · Hermes
              └ Deep Research：OpenAI Deep Research · open-deep-research(HF) · STORM/Co-STORM
Skills        Agent Skills 体系 · 技能设计与评测 · AutoSkill 技能自迭代 · 与 RAG/微调对比
```

开源模型以技术报告 / 论文为准，闭源模型以官方博客 / 模型卡为准。

## 本地开发

```bash
npm install
npm run docs:dev      # 开发预览 http://localhost:5173/llm-atlas/
npm run docs:build    # 构建（含死链检查），push 前建议先跑一遍
npm run docs:preview  # 以生产路径预览构建产物
```

push 到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages（仓库 Settings → Pages → Source 需选择 **GitHub Actions**）。

## 内容组织约定

- **目录 = URL = 侧边栏分组**：每个算法版块一个顶层目录（如 `docs/dpo/`），版块内每个算法一个 `.md` 文件，版块必有 `index.md` 总览页（含家族演化 Mermaid 图与变体对比表）。
- **文件命名**：小写连字符，即 URL 路径（`reinforce-plus-plus.md` → `/rlhf/reinforce-plus-plus`）。
- **站内链接**：写不含 base 的绝对路径（如 `/dpo/dpo`），**不要**手写 `/llm-atlas/` 前缀。
- **数学公式**：`$...$` 行内、`$$...$$` 块级，记号遵循 [符号约定](docs/guide/notation.md)。
- **图表**：Mermaid 代码块直接写在 md 中。

### 算法页标准结构

每个算法页遵循统一模板：

1. 一句话定义 + 论文/年份 + 前置阅读链接
2. 直觉与动机（它解决了什么问题）
3. 方法与公式（核心公式必须给出）
4. 与 baseline 对比（表格）
5. 实现要点与伪代码
6. 实验与调参经验
7. 参考文献

### 双语规则

- **中文（`docs/`）为 source of truth**，英文镜像位于 `docs/en/`，路径与中文严格一致（语言切换按钮依赖此约定）。
- 英文页 frontmatter 用 `translation: pending | synced` 标记翻译状态。
- 修改中文内容时，至少同步英文页的标题与小节结构，正文可后补并标 `pending`。
- 新增页面需同时更新 `docs/.vitepress/config/zh.ts` 与 `en.ts` 的侧边栏。

## 技术栈

[VitePress](https://vitepress.dev/) · markdown-it-mathjax3（数学公式）· vitepress-plugin-mermaid（图表）· 内置本地搜索 · GitHub Actions + GitHub Pages

## License

代码采用 [MIT](LICENSE) 许可；文档内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可。
