import type { DefaultTheme, LocaleSpecificConfig } from 'vitepress'

export const zh: LocaleSpecificConfig<DefaultTheme.Config> = {
  description: 'LLM 训练算法知识图谱：基模 / SFT / LoRA / DPO / RLHF / 蒸馏 / 推理 / Agent',
  themeConfig: {
    nav: [
      { text: '速览', link: '/' },
      { text: '基础模型', link: '/base-models/' },
      { text: '模型架构', link: '/architecture/' },
      {
        text: '训练',
        items: [
          { text: 'SFT 监督微调', link: '/sft/' },
          { text: 'LoRA 及变体', link: '/lora/' },
          { text: 'DPO 系列', link: '/dpo/' },
          { text: 'PPO / GRPO 系列', link: '/rlhf/' },
          { text: '蒸馏', link: '/distillation/' }
        ]
      },
      { text: '推理', link: '/inference/' },
      {
        text: 'Agent',
        items: [
          { text: 'Harness', link: '/harness/' },
          { text: 'Agent', link: '/agent/' },
          { text: 'Skills', link: '/skills/' }
        ]
      },
      { text: '导读', link: '/guide/' },
      { text: '关于', link: '/about' }
    ],
    sidebar: [
      {
        text: '导读',
        collapsed: false,
        items: [
          { text: '全景速览', link: '/' },
          { text: '如何使用本知识库', link: '/guide/' },
          { text: '符号约定', link: '/guide/notation' },
          { text: '前沿博客与资源', link: '/guide/blogs' }
        ]
      },
      {
        text: '基础模型',
        collapsed: false,
        items: [
          { text: '总览与选型', link: '/base-models/' },
          { text: 'Qwen（阿里）', link: '/base-models/qwen' },
          { text: 'DeepSeek（深度求索）', link: '/base-models/deepseek' },
          { text: 'GLM（智谱）', link: '/base-models/glm' },
          { text: 'Llama（Meta）', link: '/base-models/llama' },
          { text: 'Kimi（月之暗面）', link: '/base-models/kimi' },
          { text: 'MiniMax', link: '/base-models/minimax' },
          { text: 'Step（阶跃星辰）', link: '/base-models/stepfun' },
          { text: 'Gemini（Google）', link: '/base-models/gemini' },
          { text: 'Claude（Anthropic）', link: '/base-models/claude' },
          { text: 'GPT（OpenAI）', link: '/base-models/openai' }
        ]
      },
      {
        text: '模型架构',
        collapsed: true,
        items: [
          { text: '总览', link: '/architecture/' },
          { text: 'Transformer 基础架构', link: '/architecture/transformer' },
          { text: '注意力变体（MHA/MQA/GQA/MLA）', link: '/architecture/attention' },
          { text: '稀疏与线性注意力', link: '/architecture/sparse-attention' },
          { text: '位置编码与归一化', link: '/architecture/positional-norm' },
          { text: 'MoE 混合专家', link: '/architecture/moe' },
          { text: 'VLM 多模态结构', link: '/architecture/vlm' },
          { text: 'Omni 全模态架构', link: '/architecture/omni' }
        ]
      },
      {
        text: 'SFT 监督微调',
        collapsed: true,
        items: [
          { text: '总览', link: '/sft/' },
          { text: '全量微调', link: '/sft/full-finetuning' },
          { text: '数据构造', link: '/sft/data-construction' },
          { text: 'Chat Template', link: '/sft/chat-template' },
          { text: '序列 Packing', link: '/sft/packing' },
          { text: 'Loss Masking', link: '/sft/loss-masking' }
        ]
      },
      {
        text: 'LoRA 及变体',
        collapsed: true,
        items: [
          { text: '总览与变体对比', link: '/lora/' },
          { text: 'LoRA', link: '/lora/lora' },
          { text: 'QLoRA', link: '/lora/qlora' },
          { text: 'DoRA', link: '/lora/dora' },
          { text: 'AdaLoRA', link: '/lora/adalora' },
          { text: 'rsLoRA', link: '/lora/rslora' },
          { text: 'LoRA+', link: '/lora/lora-plus' },
          { text: 'PiSSA', link: '/lora/pissa' }
        ]
      },
      {
        text: 'DPO 系列',
        collapsed: true,
        items: [
          { text: '总览', link: '/dpo/' },
          { text: 'DPO', link: '/dpo/dpo' },
          { text: 'IPO', link: '/dpo/ipo' },
          { text: 'KTO', link: '/dpo/kto' },
          { text: 'ORPO', link: '/dpo/orpo' },
          { text: 'SimPO', link: '/dpo/simpo' },
          { text: 'CPO', link: '/dpo/cpo' }
        ]
      },
      {
        text: 'PPO / GRPO 系列',
        collapsed: true,
        items: [
          { text: '总览', link: '/rlhf/' },
          { text: 'Reward Model', link: '/rlhf/reward-model' },
          { text: 'PPO', link: '/rlhf/ppo' },
          { text: 'GRPO', link: '/rlhf/grpo' },
          { text: 'DAPO', link: '/rlhf/dapo' },
          { text: 'GSPO', link: '/rlhf/gspo' },
          { text: 'RLOO', link: '/rlhf/rloo' },
          { text: 'REINFORCE++', link: '/rlhf/reinforce-plus-plus' }
        ]
      },
      {
        text: '蒸馏',
        collapsed: true,
        items: [
          { text: '总览', link: '/distillation/' },
          { text: '黑盒蒸馏（数据/CoT）', link: '/distillation/black-box' },
          { text: '白盒蒸馏（logits KL）', link: '/distillation/white-box' },
          { text: '推理蒸馏（R1-Distill/s1/LIMO）', link: '/distillation/reasoning' }
        ]
      },
      {
        text: '推理与解码',
        collapsed: true,
        items: [
          { text: '总览', link: '/inference/' },
          { text: 'KV Cache 与 PagedAttention', link: '/inference/kv-cache' },
          { text: '量化（GPTQ/AWQ/FP8）', link: '/inference/quantization' },
          { text: '投机解码（含 MTP）', link: '/inference/speculative-decoding' },
          { text: '推理框架与服务引擎', link: '/inference/frameworks' }
        ]
      },
      {
        text: 'Harness',
        collapsed: true,
        items: [
          { text: '总览', link: '/harness/' },
          { text: '执行循环与上下文管理', link: '/harness/agent-loop' },
          { text: '沙箱与工具执行', link: '/harness/sandbox' },
          { text: '代表系统对比', link: '/harness/systems' },
          {
            text: '自主科研与自动化 Agent',
            collapsed: true,
            items: [
              { text: '总览', link: '/harness/auto-agents/' },
              { text: 'AI Scientist', link: '/harness/auto-agents/ai-scientist' },
              { text: 'Agent Laboratory', link: '/harness/auto-agents/agent-laboratory' },
              { text: 'AIDE（ML 工程 Agent）', link: '/harness/auto-agents/aide' },
              { text: 'AI co-scientist', link: '/harness/auto-agents/ai-co-scientist' }
            ]
          }
        ]
      },
      {
        text: 'Agent',
        collapsed: true,
        items: [
          { text: '总览', link: '/agent/' },
          { text: 'Tool Use 训练', link: '/agent/tool-use' },
          {
            text: 'Agentic RL',
            collapsed: true,
            items: [
              { text: '总览', link: '/agent/agentic-rl/' },
              { text: '检索与工具 RL（Search-R1 系）', link: '/agent/agentic-rl/search-rl' },
              { text: '软件工程 RL（SWE-RL）', link: '/agent/agentic-rl/swe-rl' },
              { text: 'Web 长程导航 RL', link: '/agent/agentic-rl/web-agent-rl' },
              { text: '训练稳定性', link: '/agent/agentic-rl/stability' }
            ]
          },
          {
            text: '代表性 Agent 框架',
            collapsed: true,
            items: [
              { text: '总览与对比', link: '/agent/frameworks/' },
              { text: 'LangChain', link: '/agent/frameworks/langchain' },
              { text: 'LangGraph', link: '/agent/frameworks/langgraph' },
              { text: 'LlamaIndex', link: '/agent/frameworks/llamaindex' },
              { text: 'AutoGen', link: '/agent/frameworks/autogen' },
              { text: 'CrewAI', link: '/agent/frameworks/crewai' },
              { text: 'MetaGPT', link: '/agent/frameworks/metagpt' },
              { text: 'Claude Agent SDK', link: '/agent/frameworks/claude-agent-sdk' },
              { text: 'Claude Code', link: '/agent/frameworks/claude-code' },
              { text: 'Codex', link: '/agent/frameworks/codex' },
              { text: 'OpenClaw', link: '/agent/frameworks/openclaw' },
              { text: 'Hermes Agent', link: '/agent/frameworks/hermes' }
            ]
          },
          {
            text: 'Deep Research',
            collapsed: true,
            items: [
              { text: '总览', link: '/agent/deep-research/' },
              { text: 'OpenAI Deep Research', link: '/agent/deep-research/openai-deep-research' },
              { text: 'Tongyi DeepResearch（阿里）', link: '/agent/deep-research/tongyi-deepresearch' },
              { text: 'REDSearcher（小红书）', link: '/agent/deep-research/redsearcher' },
              { text: 'open-deep-research（HF）', link: '/agent/deep-research/open-deep-research' },
              { text: 'STORM / Co-STORM', link: '/agent/deep-research/storm' }
            ]
          },
          { text: '多智能体', link: '/agent/multi-agent' }
        ]
      },
      {
        text: 'Skills',
        collapsed: true,
        items: [
          { text: 'Agent Skills 体系', link: '/skills/' },
          { text: '技能设计与评测', link: '/skills/design' },
          {
            text: 'AutoSkill：技能自迭代',
            collapsed: true,
            items: [
              { text: '总览', link: '/skills/autoskill/' },
              { text: 'SkillOS（技能策展 RL）', link: '/skills/autoskill/skillos' },
              { text: 'SkillOpt（技能即权重优化）', link: '/skills/autoskill/skillopt' },
              { text: 'SkillOps（技能库工程化运维）', link: '/skills/autoskill/skillops' },
              { text: 'OpenSkill（开放世界自演化）', link: '/skills/autoskill/openskill' }
            ]
          },
          { text: 'Skills vs RAG vs 微调', link: '/skills/vs-rag-finetune' }
        ]
      }
    ],
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    langMenuLabel: '切换语言',
    editLink: {
      pattern: 'https://github.com/zhoujx4/llm-atlas/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    }
  }
}
