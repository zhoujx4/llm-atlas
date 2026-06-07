import type { DefaultTheme, LocaleSpecificConfig } from 'vitepress'

export const zh: LocaleSpecificConfig<DefaultTheme.Config> = {
  description: 'LLM 训练算法知识图谱：SFT / LoRA / DPO / RLHF / Agent',
  themeConfig: {
    nav: [
      { text: '导读', link: '/guide/' },
      { text: 'SFT', link: '/sft/' },
      { text: 'LoRA', link: '/lora/' },
      { text: '偏好优化', link: '/dpo/' },
      { text: 'RLHF / RL', link: '/rlhf/' },
      { text: 'Agent', link: '/agent/' }
    ],
    sidebar: [
      {
        text: '导读',
        collapsed: false,
        items: [
          { text: '全景速览', link: '/' },
          { text: '如何使用本知识库', link: '/guide/' },
          { text: '符号约定', link: '/guide/notation' }
        ]
      },
      {
        text: 'SFT 监督微调',
        collapsed: false,
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
        text: '偏好优化（DPO 家族）',
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
        text: 'RLHF / 强化学习',
        collapsed: true,
        items: [
          { text: 'RLHF 总览', link: '/rlhf/' },
          { text: 'Reward Model', link: '/rlhf/reward-model' },
          { text: 'PPO', link: '/rlhf/ppo' },
          { text: 'GRPO', link: '/rlhf/grpo' },
          { text: 'RLOO', link: '/rlhf/rloo' },
          { text: 'REINFORCE++', link: '/rlhf/reinforce-plus-plus' }
        ]
      },
      {
        text: 'Agent 与 Skill',
        collapsed: true,
        items: [
          { text: '总览', link: '/agent/' },
          { text: 'Tool Use 训练', link: '/agent/tool-use' },
          { text: 'Agent Skills', link: '/agent/agent-skills' },
          { text: 'Agentic RL', link: '/agent/agentic-rl' },
          { text: '多智能体', link: '/agent/multi-agent' }
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
