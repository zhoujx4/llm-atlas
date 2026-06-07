import type { DefaultTheme, LocaleSpecificConfig } from 'vitepress'

export const en: LocaleSpecificConfig<DefaultTheme.Config> = {
  description: 'A knowledge atlas of LLM training algorithms: SFT / LoRA / DPO / RLHF / Agent',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/en/guide/' },
      { text: 'SFT', link: '/en/sft/' },
      { text: 'LoRA', link: '/en/lora/' },
      { text: 'Preference Opt.', link: '/en/dpo/' },
      { text: 'RLHF / RL', link: '/en/rlhf/' },
      { text: 'Agent', link: '/en/agent/' }
    ],
    sidebar: [
      {
        text: 'Guide',
        collapsed: false,
        items: [
          { text: 'At a Glance', link: '/en/' },
          { text: 'How to Read This Atlas', link: '/en/guide/' },
          { text: 'Notation', link: '/en/guide/notation' }
        ]
      },
      {
        text: 'Supervised Fine-Tuning',
        collapsed: false,
        items: [
          { text: 'Overview', link: '/en/sft/' },
          { text: 'Full Fine-Tuning', link: '/en/sft/full-finetuning' },
          { text: 'Data Construction', link: '/en/sft/data-construction' },
          { text: 'Chat Template', link: '/en/sft/chat-template' },
          { text: 'Sequence Packing', link: '/en/sft/packing' },
          { text: 'Loss Masking', link: '/en/sft/loss-masking' }
        ]
      },
      {
        text: 'LoRA & Variants',
        collapsed: true,
        items: [
          { text: 'Overview & Comparison', link: '/en/lora/' },
          { text: 'LoRA', link: '/en/lora/lora' },
          { text: 'QLoRA', link: '/en/lora/qlora' },
          { text: 'DoRA', link: '/en/lora/dora' },
          { text: 'AdaLoRA', link: '/en/lora/adalora' },
          { text: 'rsLoRA', link: '/en/lora/rslora' },
          { text: 'LoRA+', link: '/en/lora/lora-plus' },
          { text: 'PiSSA', link: '/en/lora/pissa' }
        ]
      },
      {
        text: 'Preference Optimization (DPO Family)',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/dpo/' },
          { text: 'DPO', link: '/en/dpo/dpo' },
          { text: 'IPO', link: '/en/dpo/ipo' },
          { text: 'KTO', link: '/en/dpo/kto' },
          { text: 'ORPO', link: '/en/dpo/orpo' },
          { text: 'SimPO', link: '/en/dpo/simpo' },
          { text: 'CPO', link: '/en/dpo/cpo' }
        ]
      },
      {
        text: 'RLHF / Reinforcement Learning',
        collapsed: true,
        items: [
          { text: 'RLHF Overview', link: '/en/rlhf/' },
          { text: 'Reward Model', link: '/en/rlhf/reward-model' },
          { text: 'PPO', link: '/en/rlhf/ppo' },
          { text: 'GRPO', link: '/en/rlhf/grpo' },
          { text: 'RLOO', link: '/en/rlhf/rloo' },
          { text: 'REINFORCE++', link: '/en/rlhf/reinforce-plus-plus' }
        ]
      },
      {
        text: 'Agent & Skills',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/agent/' },
          { text: 'Tool Use Training', link: '/en/agent/tool-use' },
          { text: 'Agent Skills', link: '/en/agent/agent-skills' },
          { text: 'Agentic RL', link: '/en/agent/agentic-rl' },
          { text: 'Multi-Agent', link: '/en/agent/multi-agent' }
        ]
      }
    ],
    outline: { label: 'On this page', level: [2, 3] },
    editLink: {
      pattern: 'https://github.com/zhoujx4/llm-atlas/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    }
  }
}
