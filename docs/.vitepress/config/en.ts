import type { DefaultTheme, LocaleSpecificConfig } from 'vitepress'

export const en: LocaleSpecificConfig<DefaultTheme.Config> = {
  description: 'A knowledge atlas of LLM training algorithms: base models / SFT / LoRA / DPO / RLHF / distillation / inference / agents',
  themeConfig: {
    nav: [
      { text: 'At a Glance', link: '/en/' },
      { text: 'Base Models', link: '/en/base-models/' },
      {
        text: 'Training',
        items: [
          { text: 'SFT', link: '/en/sft/' },
          { text: 'LoRA & Variants', link: '/en/lora/' },
          { text: 'DPO Family', link: '/en/dpo/' },
          { text: 'PPO / GRPO Family', link: '/en/rlhf/' },
          { text: 'Distillation', link: '/en/distillation/' }
        ]
      },
      { text: 'Inference', link: '/en/inference/' },
      {
        text: 'Agent',
        items: [
          { text: 'Harness', link: '/en/harness/' },
          { text: 'Agent', link: '/en/agent/' },
          { text: 'Skills', link: '/en/skills/' }
        ]
      },
      { text: 'Guide', link: '/en/guide/' },
      { text: 'About', link: '/en/about' }
    ],
    sidebar: [
      {
        text: 'Guide',
        collapsed: false,
        items: [
          { text: 'At a Glance', link: '/en/' },
          { text: 'How to Read This Atlas', link: '/en/guide/' },
          { text: 'Notation', link: '/en/guide/notation' },
          { text: 'Information Theory Basics', link: '/en/guide/info-theory' },
          { text: 'GPU Basics', link: '/en/guide/gpu' },
          { text: 'A/B Testing & Statistics', link: '/en/guide/ab-testing' }
        ]
      },
      {
        text: 'Base Models',
        collapsed: false,
        items: [
          { text: 'Overview', link: '/en/base-models/' },
          { text: 'Qwen (Alibaba)', link: '/en/base-models/qwen' },
          { text: 'DeepSeek', link: '/en/base-models/deepseek' },
          { text: 'GLM (Zhipu)', link: '/en/base-models/glm' },
          { text: 'Llama (Meta)', link: '/en/base-models/llama' },
          { text: 'Kimi (Moonshot)', link: '/en/base-models/kimi' },
          { text: 'MiniMax', link: '/en/base-models/minimax' },
          { text: 'Step (StepFun)', link: '/en/base-models/stepfun' },
          { text: 'Gemini (Google)', link: '/en/base-models/gemini' },
          { text: 'Claude (Anthropic)', link: '/en/base-models/claude' },
          { text: 'GPT (OpenAI)', link: '/en/base-models/openai' }
        ]
      },
      {
        text: 'Supervised Fine-Tuning',
        collapsed: true,
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
        text: 'DPO Family',
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
        text: 'PPO / GRPO Family',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/rlhf/' },
          { text: 'Reward Model', link: '/en/rlhf/reward-model' },
          { text: 'PPO', link: '/en/rlhf/ppo' },
          { text: 'GRPO', link: '/en/rlhf/grpo' },
          { text: 'DAPO', link: '/en/rlhf/dapo' },
          { text: 'GSPO', link: '/en/rlhf/gspo' },
          { text: 'RLOO', link: '/en/rlhf/rloo' },
          { text: 'REINFORCE++', link: '/en/rlhf/reinforce-plus-plus' }
        ]
      },
      {
        text: 'Distillation',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/distillation/' },
          { text: 'Black-Box (Data/CoT)', link: '/en/distillation/black-box' },
          { text: 'White-Box (Logits KL)', link: '/en/distillation/white-box' }
        ]
      },
      {
        text: 'Inference & Decoding',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/inference/' },
          { text: 'KV Cache & PagedAttention', link: '/en/inference/kv-cache' },
          { text: 'Quantization (GPTQ/AWQ/FP8)', link: '/en/inference/quantization' },
          { text: 'Speculative Decoding', link: '/en/inference/speculative-decoding' }
        ]
      },
      {
        text: 'Harness',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/harness/' },
          { text: 'Agent Loop & Context', link: '/en/harness/agent-loop' },
          { text: 'Sandbox & Tool Execution', link: '/en/harness/sandbox' },
          { text: 'Systems Compared', link: '/en/harness/systems' }
        ]
      },
      {
        text: 'Agent',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/agent/' },
          { text: 'Tool Use Training', link: '/en/agent/tool-use' },
          { text: 'Agentic RL', link: '/en/agent/agentic-rl' },
          {
            text: 'Deep Research',
            collapsed: true,
            items: [
              { text: 'Overview', link: '/en/agent/deep-research/' },
              { text: 'Mind DeepResearch (Li Auto)', link: '/en/agent/deep-research/mind-deepresearch' },
              { text: 'AgentDisCo', link: '/en/agent/deep-research/agentdisco' },
              { text: 'DR-Rubric', link: '/en/agent/deep-research/dr-rubric' }
            ]
          },
          { text: 'Multi-Agent', link: '/en/agent/multi-agent' }
        ]
      },
      {
        text: 'Skills',
        collapsed: true,
        items: [
          { text: 'Agent Skills', link: '/en/skills/' },
          { text: 'Skill Design & Evaluation', link: '/en/skills/design' },
          { text: 'Skills vs RAG vs Fine-Tuning', link: '/en/skills/vs-rag-finetune' }
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
