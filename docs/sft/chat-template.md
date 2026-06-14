---
title: Chat Template 对话模板
---

# Chat Template 对话模板

> **一句话**：Chat Template 是把多轮对话（system / user / assistant）序列化成单条 token 序列的格式约定，训练与推理必须使用**完全一致**的同一套模板，否则模型行为会严重错位。
> 代表工作年份：2023（OpenAI ChatML 随 ChatGPT/Whisper API 于 2023-03 推出，奠定主流对话模板格式）· Chat Template 本身为通用工程约定，无单一论文年份
>
> 前置阅读：[SFT 总览](/sft/)、[全量微调](/sft/full-finetuning)

## 直觉与动机

预训练模型只会"续写"：给一段前缀，预测下一个 token。它本身并不知道"现在是谁在说话""哪里是用户的话、哪里该自己回答""一句话什么时候结束"。要把它变成对话助手，必须人为引入一套结构化的格式，告诉模型：

- **角色边界**：哪些 token 属于 system 指令、哪些属于 user 提问、哪些属于 assistant 回答；
- **回合边界**：一轮对话从哪里开始、到哪里结束，从而知道何时停止生成；
- **生成起点**：推理时把对话拼到 assistant 角色头部，模型从这里接着写。

Chat Template 就是这套约定的具体实现。它通常借助若干 **special token**（如 `<|im_start|>`、`<|im_end|>`）把角色和回合显式标记出来。关键纪律只有一条：**训练时怎么拼，推理时就必须怎么拼**。模板不一致是 SFT 工程中最高频、也最隐蔽的事故来源——loss 曲线看起来完全正常，但部署后模型胡言乱语、停不下来、或复读角色标记。

## 方法与公式

### 模板的本质：从结构化对话到 token 序列

一段对话是一个消息列表 $M = [(\text{role}_1, c_1), \dots, (\text{role}_n, c_n)]$，模板是一个确定性函数 $T$，把它映射成 token 序列：

$$
T(M) = \text{prefix}_1 \oplus c_1 \oplus \text{suffix}_1 \oplus \cdots \oplus \text{prefix}_n \oplus c_n \oplus \text{suffix}_n
$$

其中 $\oplus$ 表示拼接，每个角色的 prefix/suffix 由模板规定。SFT 的训练目标仍是标准 NLL，但只在 assistant 内容段计算 loss（见 [Loss Masking](/sft/loss-masking)）。

### 主流模板格式

**ChatML**（Qwen、众多开源模型采用）：用 `<|im_start|>role` 开头、`<|im_end|>` 结尾，结构清晰、角色可扩展，是目前最通用的格式。

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
帮我把这句话翻译成英文。<|im_end|>
<|im_start|>assistant
Sure, here is the translation.<|im_end|>
```

**Llama 系**：Llama 2 使用 `[INST] ... [/INST]` 包裹用户指令，system prompt 放进首个 `[INST]` 内的 `<<SYS>>` 块；Llama 3 改用 header token 形式 `<|start_header_id|>role<|end_header_id|>` 加 `<|eot_id|>` 结束符。不同代际差异很大，迁移时需重写模板。

**其他**：DeepSeek、GLM、Kimi 等各家有自定义 special token 与首/尾标记，但核心思想一致——角色前缀 + 内容 + 回合结束符。具体见各 [基座模型](/base-models/) 页。

### system prompt 与 generation prompt

- **system prompt**：放在序列最前，承载角色设定、安全约束、工具说明。有的模板即使用户没传 system 也会注入默认值，需注意训练与推理保持一致。
- **generation prompt**：推理时模板要在末尾补上 assistant 的起始前缀（如 `<|im_start|>assistant\n`），让模型从这里开始生成。HuggingFace 中由 `apply_chat_template(..., add_generation_prompt=True)` 控制；**训练拼接时不能加**这个尾巴，否则会把它当成要学习的内容。

## 与 baseline 对比

| 维度 | 无模板（纯续写） | 手写字符串拼接 | `apply_chat_template`（jinja） |
| --- | --- | --- | --- |
| 角色/回合区分 | 无 | 依赖人工约定 | 模板内置，统一 |
| 训练/推理一致性 | 不适用 | 极易漏掉空格/换行不一致 | 同一模板保证一致 |
| special token 处理 | 不适用 | 易漏加或重复加 | 自动处理 |
| 多模型迁移 | 不适用 | 每次重写 | 换 tokenizer 即换模板 |
| 出错隐蔽性 | 高 | 高 | 低 |

结论：除非有特殊定制需求，**优先使用模型自带的 `tokenizer.chat_template`**，不要手搓字符串。

## 实现要点

**1. special token 与 embedding 初始化**。若模板引入了词表中原本没有的新 token（如自定义角色符），需要 `add_special_tokens` 并 `resize_token_embeddings`。新增 embedding 行默认随机初始化，常见做法是用已有 token embedding 的均值初始化，以减少初期扰动。复用模型原生模板则无此问题。

**2. 用官方接口而非字符串拼接**：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "帮我翻译这句话。"},
    {"role": "assistant", "content": "Sure, here it is."},
]

# 训练：拿到完整序列，不加 generation prompt
ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
)

# 推理：只到最后一条 user，补 assistant 起始前缀
prompt_ids = tokenizer.apply_chat_template(
    messages[:-1], tokenize=True, add_generation_prompt=True,
)
```

**3. 与 Loss Masking 配合**。多轮对话拼成一条序列后，需要知道哪些 token 区间属于 assistant 内容才能正确算 loss。可逐轮调用模板、记录每段的 token 偏移，或用支持返回 assistant mask 的模板能力（部分实现支持 `return_assistant_tokens_mask`）。模板的 prefix/suffix（角色标记、`<|im_end|>` 等）通常**不**计入 loss，但回合结束符是否计入需与团队约定统一——让模型学会输出结束符往往是有益的。详见 [Loss Masking](/sft/loss-masking)。

**4. 与 Packing 配合**。多条样本拼进一条 context 时，每条样本各自是一段完整的模板序列，样本之间需要 attention 隔离，避免 A 样本"看见"B 样本的对话。详见 [序列 Packing](/sft/packing)。

## 调参与实践经验

- **模板错位的典型症状**：① 推理时模型把角色标记（如 `<|im_start|>user`）也输出出来——通常是 generation prompt 没加对或 EOS 配置错；② 停不下来、一直续写后续轮次——回合结束符没被学到或解码时没设对 `eos_token_id`；③ 输出风格漂移、不遵循 system——训练模板里 system 的位置/格式与推理不一致。
- **最常见的坑**：换行与空格。`<|im_start|>assistant` 后面到底有没有 `\n`、内容前后有没有空格，模板里写死的细节必须训练推理逐字节一致。强烈建议训练前打印一条完整拼接序列肉眼核对。
- **EOS / 结束符配置**：确认 `eos_token` 与模板的回合结束符一致；vLLM/SGLang 部署时把模板的结束符加入 `stop` 或 `stop_token_ids`，否则会越过边界继续生成。
- **多轮一致性**：若训练只用单轮数据，推理却走多轮，需确认模板在拼接历史轮次时与训练分布一致；否则长对话会逐渐崩坏。
- **工具调用 / 多模态**：function calling、图文混排会扩展模板（新增 tool 角色、特殊占位符）。这类模板更复杂，更要严格对齐官方实现，不要自行改写。

## 参考文献

- HuggingFace Transformers 文档. *Chat Templates*.
- OpenAI. *ChatML*（Chat Markup Language）.
- Touvron et al., 2023. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288.
- Grattafiori et al., 2024. *The Llama 3 Herd of Models*. arXiv:2407.21783.
