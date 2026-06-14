---
title: QLoRA：量化基座上的低秩微调
---

# QLoRA

> **一句话**：把冻结的基座权重量化成 4-bit（NF4）存储、前向时按块反量化参与计算，LoRA 适配器仍用 bf16 训练，使单卡也能微调几十 B 级别的大模型。论文 *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)。
>
> 提出年份：2023（arXiv） · 机构/团队：University of Washington · 会议/来源：NeurIPS 2023 / arXiv:2305.14314
>
> 前置阅读：[LoRA](/lora/lora) · [量化](/inference/quantization)

## 直觉与动机

LoRA 已经把优化器状态和梯度的显存压到极小，但还有一座大山没动：**基座权重本身**。bf16 存一份 65B 模型就要约 130GB，光是把它放进显存就超出了单卡容量——可训练参数再少也没用。

QLoRA 的洞察是：基座是冻结的，它**只参与前向、不需要梯度**，因此完全没必要用 bf16 高精度存储。把它压成 4-bit，显存占用直接降到约四分之一；前向计算时再按需反量化回 bf16 参与矩阵乘。量化在这里只服务于「存储」，不改变计算精度——梯度仍然只流向高精度的 LoRA 分支。三项工程设计让这条思路真正落地：NF4 数据类型、双重量化、分页优化器。

## 方法与公式

前向计算在使用基座权重前先做反量化：

$$
h = \mathrm{dequant}(W_0^{\text{NF4}})\, x + \frac{\alpha}{r} B A x
$$

其中 $W_0^{\text{NF4}}$ 是 4-bit 量化后的冻结基座，$B,A$ 是 bf16 的 LoRA 矩阵。梯度只对 $B,A$ 计算，$W_0$ 全程不更新。三个核心组件：

**1. NF4（4-bit NormalFloat）**——一种面向「零均值正态分布」数据的信息论最优 4-bit 数据类型。神经网络权重经验上近似服从正态分布，普通的均匀量化（INT4）会在分布尾部浪费量化级。NF4 的量化分位点按标准正态分布的分位数设计，使每个 bin 内落入的权重数量大致相等（quantile quantization），在相同比特数下显著降低量化误差。量化以「块」为单位（如每 64 个权重一块），每块单独估计一个缩放常数。

**2. Double Quantization（双重量化）**——上一步每块都要存一个 fp32 的量化常数，当块很小时这些常数本身也成了不小的开销（约 0.5 bit/参数）。双重量化对这些量化常数再做一次 8-bit 量化，把它们的平均开销压到约 0.13 bit/参数，几乎免费地省下一截显存。

**3. Paged Optimizer（分页优化器）**——利用 NVIDIA 统一内存，把优化器状态分页。当训练中出现显存峰值（如长序列 batch 引发的梯度检查点尖峰）时，自动把部分优化器分页换出到 CPU 内存，避免 OOM；需要时再换回。这让显存接近上限的训练不至于因偶发尖峰崩溃。

合起来，QLoRA 让 65B 模型可在单张 48GB GPU 上微调，而论文报告其微调效果可与 16-bit 全精度 LoRA / 全量微调持平。

![不同微调方式及其显存占用对比：QLoRA 将模型量化到 4-bit 并配合分页优化器处理显存峰值](/papers/qlora/qlora-arch.png)

> 图源：Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314（用于学习注解，版权归原作者）

## 与 baseline 对比

| 维度 | LoRA（bf16 基座） | QLoRA（NF4 基座） |
| --- | --- | --- |
| 基座存储 | 2 bytes/param | ~0.5 bytes/param（+双重量化更省） |
| 单卡可微调规模 | 受限于 bf16 基座 | 同卡可大 ~4 倍 |
| 训练速度 | 快 | 慢（每次前向有反量化开销） |
| 梯度精度 | bf16 | bf16（梯度只走 LoRA 分支） |
| 效果 | 基线 | 论文称与 16-bit 持平 |
| 部署合并 | 直接合并 | 需反量化后合并，存在量化误差 |

## 实现要点

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 数据类型
    bnb_4bit_use_double_quant=True,       # 双重量化
    bnb_4bit_compute_dtype="bfloat16",    # 反量化后的计算精度
)
model = AutoModelForCausalLM.from_pretrained(
    "your/base-model", quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)  # 开梯度检查点、稳定 LN

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
# 优化器用 paged_adamw_8bit 配合分页显存
```

要点：
- **compute_dtype 用 bf16**：反量化后参与矩阵乘的精度，bf16 是稳妥选择。
- **target_modules 尽量全覆盖**：QLoRA 论文强调把 LoRA 加到所有 Linear 层（含 MLP）对恢复效果很关键，这弥补了基座低精度带来的损失。
- **配合梯度检查点**：`prepare_model_for_kbit_training` 会开启，进一步省激活显存。
- **合并需谨慎**：把 LoRA 合回 4-bit 基座要先反量化，得到的合并权重带量化误差；若追求无损部署，可在 bf16 基座上重新加载同一份 adapter 再合并。

## 调参与实践经验

- **最适用场景**：显存极度受限、想在单卡上微调大模型。显存宽裕时优先用 bf16 基座的标准 [LoRA](/lora/lora)，训练快得多。
- **速度代价**：每次前向都要反量化，吞吐通常明显低于同配置的 bf16 LoRA；这是用算力换显存的本质权衡。
- **秩可略大**：因为基座精度更低，适当增大 $r$（如 16~64）并扩大目标模块有助于补回效果。
- **与变体叠加**：QLoRA 是「量化基座」这一维度的改进，可与 [rsLoRA](/lora/rslora) 缩放、[LoRA+](/lora/lora-plus) 学习率正交组合；近年也有 QLoRA + [DoRA](/lora/dora) 的量化版实现。
- **常见坑**：4-bit 下数值更敏感，遇到 loss 不降先检查 compute_dtype、是否漏开梯度检查点、目标模块是否覆盖 MLP。

## 参考文献

- Dettmers et al., 2023. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- Dettmers et al., 2022. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
