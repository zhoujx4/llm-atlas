---
title: Data Construction
translation: synced
---

# SFT Data Construction

> **In one sentence**: TODO — the ceiling of SFT is mostly determined by data quality; this page covers where data comes from, how to clean it, and how to mix it.
>
> Prerequisites: [SFT Overview](/en/sft/)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Evidence for "quality > quantity" (LIMA, etc.)
- [ ] Instruction diversity vs single-task depth

## 2. Data Sources

TODO:

- [ ] Human annotation (designing annotation guidelines)
- [ ] Survey of open-source datasets and license caveats
- [ ] Synthetic data: self-instruct, distilling strong models, rejection sampling

## 3. Quality Filtering and Deduplication

TODO:

- [ ] Rule-based filtering, model-scored filtering
- [ ] Exact deduplication / approximate deduplication (MinHash)
- [ ] Contamination checks against evaluation sets

## 4. Mixing Ratios and Curriculum

TODO:

- [ ] Task-type ratios (code / math / dialogue / safety)
- [ ] Multilingual ratios
- [ ] Multi-stage SFT (general first, then domain-specific)

## 5. Experiments and Tuning Experience

TODO: empirical curves of data scale vs performance; common dirty-data patterns.

## 6. References

- [ ] Wang et al., 2022. *Self-Instruct*
- [ ] Zhou et al., 2023. *LIMA*
