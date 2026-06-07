---
title: Agent Skills
translation: synced
---

# Agent Skills

> **In one sentence**: TODO — package domain workflows, tool usage, and conventions into "skill packs" (instructions + scripts + resources) the model can load on demand, extending agent capabilities without changing weights.
>
> Prerequisites: [Agent Overview](/en/agent/), [Tool-Use Training](/en/agent/tool-use)

::: warning Status
🚧 This page is a placeholder outline; the full text has not been written yet.
:::

## 1. Intuition and Motivation

TODO:

- [ ] Vs fine-tuning: faster knowledge updates, composable, distributable
- [ ] Vs RAG: skills are "how to do", RAG is "what is"
- [ ] Progressive disclosure: load on demand to save context

## 2. Anatomy of a Skill

TODO:

- [ ] Typical composition: metadata (name/trigger description) + instruction body + executable scripts + reference resources
- [ ] Trigger mechanism: the model autonomously decides which skill to load based on the task description
- [ ] Example: the Anthropic Agent Skills / SKILL.md convention

```text
skill/
├── SKILL.md          # metadata + instructions
├── scripts/          # executable scripts
└── references/       # reference docs
```

## 3. Comparison with Adjacent Concepts

| Dimension | Fine-tuning | RAG | Skill |
| --- | --- | --- | --- |
| Knowledge carrier | Weights | Document chunks | Instructions + code |
| Update cost | High | Low | Low |
| Procedural knowledge | Possible | Weak | Strong |

## 4. Design Considerations

TODO:

- [ ] How to write skill descriptions so they trigger accurately
- [ ] Conflicts and composition between skills
- [ ] Evaluation: trigger accuracy, task completion rate

## 5. References

- [ ] Anthropic, 2025. *Agent Skills* engineering blog
