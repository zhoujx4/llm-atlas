<script setup lang="ts">
const links = [
  { text: '知乎', href: 'https://www.zhihu.com/people/zhoujx4' },
  { text: '小红书', href: 'https://www.xiaohongshu.com/user/profile/64b94e39000000001403fb79' },
  { text: 'X', href: 'https://x.com/zhoujx4fox35923' },
  { text: 'GitHub', href: 'https://github.com/zhoujx4' }
]
</script>

<template>
  <footer class="atlas-footer">
    <div class="atlas-footer-inner">
      <p class="atlas-footer-links">
        <a
          v-for="l in links"
          :key="l.href"
          :href="l.href"
          target="_blank"
          rel="noopener noreferrer"
          >{{ l.text }}</a
        >
      </p>
      <p class="atlas-footer-copy">
        © 2026 junius · 微信 zhoujx4 · LLM Atlas — LLM 训练算法知识图谱 ·
        <a href="/llm-atlas/about">关于本站</a>
      </p>
    </div>
  </footer>
</template>

<style scoped>
.atlas-footer {
  border-top: 1px solid var(--vp-c-divider);
  padding: 24px 24px 32px;
  background: var(--vp-c-bg);
}
.atlas-footer-inner {
  max-width: 1152px;
  margin: 0 auto;
  text-align: center;
}
.atlas-footer-links {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin: 0 0 8px;
}
.atlas-footer-links a {
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  transition: color 0.2s;
}
.atlas-footer-links a:hover {
  color: var(--vp-c-brand-1);
}
.atlas-footer-copy {
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-3);
  margin: 0;
}
.atlas-footer-copy a {
  color: var(--vp-c-text-2);
  text-decoration: none;
}
.atlas-footer-copy a:hover {
  color: var(--vp-c-brand-1);
}
</style>
