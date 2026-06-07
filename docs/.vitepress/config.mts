import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { shared } from './config/shared'
import { zh } from './config/zh'
import { en } from './config/en'

export default withMermaid({
  ...defineConfig({
    ...shared,
    locales: {
      root: { label: '简体中文', lang: 'zh-CN', ...zh },
      en: { label: 'English', lang: 'en-US', link: '/en/', ...en }
    }
  }),
  // 与「学术白」主题一致：白底节点 / 墨黑描边 / 衬线字体
  mermaid: {
    theme: 'base',
    themeVariables: {
      fontFamily: "'Fraunces', 'Noto Serif SC', serif",
      primaryColor: '#ffffff',
      primaryTextColor: '#111111',
      primaryBorderColor: '#111111',
      secondaryColor: '#f2f2f2',
      tertiaryColor: '#fafafa',
      lineColor: '#444444',
      edgeLabelBackground: '#ffffff',
      clusterBkg: '#fafafa',
      clusterBorder: '#888888'
    }
  }
})
