import type { UserConfig } from 'vitepress'
import type { DefaultTheme } from 'vitepress'

export const shared: UserConfig<DefaultTheme.Config> = {
  title: 'LLM Atlas',
  base: '/llm-atlas/',
  lastUpdated: true,
  cleanUrls: true,
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    [
      'link',
      {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,560;0,9..144,720;1,9..144,400&family=IBM+Plex+Mono:wght@400;500&family=Noto+Serif+SC:wght@500;640;760&display=swap'
      }
    ]
  ],
  markdown: {
    math: true
  },
  themeConfig: {
    socialLinks: [{ icon: 'github', link: 'https://github.com/zhoujx4/llm-atlas' }],
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
              modal: {
                noResultsText: '未找到相关结果',
                resetButtonTitle: '清除查询',
                footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
              }
            }
          }
        }
      }
    }
  }
}
