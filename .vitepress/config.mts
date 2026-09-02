import { defineConfig } from 'vitepress'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import katex from 'markdown-it-katex'

const root = path.dirname(fileURLToPath(import.meta.url))

/** 从 solutions/<diff>/ 目录读取题目列表：题号排序，标题取自每篇 H1 */
function loadGroup(diff: string) {
  const abs = path.resolve(root, '../solutions', diff)
  if (!fs.existsSync(abs)) return []
  return fs.readdirSync(abs)
    .filter(f => f.endsWith('.md'))
    .sort((a, b) => parseInt(a) - parseInt(b))
    .map(f => {
      const num = f.match(/^(\d+)-/)?.[1] ?? ''
      const h1 = fs.readFileSync(path.join(abs, f), 'utf-8').match(/^#\s+(.+)$/m)?.[1] ?? f
      const title = h1.replace(/^LeetGPU\s*/i, '').replace(/\s*题解\s*$/, '')
      return { text: `#${num} ${title}`, link: `/solutions/${diff}/${f.replace(/\.md$/, '')}` }
    })
}

// 全局题目顺序（easy → medium → hard），用于生成 上一题/下一题
const order = ['easy', 'medium', 'hard'].flatMap(d => loadGroup(d))

export default defineConfig({
  title: 'LeetGPU 题解',
  description: 'CUDA Kernel 编程题解合集',
  lang: 'zh-CN',
  base: '/leetgpu/',
  // 旧版题解目录与文档不纳入站点构建（保留在仓库中作存档）
  srcExclude: [
    'easy/**', 'medium/**', 'hard/**',
    'README.md', 'SKILL.md', 'cuda-interview-notes.md',
    'build/**', 'static/**'
  ],
  outDir: './dist',
  ignoreDeadLinks: true, // 正文里有指向站外其他仓库的相对链接
  lastUpdated: false,

  head: [
    ['link', { rel: 'icon', href: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="0.9em" font-size="90">⚡</text></svg>' }]
  ],

  markdown: {
    config: (md) => { md.use(katex) },
    lineNumbers: true,
    languageAlias: { cuda: 'cpp' }
  },

  // 无侧边栏 + 无右侧目录：题解页为纯单栏阅读；按题号顺序自动推导 上一题/下一题
  transformPageData(pageData) {
    const idx = order.findIndex(o => pageData.relativePath === o.link.slice(1) + '.md')
    if (idx >= 0) {
      pageData.frontmatter.aside = false
      const prev = order[idx - 1]
      const next = order[idx + 1]
      pageData.frontmatter.prev = prev ? { text: prev.text, link: prev.link } : false
      pageData.frontmatter.next = next ? { text: next.text, link: next.link } : false
    }
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'Easy', link: '/easy.html' },
      { text: 'Medium', link: '/medium.html' },
      { text: 'Hard', link: '/hard.html' },
      { text: 'GitHub', link: 'https://github.com/hzchenxiaobin/leetgpu' }
    ],

    sidebar: false,

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详细列表',
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    docFooter: { prev: '上一题', next: '下一题' },
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部'
  }
})
