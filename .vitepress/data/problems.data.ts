import { createContentLoader } from 'vitepress'

export interface Problem {
  url: string
  diff: 'easy' | 'medium' | 'hard'
  num: number
  title: string
  tags: string
}

declare const data: Problem[]
export { data }

export default createContentLoader('solutions/*/*.md', {
  includeSrc: true,
  transform(raw): Problem[] {
    return raw
      .map(page => {
        const m = page.url.match(/\/solutions\/(easy|medium|hard)\/(\d+)-(.+)/)
        if (!m) return null
        const src = page.src ?? ''
        const h1 = src.match(/^#\s+(.+)$/m)?.[1] ?? m[3]
        const title = h1.replace(/^LeetGPU\s*/i, '').replace(/\s*题解\s*$/, '').trim()
        const tags = (src.match(/\*\*标签\*\*[：:]\s*(.+)/)?.[1] ?? '').replace(/`/g, '').trim()
        return {
          url: page.url,
          diff: m[1] as Problem['diff'],
          num: parseInt(m[2]),
          title,
          tags
        }
      })
      .filter((p): p is Problem => p !== null)
      .sort((a, b) => a.num - b.num)
  }
})
