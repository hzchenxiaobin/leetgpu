import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import ProblemList from './ProblemList.vue'
import BackLink from './BackLink.vue'
import 'katex/dist/katex.min.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ProblemList', ProblemList)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => h(BackLink)
    })
  }
}
