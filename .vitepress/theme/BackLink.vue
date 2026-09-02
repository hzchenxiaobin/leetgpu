<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, withBase } from 'vitepress'

const route = useRoute()
const label: Record<string, string> = {
  easy: 'Easy · 简单',
  medium: 'Medium · 中等',
  hard: 'Hard · 困难'
}
const diff = computed(() => route.path.match(/\/solutions\/(easy|medium|hard)\//)?.[1])
</script>

<template>
  <nav v-if="diff" class="back-nav">
    <a :href="withBase(`/${diff}.html`)">← 返回 {{ label[diff] }} 列表</a>
  </nav>
</template>

<style scoped>
.back-nav { margin-bottom: 20px; }
.back-nav a {
  font-size: .88rem;
  color: var(--vp-c-text-2);
  text-decoration: none;
  transition: color .2s;
}
.back-nav a:hover { color: var(--vp-c-brand-1); }
</style>
