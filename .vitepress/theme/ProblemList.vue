<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../data/problems.data'

const props = defineProps<{ diff: 'easy' | 'medium' | 'hard' }>()

const meta = {
  easy:   { title: 'Easy · 简单',   icon: '🟢', desc: '入门题：kernel 基本写法与 coalesced 访存' },
  medium: { title: 'Medium · 中等', icon: '🟡', desc: '进阶题：tiling、warp shuffle 与归约范式' },
  hard:   { title: 'Hard · 困难',   icon: '🔴', desc: '高阶题：逼近手写 kernel 的性能极限' }
}

const info = computed(() => meta[props.diff])
const list = computed(() => data.filter(p => p.diff === props.diff))
</script>

<template>
  <div class="pl-wrap">
    <h1 class="pl-title">{{ info.icon }} {{ info.title }}</h1>
    <p class="pl-desc">{{ info.desc }} · 共 {{ list.length }} 题</p>
    <div class="pl-list">
      <a v-for="p in list" :key="p.url" :href="withBase(p.url)" class="pl-card">
        <span class="pl-num">#{{ p.num }}</span>
        <span class="pl-body">
          <span class="pl-name">{{ p.title }}</span>
          <span v-if="p.tags" class="pl-tags">{{ p.tags }}</span>
        </span>
        <span class="pl-arrow">→</span>
      </a>
    </div>
  </div>
</template>

<style scoped>
.pl-wrap { max-width: 1024px; margin: 0 auto; padding: 48px 24px 64px; }
.pl-title { font-size: 2rem; font-weight: 700; margin: 0 0 8px; color: var(--vp-c-text-1); }
.pl-desc { color: var(--vp-c-text-2); margin: 0 0 32px; }
.pl-list { display: flex; flex-direction: column; gap: 10px; }
.pl-card {
  display: flex; align-items: center; gap: 16px;
  padding: 14px 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  text-decoration: none;
  transition: border-color .2s, background-color .2s, transform .15s;
}
.pl-card:hover {
  border-color: var(--vp-c-brand-1);
  background-color: var(--vp-c-bg-soft);
  transform: translateX(4px);
}
.pl-num {
  flex-shrink: 0;
  font-family: var(--vp-font-family-mono);
  font-size: .9rem; font-weight: 600;
  color: var(--vp-c-brand-1);
  min-width: 44px;
}
.pl-body { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.pl-name { font-weight: 600; color: var(--vp-c-text-1); }
.pl-tags {
  font-size: .8rem; color: var(--vp-c-text-3);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pl-arrow { margin-left: auto; color: var(--vp-c-text-3); transition: color .2s; }
.pl-card:hover .pl-arrow { color: var(--vp-c-brand-1); }
</style>
