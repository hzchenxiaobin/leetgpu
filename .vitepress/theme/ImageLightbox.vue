<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const MIN_SCALE = 0.25
const MAX_SCALE = 8
const BTN_FACTOR = 1.25

const open = ref(false)
const src = ref('')
const alt = ref('')
const scale = ref(1)
const tx = ref(0)
const ty = ref(0)
const dragging = ref(false)

const imgStyle = computed(() => ({
  transform: `translate(${tx.value}px, ${ty.value}px) scale(${scale.value})`
}))

const pointers = new Map<number, { x: number; y: number }>()
let lastDist = 0
let moved = false
let downX = 0
let downY = 0

function openImage(img: HTMLImageElement) {
  src.value = img.currentSrc || img.getAttribute('src') || ''
  alt.value = img.alt || ''
  scale.value = 1
  tx.value = 0
  ty.value = 0
  open.value = true
  document.documentElement.classList.add('lightbox-open')
}

function close() {
  open.value = false
  document.documentElement.classList.remove('lightbox-open')
}

function zoomAt(factor: number, px?: number, py?: number) {
  const cx = px ?? window.innerWidth / 2
  const cy = py ?? window.innerHeight / 2
  const s0 = scale.value
  const s1 = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s0 * factor))
  if (s1 === s0) return
  const midX = window.innerWidth / 2
  const midY = window.innerHeight / 2
  tx.value = (cx - midX) - (s1 / s0) * ((cx - midX) - tx.value)
  ty.value = (cy - midY) - (s1 / s0) * ((cy - midY) - ty.value)
  scale.value = s1
  moved = true
}

function onWheel(e: WheelEvent) {
  zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY)
}

function onImgPointerDown(e: PointerEvent) {
  if (pointers.size === 0) {
    downX = e.clientX
    downY = e.clientY
    moved = false
  }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
  dragging.value = true
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()]
    lastDist = Math.hypot(a.x - b.x, a.y - b.y)
    moved = true
  }
}

function onImgPointerMove(e: PointerEvent) {
  if (!pointers.has(e.pointerId)) return
  const prev = pointers.get(e.pointerId)!
  if (pointers.size === 1) {
    tx.value += e.clientX - prev.x
    ty.value += e.clientY - prev.y
  }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()]
    const dist = Math.hypot(a.x - b.x, a.y - b.y)
    if (lastDist > 0 && dist > 0) {
      zoomAt(dist / lastDist, (a.x + b.x) / 2, (a.y + b.y) / 2)
    }
    lastDist = dist
  }
}

function onImgPointerUp(e: PointerEvent) {
  pointers.delete(e.pointerId)
  lastDist = 0
  if (pointers.size === 0) {
    dragging.value = false
    if (!moved && Math.hypot(e.clientX - downX, e.clientY - downY) < 5) close()
  }
}

function onKeydown(e: KeyboardEvent) {
  if (!open.value) return
  if (e.key === 'Escape') close()
  else if (e.key === '+' || e.key === '=') zoomAt(BTN_FACTOR)
  else if (e.key === '-') zoomAt(1 / BTN_FACTOR)
}

function onDocClick(e: MouseEvent) {
  if (open.value) return
  const t = e.target as HTMLElement
  if (t.tagName === 'IMG' && t.closest('.vp-doc')) {
    e.preventDefault()
    openImage(t as HTMLImageElement)
  }
}

onMounted(() => {
  document.addEventListener('click', onDocClick, true)
  window.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocClick, true)
  window.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="lightbox">
      <div v-if="open" class="image-lightbox" @wheel.prevent="onWheel" @click.self="close">
        <img
          :src="src"
          :alt="alt"
          :style="imgStyle"
          class="lb-img"
          :class="{ 'lb-grabbing': dragging }"
          draggable="false"
          @pointerdown="onImgPointerDown"
          @pointermove="onImgPointerMove"
          @pointerup="onImgPointerUp"
          @pointercancel="onImgPointerUp"
        >
        <div v-if="alt" class="lb-caption">{{ alt }}</div>
        <div class="lb-toolbar">
          <span class="lb-scale">{{ Math.round(scale * 100) }}%</span>
          <button class="lb-btn" title="缩小" @click="zoomAt(1 / BTN_FACTOR)">−</button>
          <button class="lb-btn" title="放大" @click="zoomAt(BTN_FACTOR)">+</button>
          <button class="lb-btn lb-close" title="关闭 (Esc)" @click="close">×</button>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.image-lightbox {
  position: fixed;
  inset: 0;
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.82);
  cursor: zoom-out;
}

.lb-img {
  max-width: calc(100vw - 96px);
  max-height: calc(100vh - 96px);
  object-fit: contain;
  cursor: grab;
  touch-action: none;
  user-select: none;
  will-change: transform;
}

.lb-grabbing { cursor: grabbing; }

.lb-caption {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 80vw;
  padding: 4px 14px;
  border-radius: 6px;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.85);
  background: rgba(0, 0, 0, 0.45);
  pointer-events: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lb-toolbar {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.lb-scale {
  min-width: 48px;
  text-align: center;
  font-size: 0.8rem;
  font-variant-numeric: tabular-nums;
  color: rgba(255, 255, 255, 0.85);
  background: rgba(0, 0, 0, 0.45);
  border-radius: 6px;
  padding: 5px 8px;
}

.lb-btn {
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 6px;
  font-size: 1.15rem;
  line-height: 1;
  color: rgba(255, 255, 255, 0.9);
  background: rgba(0, 0, 0, 0.45);
  cursor: pointer;
  transition: background 0.2s;
}

.lb-btn:hover { background: rgba(255, 255, 255, 0.2); }

.lb-close { font-size: 1.4rem; }

.lightbox-enter-active,
.lightbox-leave-active { transition: opacity 0.2s ease; }
.lightbox-enter-from,
.lightbox-leave-to { opacity: 0; }
</style>
