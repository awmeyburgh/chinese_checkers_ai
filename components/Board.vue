<template>
  <div class="relative h-full aspect-square bg-white dark:bg-gray-800 rounded-lg shadow-lg">
    <canvas ref="boardCanvas" class="w-full h-full"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const boardCanvas = ref<HTMLCanvasElement | null>(null)
let ctx: CanvasRenderingContext2D | null = null

onMounted(() => {
  if (boardCanvas.value) {
    ctx = boardCanvas.value.getContext('2d')
    if (ctx) {
      // Drawing code goes here
      // Example:
      // ctx.fillStyle = 'red'
      // ctx.fillRect(0, 0, boardCanvas.value.width, boardCanvas.value.height)
    }
  }
})
</script>