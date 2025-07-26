<template>
  <div class="relative h-full aspect-square bg-white dark:bg-gray-800 rounded-lg shadow-lg">
    <canvas ref="boardCanvas" class="w-full h-full"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useBoard } from '@/lib/useBoard'

const boardCanvas = ref<HTMLCanvasElement | null>(null)
let ctx: CanvasRenderingContext2D | null = null

const { board } = useBoard()

const playerColors = [
  { piece: '#EF4444', base: '#FCA5A5' }, // red-500, red-300
  { piece: '#000000', base: '#6B7280' }, // black, gray-500
  { piece: '#3B82F6', base: '#93C5FD' }, // blue-500, blue-300
  { piece: '#22C55E', base: '#86EFAC' }, // green-500, green-300
  { piece: '#A855F7', base: '#DDA0DD' }, // purple-500, plum (approx)
  { piece: '#EAB308', base: '#FDE047' }, // yellow-500, yellow-300
]

const drawBoard = () => {
  if (!ctx || !boardCanvas.value) return

  const canvas = boardCanvas.value
  const size = Math.min(canvas.width, canvas.height)

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  board.value.base.points.forEach(point => {
    minX = Math.min(minX, point.position[0]);
    maxX = Math.max(maxX, point.position[0]);
    minY = Math.min(minY, point.position[1]);
    maxY = Math.max(maxY, point.position[1]);
  });

  const boardWidth = (maxX - minX+2);
  const boardHeight = (maxY - minY+2);

  const scale = Math.min(canvas.width / boardWidth, canvas.height / boardHeight)/2; // 0.9 to add some padding
  const centerX = canvas.width / 2 - (minX + maxX) / 2 * scale;
  const centerY = canvas.height / 2 + (minY + maxY) / 2 * scale; // + because canvas y is inverted

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Draw board base
  ctx.fillStyle = '#D2B48C' // Light brown
  board.value.base.points.forEach(point => {
    const x = centerX + point.position[0] * scale
    const y = centerY - point.position[1] * scale // -y is bottom
    ctx.beginPath()
    ctx.arc(x, y, scale, 0, Math.PI * 2) // Radius 0.5 for base
    ctx.fill()
    console.log(point)
  })

  // Draw player bases
  board.value.players.forEach((player, index) => {
    if (player.base) {
      ctx.fillStyle = playerColors[index].base
      player.base.points.forEach(point => {
        const x = centerX + point.position[0] * scale
        const y = centerY - point.position[1] * scale
        ctx.beginPath()
        ctx.arc(x, y, scale, 0, Math.PI * 2) // Radius 0.5 for base
        ctx.fill()
      })
    }
  })

  // Draw pieces
  board.value.players.forEach((player, index) => {
    if (player.pieces) {
      ctx.fillStyle = playerColors[index].piece
      player.pieces.points.forEach(point => {
        const x = centerX + point.position[0] * scale
        const y = centerY - point.position[1] * scale
        ctx.beginPath()
        ctx.arc(x, y, 0.8*scale, 0, Math.PI * 2) // Radius 0.4 for pieces (slightly smaller)
        ctx.fill()
      })
    }
  })
}

onMounted(() => {
  if (boardCanvas.value) {
    ctx = boardCanvas.value.getContext('2d')
    if (ctx) {
      // Set canvas dimensions to match display size for proper scaling
      boardCanvas.value.width = boardCanvas.value.offsetWidth
      boardCanvas.value.height = boardCanvas.value.offsetHeight
      drawBoard()
    }
  }
})

watch(board, drawBoard, { deep: true })
</script>