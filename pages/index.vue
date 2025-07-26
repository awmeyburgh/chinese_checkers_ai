<script setup lang="ts">
import { ref, onMounted } from 'vue'
import Sidebar from '@/components/Sidebar.vue'
import Toolbar from '@/components/Toolbar.vue'
import Board from '@/components/Board.vue'
import { useBoard } from '@/lib/useBoard'
import { useGameConfig } from '@/lib/useGameConfig'

const boardRef = ref<InstanceType<typeof Board> | null>(null)
const { setPlayers } = useBoard()
const { selectedPreset } = useGameConfig()

onMounted(() => {
  setPlayers(selectedPreset.value.players);
  boardRef.value?.draw();
})
</script>

<template>
  <div class="flex h-screen">
    <Sidebar />
    <!-- Content Section -->
    <main class="flex-1 p-4 bg-gray-100 dark:bg-gray-900 relative flex items-center justify-center">
      <Board ref="boardRef" />
      <Toolbar :boardRef="boardRef" />
    </main>
  </div>
</template>