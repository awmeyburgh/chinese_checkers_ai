<script setup lang="ts">
import Sidebar from '@/components/Sidebar.vue'
import Toolbar from '@/components/Toolbar.vue'
import Board from '@/components/Board.vue'
</script>

<template>
  <div class="flex h-screen">
    <Sidebar />
    <!-- Content Section -->
    <main class="flex-1 p-4 bg-gray-100 dark:bg-gray-900 relative flex items-center justify-center">
      <Board />
      <Toolbar />
    </main>
  </div>
</template>