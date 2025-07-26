<script setup lang="ts">
import { ref } from 'vue'
import { Button } from '@/components/ui/button'
import GamePanel from './sidebar/GamePanel.vue'
import ModelsPanel from './sidebar/ModelsPanel.vue'
import { Swords, BrainCircuit, Gamepad2 } from 'lucide-vue-next'

const isCollapsed = ref(false)
const activePanel = ref<'Game' | 'Models' | null>('Game')

function selectPanel(panel: 'Game' | 'Models') {
  isCollapsed.value = false
  activePanel.value = panel
}

function toggleCollapse() {
  isCollapsed.value = true
  activePanel.value = null
}
</script>

<template>
  <div class="flex h-full">
    <div
      :class="[
        'flex flex-col justify-between border-r bg-gray-100/40 p-2 transition-all duration-300 ease-in-out',
        isCollapsed ? 'w-16' : 'w-64',
      ]"
    >
      <div>
        <Button
          :variant="activePanel === null ? 'secondary' : 'ghost'"
          class="flex w-full items-center justify-start gap-x-2"
          @click="toggleCollapse"
        >
          <Gamepad2
            :class="['h-5 w-5', { 'fill-current': activePanel === null }]"
          />
          <span v-if="!isCollapsed">Chinese Checkers AI</span>
        </Button>
        <Button
          :variant="activePanel === 'Game' ? 'secondary' : 'ghost'"
          class="flex w-full items-center justify-start gap-x-2"
          @click="selectPanel('Game')"
        >
          <Swords
            :class="['h-5 w-5', { 'fill-current': activePanel === 'Game' }]"
          />
          <span v-if="!isCollapsed">Game</span>
        </Button>
        <Button
          :variant="activePanel === 'Models' ? 'secondary' : 'ghost'"
          class="flex w-full items-center justify-start gap-x-2"
          @click="selectPanel('Models')"
        >
          <BrainCircuit
            :class="['h-5 w-5', { 'fill-current': activePanel === 'Models' }]"
          />
          <span v-if="!isCollapsed">Models</span>
        </Button>
      </div>
    </div>
    <div v-if="activePanel" class="w-80 border-r p-4">
      <GamePanel v-if="activePanel === 'Game'" />
      <ModelsPanel v-if="activePanel === 'Models'" />
    </div>
  </div>
</template>