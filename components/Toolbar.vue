<script setup lang="ts">
import { ref } from 'vue'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { RotateCcw, Play, Pause, RefreshCw, RotateCw } from 'lucide-vue-next'

const isPlaying = ref(false)

const togglePlay = () => {
  isPlaying.value = !isPlaying.value
}
</script>

<template>
  <div class="absolute bottom-4 left-1/2 -translate-x-1/2">
    <div class="flex items-center gap-2 p-2 bg-white dark:bg-black rounded-full shadow-xl border border-gray-200 dark:border-gray-700">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger as-child>
            <Button variant="ghost" size="icon">
              <RotateCcw class="w-6 h-6" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Rotate Left</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger as-child>
            <Button variant="ghost" size="icon" @click="togglePlay">
              <Pause v-if="isPlaying" class="w-6 h-6" />
              <Play v-else class="w-6 h-6" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{{ isPlaying ? 'Pause' : 'Play' }}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger as-child>
            <Button variant="ghost" size="icon">
              <RefreshCw class="w-6 h-6" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Restart</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger as-child>
            <Button variant="ghost" size="icon">
              <RotateCw class="w-6 h-6" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Rotate Right</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  </div>
</template>