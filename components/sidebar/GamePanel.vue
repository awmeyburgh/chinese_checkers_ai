<script setup lang="ts">
import { ref, watch } from 'vue'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useGameConfig } from '@/lib/useGameConfig'

const { presets, selectedPreset, players, sortedPresets, saveCustomPreset: saveGameConfigPreset } = useGameConfig()

const customPresetName = ref('')

function saveCustomPreset() {
  if (!customPresetName.value) return
  saveGameConfigPreset(customPresetName.value)
  customPresetName.value = ''
}
</script>

<template>
  <div class="p-4 space-y-4">
    <h2 class="text-lg font-bold">
      Game
    </h2>
    <div class="flex items-center justify-between">
      <Label for="preset">Preset</Label>
      <Select id="preset" v-model="selectedPreset">
        <SelectTrigger class="w-48">
          <SelectValue placeholder="Select a preset" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem v-for="preset in sortedPresets" :key="preset.name" :value="preset">
              {{ preset.name }}
            </SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>

    <div class="space-y-2">
      <div v-for="player in players" :key="player.player" class="flex items-center space-x-2">
        <Checkbox v-model="player.enabled" />
        <Label :class="player.color" class="w-20">Player {{ player.player }}</Label>
        <div class="flex-grow"></div>
        <Select v-model="player.controller">
          <SelectTrigger class="w-32">
            <SelectValue placeholder="Select a controller" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectItem value="Human">
                Human
              </SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
    </div>

    <div v-if="selectedPreset.name === 'Custom'" class="space-y-2">
      <div class="flex items-center space-x-2">
        <Input v-model="customPresetName" placeholder="Name" />
        <Button @click="saveCustomPreset">Save</Button>
      </div>
    </div>
  </div>
</template>