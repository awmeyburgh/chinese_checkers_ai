<script setup lang="ts">
import { ref, watch, onMounted, computed } from 'vue'
import Cookies from 'universal-cookie'
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

const cookies = new Cookies()

const presets = ref([
  {
    name: '1v1 - Humans',
    players: [
      { enabled: true, player: 1, color: 'text-red-500', controller: 'Human' },
      { enabled: false, player: 2, color: 'text-black', controller: 'Human' },
      { enabled: false, player: 3, color: 'text-blue-500', controller: 'Human' },
      { enabled: true, player: 4, color: 'text-green-500', controller: 'Human' },
      { enabled: false, player: 5, color: 'text-purple-500', controller: 'Human' },
      { enabled: false, player: 6, color: 'text-yellow-500', controller: 'Human' },
    ],
  },
  {
    name: 'Custom',
    players: [
      { enabled: true, player: 1, color: 'text-red-500', controller: 'Human' },
      { enabled: false, player: 2, color: 'text-black', controller: 'Human' },
      { enabled: false, player: 3, color: 'text-blue-500', controller: 'Human' },
      { enabled: true, player: 4, color: 'text-green-500', controller: 'Human' },
      { enabled: false, player: 5, color: 'text-purple-500', controller: 'Human' },
      { enabled: false, player: 6, color: 'text-yellow-500', controller: 'Human' },
    ],
  },
])

const sortedPresets = computed(() => {
  const customPreset = presets.value.find(p => p.name === 'Custom')
  const otherPresets = presets.value.filter(p => p.name !== 'Custom')
  return [...otherPresets, customPreset]
})

onMounted(() => {
  const customPresets = cookies.get('customPresets')
  if (customPresets) {
    presets.value = [...presets.value, ...customPresets]
  }
})

const selectedPreset = ref(presets.value[0])
const players = ref(JSON.parse(JSON.stringify(selectedPreset.value.players)))
const customPresetName = ref('')

watch(selectedPreset, (newPreset, oldPreset) => {
  if (newPreset.name !== 'Custom') {
    players.value = JSON.parse(JSON.stringify(newPreset.players))
  }
})

watch(players, (newPlayers) => {
  const presetPlayers = JSON.stringify(selectedPreset.value.players)
  const currentPlayers = JSON.stringify(newPlayers)

  if (presetPlayers !== currentPlayers && selectedPreset.value.name !== 'Custom') {
    const customPreset = presets.value.find(p => p.name === 'Custom')
    if (customPreset) {
      customPreset.players = JSON.parse(JSON.stringify(newPlayers))
      selectedPreset.value = customPreset
    }
  }
}, { deep: true })

function saveCustomPreset() {
  if (!customPresetName.value) return

  const newPreset = {
    name: customPresetName.value,
    players: JSON.parse(JSON.stringify(players.value)),
  }

  const customPresets = cookies.get('customPresets') || []
  customPresets.push(newPreset)
  cookies.set('customPresets', customPresets, { path: '/' })

  presets.value.push(newPreset)
  selectedPreset.value = newPreset
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