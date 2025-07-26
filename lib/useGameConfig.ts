import { ref, computed, watch } from 'vue'
import Cookies from 'universal-cookie'

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

const customPresets = cookies.get('customPresets')
if (customPresets) {
  presets.value = [...presets.value, ...customPresets]
}

const selectedPreset = ref(presets.value[0])
const players = ref(JSON.parse(JSON.stringify(selectedPreset.value.players)))

const sortedPresets = computed(() => {
  const customPreset = presets.value.find(p => p.name === 'Custom')
  const otherPresets = presets.value.filter(p => p.name !== 'Custom')
  return [...otherPresets, customPreset]
})

watch(selectedPreset, (newPreset, oldPreset) => {
  if (newPreset.name !== 'Custom') {
    players.value = JSON.parse(JSON.stringify(newPreset.players))
  }
});

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
}, { deep: true });

function saveCustomPreset(customPresetName: string) {
  if (!customPresetName) return

  const newPreset = {
    name: customPresetName,
    players: JSON.parse(JSON.stringify(players.value)),
  }

  const customPresets = cookies.get('customPresets') || []
  customPresets.push(newPreset)
  cookies.set('customPresets', customPresets, { path: '/' })

  presets.value.push(newPreset)
  selectedPreset.value = newPreset
}

export function useGameConfig() {
  return {
    presets,
    selectedPreset,
    players,
    sortedPresets,
    saveCustomPreset,
  }
}
