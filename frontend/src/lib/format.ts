/** Format a 0..1 ratio as a percentage string, or a dash when unavailable. */
export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }
  return `${(value * 100).toFixed(1)}%`
}

/** Placeholder glyph for any numeric result that does not exist yet. */
export const NO_VALUE = '—'

/** Human-readable byte size, e.g. `2.4 MB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`
}

/** `HH:MM:SS` from a timestamp, or a dash when the timestamp is missing. */
export function formatClock(iso: string | null): string {
  if (!iso) return NO_VALUE
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return NO_VALUE
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** `12 Mar 2026`, or a dash when the timestamp is missing. */
export function formatDate(iso: string | null): string {
  if (!iso) return NO_VALUE
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return NO_VALUE
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}
