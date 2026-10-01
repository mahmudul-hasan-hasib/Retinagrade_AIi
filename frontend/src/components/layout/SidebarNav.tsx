import { NAV_ITEMS } from '../../data/clinical'
import type { NavItemId } from '../../types'
import {
  DocumentIcon,
  GridIcon,
  HistoryIcon,
  LayersIcon,
  SettingsIcon,
  UploadIcon,
} from '../icons'

const NAV_ICONS = {
  dashboard: GridIcon,
  analyze: UploadIcon,
  history: HistoryIcon,
  explainability: LayersIcon,
  reports: DocumentIcon,
  settings: SettingsIcon,
} as const

export interface SidebarNavProps {
  activeItem: NavItemId
  onSelect: (id: NavItemId) => void
  onNavigate?: () => void
}

export function SidebarNav({
  activeItem,
  onSelect,
  onNavigate,
}: SidebarNavProps) {
  return (
    <nav aria-label="Primary" className="flex flex-col gap-1 p-3">
      <p className="px-3 pt-2 pb-1.5 text-[10px] font-semibold tracking-[0.12em] text-ink-400 uppercase">
        Screening
      </p>

      {NAV_ITEMS.map((item) => {
        const Icon = NAV_ICONS[item.id]
        const isActive = item.id === activeItem

        return (
          <button
            key={item.id}
            type="button"
            onClick={() => {
              onSelect(item.id)
              onNavigate?.()
            }}
            aria-current={isActive ? 'page' : undefined}
            className={[
              'group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors',
              isActive
                ? 'bg-brand-50 text-brand-800 ring-1 ring-brand-200'
                : 'text-ink-600 hover:bg-ink-50 hover:text-ink-900',
              !item.implemented ? 'opacity-70' : '',
            ].join(' ')}
          >
            <Icon
              className={[
                'size-4.5 shrink-0',
                isActive ? 'text-brand-600' : 'text-ink-400 group-hover:text-ink-600',
              ].join(' ')}
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium">
                {item.label}
              </span>
              <span className="block truncate text-[11px] text-ink-400">
                {item.hint}
              </span>
            </span>
            {!item.implemented ? (
              <span className="shrink-0 rounded-full bg-ink-100 px-1.5 py-0.5 text-[10px] font-medium text-ink-500 ring-1 ring-ink-200">
                Soon
              </span>
            ) : null}
          </button>
        )
      })}
    </nav>
  )
}
