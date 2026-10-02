import { NAV_ITEMS } from '../../data/clinical'
import type { NavItemId } from '../../types'
import { CpuIcon, LayersIcon, UploadIcon } from '../icons'

const NAV_ICONS = {
  analyze: UploadIcon,
  explainability: LayersIcon,
  model: CpuIcon,
} as const

export interface SidebarNavProps {
  activeItem: NavItemId
  onSelect: (id: NavItemId) => void
  onNavigate?: () => void
  /** Icon-only rail: the labels move to `title` / `aria-label`. */
  collapsed?: boolean
}

/**
 * The three destinations, in workflow order, and nothing else.
 *
 * Each row is one button that is either active or not, with no group headings or
 * secondary captions: the sidebar's only job is navigation. Crimson marks the
 * active row; every other row stays neutral until hovered.
 */
export function SidebarNav({
  activeItem,
  onSelect,
  onNavigate,
  collapsed = false,
}: SidebarNavProps) {
  return (
    <nav
      aria-label="Primary"
      className={
        collapsed
          ? 'flex flex-col items-center gap-1.5 p-2.5'
          : 'flex flex-col gap-1 p-3'
      }
    >
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
            // With no label on screen the accessible name has to come from
            // somewhere, so the collapsed rail carries it as a tooltip too.
            aria-label={collapsed ? item.label : undefined}
            title={collapsed ? item.label : undefined}
            className={[
              'group rounded-xl transition-colors duration-150',
              collapsed
                ? 'grid size-10 place-items-center'
                : 'flex w-full items-center gap-3 px-3 py-2.5 text-left',
              isActive
                ? 'bg-brand-500/10 text-ink-900 ring-1 ring-brand-500/25'
                : 'text-ink-400 hover:bg-white/[0.04] hover:text-ink-800',
            ].join(' ')}
          >
            <Icon
              className={[
                'size-[18px] shrink-0 transition-colors duration-150',
                isActive
                  ? 'text-brand-700'
                  : 'text-ink-500 group-hover:text-ink-400',
              ].join(' ')}
            />

            {collapsed ? null : (
              <span className="min-w-0 flex-1 truncate text-sm font-medium">
                {item.label}
              </span>
            )}
          </button>
        )
      })}
    </nav>
  )
}