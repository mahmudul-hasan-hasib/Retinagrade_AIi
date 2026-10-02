import type { NavItemId } from '../../types'
import { ShieldIcon } from '../icons'
import { Brand } from './Brand'
import { SidebarNav } from './SidebarNav'

export interface SidebarContentProps {
  activeItem: NavItemId
  onSelect: (id: NavItemId) => void
  onNavigate?: () => void
  collapsed?: boolean
  /**
   * `glass` is the docked sidebar: barely tinted and blurred, so the ambient
   * glow reads faintly through it. `strong` is for the mobile drawer, which
   * overlays live content and has to stay legible.
   */
  surface?: 'glass' | 'strong'
}

/**
 * Shown at the foot of the sidebar. This is a screening aid, not a diagnostic
 * device, and the limit belongs where every run starts rather than buried in a
 * result panel.
 */
const SCREENING_CAVEAT =
  'Screening support only. Not a diagnostic device. Always confirm findings with a qualified clinician.'

/** Shared by the desktop sidebar and the mobile drawer. */
export function SidebarContent({
  activeItem,
  onSelect,
  onNavigate,
  collapsed = false,
  surface = 'glass',
}: SidebarContentProps) {
  return (
    <div
      className={[
        'flex h-full flex-col backdrop-blur-xl',
        surface === 'glass' ? 'bg-surface' : 'bg-surface-strong',
      ].join(' ')}
    >
      <div
        className={`border-b border-line ${collapsed ? 'px-2.5 py-4' : 'px-4 py-4'}`}
      >
        <Brand collapsed={collapsed} />
      </div>

      <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto">
        <SidebarNav
          activeItem={activeItem}
          onSelect={onSelect}
          onNavigate={onNavigate}
          collapsed={collapsed}
        />
      </div>

      <div
        className={`border-t border-line ${collapsed ? 'px-2.5 py-4' : 'px-4 py-4'}`}
      >
        {collapsed ? (
          <span
            title={SCREENING_CAVEAT}
            aria-label={SCREENING_CAVEAT}
            role="img"
            className="mx-auto grid size-9 place-items-center rounded-xl text-ink-500"
          >
            <ShieldIcon className="size-4" />
          </span>
        ) : (
          <div className="flex items-start gap-2 text-[11px] leading-relaxed text-ink-500">
            <ShieldIcon className="mt-px size-3.5 shrink-0 text-ink-500" />
            <p>{SCREENING_CAVEAT}</p>
          </div>
        )}
      </div>
    </div>
  )
}