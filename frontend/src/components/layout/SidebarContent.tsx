import type { NavItemId } from '../../types'
import { ShieldIcon } from '../icons'
import { Brand } from './Brand'
import { SidebarNav } from './SidebarNav'

export interface SidebarContentProps {
  activeItem: NavItemId
  onSelect: (id: NavItemId) => void
  onNavigate?: () => void
}

/** Shared by the desktop sidebar and the mobile drawer. */
export function SidebarContent({
  activeItem,
  onSelect,
  onNavigate,
}: SidebarContentProps) {
  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="border-b border-line px-4 py-4">
        <Brand />
      </div>

      <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto">
        <SidebarNav
          activeItem={activeItem}
          onSelect={onSelect}
          onNavigate={onNavigate}
        />
      </div>

      <div className="border-t border-line px-4 py-4">
        <div className="flex items-start gap-2 text-[11px] leading-relaxed text-ink-500">
          <ShieldIcon className="mt-px size-3.5 shrink-0 text-ink-400" />
          <p>
            Screening support only. Not a diagnostic device. Always confirm
            findings with a qualified clinician.
          </p>
        </div>
      </div>
    </div>
  )
}