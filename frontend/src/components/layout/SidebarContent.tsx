import { APP_VERSION, BACKEND_BASE_URL } from '../../data/clinical'
import type { NavItemId } from '../../types'
import { CpuIcon, ShieldIcon } from '../icons'
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

      <div className="space-y-3 border-t border-line px-4 py-4">
        <div className="rounded-xl bg-ink-50/80 px-3 py-2.5 ring-1 ring-line">
          <p className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-700">
            <CpuIcon className="size-3.5 text-ink-400" />
            Inference device
          </p>
          <p className="mt-1 font-mono text-[11px] text-ink-500">
            CPU only &middot; CUDA unused
          </p>
        </div>

        <div className="flex items-start gap-2 text-[11px] leading-relaxed text-ink-500">
          <ShieldIcon className="mt-px size-3.5 shrink-0 text-ink-400" />
          <p>
            Research tool. Not a diagnostic device. Always confirm findings with
            a qualified clinician.
          </p>
        </div>

        <div className="space-y-0.5 text-[10px] text-ink-400">
          <p className="font-mono">API {BACKEND_BASE_URL}</p>
          <p className="font-mono">v{APP_VERSION} &middot; UI preview</p>
        </div>
      </div>
    </div>
  )
}
