import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { NavItemId } from '../../types'
import { AppBackground } from '../background/AppBackground'
import { CloseIcon } from '../icons'
import { SidebarContent } from './SidebarContent'
import { TopBar } from './TopBar'

export interface AppShellProps {
  activeItem: NavItemId
  onSelectItem: (id: NavItemId) => void
  title: string
  subtitle?: string
  children: ReactNode
}

/**
 * Session-scoped, so a reload keeps the choice but a new tab starts wide open.
 * `sessionStorage` (not `localStorage`) keeps it out of the user's long-term
 * preferences - this is a per-visit layout choice, not a setting.
 */
const SIDEBAR_STORAGE_KEY = 'retinagrade:sidebar-collapsed'

/** Expanded width, and the narrow rail: enough for a 40px icon button. */
const SIDEBAR_WIDTH = 'w-64'
const SIDEBAR_RAIL_WIDTH = 'w-[4.5rem]'
const CONTENT_INSET = 'lg:pl-64'
const CONTENT_INSET_RAIL = 'lg:pl-[4.5rem]'

function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

function readCollapsedPreference(): boolean {
  try {
    return window.sessionStorage.getItem(SIDEBAR_STORAGE_KEY) === '1'
  } catch {
    // Storage can be blocked (private mode, sandboxed frame). Defaulting to the
    // expanded sidebar is the safe reading; nothing else depends on this.
    return false
  }
}

function writeCollapsedPreference(collapsed: boolean): void {
  try {
    window.sessionStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? '1' : '0')
  } catch {
    // Same as above: failing to remember the preference must not break layout.
  }
}

/**
 * Responsive application shell.
 *
 * Desktop (lg and up): a fixed sidebar that toggles between a full 16rem column
 * and a narrow icon-only rail, with the content inset animating alongside it.
 * Phone / tablet: no docked sidebar at all - the same toggle opens an overlay
 * drawer that closes as soon as a destination is chosen.
 */
export function AppShell({
  activeItem,
  onSelectItem,
  title,
  subtitle,
  children,
}: AppShellProps) {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(readCollapsedPreference)

  useEffect(() => {
    writeCollapsedPreference(isSidebarCollapsed)
  }, [isSidebarCollapsed])

  const toggleSidebar = useCallback(() => {
    setIsSidebarCollapsed((collapsed) => !collapsed)
  }, [])

  // Close the drawer on Escape.
  useEffect(() => {
    if (!isDrawerOpen) return
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setIsDrawerOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isDrawerOpen])

  // Prevent body scroll while the drawer is open.
  useEffect(() => {
    if (!isDrawerOpen) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [isDrawerOpen])

  return (
    /* Transparent shell: the ambient background paints the page, so the cards
       float over it instead of sitting on an opaque fill. */
    <div className="relative min-h-dvh">
      <AppBackground />

      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-brand-600 focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-white"
      >
        Skip to content
      </a>

      <aside
        className={cx(
          'fixed inset-y-0 left-0 z-40 hidden border-r border-line lg:block',
          'transition-[width] duration-200 ease-out',
          isSidebarCollapsed ? SIDEBAR_RAIL_WIDTH : SIDEBAR_WIDTH,
        )}
      >
        <SidebarContent
          activeItem={activeItem}
          onSelect={onSelectItem}
          collapsed={isSidebarCollapsed}
        />
      </aside>

      {isDrawerOpen ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close navigation menu"
            onClick={() => setIsDrawerOpen(false)}
            className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
          />
          <div className="animate-fade-in absolute inset-y-0 left-0 flex w-[17rem] max-w-[85vw] flex-col border-r border-line shadow-[var(--shadow-raised)]">
            <button
              type="button"
              onClick={() => setIsDrawerOpen(false)}
              aria-label="Close navigation menu"
              className="absolute top-4 right-3 grid size-8 place-items-center rounded-lg text-ink-500 transition-colors hover:bg-white/[0.06] hover:text-ink-900"
            >
              <CloseIcon className="size-4.5" />
            </button>
            {/* An overlay always opens wide: the rail is for docked navigation. */}
            <SidebarContent
              activeItem={activeItem}
              onSelect={onSelectItem}
              onNavigate={() => setIsDrawerOpen(false)}
              surface="strong"
            />
          </div>
        </div>
      ) : null}

      <div
        className={cx(
          'transition-[padding] duration-200 ease-out',
          isSidebarCollapsed ? CONTENT_INSET_RAIL : CONTENT_INSET,
        )}
      >
        <TopBar
          title={title}
          subtitle={subtitle}
          sidebarCollapsed={isSidebarCollapsed}
          onToggleSidebar={toggleSidebar}
          isDrawerOpen={isDrawerOpen}
          onOpenDrawer={() => setIsDrawerOpen(true)}
        />

        <main id="main-content" className="px-4 py-8 sm:px-6 sm:py-10">
          <div className="mx-auto w-full max-w-[1500px]">{children}</div>
        </main>

        <footer className="border-t border-line px-4 py-5 sm:px-6">
          <p className="mx-auto max-w-[1500px] text-[11px] leading-relaxed text-ink-400">
            RetinaGrade AI &middot; Diabetic Retinopathy Screening
          </p>
        </footer>
      </div>
    </div>
  )
}