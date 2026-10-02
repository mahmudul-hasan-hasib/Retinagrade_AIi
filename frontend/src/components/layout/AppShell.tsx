import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { NavItemId } from '../../types'
import { CloseIcon } from '../icons'
import { SidebarContent } from './SidebarContent'
import { TopBar } from './TopBar'

export interface AppShellProps {
  activeItem: NavItemId
  onSelectItem: (id: NavItemId) => void
  title: string
  subtitle?: string
  topBarBadge?: ReactNode
  children: ReactNode
}

/**
 * Responsive application shell.
 *
 * Desktop (lg and up): fixed 16rem sidebar + sticky top bar.
 * Phone / tablet: sticky top bar with a slide-in drawer navigation.
 */
export function AppShell({
  activeItem,
  onSelectItem,
  title,
  subtitle,
  topBarBadge,
  children,
}: AppShellProps) {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)

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
    <div className="min-h-dvh bg-canvas">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-brand-600 focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-white"
      >
        Skip to content
      </a>

      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 border-r border-line lg:block">
        <SidebarContent activeItem={activeItem} onSelect={onSelectItem} />
      </aside>

      {isDrawerOpen ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close navigation menu"
            onClick={() => setIsDrawerOpen(false)}
            className="absolute inset-0 bg-ink-900/40 backdrop-blur-[2px]"
          />
          <div className="animate-fade-in absolute inset-y-0 left-0 flex w-[17rem] max-w-[85vw] flex-col border-r border-line shadow-[var(--shadow-raised)]">
            <button
              type="button"
              onClick={() => setIsDrawerOpen(false)}
              aria-label="Close navigation menu"
              className="absolute top-4 right-3 grid size-8 place-items-center rounded-lg text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            >
              <CloseIcon className="size-4.5" />
            </button>
            <SidebarContent
              activeItem={activeItem}
              onSelect={onSelectItem}
              onNavigate={() => setIsDrawerOpen(false)}
            />
          </div>
        </div>
      ) : null}

      <div className="lg:pl-64">
        <TopBar
          title={title}
          subtitle={subtitle}
          badge={topBarBadge}
          onOpenMenu={() => setIsDrawerOpen(true)}
        />

        <main id="main-content" className="px-4 py-6 sm:px-6 sm:py-8">
          <div className="mx-auto w-full max-w-[1500px] space-y-5 sm:space-y-6">
            {children}
          </div>
        </main>

        <footer className="border-t border-line px-4 py-5 sm:px-6">
          <p className="mx-auto max-w-[1500px] text-[11px] leading-relaxed text-ink-400">
            RetinaGrade AI &middot; AI-generated interpretation. Not a standalone
            medical diagnosis.
          </p>
        </footer>
      </div>
    </div>
  )
}
