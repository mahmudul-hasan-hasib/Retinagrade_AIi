import { MenuIcon } from '../icons'
import { Brand } from './Brand'

export interface TopBarProps {
  /** Empty on screens that render their own page header. */
  title: string
  subtitle?: string
  /** Desktop: expands / collapses the fixed sidebar. */
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
  /** Small screens: opens the overlay drawer. */
  isDrawerOpen: boolean
  onOpenDrawer: () => void
}

const TOGGLE_CLASSES =
  'grid size-9 shrink-0 place-items-center rounded-lg text-ink-400 transition-colors duration-150 hover:bg-white/[0.06] hover:text-ink-900'

/**
 * Slim sticky bar holding the navigation toggle at the very top-left.
 *
 * The toggle is the same three-line mark at both breakpoints but does two
 * different things, so it is rendered as two buttons that occupy the same slot:
 * below `lg` the sidebar is an overlay and the button opens the drawer, from
 * `lg` up the sidebar is docked and the button collapses or expands it. Each
 * button therefore reports the state of the thing it actually controls.
 *
 * A screen that draws its own page header passes no title, and the bar then
 * carries only the toggle (plus the mark on small screens, where the page header
 * is not in view yet).
 */
export function TopBar({
  title,
  subtitle,
  sidebarCollapsed,
  onToggleSidebar,
  isDrawerOpen,
  onOpenDrawer,
}: TopBarProps) {
  const hasTitle = title.trim().length > 0

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/85 backdrop-blur-xl">
      <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!sidebarCollapsed}
          className={`${TOGGLE_CLASSES} hidden lg:grid`}
        >
          <MenuIcon className="size-5" />
        </button>

        <button
          type="button"
          onClick={onOpenDrawer}
          aria-label="Open navigation menu"
          aria-expanded={isDrawerOpen}
          className={`${TOGGLE_CLASSES} lg:hidden`}
        >
          <MenuIcon className="size-5" />
        </button>

        {hasTitle ? (
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-[14px] font-semibold tracking-tight text-ink-900 lg:text-[15px]">
              {title}
            </h1>
            {subtitle ? (
              <p className="truncate text-[12px] text-ink-500">{subtitle}</p>
            ) : null}
          </div>
        ) : (
          <div className="lg:hidden">
            <Brand size="sm" />
          </div>
        )}
      </div>
    </header>
  )
}