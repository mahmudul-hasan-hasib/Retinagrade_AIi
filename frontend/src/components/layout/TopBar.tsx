import type { ReactNode } from 'react'
import { Button } from '../ui/Button'
import { MenuIcon } from '../icons'
import { Brand } from './Brand'

export interface TopBarProps {
  /** Empty on screens that render their own page header. */
  title: string
  subtitle?: string
  badge?: ReactNode
  onOpenMenu: () => void
}

export function TopBar({ title, subtitle, badge, onOpenMenu }: TopBarProps) {
  const hasTitle = title.trim().length > 0

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-surface/85 backdrop-blur-md">
      <div className="flex h-16 items-center gap-3 px-4 sm:px-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenMenu}
          aria-label="Open navigation menu"
          icon={<MenuIcon />}
          className="lg:hidden"
        />

        <div className="lg:hidden">
          <Brand size="sm" showTagline={false} />
        </div>

        {hasTitle ? (
          <div className="hidden min-w-0 flex-1 lg:block">
            <div className="flex items-center gap-2.5">
              <h1 className="truncate text-base font-semibold tracking-tight text-ink-900">
                {title}
              </h1>
              {badge}
            </div>
            {subtitle ? (
              <p className="truncate text-xs text-ink-500">{subtitle}</p>
            ) : null}
          </div>
        ) : (
          <div className="hidden min-w-0 flex-1 lg:block" />
        )}

        {badge && !hasTitle ? <div className="ml-auto">{badge}</div> : null}
      </div>

      {hasTitle ? (
        <div className="border-t border-line px-4 py-2 lg:hidden">
          <h1 className="truncate text-sm font-semibold tracking-tight text-ink-900">
            {title}
          </h1>
          {subtitle ? (
            <p className="truncate text-[11px] text-ink-500">{subtitle}</p>
          ) : null}
        </div>
      ) : null}
    </header>
  )
}