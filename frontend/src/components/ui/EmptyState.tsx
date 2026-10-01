import type { ReactNode } from 'react'

export interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description: string
  action?: ReactNode
  className?: string
  compact?: boolean
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  compact = false,
}: EmptyStateProps) {
  return (
    <div
      className={[
        'flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-200 bg-ink-50/50 text-center',
        compact ? 'gap-2 px-4 py-6' : 'gap-3 px-6 py-12',
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {icon ? (
        <span className="grid size-10 place-items-center rounded-xl bg-surface text-ink-400 ring-1 ring-line [&>svg]:size-5">
          {icon}
        </span>
      ) : null}
      <div className="max-w-sm space-y-1">
        <p className="text-sm font-semibold text-ink-700">{title}</p>
        <p className="text-xs leading-relaxed text-ink-500">{description}</p>
      </div>
      {action}
    </div>
  )
}
