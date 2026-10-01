import type { ReactNode } from 'react'

export interface PlaceholderProps {
  label?: string
  value?: ReactNode
  hint?: ReactNode
  align?: 'start' | 'center'
  className?: string
}

/**
 * Renders a "value not available yet" row.
 *
 * This is the only way a result field is shown in this build: the dashed value
 * box is intentional - it marks data the backend cannot produce yet, so no
 * placeholder ever reads as a real measurement.
 */
export function Placeholder({
  label,
  value = '—',
  hint,
  align = 'start',
  className,
}: PlaceholderProps) {
  return (
    <div
      className={[
        'flex min-w-0 flex-col gap-1',
        align === 'center' ? 'items-center text-center' : '',
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {label ? (
        <span className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
          {label}
        </span>
      ) : null}
      <span
        className={[
          'grid min-h-9 w-full place-items-center rounded-lg border border-dashed border-ink-200 bg-ink-50/70 px-3',
          'font-mono text-sm font-medium text-ink-400',
        ].join(' ')}
      >
        {value}
      </span>
      {hint ? (
        <span className="text-[11px] leading-relaxed text-ink-400">{hint}</span>
      ) : null}
    </div>
  )
}

export interface NoticeProps {
  tone?: 'info' | 'warning'
  title: string
  children?: ReactNode
  icon?: ReactNode
  className?: string
}

const NOTICE_TONES = {
  info: 'bg-brand-50/70 ring-brand-200 text-brand-900',
  warning: 'bg-amber-50 ring-amber-200 text-amber-900',
} as const

/** Informational banner used to state clearly what is not wired up yet. */
export function Notice({
  tone = 'info',
  title,
  children,
  icon,
  className,
}: NoticeProps) {
  return (
    <div
      role="status"
      className={[
        'flex gap-3 rounded-xl px-3.5 py-3 text-xs leading-relaxed ring-1',
        NOTICE_TONES[tone],
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {icon ? (
        <span className="mt-0.5 shrink-0 [&>svg]:size-4">{icon}</span>
      ) : null}
      <div className="min-w-0">
        <p className="font-semibold">{title}</p>
        {children ? <div className="mt-0.5 opacity-90">{children}</div> : null}
      </div>
    </div>
  )
}
