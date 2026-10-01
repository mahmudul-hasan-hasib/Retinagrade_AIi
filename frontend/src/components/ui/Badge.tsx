import type { ReactNode } from 'react'

export type BadgeTone = 'neutral' | 'brand' | 'success' | 'warning' | 'muted'

export interface BadgeProps {
  tone?: BadgeTone
  children: ReactNode
  className?: string
  title?: string
}

const TONES: Record<BadgeTone, string> = {
  neutral: 'bg-ink-100 text-ink-700 ring-ink-200',
  brand: 'bg-brand-50 text-brand-700 ring-brand-200',
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  warning: 'bg-amber-50 text-amber-700 ring-amber-200',
  muted: 'bg-transparent text-ink-400 ring-ink-200',
}

export function Badge({
  tone = 'neutral',
  children,
  className,
  title,
}: BadgeProps) {
  return (
    <span
      title={title}
      className={[
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 whitespace-nowrap',
        TONES[tone],
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </span>
  )
}
