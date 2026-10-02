import type { ReactNode } from 'react'

export type BadgeTone = 'neutral' | 'brand' | 'success' | 'warning' | 'muted'

export interface BadgeProps {
  tone?: BadgeTone
  children: ReactNode
  className?: string
  title?: string
}

/**
 * Tinted-fill badges: a low-alpha colour over the dark surface for the fill and
 * ring, the matching light step for the text. Never a pale solid, which would
 * glare against a near-black page.
 */
const TONES: Record<BadgeTone, string> = {
  neutral: 'bg-white/[0.06] text-ink-800 ring-white/10',
  brand: 'bg-brand-500/12 text-brand-700 ring-brand-500/30',
  success: 'bg-emerald-500/12 text-emerald-300 ring-emerald-500/30',
  warning: 'bg-amber-500/12 text-amber-300 ring-amber-500/30',
  muted: 'bg-transparent text-ink-400 ring-white/10',
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
