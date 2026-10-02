import type { HTMLAttributes, ReactNode } from 'react'

export type CardTone = 'default' | 'flat' | 'accent'

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  tone?: CardTone
  children: ReactNode
}

const TONE_CLASSES: Record<CardTone, string> = {
  default:
    'bg-surface ring-1 ring-line shadow-[var(--shadow-card)] rounded-2xl',
  flat: 'bg-surface ring-1 ring-line rounded-2xl',
  accent:
    'bg-surface ring-1 ring-brand-200 shadow-[var(--shadow-card)] rounded-2xl',
}

function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

export function Card({
  tone = 'default',
  className,
  children,
  ...rest
}: CardProps) {
  return (
    <div className={cx(TONE_CLASSES[tone], className)} {...rest}>
      {children}
    </div>
  )
}

export interface CardHeaderProps {
  icon?: ReactNode
  title: string
  subtitle?: string
  action?: ReactNode
  className?: string
}

export function CardHeader({
  icon,
  title,
  subtitle,
  action,
  className,
}: CardHeaderProps) {
  return (
    <div
      className={cx(
        'flex items-start justify-between gap-3 border-b border-line px-4 py-3.5 sm:px-5',
        className,
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        {icon ? (
          <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-brand-50 text-brand-700 ring-1 ring-brand-100">
            {icon}
          </span>
        ) : null}
        <div className="min-w-0">
          <h2 className="truncate text-[15px] font-semibold tracking-tight text-ink-900">
            {title}
          </h2>
          {subtitle ? (
            <p className="mt-0.5 text-[13px] leading-relaxed text-ink-500">
              {subtitle}
            </p>
          ) : null}
        </div>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}

export function CardBody({
  className,
  roomy = false,
  children,
}: {
  className?: string
  /** Extra breathing room for a page's primary card. */
  roomy?: boolean
  children: ReactNode
}) {
  return (
    <div className={cx(roomy ? 'px-5 py-6 sm:px-7 sm:py-7' : 'px-4 py-4 sm:px-5', className)}>
      {children}
    </div>
  )
}
