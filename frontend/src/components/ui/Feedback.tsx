export interface SkeletonProps {
  className?: string
}

/** Animated placeholder block. Decorative only - never carries a value. */
export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      aria-hidden="true"
      className={`animate-shimmer-sweep rounded-md bg-ink-100 ${className}`}
    />
  )
}

export type SpinnerTone = 'light' | 'brand' | 'muted'

export interface SpinnerProps {
  tone?: SpinnerTone
  className?: string
  /** Announced to assistive tech while work is in flight. */
  label?: string
}

const SPINNER_TONES: Record<SpinnerTone, string> = {
  light: 'border-white/35 border-t-white',
  brand: 'border-brand-200 border-t-brand-600',
  muted: 'border-ink-200 border-t-ink-400',
}

/**
 * Indeterminate activity indicator.
 *
 * Deliberately has no percentage and no fake progress: it only says that
 * something is happening, never how far along it is.
 */
export function Spinner({ tone = 'brand', className = '', label }: SpinnerProps) {
  return (
    <span
      role={label ? 'status' : undefined}
      aria-label={label}
      className={`inline-block size-4 shrink-0 animate-spin rounded-full border-2 ${
        SPINNER_TONES[tone]
      } ${className}`}
    />
  )
}

export interface ProgressBarProps {
  /** 0..1. When `null` the bar renders its empty track only. */
  value: number | null
  barClassName?: string
  trackClassName?: string
  className?: string
}

export function ProgressBar({
  value,
  barClassName = 'bg-brand-500',
  trackClassName = 'bg-ink-100',
  className = '',
}: ProgressBarProps) {
  const clamped =
    value === null || Number.isNaN(value)
      ? null
      : Math.min(1, Math.max(0, value))

  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={clamped === null ? undefined : Math.round(clamped * 100)}
      aria-label="probability"
      className={`h-2 w-full overflow-hidden rounded-full ${trackClassName} ${className}`}
    >
      {clamped === null ? (
        <div className="h-full w-full rounded-full bg-[repeating-linear-gradient(135deg,var(--color-ink-100),var(--color-ink-100)_6px,var(--color-ink-200)_6px,var(--color-ink-200)_12px)]" />
      ) : (
        <div
          className={`h-full rounded-full transition-[width] duration-500 ease-out ${barClassName}`}
          style={{ width: `${clamped * 100}%` }}
        />
      )}
    </div>
  )
}
