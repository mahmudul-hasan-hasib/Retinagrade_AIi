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
