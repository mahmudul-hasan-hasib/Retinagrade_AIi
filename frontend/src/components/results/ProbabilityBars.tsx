import { DR_CLASSES, SEVERITY_STYLES } from '../../data/clinical'
import { ProgressBar } from '../ui/Feedback'

export interface ProbabilityBarsProps {
  /**
   * Per-class probabilities keyed by `DR_CLASSES[i].key`. `null` until real
   * inference runs - the bars then render as empty hatched tracks so they can
   * never be mistaken for measured values.
   */
  probabilities: Record<string, number> | null
  /** Index of the predicted class, or `null` when unknown. */
  predictedIndex?: number | null
}

export function ProbabilityBars({
  probabilities,
  predictedIndex = null,
}: ProbabilityBarsProps) {
  return (
    <ul className="space-y-3.5">
      {DR_CLASSES.map((drClass) => {
        const value = probabilities?.[drClass.key] ?? null
        const severity = SEVERITY_STYLES[drClass.severity]
        const isPredicted = predictedIndex === drClass.index

        return (
          <li key={drClass.key} className="space-y-1.5">
            <div className="flex items-baseline justify-between gap-3">
              <span className="flex min-w-0 items-center gap-2">
                <span className={`size-1.5 shrink-0 rounded-full ${severity.dot} opacity-40`} />
                <span
                  className={`truncate text-xs font-medium ${
                    isPredicted ? 'text-ink-900' : 'text-ink-600'
                  }`}
                >
                  <span className="mr-1.5 font-mono text-[11px] text-ink-400">
                    {drClass.index}
                  </span>
                  {drClass.label}
                </span>
                {isPredicted ? (
                  <span className="rounded-full bg-brand-50 px-1.5 py-px text-[10px] font-semibold text-brand-700 ring-1 ring-brand-200">
                    top
                  </span>
                ) : null}
              </span>

              <span className="shrink-0 font-mono text-xs text-ink-500">
                {value === null ? (
                  <span className="text-ink-300">—</span>
                ) : (
                  `${(value * 100).toFixed(1)}%`
                )}
              </span>
            </div>

            <ProgressBar
              value={value}
              barClassName={severity.bar}
              className="h-1.5"
            />
          </li>
        )
      })}
    </ul>
  )
}
