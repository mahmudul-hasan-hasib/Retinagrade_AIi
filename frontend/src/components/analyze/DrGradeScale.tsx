import { DR_CLASSES } from '../../data/clinical'

export interface DrGradeScaleProps {
  /**
   * Index of the predicted class, or `null` when it could not be resolved from
   * the response. Nothing is highlighted in that case - the scale never guesses.
   */
  activeIndex: number | null
}

/**
 * The five ICDR grades as one compact, ordered row.
 *
 * This is a legend for the headline grade, not a chart: every grade is always
 * shown at equal size, the predicted one is only tinted, and no width, length or
 * position encodes a number. Confidence and the 5-class distribution are
 * deliberately absent - they live in the API response and the Model screen.
 *
 * Labels are the user-facing grade names, never `class_0`-style keys. Below the
 * width where all five fit, the row scrolls horizontally instead of wrapping or
 * truncating, so the order 0 -> 4 always reads left to right.
 */
export function DrGradeScale({ activeIndex }: DrGradeScaleProps) {
  return (
    <div>
      <p className="text-[10px] font-semibold tracking-[0.14em] text-ink-500 uppercase">
        DR Grade
      </p>

      <ol className="scrollbar-none -mx-1 mt-3 flex gap-1.5 overflow-x-auto px-1 pb-1">
        {DR_CLASSES.map((drClass) => {
          const isActive = drClass.index === activeIndex

          return (
            <li key={drClass.index} className="shrink-0">
              <span
                aria-current={isActive ? 'true' : undefined}
                className={[
                  'inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12px] font-medium whitespace-nowrap transition-colors duration-150',
                  isActive
                    ? // The predicted step is the only crimson thing in the row;
                      // everything else stays neutral.
                      'bg-brand-500/12 text-ink-900 ring-1 ring-brand-500/40'
                    : 'bg-white/[0.03] text-ink-400 ring-1 ring-white/8',
                ].join(' ')}
              >
                <span className={isActive ? 'text-brand-700' : 'text-ink-500'}>
                  {drClass.index}
                </span>
                {drClass.shortLabel}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}