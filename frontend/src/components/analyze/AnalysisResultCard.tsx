import type { DrClass } from '../../types'
import { Badge } from '../ui/Badge'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { DrGradeScale } from './DrGradeScale'

export interface AnalysisResultCardProps {
  /** Final predicted grade, e.g. `Moderate Diabetic Retinopathy`. */
  gradeTitle: string | null
  /** The matching entry from `DR_CLASSES`, or `null` if unresolved. */
  predictedClass: DrClass | null
  /** Short modality name shown as a badge, e.g. `CFP`. */
  modalityLabel: string
  fileName: string | null
  /** Navigates to the Explainability screen. */
  onViewExplainability: () => void
}

/**
 * The outcome of one screening, in plain language.
 *
 * Reading order is deliberate: the grade is the largest text on the card and is
 * set in near-white, the compact `Grade 2 · Moderate` repeats it one step below
 * as secondary copy, and the five-step scale underneath is a quiet legend with
 * the predicted step picked out in crimson. Confidence, per-class probabilities,
 * inference time, checkpoint name and the raw response are all absent - none of
 * them changes what a clinician needs to read here.
 *
 * The grade string is the server's own label; nothing here re-grades, re-ranks or
 * re-interprets it. The card is rendered only once `/predict` has succeeded.
 */
export function AnalysisResultCard({
  gradeTitle,
  predictedClass,
  modalityLabel,
  fileName,
  onViewExplainability,
}: AnalysisResultCardProps) {
  return (
    <Card>
      <CardHeader
        title="Analysis Result"
        action={<Badge tone="brand">{modalityLabel}</Badge>}
      />

      <CardBody className="space-y-7">
        <div className="rounded-2xl border border-line bg-white/[0.03] px-5 py-8 text-center">
          <h3 className="text-[13px] font-medium tracking-[0.14em] text-ink-500 uppercase">
            Predicted grade
          </h3>

          <p className="mt-3 text-[30px] leading-[1.12] font-bold tracking-tight text-balance text-ink-900 sm:text-[40px]">
            {gradeTitle ?? '—'}
          </p>

          {predictedClass ? (
            <p className="mt-3 text-[14px] font-medium text-ink-500">
              Grade {predictedClass.index} &middot; {predictedClass.label}
            </p>
          ) : null}
        </div>

        <DrGradeScale activeIndex={predictedClass?.index ?? null} />

        <div className="flex flex-col items-center gap-4 border-t border-line pt-5 sm:flex-row sm:justify-between">
          <p
            className="truncate text-[12px] text-ink-400"
            title={fileName ?? undefined}
          >
            {fileName}
          </p>

          <button
            type="button"
            onClick={onViewExplainability}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-2 text-[13px] font-medium text-brand-700 transition-colors duration-150 hover:bg-brand-500/10 hover:text-brand-800"
          >
            View Explainability
            <span aria-hidden="true">&rarr;</span>
          </button>
        </div>
      </CardBody>
    </Card>
  )
}