import { Badge } from '../ui/Badge'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { CheckIcon } from '../icons'

export interface AnalysisResultCardProps {
  /** Final predicted grade, e.g. `Moderate Diabetic Retinopathy`. */
  gradeTitle: string | null
  /** Short modality name shown as a badge, e.g. `CFP`. */
  modalityLabel: string
  fileName: string | null
  /** Navigates to the Explainability screen. */
  onViewExplainability: () => void
}

/**
 * The headline outcome of one screening.
 *
 * Exactly one thing is shown: the grade the model predicted. Confidence,
 * per-class probabilities, inference time, checkpoint name and the raw response
 * are deliberately absent - they remain available through the API and the Model
 * screen, and none of them changes what a clinician needs to read here.
 *
 * The grade string is the server's own label; nothing here re-grades, re-ranks or
 * re-interprets it.
 */
export function AnalysisResultCard({
  gradeTitle,
  modalityLabel,
  fileName,
  onViewExplainability,
}: AnalysisResultCardProps) {
  return (
    <Card tone="accent">
      <CardHeader
        title="Analysis Result"
        action={<Badge tone="brand">{modalityLabel}</Badge>}
      />

      <CardBody className="space-y-5">
        <div className="rounded-[14px] border border-brand-100 bg-brand-50/40 px-5 py-6 text-center">
          <p className="flex items-center justify-center gap-1.5 text-[11px] font-medium tracking-wide text-brand-700 uppercase">
            <CheckIcon className="size-3.5" />
            Predicted grade
          </p>
          <p className="mt-2.5 text-3xl font-bold tracking-tight text-balance text-ink-900 sm:text-4xl">
            {gradeTitle ?? '—'}
          </p>
          <p className="mt-3 text-[13px] leading-relaxed text-ink-500">
            The AI model identified patterns associated with this grade.
          </p>
        </div>

        <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-between">
          <p
            className="truncate text-[12px] text-ink-400"
            title={fileName ?? undefined}
          >
            {fileName}
          </p>
          <button
            type="button"
            onClick={onViewExplainability}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg text-[13px] font-medium text-brand-700 transition-colors hover:text-brand-800"
          >
            View Explainability
            <span aria-hidden="true">&rarr;</span>
          </button>
        </div>
      </CardBody>
    </Card>
  )
}