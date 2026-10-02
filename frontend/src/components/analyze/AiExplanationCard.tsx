import type { ReactNode } from 'react'
import type { ExplainApiResponse } from '../../api/client'
import type { ExplainStatus } from '../reports/AiReportCard'
import { SparkIcon } from '../icons'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { Skeleton, Spinner } from '../ui/Feedback'

export interface AiExplanationCardProps {
  /** Lifecycle of `POST /explain`. */
  status: ExplainStatus
  /** Gemini's prose, rendered verbatim. */
  explanation: ExplainApiResponse | null
  /** Whether a retry request can be sent right now. */
  canRetry: boolean
  onRetry: () => void
}

/** Fixed, deliberately small caveat. Never a headline, never a banner. */
const DISCLAIMER = 'AI-generated interpretation. Not a standalone medical diagnosis.'

/**
 * Short Gemini explanation of the grade shown directly above.
 *
 * This card carries prose and nothing else. The prompt, the request payload, the
 * model name, latency, confidence and per-class probabilities behind the text
 * stay out of the screening view, and the clinical wording is never composed,
 * trimmed or paraphrased in the browser - `explanation.explanation` is shown
 * exactly as returned.
 *
 * Independent of the prediction: while Gemini is still answering, the grade
 * above is already final and remains visible, and a Gemini failure never
 * invalidates it.
 */
export function AiExplanationCard({
  status,
  explanation,
  canRetry,
  onRetry,
}: AiExplanationCardProps) {
  const prose = status === 'ready' ? (explanation?.explanation ?? null) : null
  const isUnavailable =
    status === 'error' || status === 'unavailable' || (status === 'ready' && !prose)

  return (
    <Card>
      <CardHeader
        icon={<SparkIcon />}
        title="AI Explanation"
        subtitle="A concise explanation based on the model result."
      />

      <CardBody className="space-y-4">
        {prose ? (
          <p className="text-[15px] leading-relaxed whitespace-pre-line text-ink-700">
            {prose}
          </p>
        ) : null}

        {!prose && isUnavailable ? (
          <div className="space-y-1.5">
            <p className="text-[15px] font-medium text-ink-800">
              AI explanation unavailable
            </p>
            <p className="text-[13px] leading-relaxed text-ink-500">
              Your analysis result is still available.
            </p>
          </div>
        ) : null}

        {!prose && !isUnavailable ? <PendingExplanation /> : null}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3.5">
          <p className="text-[11px] leading-relaxed text-ink-400">{DISCLAIMER}</p>
          {prose ? (
            <RetryAction canRetry={canRetry} onRetry={onRetry} label="Regenerate" />
          ) : null}
          {isUnavailable ? (
            <RetryAction canRetry={canRetry} onRetry={onRetry} label="Try again" />
          ) : null}
        </div>
      </CardBody>
    </Card>
  )
}

/** Loading state. A spinner plus skeletons - never a percentage. */
function PendingExplanation() {
  return (
    <div className="space-y-3" aria-live="polite">
      <p className="flex items-center gap-2 text-[13px] font-medium text-ink-600">
        <Spinner tone="brand" />
        Generating AI explanation...
      </p>
      <div className="space-y-2">
        <Skeleton className="h-2.5 w-full" />
        <Skeleton className="h-2.5 w-11/12" />
        <Skeleton className="h-2.5 w-4/5" />
      </div>
    </div>
  )
}

/** Small secondary action, kept out of the prose area. */
function RetryAction({
  canRetry,
  onRetry,
  label,
}: {
  canRetry: boolean
  onRetry: () => void
  label: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onRetry}
      disabled={!canRetry}
      className="rounded-lg text-[13px] font-medium text-brand-700 transition-colors hover:text-brand-800 disabled:cursor-not-allowed disabled:text-ink-300"
    >
      {label}
    </button>
  )
}