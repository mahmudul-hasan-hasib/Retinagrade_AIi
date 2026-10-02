import type { ExplainApiResponse } from '../../api/client'
import { getModality } from '../../data/clinical'
import { formatPercent, NO_VALUE } from '../../lib/format'
import type { ModalityKey, PredictionStatus, PredictApiPrediction } from '../../types'
import { AlertIcon, CheckIcon, ClockIcon, CpuIcon, DocumentIcon, SparkIcon } from '../icons'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { Skeleton } from '../ui/Feedback'
import { Notice } from '../ui/Placeholder'

/** Lifecycle of `GET /health`, which decides if the action can be offered. */
export type GeminiAvailability = 'unknown' | 'ready' | 'unavailable' | 'error'

/** Lifecycle of `POST /explain`, independent of `/predict` and `/gradcam`. */
export type ExplainStatus = 'idle' | 'loading' | 'ready' | 'error' | 'unavailable'

export interface AiReportCardProps {
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
  /** The graded result from `/predict`. Null until an analysis has succeeded. */
  prediction: PredictApiPrediction | null
  /** Whether `/gradcam` produced a heatmap for this same capture. */
  gradCamAvailable: boolean
  /** Whether `/gradcam` is still running for this capture. */
  gradCamPending: boolean
  availability: GeminiAvailability
  /** Set when `GET /health` itself could not be reached. */
  availabilityError: string | null
  explainStatus: ExplainStatus
  explanation: ExplainApiResponse | null
  explainError: string | null
  onGenerate: () => void
}

const EXPLAIN_BADGE: Record<
  ExplainStatus,
  { tone: 'brand' | 'success' | 'warning' | 'muted'; label: string }
> = {
  idle: { tone: 'muted', label: 'Not generated' },
  loading: { tone: 'brand', label: 'Generating' },
  ready: { tone: 'success', label: 'Generated' },
  error: { tone: 'warning', label: 'Failed' },
  unavailable: { tone: 'warning', label: 'Unavailable' },
}

/**
 * Gemini prose explanation of the current prediction, from `POST /explain`.
 *
 * The card is a projection of the server payload: every grade, confidence,
 * latency and model name below is read straight off the response, and the prose
 * block is rendered verbatim as whitespace-preformatted text. No clinical text
 * is composed, paraphrased or invented in this tree - when there is nothing to
 * show, a state is shown instead.
 *
 * Gemini is entirely optional and deliberately non-blocking. `GET /health` is
 * consulted first, so a server without a key reports that plainly and never
 * fires a request that is guaranteed to 503. Any failure is confined to this
 * card: the DR grade from `/predict` and the heatmap from `/gradcam` are held in
 * separate state by `App` and are neither cleared nor invalidated by anything
 * shown here.
 */
export function AiReportCard({
  fileName,
  modality,
  status,
  prediction,
  gradCamAvailable,
  gradCamPending,
  availability,
  availabilityError,
  explainStatus,
  explanation,
  explainError,
  onGenerate,
}: AiReportCardProps) {
  const hasRun = status === 'analyzed'
  const hasPrediction = Boolean(prediction)
  const modalityLabel = getModality(modality).label
  const badge = EXPLAIN_BADGE[explainStatus]

  const gradCamLabel = gradCamAvailable
    ? 'Available'
    : gradCamPending
      ? 'Computing'
      : 'Not available'

  const gradCamHint = gradCamAvailable
    ? 'Metadata only is sent to Gemini; no pixels leave the server.'
    : gradCamPending
      ? 'The heatmap is still being computed for this capture.'
      : 'The explanation will be generated from the grade and probabilities alone.'

  // The action is offered only when it can succeed: a graded prediction to
  // explain, and a server that reports Gemini as configured.
  const canGenerate =
    hasRun && hasPrediction && availability === 'ready' && explainStatus !== 'loading'

  const metadata = [
    { label: 'Generator', value: explanation?.model ?? NO_VALUE },
    {
      label: 'Gemini latency',
      value:
        typeof explanation?.latency_ms === 'number'
          ? `${explanation.latency_ms.toFixed(0)} ms`
          : NO_VALUE,
    },
    { label: 'Grad-CAM sent', value: gradCamAvailable ? 'Yes' : 'No' },
    { label: 'Predicted grade', value: prediction?.label ?? NO_VALUE },
  ]

  return (
    <Card>
      <CardHeader
        icon={<DocumentIcon />}
        title="AI Report"
        subtitle="Gemini explanation of the prediction above"
        action={<Badge tone={badge.tone}>{badge.label}</Badge>}
      />

      <CardBody className="space-y-4">
        {availability === 'unavailable' ? (
          <Notice tone="warning" icon={<AlertIcon />} title="Gemini is not configured on this server">
            <code>GEMINI_API_KEY</code> is not set in the backend environment, so{' '}
            <code>POST /explain</code> cannot run here. The DR grade and the Grad-CAM
            heatmap are unaffected. Set the key on the server to enable this panel.
          </Notice>
        ) : null}

        {availability === 'error' ? (
          <Notice tone="warning" icon={<AlertIcon />} title="Backend health could not be read">
            {availabilityError ??
              'The health check failed, so it is unknown whether Gemini is configured.'}{' '}
            No explanation request was sent. The grade and heatmap are unaffected.
          </Notice>
        ) : null}

        {explainStatus === 'unavailable' ? (
          <Notice tone="warning" icon={<AlertIcon />} title="No explanation was requested">
            The server reports Gemini as not configured, so the request was not sent
            rather than being fired at an endpoint that would reject it. The DR grade
            and the Grad-CAM heatmap are unaffected.
          </Notice>
        ) : null}

        {explainStatus === 'error' ? (
          <Notice tone="warning" icon={<AlertIcon />} title="The explanation could not be generated">
            {explainError ?? 'The request to /explain failed.'} The DR grade above and the
            Grad-CAM heatmap are unaffected.
          </Notice>
        ) : null}

        {/* The returned prose. Rendered verbatim - never edited client-side. */}
        {explainStatus === 'ready' && explanation?.explanation ? (
          <div className="space-y-3">
            <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
              Explanation
            </p>
            <div className="rounded-xl border border-line bg-ink-50/60 px-3.5 py-3">
              <p className="text-sm leading-relaxed whitespace-pre-line text-ink-700">
                {explanation.explanation}
              </p>
            </div>
            {explanation.disclaimer ? (
              <p className="text-[11px] leading-relaxed text-ink-400">
                {explanation.disclaimer}
              </p>
            ) : null}
          </div>
        ) : null}

        {explainStatus === 'loading' ? (
          <div className="space-y-3">
            <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
              Explanation
            </p>
            <div className="space-y-2 rounded-xl border border-line bg-ink-50/60 px-3.5 py-3">
              <Skeleton className="h-2 w-full" />
              <Skeleton className="h-2 w-11/12" />
              <Skeleton className="h-2 w-4/5" />
            </div>
            <p className="text-[11px] leading-relaxed text-ink-400">
              Waiting for Gemini. This is a network call only - no model of ours runs
              again, and the grade is already final.
            </p>
          </div>
        ) : null}

        {explainStatus === 'idle' && !hasRun ? (
          <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3.5 py-8 text-center">
            <span className="grid size-12 place-items-center rounded-2xl bg-white text-ink-400 ring-1 ring-ink-200">
              <SparkIcon className="size-5.5" />
            </span>
            <p className="text-sm font-medium text-ink-700">No prediction to explain yet</p>
            <p className="max-w-sm text-xs leading-relaxed text-ink-500">
              Run the analysis on the Analyze page. The explanation is generated from
              the resulting grade, so it cannot be requested before that.
            </p>
          </div>
        ) : null}

        {explainStatus === 'idle' && hasRun ? (
          <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3.5 py-8 text-center">
            <span className="grid size-12 place-items-center rounded-2xl bg-white text-ink-400 ring-1 ring-ink-200">
              <SparkIcon className="size-5.5" />
            </span>
            <p className="text-sm font-medium text-ink-700">Ready to explain</p>
            <p className="max-w-sm text-xs leading-relaxed text-ink-500">
              {availability === 'ready'
                ? 'Gemini will describe this grade in prose. The retinal image is not sent.'
                : 'Waiting for the backend health check to confirm Gemini is available.'}
            </p>
          </div>
        ) : null}

        {/* The numbers sent to /explain, echoed back by the server under `source`. */}
        <div>
          <p className="mb-2 flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            <CpuIcon className="size-3" />
            Explanation inputs
          </p>
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {metadata.map((entry) => (
              <div
                key={entry.label}
                className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line"
              >
                <dt className="truncate text-[11px] text-ink-500">{entry.label}</dt>
                <dd className="mt-0.5 font-mono text-xs font-medium text-ink-700">
                  {entry.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="space-y-2 border-t border-line pt-3">
          <div className="flex items-center justify-between gap-3 text-[11px]">
            <span className="text-ink-500">Source capture</span>
            <span className="truncate font-mono text-ink-400" title={fileName ?? ''}>
              {fileName ? `${fileName} (${modalityLabel})` : NO_VALUE}
            </span>
          </div>
          <div className="flex items-center justify-between gap-3 text-[11px]">
            <span className="text-ink-500">Confidence</span>
            <span className="font-mono text-ink-400">
              {formatPercent(prediction?.confidence)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-3 text-[11px]">
            <span className="text-ink-500">Grad-CAM</span>
            <span
              className={`inline-flex items-center gap-1 font-mono ${
                gradCamAvailable ? 'text-emerald-600' : 'text-ink-400'
              }`}
            >
              {gradCamAvailable ? <CheckIcon className="size-3" /> : null}
              {gradCamLabel}
            </span>
          </div>
          <p className="text-[11px] leading-relaxed text-ink-400">{gradCamHint}</p>

          <Button
            variant="primary"
            size="sm"
            block
            onClick={onGenerate}
            disabled={!canGenerate}
            icon={explainStatus === 'loading' ? <ClockIcon /> : <SparkIcon />}
          >
            {explainStatus === 'loading'
              ? 'Generating…'
              : explainStatus === 'ready'
                ? 'Regenerate explanation'
                : 'Generate explanation'}
          </Button>

          {!canGenerate ? (
            <p className="text-[11px] leading-relaxed text-ink-400">
              {!hasRun
                ? 'Run the analysis first: the explanation is generated from its result.'
                : availability === 'ready'
                  ? 'Wait for the current request to finish.'
                  : 'Disabled until the backend confirms Gemini is configured.'}
            </p>
          ) : null}
        </div>
      </CardBody>
    </Card>
  )
}