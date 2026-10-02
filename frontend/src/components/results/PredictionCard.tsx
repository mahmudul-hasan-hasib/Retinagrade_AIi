import { formatPercent } from '../../lib/format'
import { ProgressBar } from '../ui/Feedback'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { CheckIcon, CpuIcon, SparkIcon, ClockIcon } from '../icons'

export interface PredictionCardProps {
  modalityLabel: string
  fileName: string | null
  /** Predicted class label. `null` = not computed yet. */
  label: string | null
  /** 0..1 confidence. `null` = not computed yet. */
  confidence: number | null
  modelName: string
  /** Server-reported milliseconds spent in `predict()`. `null` = unavailable. */
  inferenceMs?: number | null
  /** Full grade description from the backend, e.g. `Moderate non-proliferative DR`. */
  description?: string | null
}

/**
 * Headline result card.
 *
 * Values come straight from the `/predict` response; nothing is computed or
 * rounded here. A `null` label/confidence still renders as the dashed
 * placeholder, so a missing value can never be mistaken for a measurement.
 */
export function PredictionCard({
  modalityLabel,
  fileName,
  label,
  confidence,
  modelName,
  inferenceMs = null,
  description = null,
}: PredictionCardProps) {
  const hasGrade = label !== null

  return (
    <Card tone="accent">
      <CardHeader
        icon={<SparkIcon />}
        title="Prediction Result"
        subtitle={hasGrade ? 'Real model run (CPU)' : 'Awaiting a real model run'}
      />

      <CardBody className="space-y-4">
        <div
          className={`rounded-xl border px-4 py-4 text-center ${
            hasGrade
              ? 'border-brand-200 bg-brand-50/40'
              : 'border-dashed border-brand-200 bg-brand-50/40'
          }`}
        >
          <p className="text-[11px] font-medium tracking-wide text-brand-600 uppercase">
            Diabetic retinopathy grade
          </p>
          <p
            className={`mt-2 font-mono text-2xl font-bold ${
              hasGrade ? 'text-ink-900' : 'text-ink-300'
            }`}
          >
            {label ?? '—'}
          </p>
          <p className="mt-1.5 text-[11px] leading-relaxed text-ink-400">
            {description ??
              'No grade has been produced. Inference is not connected in this build.'}
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
              Confidence
            </span>
            <span className="font-mono text-sm font-semibold text-ink-400">
              {confidence === null ? '—' : formatPercent(confidence)}
            </span>
          </div>
          <ProgressBar value={confidence} className="h-2.5" />
          <p className="text-[11px] text-ink-400">
            Softmax probability of the top class.
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line">
            <dt className="flex items-center gap-1 text-ink-500">
              <CheckIcon className="size-3" />
              Modality
            </dt>
            <dd className="mt-0.5 font-mono font-medium text-ink-700">
              {modalityLabel}
            </dd>
          </div>
          <div className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line">
            <dt className="flex items-center gap-1 text-ink-500">
              <CpuIcon className="size-3" />
              Model
            </dt>
            <dd className="mt-0.5 truncate font-mono font-medium text-ink-700" title={modelName}>
              {modelName}
            </dd>
          </div>
          <div className="col-span-2 rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line">
            <dt className="flex items-center gap-1 text-ink-500">
              <ClockIcon className="size-3" />
              Inference time
            </dt>
            <dd className="mt-0.5 font-mono font-medium text-ink-700">
              {inferenceMs === null ? '—' : `${inferenceMs.toFixed(2)} ms`}
            </dd>
          </div>
        </dl>

        {fileName ? (
          <p className="truncate font-mono text-[11px] text-ink-400" title={fileName}>
            Source: {fileName}
          </p>
        ) : null}
      </CardBody>
    </Card>
  )
}
