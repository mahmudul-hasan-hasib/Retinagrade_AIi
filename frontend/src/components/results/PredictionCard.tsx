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
}

/**
 * Headline result card. Both result fields stay `null` until the backend can
 * actually run a model, so both render as dashed placeholders.
 */
export function PredictionCard({
  modalityLabel,
  fileName,
  label,
  confidence,
  modelName,
}: PredictionCardProps) {
  return (
    <Card tone="accent">
      <CardHeader
        icon={<SparkIcon />}
        title="Prediction Result"
        subtitle="Awaiting a real model run"
      />

      <CardBody className="space-y-4">
        <div className="rounded-xl border border-dashed border-brand-200 bg-brand-50/40 px-4 py-4 text-center">
          <p className="text-[11px] font-medium tracking-wide text-brand-600 uppercase">
            Diabetic retinopathy grade
          </p>
          <p className="mt-2 font-mono text-2xl font-bold text-ink-300">
            {label ?? '—'}
          </p>
          <p className="mt-1.5 text-[11px] leading-relaxed text-ink-400">
            No grade has been produced. Inference is not connected in this build.
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
            <dd className="mt-0.5 font-mono font-medium text-ink-700">—</dd>
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
