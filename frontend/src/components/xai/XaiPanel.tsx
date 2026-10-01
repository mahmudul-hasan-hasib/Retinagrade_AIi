import { useState } from 'react'
import { getModality } from '../../data/clinical'
import type { ModalityKey, PredictionStatus } from '../../types'
import { EyeIcon, ImageIcon, LayersIcon, SparkIcon } from '../icons'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { EmptyState } from '../ui/EmptyState'
import { Skeleton } from '../ui/Feedback'
import { Placeholder } from '../ui/Placeholder'

export interface XaiPanelProps {
  previewUrl: string | null
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
}

type EvidenceView = 'original' | 'gradcam' | 'overlay'

const VIEWS: readonly { key: EvidenceView; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'gradcam', label: 'Grad-CAM' },
  { key: 'overlay', label: 'Overlay' },
]

/** Metadata that a real Grad-CAM implementation would have to fill in. */
const GRAD_CAM_META = [
  { label: 'Target layer', value: '-' },
  { label: 'Alpha (blend)', value: '-' },
  { label: 'Max activation', value: '-' },
  { label: 'Upsample', value: '-' },
] as const

/**
 * Explainability section: Grad-CAM heatmap plus companion evidence maps.
 *
 * The image tiles are inert placeholders - no activation hook, no gradient
 * capture, no heatmap rendering happens anywhere in this build.
 */
export function XaiPanel({
  previewUrl,
  fileName,
  modality,
  status,
}: XaiPanelProps) {
  const [view, setView] = useState<EvidenceView>('gradcam')
  const hasImage = Boolean(previewUrl)
  const hasRun = status === 'analyzed'

  return (
    <Card>
      <CardHeader
        icon={<LayersIcon />}
        title="Explainability (XAI)"
        subtitle="Grad-CAM and saliency evidence"
        action={
          <div
            role="radiogroup"
            aria-label="Evidence view"
            className="hidden rounded-lg bg-ink-100 p-0.5 sm:flex"
          >
            {VIEWS.map((option) => {
              const isSelected = option.key === view
              return (
                <button
                  key={option.key}
                  type="button"
                  role="radio"
                  aria-checked={isSelected}
                  onClick={() => setView(option.key)}
                  className={[
                    'rounded-[7px] px-2.5 py-1 text-[11px] font-medium transition-colors',
                    isSelected
                      ? 'bg-surface text-ink-900 shadow-sm'
                      : 'text-ink-500 hover:text-ink-700',
                  ].join(' ')}
                >
                  {option.label}
                </button>
              )
            })}
          </div>
        }
      />

      <CardBody className="space-y-4">
        <div className="relative grid min-h-64 place-items-center overflow-hidden rounded-xl border border-line bg-ink-900 p-3">
          {hasImage && view === 'original' ? (
            <img
              src={previewUrl ?? ''}
              alt={`Original retinal image: ${fileName ?? 'retinal image'}`}
              className="max-h-[24rem] w-auto rounded-lg object-contain"
            />
          ) : null}

          {hasImage && view !== 'original' ? (
            <div className="flex w-full max-w-sm flex-col items-center gap-3 py-12 text-center">
              <span className="relative grid size-28 place-items-center">
                <span className="absolute inset-0 animate-shimmer-sweep rounded-full border-2 border-dashed border-ink-600" />
                <span className="size-16 rounded-full border border-ink-700 bg-ink-800" />
              </span>
              <p className="text-xs font-medium text-ink-400">
                {view === 'gradcam' ? 'Grad-CAM heatmap' : 'Blended overlay'}
              </p>
              <p className="text-[11px] leading-relaxed text-ink-500">
                {hasRun
                  ? 'The heatmap is still empty: Grad-CAM is not implemented.'
                  : 'Run the analysis to request a heatmap. None is produced yet.'}
              </p>
            </div>
          ) : null}

          {!hasImage ? (
            <div className="flex w-full max-w-sm flex-col items-center gap-3 py-12 text-center">
              <span className="grid size-12 place-items-center rounded-2xl bg-white/5 text-ink-400 ring-1 ring-white/10">
                <ImageIcon className="size-5.5" />
              </span>
              <p className="text-sm font-medium text-ink-300">No image loaded</p>
              <p className="text-xs leading-relaxed text-ink-500">
                Select a {getModality(modality).label} capture on the Analyze page
                to preview evidence maps here.
              </p>
            </div>
          ) : null}

          {hasImage ? (
            <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur-sm">
              <EyeIcon className="size-3" />
              Local preview only
            </span>
          ) : null}
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <Placeholder
            label="Saliency map"
            value="-"
            hint="Pixel-attribution pass, not implemented."
          />
          <Placeholder
            label="Class activation map"
            value="-"
            hint="Per-class activation grid, not implemented."
          />
        </div>

        <div>
          <p className="mb-2 text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            Grad-CAM parameters
          </p>
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {GRAD_CAM_META.map((entry) => (
              <div
                key={entry.label}
                className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line"
              >
                <dt className="truncate text-[11px] text-ink-500">{entry.label}</dt>
                <dd className="mt-0.5 font-mono text-xs font-medium text-ink-400">
                  {entry.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3 py-3">
            <p className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-700">
              <SparkIcon className="size-3.5 text-ink-400" />
              Explainability summary
            </p>
            <div className="mt-2 space-y-1.5">
              <Skeleton className="h-2 w-full" />
              <Skeleton className="h-2 w-11/12" />
              <Skeleton className="h-2 w-3/4" />
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-ink-400">
              Narrative rationale is not generated.
            </p>
          </div>

          <div className="rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3 py-3">
            <p className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-700">
              <LayersIcon className="size-3.5 text-ink-400" />
              Attention consensus
            </p>
            <div className="mt-2 flex items-center gap-1.5">
              <span className="font-mono text-xs font-medium text-ink-400">-</span>
              <span className="text-[11px] text-ink-400">
                agreement between Grad-CAM and clinician marking is unavailable.
              </span>
            </div>
          </div>
        </div>

        {!hasImage ? (
          <EmptyState
            compact
            icon={<LayersIcon />}
            title="Evidence maps need an image"
            description="Upload a fundus capture first. Grad-CAM will then run against the model's target convolutional layer."
          />
        ) : null}
      </CardBody>
    </Card>
  )
}
