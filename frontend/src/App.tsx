import { useCallback, useRef, useState } from 'react'
import { predictImage, requestGradCam } from './api/client'
import type { GradCamApiResponse } from './api/client'
import { LayersIcon } from './components/icons'
import { AppShell } from './components/layout/AppShell'
import { Badge } from './components/ui/Badge'
import { Card, CardBody, CardHeader } from './components/ui/Card'
import { Skeleton } from './components/ui/Feedback'
import { useImageSelection } from './hooks/useImageSelection'
import { formatPercent, NO_VALUE } from './lib/format'
import { AnalyzePage } from './pages/AnalyzePage'
import { DashboardPage } from './pages/DashboardPage'
import { ExplainabilityPage } from './pages/ExplainabilityPage'
import { ReportsPage } from './pages/ReportsPage'
import { UnavailablePage } from './pages/UnavailablePage'
import type {
  ModalityKey,
  NavItemId,
  PredictApiResponse,
  PredictionStatus,
} from './types'

const PAGE_META: Record<NavItemId, { title: string; subtitle: string }> = {
  dashboard: {
    title: 'Dashboard',
    subtitle: 'Screening workspace overview',
  },
  analyze: {
    title: 'Analyze Image',
    subtitle: 'Upload a retinal capture and grade diabetic retinopathy',
  },
  history: {
    title: 'Screening History',
    subtitle: 'Past studies and PDF export',
  },
  explainability: {
    title: 'Explainability',
    subtitle: 'Grad-CAM overlays and saliency evidence',
  },
  reports: {
    title: 'AI Reports',
    subtitle: 'Generated clinical summaries',
  },
  settings: {
    title: 'Settings',
    subtitle: 'Model selection and inference device',
  },
}

/** Lifecycle of the follow-up Grad-CAM call, which is independent of /predict. */
type GradCamLoadStatus = 'idle' | 'loading' | 'ready' | 'error'

interface GradCamCardProps {
  gradCam: GradCamApiResponse | null
  status: GradCamLoadStatus
  error: string | null
}

const GRADCAM_BADGE: Record<
  GradCamLoadStatus,
  { tone: 'brand' | 'success' | 'warning' | 'muted'; label: string }
> = {
  idle: { tone: 'muted', label: 'Idle' },
  loading: { tone: 'brand', label: 'Computing' },
  ready: { tone: 'success', label: 'Ready' },
  error: { tone: 'warning', label: 'Failed' },
}

/**
 * The real Grad-CAM heatmap returned by `POST /gradcam`.
 *
 * The base64 PNG data URI the backend sends is used directly as the `<img src>`;
 * nothing is decoded, colour-mapped or blended client-side. Every value shown
 * is read straight off the response - no grade, confidence or activation is
 * derived here.
 *
 * A Grad-CAM failure is deliberately non-fatal and non-blocking: the grade from
 * `/predict` is already on screen, so a missing heatmap is reported in its own
 * card and never replaces or invalidates the prediction.
 */
function GradCamCard({ gradCam, status, error }: GradCamCardProps) {
  const dataUri = gradCam?.cam?.data_uri ?? null
  const hasImage = Boolean(dataUri)
  const badge = GRADCAM_BADGE[status]

  const metadata = [
    {
      label: 'Explained grade',
      value: gradCam?.label ?? NO_VALUE,
    },
    {
      label: 'Confidence',
      value: formatPercent(gradCam?.confidence),
    },
    { label: 'Target layer', value: gradCam?.target_layer ?? NO_VALUE },
    {
      label: 'CAM grid',
      value: Array.isArray(gradCam?.grid_shape)
        ? `[${gradCam.grid_shape.join(' × ')}]`
        : NO_VALUE,
    },
    {
      label: 'Model input',
      value: typeof gradCam?.image_size === 'number' ? `${gradCam.image_size} px` : NO_VALUE,
    },
    {
      label: 'Inference',
      value: typeof gradCam?.inference_ms === 'number' ? `${gradCam.inference_ms} ms` : NO_VALUE,
    },
  ]

  return (
    <Card>
      <CardHeader
        icon={<LayersIcon />}
        title="Grad-CAM Heatmap"
        subtitle="Class activation map for the grade above, computed on the CPU"
        action={<Badge tone={badge.tone}>{hasImage ? 'Ready' : badge.label}</Badge>}
      />

      <CardBody className="space-y-4">
        <div className="relative grid min-h-64 place-items-center overflow-hidden rounded-xl border border-line bg-ink-900 p-3">
          {hasImage ? (
            <img
              src={dataUri ?? ''}
              alt={`Grad-CAM heatmap explaining the ${gradCam?.label ?? 'predicted'} grade`}
              className="max-h-64 w-auto rounded-lg object-contain"
            />
          ) : null}

          {status === 'loading' ? (
            <div className="flex w-full max-w-sm flex-col items-center gap-3 py-12 text-center">
              <div className="w-40">
                <Skeleton className="h-32 w-full" />
              </div>
              <p className="text-xs font-medium text-ink-400">Computing heatmap</p>
              <p className="text-[11px] leading-relaxed text-ink-500">
                The backend runs a forward and backward pass over the same
                checkpoint used for the grade.
              </p>
            </div>
          ) : null}

          {status === 'error' ? (
            <div className="flex w-full max-w-sm flex-col items-center gap-2 py-12 text-center">
              <p className="text-xs font-medium text-amber-300">
                The heatmap could not be produced
              </p>
              <p className="text-[11px] leading-relaxed text-ink-500">
                {error ?? 'The request to /gradcam failed.'}
              </p>
              <p className="text-[11px] leading-relaxed text-ink-400">
                The DR grade above is unaffected.
              </p>
            </div>
          ) : null}

          {status === 'idle' ? (
            <div className="flex w-full max-w-sm flex-col items-center gap-2 py-12 text-center">
              <span className="grid size-12 place-items-center rounded-2xl bg-white/5 text-ink-400 ring-1 ring-white/10">
                <LayersIcon className="size-5.5" />
              </span>
              <p className="text-sm font-medium text-ink-300">No heatmap yet</p>
              <p className="text-xs leading-relaxed text-ink-500">
                Run the analysis to request a Grad-CAM map for the same capture.
              </p>
            </div>
          ) : null}

          {hasImage ? (
            <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur-sm">
              {gradCam?.target_layer ?? 'Grad-CAM'}
            </span>
          ) : null}
        </div>

        <div>
          <p className="mb-2 text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            Grad-CAM parameters
          </p>
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
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

        <p className="text-[11px] leading-relaxed text-ink-400">
          The heatmap is the raw activation map in the model-input frame, not
          blended with the photograph. Redder areas contributed more to the
          explained grade.
        </p>
      </CardBody>
    </Card>
  )
}

/**
 * Single-page UI shell.
 *
 * The modality, selected image and run status live here rather than inside
 * `AnalyzePage` so that switching to Explainability or AI Reports keeps the
 * same capture on screen instead of resetting the workspace.
 *
 * "Analyze Image" is the only place that talks to the API. One click runs
 * `POST /predict` for the DR grade and, once that succeeds, `POST /gradcam` for
 * the heatmap explaining it - same file, same modality. The two responses are
 * stored separately and arrive independently, so a slow or failing heatmap never
 * delays or discards the grade. No grade, confidence, probability or activation
 * is computed in this tree - the panels only project fields the server sent.
 */
export default function App() {
  const [activeItem, setActiveItem] = useState<NavItemId>('analyze')
  const [modality, setModality] = useState<ModalityKey>('cfp')
  const [status, setStatus] = useState<PredictionStatus>('idle')
  const [response, setResponse] = useState<PredictApiResponse | null>(null)
  const [httpStatus, setHttpStatus] = useState<number | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [gradCam, setGradCam] = useState<GradCamApiResponse | null>(null)
  const [gradCamStatus, setGradCamStatus] = useState<GradCamLoadStatus>('idle')
  const [gradCamError, setGradCamError] = useState<string | null>(null)
  const image = useImageSelection()

  /**
   * Monotonic request generation.
   *
   * The image and the modality can be changed while `/predict` or `/gradcam` is
   * still in flight, so a response can arrive after its inputs are gone. Every
   * input change and every new request bumps this counter; a pending request
   * that resolves against a different value is stale and is dropped instead of
   * overwriting the current state. The heatmap is guarded by the same counter
   * as the grade, so the two can never end up describing different images.
   */
  const requestGeneration = useRef(0)

  // Any change to the inputs invalidates the result panels.
  const resetResults = useCallback(() => {
    requestGeneration.current += 1
    setStatus('idle')
    setResponse(null)
    setHttpStatus(null)
    setRequestError(null)
    setGradCam(null)
    setGradCamStatus('idle')
    setGradCamError(null)
  }, [])

  const handleModalityChange = useCallback(
    (next: ModalityKey) => {
      setModality(next)
      resetResults()
    },
    [resetResults],
  )

  const handleSelectImage = useCallback(
    (file: File | null) => {
      image.select(file)
      resetResults()
    },
    [image, resetResults],
  )

  const handleClearImage = useCallback(() => {
    image.clear()
    resetResults()
  }, [image, resetResults])

  const handleAnalyze = useCallback(async () => {
    const file = image.file
    if (!file) {
      return
    }

    // Claim a generation: this also supersedes any request still in flight.
    const generation = (requestGeneration.current += 1)

    setStatus('analyzing')
    setResponse(null)
    setHttpStatus(null)
    setRequestError(null)
    setGradCam(null)
    setGradCamStatus('idle')
    setGradCamError(null)

    try {
      const result = await predictImage(file, modality)
      if (requestGeneration.current !== generation) {
        return
      }
      setHttpStatus(result.httpStatus)
      setResponse(result.data)
      setStatus('analyzed')
    } catch (error) {
      if (requestGeneration.current !== generation) {
        return
      }
      const failed = error as { httpStatus?: number; message?: string }
      setHttpStatus(failed.httpStatus ?? null)
      setResponse(null)
      setRequestError(
        failed.message ?? 'The request to the backend failed for an unknown reason.',
      )
      setStatus('error')
      return
    }

    // The grade is on screen, so the heatmap is requested as a follow-up rather
    // than in parallel: no point running a second CPU inference pass for a grade
    // that has not been accepted yet. It reuses the file handle and modality the
    // caller just graded, and sends no target_class, so the backend explains its
    // own predicted grade.
    setGradCamStatus('loading')
    setGradCamError(null)
    try {
      const heatmap = await requestGradCam(file, modality)
      if (requestGeneration.current !== generation) {
        return
      }
      setGradCam(heatmap.data)
      setGradCamStatus('ready')
    } catch (error) {
      if (requestGeneration.current !== generation) {
        return
      }
      const failed = error as { httpStatus?: number; message?: string }
      setGradCamError(
        failed.message ?? 'The request for the Grad-CAM heatmap failed.',
      )
      setGradCamStatus('error')
    }
  }, [image.file, modality])

  const meta = PAGE_META[activeItem]

  return (
    <AppShell
      activeItem={activeItem}
      onSelectItem={setActiveItem}
      title={meta.title}
      subtitle={meta.subtitle}
    >
      {activeItem === 'analyze' ? (
        <>
          <AnalyzePage
            modality={modality}
            onModalityChange={handleModalityChange}
            file={image.file}
            previewUrl={image.previewUrl}
            uploadError={image.error}
            onSelectImage={handleSelectImage}
            onClearImage={handleClearImage}
            status={status}
            response={response}
            httpStatus={httpStatus}
            requestError={requestError}
            onAnalyze={handleAnalyze}
          />

          {/* Real heatmap, rendered in the result area directly below the
              analyze workflow. The placeholder XAI panel above keeps its own
              place in the layout. */}
          <GradCamCard
            gradCam={gradCam}
            status={gradCamStatus}
            error={gradCamError}
          />
        </>
      ) : null}

      {activeItem === 'dashboard' ? <DashboardPage /> : null}

      {activeItem === 'explainability' ? (
        <>
          <ExplainabilityPage
            previewUrl={image.previewUrl}
            fileName={image.file?.name ?? null}
            modality={modality}
            status={status}
          />

          <GradCamCard
            gradCam={gradCam}
            status={gradCamStatus}
            error={gradCamError}
          />
        </>
      ) : null}

      {activeItem === 'reports' ? (
        <ReportsPage
          fileName={image.file?.name ?? null}
          modality={modality}
          status={status}
        />
      ) : null}

      {activeItem === 'history' ? <UnavailablePage itemId="history" /> : null}

      {activeItem === 'settings' ? <UnavailablePage itemId="settings" /> : null}
    </AppShell>
  )
}
