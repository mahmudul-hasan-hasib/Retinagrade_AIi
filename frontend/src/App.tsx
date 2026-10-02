import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchHealth, predictImage, requestExplanation, requestGradCam } from './api/client'
import type {
  ExplainApiRequest,
  ExplainApiResponse,
  GradCamApiResponse,
} from './api/client'
import { LayersIcon } from './components/icons'
import { AppShell } from './components/layout/AppShell'
import { Badge } from './components/ui/Badge'
import { Card, CardBody, CardHeader } from './components/ui/Card'
import { Skeleton } from './components/ui/Feedback'
import { useImageSelection } from './hooks/useImageSelection'
import { formatPercent, NO_VALUE } from './lib/format'
import { AnalyzePage } from './pages/AnalyzePage'
import { DashboardPage } from './pages/DashboardPage'
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
    subtitle: 'Grad-CAM activation map for the predicted grade',
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

/** Lifecycle of `GET /health`, which decides whether Gemini can be offered. */
type GeminiAvailability = 'unknown' | 'ready' | 'unavailable' | 'error'

/** Lifecycle of `POST /explain`, which runs no model of ours. */
type ExplainStatus = 'idle' | 'loading' | 'ready' | 'error' | 'unavailable'

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
  const [availability, setAvailability] = useState<GeminiAvailability>('unknown')
  const [availabilityError, setAvailabilityError] = useState<string | null>(null)
  const [explainStatus, setExplainStatus] = useState<ExplainStatus>('idle')
  const [explanation, setExplanation] = useState<ExplainApiResponse | null>(null)
  const [explainError, setExplainError] = useState<string | null>(null)
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
    setExplainStatus('idle')
    setExplanation(null)
    setExplainError(null)
  }, [])

  /**
   * Read `gemini_configured` from `GET /health` once on load.
   *
   * The flag is a server-side fact that does not change while the page is open,
   * so it is fetched once rather than per request. It runs no model and never
   * touches the prediction state. A failure here only means the explanation
   * action stays disabled - it is recorded, not thrown, so the analyze workflow
   * is unaffected.
   */
  useEffect(() => {
    let active = true

    fetchHealth()
      .then((result) => {
        if (!active) {
          return
        }
        setAvailability(result.data.gemini_configured === true ? 'ready' : 'unavailable')
      })
      .catch((error: unknown) => {
        if (!active) {
          return
        }
        const failed = error as { message?: string }
        setAvailability('error')
        setAvailabilityError(
          failed.message ?? 'The health check request to the backend failed.',
        )
      })

    return () => {
      active = false
    }
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

  /**
   * Ask Gemini to explain the prediction that is already on screen.
   *
   * Only the numbers `/predict` returned are sent - modality, predicted class,
   * confidence and the full 5-class distribution - plus Grad-CAM *metadata*
   * from `/gradcam` when a heatmap exists. The heatmap bitmap is deliberately
   * not included, and no image is re-uploaded: `/explain` runs no model of ours
   * and only describes what it is given.
   *
   * Two guards, in order:
   *   1. there must be a graded prediction with a class and a confidence to
   *      describe - the endpoint explains numbers, so it cannot run before that;
   *   2. `GET /health` must have reported `gemini_configured: true`. Anything
   *      else - false, absent or unreadable - is treated as "not known to be
   *      configured" and never as permission, so a server with no key is
   *      reported as unavailable instead of being sent a request that is
   *      guaranteed to answer 503.
   *
   * The request is guarded by the same monotonic counter as the grade and the
   * heatmap, so an explanation can never outlive the image it describes. Any
   * failure is confined to its own state: `status`, `response`, `gradCam` and
   * `gradCamStatus` are never touched here.
   */
  const handleGenerateExplanation = useCallback(async () => {
    const prediction = response?.prediction

    if (status !== 'analyzed' || !prediction) {
      return
    }

    if (typeof prediction.predicted_class !== 'number' || typeof prediction.confidence !== 'number') {
      setExplainStatus('error')
      setExplanation(null)
      setExplainError(
        'The prediction result is missing its class or confidence, so there is nothing to explain.',
      )
      return
    }

    if (availability !== 'ready') {
      setExplainStatus('unavailable')
      setExplanation(null)
      setExplainError(null)
      return
    }

    const generation = requestGeneration.current

    const payload: ExplainApiRequest = {
      modality: prediction.modality ?? modality,
      predicted_class: prediction.predicted_class,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities ?? {},
    }

    // Only metadata is sent when a real heatmap exists; the bitmap is left out
    // so no pixels leave the machine. Same `data_uri` test the card shows, so
    // what is sent and what is displayed cannot disagree.
    if (gradCam?.cam?.data_uri) {
      payload.gradcam = {
        available: true,
        target_layer: gradCam.target_layer ?? null,
        grid_shape: Array.isArray(gradCam.grid_shape) ? gradCam.grid_shape : null,
        image_size: typeof gradCam.image_size === 'number' ? gradCam.image_size : null,
        gradients_are_nonzero:
          typeof gradCam.gradients_are_nonzero === 'boolean'
            ? gradCam.gradients_are_nonzero
            : null,
        is_predicted_class:
          typeof gradCam.is_predicted_class === 'boolean'
            ? gradCam.is_predicted_class
            : null,
      }
    } else {
      payload.gradcam = { available: false }
    }

    setExplainStatus('loading')
    setExplainError(null)

    try {
      const result = await requestExplanation(payload)
      if (requestGeneration.current !== generation) {
        return
      }
      setExplanation(result.data)
      setExplainStatus('ready')
    } catch (error) {
      if (requestGeneration.current !== generation) {
        return
      }
      const failed = error as { httpStatus?: number; message?: string }
      setExplanation(null)
      setExplainError(failed.message ?? 'The request to /explain failed.')
      setExplainStatus('error')
    }
  }, [response, status, modality, availability, gradCam])

  const meta = PAGE_META[activeItem]

  /**
   * Whether a real heatmap exists for the current capture. Derived from the
   * response rather than the status alone, so a "ready" state with no image can
   * never be reported to `/explain` as available Grad-CAM evidence.
   */
  const gradCamReady = gradCamStatus === 'ready' && Boolean(gradCam?.cam?.data_uri)

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
            gradCamAvailable={gradCamReady}
            gradCamPending={gradCamStatus === 'loading'}
            availability={availability}
            availabilityError={availabilityError}
            explainStatus={explainStatus}
            explanation={explanation}
            explainError={explainError}
            onGenerateExplanation={handleGenerateExplanation}
          />

          {/* The single real Grad-CAM surface, drawn directly below the analyze
              workflow. It is the only explainability panel in the app. */}
          <GradCamCard
            gradCam={gradCam}
            status={gradCamStatus}
            error={gradCamError}
          />
        </>
      ) : null}

      {activeItem === 'dashboard' ? <DashboardPage /> : null}

      {activeItem === 'explainability' ? (
        <GradCamCard
          gradCam={gradCam}
          status={gradCamStatus}
          error={gradCamError}
        />
      ) : null}

      {activeItem === 'reports' ? (
        <ReportsPage
          fileName={image.file?.name ?? null}
          modality={modality}
          status={status}
          prediction={response?.prediction ?? null}
          gradCamAvailable={gradCamReady}
          gradCamPending={gradCamStatus === 'loading'}
          availability={availability}
          availabilityError={availabilityError}
          explainStatus={explainStatus}
          explanation={explanation}
          explainError={explainError}
          onGenerateExplanation={handleGenerateExplanation}
        />
      ) : null}

      {activeItem === 'history' ? <UnavailablePage itemId="history" /> : null}

      {activeItem === 'settings' ? <UnavailablePage itemId="settings" /> : null}
    </AppShell>
  )
}
