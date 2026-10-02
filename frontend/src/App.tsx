import { useCallback, useRef, useState } from 'react'
import { predictImage } from './api/client'
import { AppShell } from './components/layout/AppShell'
import { useImageSelection } from './hooks/useImageSelection'
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

/**
 * Single-page UI shell.
 *
 * The modality, selected image and run status live here rather than inside
 * `AnalyzePage` so that switching to Explainability or AI Reports keeps the
 * same capture on screen instead of resetting the workspace.
 *
 * "Analyze Image" is the only place that talks to the API: it POSTs the capture
 * and the modality to `/predict` and stores the response verbatim. No grade,
 * confidence or probability is computed in this tree - the analyze page only
 * projects fields the server sent.
 */
export default function App() {
  const [activeItem, setActiveItem] = useState<NavItemId>('analyze')
  const [modality, setModality] = useState<ModalityKey>('cfp')
  const [status, setStatus] = useState<PredictionStatus>('idle')
  const [response, setResponse] = useState<PredictApiResponse | null>(null)
  const [httpStatus, setHttpStatus] = useState<number | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const image = useImageSelection()

  /**
   * Monotonic request generation.
   *
   * The image and the modality can be changed while `/predict` is still in
   * flight, so a response can arrive after its inputs are gone. Every input
   * change and every new request bumps this counter; a pending request that
   * resolves against a different value is stale and is dropped instead of
   * overwriting the current state.
   */
  const requestGeneration = useRef(0)

  // Any change to the inputs invalidates the result panels.
  const resetResults = useCallback(() => {
    requestGeneration.current += 1
    setStatus('idle')
    setResponse(null)
    setHttpStatus(null)
    setRequestError(null)
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
      ) : null}

      {activeItem === 'dashboard' ? <DashboardPage /> : null}

      {activeItem === 'explainability' ? (
        <ExplainabilityPage
          previewUrl={image.previewUrl}
          fileName={image.file?.name ?? null}
          modality={modality}
          status={status}
        />
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
