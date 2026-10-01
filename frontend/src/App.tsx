import { useCallback, useState } from 'react'
import { AppShell } from './components/layout/AppShell'
import { useImageSelection } from './hooks/useImageSelection'
import { AnalyzePage } from './pages/AnalyzePage'
import { DashboardPage } from './pages/DashboardPage'
import { ExplainabilityPage } from './pages/ExplainabilityPage'
import { ReportsPage } from './pages/ReportsPage'
import { UnavailablePage } from './pages/UnavailablePage'
import type { ModalityKey, NavItemId, PredictionStatus } from './types'

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
 * Nothing in this tree performs inference or contacts the API: `status` only
 * controls which placeholder panels are revealed.
 */
export default function App() {
  const [activeItem, setActiveItem] = useState<NavItemId>('analyze')
  const [modality, setModality] = useState<ModalityKey>('cfp')
  const [status, setStatus] = useState<PredictionStatus>('idle')
  const image = useImageSelection()

  // Any change to the inputs invalidates the result panels.
  const resetResults = useCallback(() => setStatus('idle'), [])

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

  const handleAnalyze = useCallback(() => {
    setStatus('analyzed')
  }, [])

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
          onClearImage={image.clear}
          status={status}
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
