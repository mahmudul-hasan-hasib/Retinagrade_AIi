import { XaiPanel } from '../components/xai/XaiPanel'
import type { ModalityKey, PredictionStatus } from '../types'

export interface ExplainabilityPageProps {
  previewUrl: string | null
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
}

/** Explainability section on its own route, reusing the analyze-page panel. */
export function ExplainabilityPage({
  previewUrl,
  fileName,
  modality,
  status,
}: ExplainabilityPageProps) {
  return (
    <XaiPanel
      previewUrl={previewUrl}
      fileName={fileName}
      modality={modality}
      status={status}
    />
  )
}
