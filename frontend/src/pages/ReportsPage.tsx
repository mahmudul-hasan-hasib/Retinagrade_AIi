import type { ExplainApiResponse } from '../api/client'
import { AiReportCard } from '../components/reports/AiReportCard'
import type {
  ExplainStatus,
  GeminiAvailability,
  ModalityKey,
  PredictionStatus,
  PredictApiPrediction,
} from '../types'

export interface ReportsPageProps {
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
  prediction: PredictApiPrediction | null
  gradCamAvailable: boolean
  gradCamPending: boolean
  availability: GeminiAvailability
  availabilityError: string | null
  explainStatus: ExplainStatus
  explanation: ExplainApiResponse | null
  explainError: string | null
  onGenerateExplanation: () => void
}

/** AI report section on its own route, reusing the analyze-page panel. */
export function ReportsPage({
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
  onGenerateExplanation,
}: ReportsPageProps) {
  return (
    <AiReportCard
      fileName={fileName}
      modality={modality}
      status={status}
      prediction={prediction}
      gradCamAvailable={gradCamAvailable}
      gradCamPending={gradCamPending}
      availability={availability}
      availabilityError={availabilityError}
      explainStatus={explainStatus}
      explanation={explanation}
      explainError={explainError}
      onGenerate={onGenerateExplanation}
    />
  )
}