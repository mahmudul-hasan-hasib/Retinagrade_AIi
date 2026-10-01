import { AiReportCard } from '../components/reports/AiReportCard'
import type { ModalityKey, PredictionStatus } from '../types'

export interface ReportsPageProps {
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
}

/** AI report section on its own route, reusing the analyze-page panel. */
export function ReportsPage({ fileName, modality, status }: ReportsPageProps) {
  return <AiReportCard fileName={fileName} modality={modality} status={status} />
}
