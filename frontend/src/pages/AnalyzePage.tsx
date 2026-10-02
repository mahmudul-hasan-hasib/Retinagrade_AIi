import { useEffect } from 'react'
import type { ExplainApiResponse } from '../api/client'
import { AnalysisResultCard } from '../components/analyze/AnalysisResultCard'
import { AiExplanationCard } from '../components/analyze/AiExplanationCard'
import { ImageDropzone } from '../components/analyze/ImageDropzone'
import { ModalitySelector } from '../components/analyze/ModalitySelector'
import { AlertIcon, CheckIcon, SparkIcon, UploadIcon } from '../components/icons'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { Spinner } from '../components/ui/Feedback'
import { getDrClass, getModality } from '../data/clinical'
import type {
  ExplainStatus,
  GeminiAvailability,
  ModalityKey,
  PredictApiResponse,
  PredictionStatus,
} from '../types'

export interface AnalyzePageProps {
  modality: ModalityKey
  onModalityChange: (next: ModalityKey) => void
  file: File | null
  previewUrl: string | null
  uploadError: string | null
  onSelectImage: (file: File | null) => void
  status: PredictionStatus
  response: PredictApiResponse | null
  onAnalyze: () => void
  /** Whether `/gradcam` is still running for this capture. */
  gradCamPending: boolean
  availability: GeminiAvailability
  explainStatus: ExplainStatus
  explanation: ExplainApiResponse | null
  onGenerateExplanation: () => void
  /** Opens the Explainability screen. */
  onViewExplainability: () => void
}

/** Friendly copy shown when a prediction request fails. */
const ERROR_TITLE = 'Unable to analyze this image.'
const ERROR_HINT = 'Please check the image and try again.'

/**
 * The whole screening journey on one screen: choose a modality, load a capture,
 * analyse it, read the grade, read a short explanation of it.
 *
 * Nothing here derives a grade, a probability or a sentence. `label` is the
 * server's own predicted class (only the display name is expanded, from
 * `DR_CLASSES`), and the explanation is Gemini's prose from `/explain`.
 *
 * Model internals, probabilities, confidences, raw responses and Grad-CAM are
 * intentionally not rendered here. They remain available on the Explainability
 * and Model screens, and in the API response itself.
 */
export function AnalyzePage({
  modality,
  onModalityChange,
  file,
  previewUrl,
  uploadError,
  onSelectImage,
  status,
  response,
  onAnalyze,
  gradCamPending,
  availability,
  explainStatus,
  explanation,
  onGenerateExplanation,
  onViewExplainability,
}: AnalyzePageProps) {
  const modalityInfo = getModality(modality)
  const hasImage = Boolean(file)
  const isAnalyzing = status === 'analyzing'
  const isFailed = status === 'error'

  const prediction = response?.prediction ?? null
  const gradeTitle = prediction?.label
    ? (getDrClass(prediction.label)?.fullLabel ?? prediction.label)
    : null

  /**
   * Ask for the explanation as soon as the grade is final.
   *
   * Two conditions are waited on, both of them about request quality rather
   * than availability: the health check must have reported on Gemini, and the
   * Grad-CAM follow-up must have settled so its metadata can be part of the
   * explanation request. The grade itself is already on screen throughout.
   *
   * The same `onGenerateExplanation` handler as the AI Reports panel is used -
   * no second Gemini path exists.
   */
  useEffect(() => {
    if (status !== 'analyzed') return
    if (explainStatus !== 'idle') return
    if (availability === 'unknown') return
    if (gradCamPending) return
    onGenerateExplanation()
  }, [
    status,
    explainStatus,
    availability,
    gradCamPending,
    onGenerateExplanation,
  ])

  return (
    <div className="mx-auto w-full max-w-[1180px] space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-[26px] font-semibold tracking-tight text-balance text-ink-900 sm:text-[28px]">
            Analyze Retinal Image
          </h1>
          <p className="mt-1.5 text-[14px] leading-relaxed text-ink-500">
            Upload a retinal image to assess diabetic retinopathy severity.
          </p>
        </div>

        {availability === 'unavailable' || availability === 'error' ? null : (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-surface px-3 py-1.5 text-[12px] font-medium text-ink-600 ring-1 ring-line">
            <CheckIcon className="size-3.5 text-emerald-600" />
            AI Ready
          </span>
        )}
      </header>

      <Card>
        <CardHeader
          title="Analyze Image"
          subtitle="Select the imaging modality and upload a retinal image."
        />

        <CardBody roomy className="space-y-6">
          <ModalitySelector value={modality} onChange={onModalityChange} />

          <ImageDropzone
            file={file}
            previewUrl={previewUrl}
            modalityLabel={modalityInfo.label}
            error={uploadError}
            onSelect={onSelectImage}
          />

          <Button
            size="lg"
            block
            onClick={onAnalyze}
            disabled={!hasImage || isAnalyzing}
            aria-busy={isAnalyzing || undefined}
            icon={isAnalyzing ? undefined : <SparkIcon />}
            iconRight={isAnalyzing ? <Spinner tone="light" label="Analyzing" /> : undefined}
          >
            {isAnalyzing ? 'Analyzing retinal image...' : 'Analyze Image →'}
          </Button>

          {isFailed ? (
            <div
              role="alert"
              className="flex flex-col items-start gap-3 rounded-xl bg-red-50 px-4 py-3.5 ring-1 ring-red-200 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex items-start gap-2.5">
                <AlertIcon className="mt-0.5 size-4 shrink-0 text-red-600" />
                <div>
                  <p className="text-[14px] font-semibold text-red-800">
                    {ERROR_TITLE}
                  </p>
                  <p className="text-[13px] leading-relaxed text-red-700">
                    {ERROR_HINT}
                  </p>
                </div>
              </div>
              <Button variant="secondary" size="sm" onClick={onAnalyze}>
                Try again
              </Button>
            </div>
          ) : null}

          {!hasImage ? (
            <p className="flex items-center justify-center gap-1.5 text-[12px] text-ink-400">
              <UploadIcon className="size-3.5" />
              Upload an image to enable analysis.
            </p>
          ) : null}
        </CardBody>
      </Card>

      {status === 'analyzed' ? (
        <AnalysisResultCard
          gradeTitle={gradeTitle}
          modalityLabel={modalityInfo.label}
          fileName={file?.name ?? null}
          onViewExplainability={onViewExplainability}
        />
      ) : null}

      {prediction ? (
        <AiExplanationCard
          status={explainStatus}
          explanation={explanation}
          canRetry={availability === 'ready' && explainStatus !== 'loading'}
          onRetry={onGenerateExplanation}
        />
      ) : null}
    </div>
  )
}