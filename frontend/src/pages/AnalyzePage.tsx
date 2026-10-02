import { AnalysisResultCard } from '../components/analyze/AnalysisResultCard'
import { ImageDropzone } from '../components/analyze/ImageDropzone'
import { ModalitySelector } from '../components/analyze/ModalitySelector'
import { AlertIcon } from '../components/icons'
import { Button } from '../components/ui/Button'
import { Spinner } from '../components/ui/Feedback'
import { DR_CLASSES, getDrClass, getModality } from '../data/clinical'
import type { ModalityKey, PredictApiResponse, PredictionStatus } from '../types'

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
  /** Opens the Explainability screen. */
  onViewExplainability: () => void
}

/** Friendly copy shown when a prediction request fails. */
const ERROR_TITLE = 'Unable to analyze this image.'
const ERROR_HINT = 'Please check the image and try again.'

/**
 * The screening journey, in three steps: choose a modality, load a capture,
 * analyse it.
 *
 * That is the whole screen. Everything the run produces arrives below the
 * button as its own card, so nothing competes with the decision being made
 * above the fold, and no explanatory, research or debug copy sits on the page
 * itself. The result card is the last thing on the page: one grade, the scale
 * it sits on, and the single link to the Grad-CAM view.
 *
 * Nothing here derives a grade, a probability or a sentence. `label` is the
 * server's own predicted class (only the display name is expanded, from
 * `DR_CLASSES`).
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
  onViewExplainability,
}: AnalyzePageProps) {
  const modalityInfo = getModality(modality)
  const hasImage = Boolean(file)
  const isAnalyzing = status === 'analyzing'
  const isFailed = status === 'error'

  const prediction = response?.prediction ?? null

  /**
   * The predicted class, resolved from the response's own `predicted_class`
   * index and only falling back to a label lookup if that field is absent. Both
   * come straight from `/predict`; nothing is re-derived from the probabilities.
   */
  const predictedClass =
    typeof prediction?.predicted_class === 'number'
      ? (DR_CLASSES.find(
          (drClass) => drClass.index === prediction.predicted_class,
        ) ?? null)
      : prediction?.label
        ? (getDrClass(prediction.label) ?? null)
        : null

  const gradeTitle = prediction?.label
    ? (getDrClass(prediction.label)?.fullLabel ?? prediction.label)
    : null

  return (
    <div className="mx-auto w-full max-w-[880px]">
      <header>
        <h1 className="text-[28px] leading-[1.15] font-semibold tracking-tight text-balance text-ink-900 sm:text-[34px]">
          Analyze Retinal Image
        </h1>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-500">
          Upload a retinal image to assess diabetic retinopathy.
        </p>
      </header>

      <div className="mt-9">
        <ModalitySelector value={modality} onChange={onModalityChange} />
      </div>

      <div className="mt-9">
        <ImageDropzone
          file={file}
          previewUrl={previewUrl}
          modalityLabel={modalityInfo.label}
          error={uploadError}
          onSelect={onSelectImage}
        />
      </div>

      <div className="mt-9">
        <Button
          size="xl"
          block
          onClick={onAnalyze}
          disabled={!hasImage || isAnalyzing}
          aria-busy={isAnalyzing || undefined}
          iconRight={isAnalyzing ? <Spinner tone="light" label="Analyzing" /> : null}
        >
          {isAnalyzing ? (
            'Analyzing...'
          ) : (
            <>
              Analyze Image <span aria-hidden="true">&rarr;</span>
            </>
          )}
        </Button>

        {isFailed ? (
          <div
            role="alert"
            className="mt-4 flex items-start gap-3 rounded-2xl border border-brand-500/25 bg-brand-500/[0.07] px-4 py-3.5"
          >
            <AlertIcon className="mt-0.5 size-4 shrink-0 text-brand-700" />
            <div className="min-w-0 flex-1">
              <p className="text-[14px] font-semibold text-ink-900">
                {ERROR_TITLE}
              </p>
              <p className="mt-0.5 text-[13px] leading-relaxed text-ink-500">
                {ERROR_HINT}
              </p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={onAnalyze}
              className="self-center"
            >
              Retry
            </Button>
          </div>
        ) : null}
      </div>

      {/* The result card is the end of the page. */}
      {status === 'analyzed' ? (
        <div className="mt-10">
          <AnalysisResultCard
            gradeTitle={gradeTitle}
            predictedClass={predictedClass}
            modalityLabel={modalityInfo.label}
            fileName={file?.name ?? null}
            onViewExplainability={onViewExplainability}
          />
        </div>
      ) : null}
    </div>
  )
}