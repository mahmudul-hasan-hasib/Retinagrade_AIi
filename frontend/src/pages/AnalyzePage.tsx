import { ImageDropzone } from '../components/analyze/ImageDropzone'
import { ImagePreview } from '../components/analyze/ImagePreview'
import { ModalitySelector } from '../components/analyze/ModalitySelector'
import { AiReportCard } from '../components/reports/AiReportCard'
import { ApiResponseCard } from '../components/results/ApiResponseCard'
import { PredictionCard } from '../components/results/PredictionCard'
import { ProbabilityCard } from '../components/results/ProbabilityCard'
import { XaiPanel } from '../components/xai/XaiPanel'
import { AlertIcon, SparkIcon } from '../components/icons'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Notice } from '../components/ui/Placeholder'
import { DR_CLASSES, MODEL_FILE_NAMES, getModality } from '../data/clinical'
import type {
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
  onClearImage: () => void
  status: PredictionStatus
  /** Raw `POST /predict` body, `null` before the first successful call. */
  response: PredictApiResponse | null
  httpStatus: number | null
  requestError: string | null
  onAnalyze: () => void
}

/**
 * Primary workflow: pick a modality, load an image, request an analysis.
 *
 * "Analyze Image" POSTs the capture plus the modality to `/predict` and the
 * response is shown verbatim in the Backend Response card. No grade, confidence
 * or probability is computed on this page - the clinical panels stay
 * placeholders until the backend can actually run a model.
 */
export function AnalyzePage({
  modality,
  onModalityChange,
  file,
  previewUrl,
  uploadError,
  onSelectImage,
  onClearImage,
  status,
  response,
  httpStatus,
  requestError,
  onAnalyze,
}: AnalyzePageProps) {
  const modalityInfo = getModality(modality)
  const hasImage = Boolean(file)
  const hasRun = status === 'analyzed'
  const isAnalyzing = status === 'analyzing'
  const isFailed = status === 'error'

  // Read straight from the API body. Nothing here derives a grade: if the server
  // did not send a value the panel keeps its `null` placeholder.
  const prediction = response?.prediction ?? null
  const label = prediction?.label ?? null
  const confidence =
    typeof prediction?.confidence === 'number' ? prediction.confidence : null
  const probabilities = prediction?.probabilities ?? null
  const predictedIndex =
    typeof prediction?.predicted_class === 'number' ? prediction.predicted_class : null
  const inferenceMs =
    typeof response?.inference_ms === 'number' ? response.inference_ms : null

  // Display-only integrity check for the softmax head; not a clinical value.
  const probabilitySum = probabilities
    ? DR_CLASSES.reduce((total, drClass) => total + (probabilities[drClass.key] ?? 0), 0)
    : null

  return (
    <>
      <Notice
        tone="warning"
        icon={<AlertIcon />}
        title="Real inference is running"
      >
        "Analyze Image" uploads the capture to POST /predict and the returned DR
        grade, confidence and per-class probabilities are shown below, as is the
        raw response. Grad-CAM maps and the AI report are still placeholders.
      </Notice>

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        <Card>
          <CardHeader
            icon={<SparkIcon />}
            title="Imaging Input"
            subtitle="Select a modality and load a retinal capture"
            action={
              <Badge
                tone={hasRun ? 'brand' : isFailed ? 'warning' : 'muted'}
              >
                {hasRun
                  ? 'Analyzed'
                  : isAnalyzing
                    ? 'Analyzing'
                    : isFailed
                      ? 'Failed'
                      : 'Idle'}
              </Badge>
            }
          />

          <CardBody className="space-y-5">
            <ModalitySelector value={modality} onChange={onModalityChange} />

            <div className="space-y-2">
              <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
                Retinal image
              </p>
              <ImageDropzone
                file={file}
                error={uploadError}
                onSelect={onSelectImage}
                onClear={onClearImage}
              />
            </div>

            <div className="flex flex-col gap-2 border-t border-line pt-4 sm:flex-row">
              <Button
                size="lg"
                block
                onClick={onAnalyze}
                disabled={!hasImage || isAnalyzing}
                icon={<SparkIcon />}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={onClearImage}
                disabled={!hasImage}
              >
                Clear
              </Button>
            </div>

            <p className="text-[11px] leading-relaxed text-ink-400">
              {hasImage
                ? `Ready to analyze with the ${modalityInfo.label} checkpoint (${MODEL_FILE_NAMES[modality]}).`
                : `Upload a ${modalityInfo.label} capture to enable the Analyze button.`}
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Image Preview"
            subtitle={`Local preview of the selected ${modalityInfo.label} capture`}
          />
          <CardBody>
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name ?? null}
              modalityLabel={modalityInfo.label}
            />
          </CardBody>
        </Card>
      </div>

      <ApiResponseCard
        payload={response}
        httpStatus={httpStatus}
        error={requestError}
      />

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        {hasRun ? (
          <>
            <PredictionCard
              modalityLabel={modalityInfo.label}
              fileName={file?.name ?? null}
              label={label}
              confidence={confidence}
              modelName={prediction?.checkpoint ?? MODEL_FILE_NAMES[modality]}
              inferenceMs={inferenceMs}
              description={prediction?.description ?? null}
            />
            <ProbabilityCard
              probabilities={probabilities}
              predictedIndex={predictedIndex}
              probabilitySum={probabilitySum}
            />
          </>
        ) : (
          <div className="xl:col-span-2">
            <EmptyState
              icon={<SparkIcon />}
              title="No analysis yet"
              description="Choose a modality, upload a retinal image and press Analyze Image. The result, confidence, class probabilities, Grad-CAM and AI report panels appear here."
              action={
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={onAnalyze}
                  disabled={!hasImage || isAnalyzing}
                >
                  {hasImage ? 'Analyze Image' : 'Upload an image first'}
                </Button>
              }
            />
          </div>
        )}
      </div>

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        <XaiPanel
          previewUrl={previewUrl}
          fileName={file?.name ?? null}
          modality={modality}
          status={status}
        />
        <AiReportCard
          fileName={file?.name ?? null}
          modality={modality}
          status={status}
        />
      </div>
    </>
  )
}
