import { ImageDropzone } from '../components/analyze/ImageDropzone'
import { ImagePreview } from '../components/analyze/ImagePreview'
import { ModalitySelector } from '../components/analyze/ModalitySelector'
import { AiReportCard } from '../components/reports/AiReportCard'
import { PredictionCard } from '../components/results/PredictionCard'
import { ProbabilityCard } from '../components/results/ProbabilityCard'
import { XaiPanel } from '../components/xai/XaiPanel'
import { AlertIcon, SparkIcon } from '../components/icons'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Notice } from '../components/ui/Placeholder'
import { MODEL_FILE_NAMES, getModality } from '../data/clinical'
import type { ModalityKey, PredictionStatus } from '../types'

export interface AnalyzePageProps {
  modality: ModalityKey
  onModalityChange: (next: ModalityKey) => void
  file: File | null
  previewUrl: string | null
  uploadError: string | null
  onSelectImage: (file: File | null) => void
  onClearImage: () => void
  status: PredictionStatus
  onAnalyze: () => void
}

/**
 * Primary workflow: pick a modality, load an image, request an analysis.
 *
 * `onAnalyze` only flips the panels from "idle" to "placeholder". No request
 * leaves the browser and no grade is computed anywhere on this page.
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
  onAnalyze,
}: AnalyzePageProps) {
  const modalityInfo = getModality(modality)
  const hasImage = Boolean(file)
  const hasRun = status === 'analyzed'

  return (
    <>
      <Notice
        tone="warning"
        icon={<AlertIcon />}
        title="Interface only - no inference is running"
      >
        This dashboard renders the full screening interface. The backend is not
        called, so the DR grade, confidence, per-class probabilities, Grad-CAM
        maps and AI report below are all placeholders.
      </Notice>

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        <Card>
          <CardHeader
            icon={<SparkIcon />}
            title="Imaging Input"
            subtitle="Select a modality and load a retinal capture"
            action={
              <Badge tone={hasRun ? 'brand' : 'muted'}>
                {hasRun ? 'Analyzed' : 'Idle'}
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
                disabled={!hasImage}
                icon={<SparkIcon />}
              >
                Analyze Image
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

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        {hasRun ? (
          <>
            <PredictionCard
              modalityLabel={modalityInfo.label}
              fileName={file?.name ?? null}
              label={null}
              confidence={null}
              modelName={MODEL_FILE_NAMES[modality]}
            />
            <ProbabilityCard
              probabilities={null}
              predictedIndex={null}
              probabilitySum={null}
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
                  disabled={!hasImage}
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
