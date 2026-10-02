import type { GradCamApiResponse } from '../api/client'
import { CpuIcon, LayersIcon, UploadIcon } from '../components/icons'
import { Badge } from '../components/ui/Badge'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import type { PredictApiPrediction } from '../types'
import {
  GRADCAM_TARGET_LAYER,
  INFERENCE_DEVICE,
  MAX_UPLOAD_BYTES,
  MODEL_ARCHITECTURE,
  MODEL_DETAILS,
  MODEL_IMPLEMENTATION,
  MODEL_NUM_CLASSES,
  MODALITIES,
  PREPROCESSING_NORMALIZATION,
  PREPROCESSING_RESIZE,
} from '../data/clinical'

export interface ModelPageProps {
  /** Result of the last successful run, when there is one. */
  prediction: PredictApiPrediction | null
  /** Grad-CAM metadata from the last run, when there is one. */
  gradCam: GradCamApiResponse | null
}

/**
 * Model and inference configuration.
 *
 * The one place that states what the system is running: architecture, per-modality
 * checkpoints, input sizes, preprocessing, device and the Grad-CAM target layer.
 * These facts are deliberately kept out of the Analyze screen, which shows only
 * the grade and a short explanation.
 *
 * Static values mirror `backend/inference/model_loader.py`, `preprocessing.py` and
 * `gradcam.py`. When a run has happened, the values the server actually reported
 * are shown next to the configured defaults, so nothing on this page can drift
 * away from what inference did.
 */
export function ModelPage({ prediction, gradCam }: ModelPageProps) {
  const liveDevice = prediction?.device ?? null
  const liveCheckpoint = prediction?.checkpoint ?? null
  const liveImageSize = prediction?.image_size ?? null
  const liveTargetLayer = gradCam?.target_layer ?? null

  return (
    <div className="mx-auto w-full max-w-[1180px] space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-[26px] font-semibold tracking-tight text-balance text-ink-900 sm:text-[28px]">
            Model
          </h1>
          <p className="mt-1.5 text-[14px] leading-relaxed text-ink-500">
            Architecture, checkpoints and inference configuration.
          </p>
        </div>
      </header>

      <Card>
        <CardHeader
          icon={<CpuIcon />}
          title="Architecture"
          subtitle="Shared by both modalities"
        />
        <CardBody>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Architecture" value={MODEL_ARCHITECTURE} />
            <Field label="Implementation" value={MODEL_IMPLEMENTATION} mono />
            <Field label="Output classes" value={`${MODEL_NUM_CLASSES} ICDR grades`} />
            <Field
              label="Inference device"
              value={liveDevice ?? INFERENCE_DEVICE}
              hint={liveDevice ? 'reported by the last run' : undefined}
            />
          </dl>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          icon={<LayersIcon />}
          title="Checkpoints"
          subtitle="One screening model per imaging modality"
        />
        <CardBody className="space-y-4">
          {MODALITIES.map((modality) => {
            const detail = MODEL_DETAILS[modality.key]
            const isLive = liveCheckpoint === detail.checkpoint
            return (
              <div
                key={modality.key}
                className="rounded-[14px] border border-line bg-white/[0.03] p-4 sm:p-5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[15px] font-semibold text-ink-900">
                    {modality.label} &middot; {modality.fullName}
                  </p>
                  <Badge tone={isLive ? 'success' : 'muted'}>
                    {isLive ? 'Last run' : 'Configured'}
                  </Badge>
                </div>

                <dl className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <Field label="Checkpoint" value={detail.checkpoint} mono wide />
                  <Field label="Head layout" value={detail.headLayout} mono />
                  <Field
                    label="Input size"
                    value={`${liveImageSize ?? detail.inputSize} px`}
                    hint={
                      isLive
                        ? 'reported by the last run'
                        : detail.inputSizeSource === 'checkpoint'
                          ? 'read from checkpoint config'
                          : 'default for this checkpoint'
                    }
                  />
                  <Field
                    label="Preprocessing"
                    value={`${PREPROCESSING_RESIZE} · ${PREPROCESSING_NORMALIZATION}`}
                  />
                </dl>
              </div>
            )
          })}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          icon={<LayersIcon />}
          title="Explainability"
          subtitle="Grad-CAM configuration used for the Explainability screen"
        />
        <CardBody>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field
              label="Target layer"
              value={liveTargetLayer ?? GRADCAM_TARGET_LAYER}
              hint={liveTargetLayer ? 'reported by the last run' : undefined}
              mono
            />
            <Field
              label="CAM grid"
              value={
                Array.isArray(gradCam?.grid_shape)
                  ? `${gradCam.grid_shape.join(' × ')} (before upsampling)`
                  : 'Not computed yet'
              }
              mono
            />
            <Field
              label="Gradients"
              value={
                typeof gradCam?.gradients_are_nonzero === 'boolean'
                  ? gradCam.gradients_are_nonzero
                    ? 'Reached the network'
                    : 'Backward pass did not reach the network'
                  : 'Not computed yet'
              }
            />
          </dl>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          icon={<UploadIcon />}
          title="Input requirements"
          subtitle="What the screening API accepts"
        />
        <CardBody>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Modalities" value="CFP, UWF" />
            <Field
              label="Max upload size"
              value={`${MAX_UPLOAD_BYTES / (1024 * 1024)} MB`}
            />
            <Field label="Accepted formats" value="JPEG, PNG, BMP, TIFF, WEBP" />
          </dl>
        </CardBody>
      </Card>
    </div>
  )
}

function Field({
  label,
  value,
  hint,
  mono = false,
  wide = false,
}: {
  label: string
  value: string
  hint?: string
  mono?: boolean
  wide?: boolean
}) {
  return (
    <div className={wide ? 'sm:col-span-2' : undefined}>
      <dt className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
        {label}
      </dt>
      <dd
        className={`mt-1 break-words text-[14px] font-medium text-ink-800 ${
          mono ? 'font-mono text-[13px]' : ''
        }`}
      >
        {value}
      </dd>
      {hint ? (
        <p className="mt-0.5 text-[11px] leading-relaxed text-ink-400">{hint}</p>
      ) : null}
    </div>
  )
}