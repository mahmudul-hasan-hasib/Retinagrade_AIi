import type { DrClass, Modality, ModalityKey, NavItemId } from '../types'

export const APP_NAME = 'RetinaGrade AI'
export const APP_TAGLINE = 'Diabetic Retinopathy Screening'
/**
 * Base URL of the RetinaGrade AI backend.
 *
 * Read from `VITE_API_URL` at build time so the same source tree can be built
 * against a local API or a deployed one:
 *
 *   development  VITE_API_URL unset or blank -> http://localhost:8000
 *               (copy `frontend/.env.example` to `frontend/.env` to set it)
 *   production   VITE_API_URL=https://<backend-host>  -> that host
 *
 * Blank counts as unset on purpose. A deployment platform that declares
 * `VITE_API_URL` as a value-to-fill (Render does, with `sync: false`) will hand
 * the build an empty string if the operator skips it, and an empty string is not
 * nullish: `??` would keep it, every endpoint would collapse to a same-origin
 * path like `/predict`, and the deployed page would quietly call itself instead
 * of the API. Falling back to localhost is the same class of silent failure, so
 * the value is validated once, here, rather than being trusted.
 *
 * Vite inlines `import.meta.env.VITE_*` into the bundle, so this must stay a
 * plain property access on `import.meta.env` - no dynamic lookup. Surrounding
 * whitespace and trailing slashes are stripped so
 * `${BACKEND_BASE_URL}/predict` never doubles up or points at nothing.
 */
const configuredApiUrl = (import.meta.env.VITE_API_URL ?? '').trim()

export const BACKEND_BASE_URL = (
  configuredApiUrl || 'http://localhost:8000'
).replace(/\/+$/, '')

/** Kept identical to the backend upload cap (25 MB). */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024

export const ACCEPTED_IMAGE_TYPES = [
  'image/jpeg',
  'image/png',
  'image/tiff',
  'image/webp',
  'image/bmp',
]

export const MODALITIES: readonly Modality[] = [
  {
    key: 'cfp',
    label: 'CFP',
    shortLabel: 'CFP',
    fullName: 'Color Fundus Photography',
    description:
      'Colour Fundus Photography. A centred circular fundus view of the posterior pole.',
    fieldOfView: 'Circular posterior pole',
  },
  {
    key: 'uwf',
    label: 'UWF',
    shortLabel: 'UWF',
    fullName: 'Ultra-Widefield',
    description:
      'Ultra-Wide Field. A single wide capture covering the full retina including the periphery.',
    fieldOfView: 'Full retina + periphery',
  },
] as const

/**
 * Logit order of the classification head (5 classes).
 * Source of truth: `backend/inference/predictor.py::DR_CLASSES`.
 */
export const DR_CLASSES: readonly DrClass[] = [
  {
    index: 0,
    key: 'No DR',
    label: 'No DR',
    shortLabel: 'No DR',
    fullLabel: 'No Diabetic Retinopathy',
    description: 'No apparent diabetic retinopathy',
    severity: 'none',
  },
  {
    index: 1,
    key: 'Mild',
    label: 'Mild',
    shortLabel: 'Mild',
    fullLabel: 'Mild Diabetic Retinopathy',
    description: 'Mild non-proliferative DR',
    severity: 'mild',
  },
  {
    index: 2,
    key: 'Moderate',
    label: 'Moderate',
    shortLabel: 'Moderate',
    fullLabel: 'Moderate Diabetic Retinopathy',
    description: 'Moderate non-proliferative DR',
    severity: 'moderate',
  },
  {
    index: 3,
    key: 'Severe',
    label: 'Severe',
    shortLabel: 'Severe',
    fullLabel: 'Severe Diabetic Retinopathy',
    description: 'Severe non-proliferative DR',
    severity: 'severe',
  },
  {
    index: 4,
    key: 'Proliferative DR',
    label: 'Proliferative DR',
    shortLabel: 'Proliferative',
    fullLabel: 'Proliferative Diabetic Retinopathy',
    description: 'Proliferative diabetic retinopathy',
    severity: 'proliferative',
  },
] as const

/** Static severity tint used for bar / chip colour. Never encodes a probability. */
export const SEVERITY_STYLES: Record<
  DrClass['severity'],
  { chip: string; bar: string; dot: string }
> = {
  none: {
    chip: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    bar: 'bg-emerald-500',
    dot: 'bg-emerald-500',
  },
  mild: {
    chip: 'bg-amber-50 text-amber-700 ring-amber-200',
    bar: 'bg-amber-400',
    dot: 'bg-amber-400',
  },
  moderate: {
    chip: 'bg-orange-50 text-orange-700 ring-orange-200',
    bar: 'bg-orange-500',
    dot: 'bg-orange-500',
  },
  severe: {
    chip: 'bg-rose-50 text-rose-700 ring-rose-200',
    bar: 'bg-rose-500',
    dot: 'bg-rose-500',
  },
  proliferative: {
    chip: 'bg-red-50 text-red-700 ring-red-200',
    bar: 'bg-red-600',
    dot: 'bg-red-600',
  },
}

export interface NavItem {
  id: NavItemId
  label: string
}

/**
 * Primary navigation.
 *
 * Exactly the three screens that exist and work, in workflow order: screen an
 * image, inspect why the model graded it, then confirm which model and
 * checkpoint produced the grade. Nothing else is listed, so nothing can be
 * opened that has no content behind it.
 */
export const NAV_ITEMS: readonly NavItem[] = [
  { id: 'analyze', label: 'Analyze Image' },
  { id: 'explainability', label: 'Explainability' },
  { id: 'model', label: 'Model' },
] as const

/**
 * Checkpoint each modality will eventually load.
 * Source of truth: `backend/inference/model_loader.py::MODEL_SPECS`.
 * (The repo README still lists the older `efficient_*.pt` names.)
 */
export const MODEL_FILE_NAMES: Record<ModalityKey, string> = {
  cfp: 'EfficientNetB0_CFP_final.pth',
  uwf: 'EfficientNetB0_UWF_final.pt',
}

/** Source of truth: `backend/inference/model_loader.py::ARCHITECTURE_NAME`. */
export const MODEL_ARCHITECTURE = 'EfficientNet-B0'

/** Source of truth: `backend/inference/model_architecture.py`. */
export const MODEL_IMPLEMENTATION = 'torchvision.models.efficientnet_b0'

/** ICDR grades produced by the classification head (`DR_CLASSES.length`). */
export const MODEL_NUM_CLASSES = DR_CLASSES.length

/** Source of truth: `backend/inference/model_loader.py::CPU_DEVICE`. */
export const INFERENCE_DEVICE = 'CPU'

/** Source of truth: `backend/inference/gradcam.py::GRADCAM_TARGET_LAYER`. */
export const GRADCAM_TARGET_LAYER = 'features.8'

/** Source of truth: `backend/inference/preprocessing.py`. */
export const PREPROCESSING_RESIZE = 'bilinear'
export const PREPROCESSING_NORMALIZATION = 'ImageNet mean / std'

export interface ModalityModelDetail {
  modality: ModalityKey
  checkpoint: string
  /** `multi_head` (CFP) or `single_head` (UWF) parameter layout. */
  headLayout: 'multi_head' | 'single_head'
  inputSize: number
  /**
   * `checkpoint` - the size is read out of the checkpoint config.
   * `default`     - the checkpoint states no size, so the fallback is used.
   */
  inputSizeSource: 'checkpoint' | 'default'
}

/**
 * Static per-modality model configuration, mirrored from the backend loader and
 * preprocessing modules. Display only: nothing here is sent to the API, and the
 * values reported by a live run take precedence on screen.
 */
export const MODEL_DETAILS: Record<ModalityKey, ModalityModelDetail> = {
  cfp: {
    modality: 'cfp',
    checkpoint: MODEL_FILE_NAMES.cfp,
    headLayout: 'multi_head',
    inputSize: 224,
    inputSizeSource: 'default',
  },
  uwf: {
    modality: 'uwf',
    checkpoint: MODEL_FILE_NAMES.uwf,
    headLayout: 'single_head',
    inputSize: 512,
    inputSizeSource: 'checkpoint',
  },
}

export function getModality(key: ModalityKey): Modality {
  const found = MODALITIES.find((m) => m.key === key)
  if (!found) {
    throw new Error(`Unknown modality: ${key}`)
  }
  return found
}

export function isSupportedModality(value: string): value is ModalityKey {
  return MODALITIES.some((m) => m.key === value)
}

/** Look up one DR grade by its stable key, e.g. `Moderate`. */
export function getDrClass(key: string): DrClass | undefined {
  return DR_CLASSES.find((drClass) => drClass.key === key)
}
