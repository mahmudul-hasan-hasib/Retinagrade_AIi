import type { DrClass, Modality, ModalityKey, NavItemId } from '../types'

export const APP_NAME = 'RetinaGrade AI'
export const APP_TAGLINE = 'Diabetic Retinopathy Screening'
export const APP_VERSION = '0.1.0-ui'
export const BACKEND_BASE_URL = 'http://127.0.0.1:8000'

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
    description:
      'Colour Fundus Photography. A centred circular fundus view of the posterior pole.',
    fieldOfView: 'Circular posterior pole',
  },
  {
    key: 'uwf',
    label: 'UWF',
    shortLabel: 'UWF',
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
    description: 'No apparent diabetic retinopathy',
    severity: 'none',
  },
  {
    index: 1,
    key: 'Mild',
    label: 'Mild',
    description: 'Mild non-proliferative DR',
    severity: 'mild',
  },
  {
    index: 2,
    key: 'Moderate',
    label: 'Moderate',
    description: 'Moderate non-proliferative DR',
    severity: 'moderate',
  },
  {
    index: 3,
    key: 'Severe',
    label: 'Severe',
    description: 'Severe non-proliferative DR',
    severity: 'severe',
  },
  {
    index: 4,
    key: 'Proliferative DR',
    label: 'Proliferative DR',
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
  hint: string
  implemented: boolean
}

export const NAV_ITEMS: readonly NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', hint: 'Overview', implemented: true },
  {
    id: 'analyze',
    label: 'Analyze Image',
    hint: 'Upload & run',
    implemented: true,
  },
  { id: 'history', label: 'Screening History', hint: 'Past studies', implemented: false },
  {
    id: 'explainability',
    label: 'Explainability',
    hint: 'Grad-CAM',
    implemented: true,
  },
  {
    id: 'reports',
    label: 'AI Reports',
    hint: 'Gemini summary',
    implemented: true,
  },
  { id: 'settings', label: 'Settings', hint: 'Models & device', implemented: false },
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
