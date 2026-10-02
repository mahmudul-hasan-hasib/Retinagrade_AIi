/**
 * Domain types for the RetinaGrade AI dashboard.
 *
 * The 5 DR classes and their index order mirror the model output contract in
 * `backend/inference/predictor.py` (`DR_CLASSES` / `PROBABILITY_KEYS`), so the
 * UI is already aligned with the order the network will emit.
 *
 * Three result surfaces are live, each from its own endpoint: the `/predict`
 * grade and 5-class distribution, the `/gradcam` heatmap, and the `/explain`
 * Gemini prose. Every field is optional, so a partial response renders with a
 * placeholder rather than crashing the page.
 */

export type ModalityKey = 'cfp' | 'uwf'

export interface Modality {
  key: ModalityKey
  /** Display name, matches the backend `MODALITY_DISPLAY_NAMES`. */
  label: string
  /** Short form used in dense UI (badges, table cells). */
  shortLabel: string
  /** Plain-language explanation shown under the selector. */
  description: string
  /** Expected framing / field of view of the capture. */
  fieldOfView: string
}

export interface DrClass {
  index: number
  /** Stable key; identical to `label` so bars and results can never drift. */
  key: string
  label: string
  description: string
  /**
   * Static, semantic severity used only for iconography and bar tinting.
   * NOT a probability.
   */
  severity: 'none' | 'mild' | 'moderate' | 'severe' | 'proliferative'
}

export type PredictionStatus = 'idle' | 'analyzing' | 'analyzed' | 'error'

/**
 * A real DR prediction, mirrored from
 * `inference/predictor.py::PredictionResult.to_dict()` via
 * `backend/main.py::PredictionModel`.
 *
 * `probabilities` is the full 5-class ICDR softmax distribution keyed by
 * `DR_CLASSES[i].key`, and `predicted_class` is its argmax index.
 */
export interface PredictApiPrediction {
  /** Display name from the backend, e.g. `CFP` / `UWF`. */
  modality?: string
  model?: string
  predicted_class?: number
  label?: string
  description?: string
  confidence?: number
  probabilities?: Record<string, number>
  image_size?: number
  device?: string
  checkpoint?: string
  preprocessing?: Record<string, unknown>
}

/**
 * Body returned by `POST /predict`.
 *
 * A successful call returns `inference_ms` plus the full `prediction` object.
 * Failures return `{ success: false, error: { code, message } }` instead. Every
 * field is optional so an unexpected server response renders instead of
 * crashing the page.
 */
export interface PredictApiResponse {
  success?: boolean
  /** Wall-clock milliseconds spent inside `predict()`. */
  inference_ms?: number
  prediction?: PredictApiPrediction
  error?: {
    code?: string
    message?: string
  }
}

/**
 * A single analysis row.
 *
 * Reserved for the screening-history surface, which is not built yet - see
 * `NAV_ITEMS` in `data/clinical.ts`. The fields are `null` until results are
 * actually persisted; they are never faked.
 */
export interface AnalysisRecord {
  id: string
  createdAt: string | null
  modality: ModalityKey
  fileName: string | null
  thumbnailUrl: string | null
  label: string | null
  confidence: number | null
  probabilities: Record<string, number> | null
}

export type NavItemId =
  | 'dashboard'
  | 'analyze'
  | 'history'
  | 'explainability'
  | 'reports'
  | 'settings'
