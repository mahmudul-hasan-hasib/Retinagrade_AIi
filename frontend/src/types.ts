/**
 * Domain types for the RetinaGrade AI dashboard.
 *
 * The 5 DR classes and their index order mirror the model output contract in
 * `backend/inference/predictor.py` (`DR_CLASSES` / `PROBABILITY_KEYS`), so the
 * UI is already aligned with the order the network will emit.
 *
 * NOTE: no inference, Grad-CAM or report value exists yet. Every result field
 * in this app is `null` and is rendered as a placeholder.
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

export type PredictionStatus = 'idle' | 'analyzed'

/**
 * A single analysis row. `label` / `confidence` / `probabilities` are
 * `null` until real inference is wired up - they are never faked.
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
