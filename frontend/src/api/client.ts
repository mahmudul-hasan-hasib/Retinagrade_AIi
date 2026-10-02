/**
 * Thin HTTP client for the RetinaGrade AI backend.
 *
 * Two calls are implemented, both against the same real checkpoints on the
 * backend's CPU:
 *   * `POST /predict` - runs the model and returns a DR grade;
 *   * `POST /gradcam` - runs a forward *and* backward pass and returns a
 *     Grad-CAM heatmap explaining that grade.
 *
 * No heatmap is computed client-side and no response field is invented here -
 * the server payload is passed through untouched. The CAM arrives as a base64
 * PNG data URI and is used directly as an `<img src>`, so no decoding, colour
 * mapping or overlay compositing happens in the browser.
 */
import { BACKEND_BASE_URL } from '../data/clinical'
import type { ModalityKey, PredictApiResponse } from '../types'

export const PREDICT_ENDPOINT = `${BACKEND_BASE_URL}/predict`
export const GRADCAM_ENDPOINT = `${BACKEND_BASE_URL}/gradcam`

export interface PredictResult {
  httpStatus: number
  data: PredictApiResponse
}

/** A non-2xx response or an unreachable server, carrying the status code. */
export class PredictApiError extends Error {
  readonly httpStatus: number
  readonly code: string | null

  constructor(message: string, httpStatus: number, code: string | null) {
    super(message)
    this.name = 'PredictApiError'
    this.httpStatus = httpStatus
    this.code = code
  }
}

type JsonRecord = Record<string, unknown>

async function readBody(response: Response): Promise<JsonRecord | null> {
  const text = await response.text()
  if (!text) {
    return null
  }
  try {
    const parsed: unknown = JSON.parse(text)
    return typeof parsed === 'object' && parsed !== null ? (parsed as JsonRecord) : null
  } catch {
    return null
  }
}

/**
 * Turn a failed response into a readable message.
 *
 * The API returns `{ success: false, error: { code, message } }`; FastAPI's own
 * validation failures come back as `{ detail: ... }` instead.
 */
function readError(body: JsonRecord | null, httpStatus: number): PredictApiError {
  const error = body?.error
  if (typeof error === 'object' && error !== null) {
    const { code, message } = error as { code?: unknown; message?: unknown }
    return new PredictApiError(
      typeof message === 'string' ? message : `Request failed (HTTP ${httpStatus}).`,
      httpStatus,
      typeof code === 'string' ? code : null,
    )
  }

  const detail = body?.detail
  if (typeof detail === 'string' && detail) {
    return new PredictApiError(detail, httpStatus, null)
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return new PredictApiError(
      'The request payload was rejected by the server (HTTP ' + httpStatus + ').',
      httpStatus,
      'invalid_request',
    )
  }

  return new PredictApiError(
    `The backend returned HTTP ${httpStatus}.`,
    httpStatus,
    null,
  )
}

/**
 * Send the selected capture to `POST /predict` as `multipart/form-data`.
 *
 * Field names match `backend/main.py::predict_image`:
 *   * `image`    - the file, sent under its original name;
 *   * `modality` - `'cfp'` or `'uwf'`.
 *
 * `Content-Type` is intentionally left unset so the browser can generate the
 * multipart boundary itself.
 */
export async function predictImage(
  file: File,
  modality: ModalityKey,
): Promise<PredictResult> {
  const form = new FormData()
  form.append('image', file, file.name)
  form.append('modality', modality)

  let response: Response
  try {
    response = await fetch(PREDICT_ENDPOINT, { method: 'POST', body: form })
  } catch {
    throw new PredictApiError(
      `Could not reach the backend at ${BACKEND_BASE_URL}. Is the API server running?`,
      0,
      'network_error',
    )
  }

  const body = await readBody(response)

  if (!response.ok) {
    throw readError(body, response.status)
  }

  return {
    httpStatus: response.status,
    data: (body ?? {}) as PredictApiResponse,
  }
}

/* ------------------------------------------------------------------ *
 * POST /gradcam
 * ------------------------------------------------------------------ */

/**
 * The Grad-CAM heatmap as the backend encodes it.
 *
 * Mirrors `backend/main.py::GradCamImage`. `data_uri` is a complete
 * `data:image/png;base64,...` string, so it is usable as an `<img src>` with no
 * further work. `width` / `height` are the decoded pixel dimensions (the model
 * input size - 224 for CFP, 512 for UWF), not the native Grad-CAM grid, which is
 * reported separately as `grid_shape`.
 */
export interface GradCamImage {
  media_type?: string
  encoding?: string
  width?: number
  height?: number
  data_uri?: string
}

/**
 * Body returned by `POST /gradcam`, mirroring `backend/main.py::GradCamResponse`.
 *
 * `predicted_class` / `label` / `confidence` describe the grade the heatmap
 * explains. When no `target_class` is requested that grade is the model's own
 * prediction; `is_predicted_class` states whether an explicitly requested grade
 * happened to agree with it.
 *
 * Like `PredictApiResponse`, every field is optional so an unexpected or partial
 * server response renders instead of crashing the page.
 */
export interface GradCamApiResponse {
  success?: boolean
  /** Display name from the backend, e.g. `CFP` / `UWF`. */
  modality?: string
  predicted_class?: number
  label?: string
  confidence?: number
  description?: string
  probabilities?: Record<string, number>
  is_predicted_class?: boolean
  /** Dotted path of the trunk layer the heatmap was computed at. */
  target_layer?: string
  /** Native Grad-CAM grid before upsampling, e.g. `[7, 7]`. */
  grid_shape?: number[]
  image_size?: number
  /** False would mean the backward pass never reached the network. */
  gradients_are_nonzero?: boolean
  inference_ms?: number
  device?: string
  checkpoint?: string
  cam?: GradCamImage
  error?: {
    code?: string
    message?: string
  }
}

export interface GradCamResult {
  httpStatus: number
  data: GradCamApiResponse
}

/**
 * Ask the backend for a Grad-CAM heatmap for one capture.
 *
 * Field names match `backend/main.py::gradcam_image`:
 *   * `image`         - the file, sent under its original name;
 *   * `modality`      - `'cfp'` or `'uwf'`;
 *   * `target_class`  - optional DR grade (0-4) to explain. Omitted entirely
 *                       when not requested, rather than sent as an empty string,
 *                       so the backend falls back to the model's own prediction.
 *
 * `Content-Type` is left unset for the same reason as in `predictImage`: the
 * browser must generate the multipart boundary itself.
 *
 * Failures reject with the same `PredictApiError` shape used by `/predict`, so
 * callers only handle one error type. That name is kept rather than split into
 * a second identical class, since it is already the error the app renders from.
 */
export async function requestGradCam(
  file: File,
  modality: ModalityKey,
  targetClass?: number,
): Promise<GradCamResult> {
  const form = new FormData()
  form.append('image', file, file.name)
  form.append('modality', modality)
  if (typeof targetClass === 'number') {
    form.append('target_class', String(targetClass))
  }

  let response: Response
  try {
    response = await fetch(GRADCAM_ENDPOINT, { method: 'POST', body: form })
  } catch {
    throw new PredictApiError(
      `Could not reach the backend at ${BACKEND_BASE_URL}. Is the API server running?`,
      0,
      'network_error',
    )
  }

  const body = await readBody(response)

  if (!response.ok) {
    throw readError(body, response.status)
  }

  return {
    httpStatus: response.status,
    data: (body ?? {}) as GradCamApiResponse,
  }
}
