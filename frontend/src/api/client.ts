/**
 * Thin HTTP client for the RetinaGrade AI backend.
 *
 * Four calls are implemented:
 *   * `GET  /health`   - liveness plus whether Gemini is configured server-side;
 *   * `POST /predict`  - runs the model and returns a DR grade;
 *   * `POST /gradcam`  - runs a forward *and* backward pass and returns a
 *     Grad-CAM heatmap explaining that grade;
 *   * `POST /explain`  - describes an existing prediction in prose via Gemini.
 *
 * `/predict` and `/gradcam` run the real checkpoints on the backend's CPU.
 * `/explain` runs no model of ours: it posts numbers the caller already has and
 * returns Gemini's prose, so it can never disagree with the grade on screen.
 *
 * No heatmap is computed client-side and no response field is invented here -
 * the server payload is passed through untouched. The CAM arrives as a base64
 * PNG data URI and is used directly as an `<img src>`, so no decoding, colour
 * mapping or overlay compositing happens in the browser.
 */
import { BACKEND_BASE_URL } from '../data/clinical'
import type { ModalityKey, PredictApiResponse } from '../types'

export const HEALTH_ENDPOINT = `${BACKEND_BASE_URL}/health`
export const PREDICT_ENDPOINT = `${BACKEND_BASE_URL}/predict`
export const GRADCAM_ENDPOINT = `${BACKEND_BASE_URL}/gradcam`
export const EXPLAIN_ENDPOINT = `${BACKEND_BASE_URL}/explain`

/** `error.code` the backend returns when GEMINI_API_KEY is absent. */
export const GEMINI_NOT_CONFIGURED_CODE = 'gemini_not_configured'

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

/* ------------------------------------------------------------------ *
 * GET /health
 * ------------------------------------------------------------------ */

/**
 * Body returned by `GET /health`, mirroring `backend/schemas/api.py::HealthResponse`.
 *
 * `gemini_configured` is the single flag that decides whether `POST /explain`
 * can work on this server. It is a bare boolean - the API key, its length and
 * any part of its value are never sent to the client - so it is safe to read
 * before offering the explanation action.
 *
 * Every field is optional so a partial or unexpected response renders instead
 * of crashing the page.
 */
export interface HealthApiResponse {
  status?: string
  /** Always `cpu` on this backend. */
  device?: string
  /** Informational only; this API never selects a CUDA device. */
  cuda_available?: boolean
  gemini_configured?: boolean
}

export interface HealthResult {
  httpStatus: number
  data: HealthApiResponse
}

/**
 * Ask the backend whether it is up and whether Gemini is configured.
 *
 * Called before any explanation request so the UI can say "Gemini is not
 * configured on this server" up front, instead of letting the user click into a
 * 503. It runs no model, so it is cheap enough to call on load.
 */
export async function fetchHealth(): Promise<HealthResult> {
  let response: Response
  try {
    response = await fetch(HEALTH_ENDPOINT, { method: 'GET' })
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
    data: (body ?? {}) as HealthApiResponse,
  }
}

/* ------------------------------------------------------------------ *
 * POST /explain
 * ------------------------------------------------------------------ */

/**
 * Grad-CAM facts handed to `POST /explain`, mirroring
 * `backend/main.py::GradCamEvidence`.
 *
 * Metadata only. The base64 heatmap is deliberately not part of this: the
 * endpoint accepts the fact that a map exists and how strong its gradients
 * were, so no pixels are ever sent off the machine.
 *
 * `available` is false when `/gradcam` failed or was not run for this capture.
 * That is a normal state, not an error - the explanation is still generated
 * from the grade, confidence and probabilities alone.
 */
export interface GradCamEvidence {
  available?: boolean
  target_layer?: string | null
  grid_shape?: number[] | null
  image_size?: number | null
  gradients_are_nonzero?: boolean | null
  is_predicted_class?: boolean | null
  summary?: string | null
}

/**
 * Body of `POST /explain`, mirroring `backend/main.py::ExplainRequest`.
 *
 * This is exactly what `POST /predict` already returned, forwarded unchanged:
 *   * `modality`         - `'cfp'` or `'uwf'`;
 *   * `predicted_class`  - DR grade index 0-4;
 *   * `confidence`       - top-class probability, 0-1;
 *   * `probabilities`    - the full 5-class distribution, keyed by grade label;
 *   * `gradcam`          - optional metadata from a prior `/gradcam` call.
 *
 * No image is sent: the endpoint explains numbers, it does not re-read the
 * photograph.
 */
export interface ExplainApiRequest {
  modality: string
  predicted_class: number
  confidence: number
  probabilities: Record<string, number>
  gradcam?: GradCamEvidence
}

/**
 * The caller's own numbers, echoed back verbatim by the server.
 *
 * This is the audit trail for the rule this endpoint exists to enforce: the
 * explanation is generated *from* these values and cannot have changed them.
 */
export interface ExplainApiSource {
  modality?: string
  predicted_class?: number
  label?: string
  description?: string
  confidence?: number
  probabilities?: Record<string, number>
  gradcam_available?: boolean
}

/**
 * Body of a successful `POST /explain`, mirroring `backend/main.py::ExplainResponse`.
 *
 * `explanation` is Gemini's prose. `disclaimer` is fixed server text, appended
 * regardless of what the model returned, so it cannot be dropped by truncation
 * or a refusal. `model` names the Gemini model for provenance and `latency_ms`
 * measures the Gemini call only - no model of ours runs here.
 *
 * Like `PredictApiResponse`, every field is optional so an unexpected or partial
 * server response renders instead of crashing the page.
 */
export interface ExplainApiResponse {
  success?: boolean
  explanation?: string
  disclaimer?: string
  model?: string
  modality?: string
  latency_ms?: number
  source?: ExplainApiSource
  error?: {
    code?: string
    message?: string
  }
}

export interface ExplainResult {
  httpStatus: number
  data: ExplainApiResponse
}

/**
 * Ask Gemini to explain a prediction that has already been graded.
 *
 * Field names match `backend/main.py::explain_prediction`. The body is JSON
 * (not multipart) because no file is uploaded - the endpoint runs no model of
 * ours and never sees the retinal image.
 *
 * Failures reject with the same `PredictApiError` shape used by `/predict` and
 * `/gradcam`, so callers only handle one error type. A server without a Gemini
 * key rejects with `code === GEMINI_NOT_CONFIGURED_CODE`; callers are expected
 * to check `GET /health` first and avoid triggering that request at all.
 */
export async function requestExplanation(
  payload: ExplainApiRequest,
): Promise<ExplainResult> {
  let response: Response
  try {
    response = await fetch(EXPLAIN_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
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
    data: (body ?? {}) as ExplainApiResponse,
  }
}
