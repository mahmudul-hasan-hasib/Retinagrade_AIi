/**
 * Thin HTTP client for the RetinaGrade AI backend.
 *
 * Scope: exactly one call is implemented, `POST /predict`, which runs the real
 * CFP or UWF model on the backend's CPU and returns a DR grade. No Grad-CAM or
 * report generation happens client-side, and no response field is invented
 * here - the server payload is passed through untouched.
 */
import { BACKEND_BASE_URL } from '../data/clinical'
import type { ModalityKey, PredictApiResponse } from '../types'

export const PREDICT_ENDPOINT = `${BACKEND_BASE_URL}/predict`

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
