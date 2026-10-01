import type { PredictApiResponse } from '../../types'
import { AlertIcon, CheckIcon, CpuIcon } from '../icons'
import { Badge } from '../ui/Badge'
import { Card, CardBody, CardHeader } from '../ui/Card'

export interface ApiResponseCardProps {
  /** `POST /predict` response body, rendered verbatim. */
  payload: PredictApiResponse | null
  httpStatus: number | null
  /** Transport / server error message, when the request did not succeed. */
  error: string | null
}

/**
 * Raw output of `POST /predict`.
 *
 * Deliberately unstructured: the endpoint performs upload validation only, so
 * this card shows the JSON exactly as returned instead of projecting it onto
 * clinical fields that do not exist yet.
 */
export function ApiResponseCard({
  payload,
  httpStatus,
  error,
}: ApiResponseCardProps) {
  const failed = Boolean(error)
  const hasResponse = payload !== null

  return (
    <Card>
      <CardHeader
        icon={<CpuIcon />}
        title="Backend Response"
        subtitle="POST /predict - upload validation only"
        action={
          <Badge tone={failed ? 'warning' : hasResponse ? 'success' : 'muted'}>
            {failed ? 'Failed' : hasResponse ? `HTTP ${httpStatus ?? '-'}` : 'No request yet'}
          </Badge>
        }
      />

      <CardBody className="space-y-4">
        {error ? (
          <p
            role="alert"
            className="flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 ring-1 ring-red-200"
          >
            <AlertIcon className="mt-px size-3.5 shrink-0" />
            {error}
          </p>
        ) : null}

        {hasResponse && !error ? (
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-700">
            <CheckIcon className="size-3.5" />
            Request completed and the response was received.
          </div>
        ) : null}

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            Response body
          </p>
          <pre className="max-h-64 overflow-auto rounded-xl border border-line bg-ink-900 px-3.5 py-3 font-mono text-[11px] leading-relaxed text-ink-100">
            {hasResponse
              ? JSON.stringify(payload, null, 2)
              : '// Press "Analyze Image" to call the backend.'}
          </pre>
        </div>
      </CardBody>
    </Card>
  )
}
