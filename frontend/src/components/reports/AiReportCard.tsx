import { getModality } from '../../data/clinical'
import type { ModalityKey, PredictionStatus } from '../../types'
import { CpuIcon, DocumentIcon, SparkIcon } from '../icons'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { Skeleton } from '../ui/Feedback'
import { Notice, Placeholder } from '../ui/Placeholder'

export interface AiReportCardProps {
  fileName: string | null
  modality: ModalityKey
  status: PredictionStatus
}

/** Blocks the generator is expected to fill, in report order. */
const REPORT_SECTIONS = [
  { key: 'findings', label: 'Imaging findings' },
  { key: 'impression', label: 'Clinical impression' },
  { key: 'recommendation', label: 'Recommended action' },
] as const

const REPORT_META = [
  { label: 'Generator', value: 'Gemini' },
  { label: 'Prompt version', value: '-' },
  { label: 'Generated at', value: '-' },
  { label: 'Tokens', value: '-' },
] as const

/**
 * AI-generated report section.
 *
 * This build performs no generation of any kind, so every prose block below is
 * a deliberately inert dashed placeholder rather than sample text.
 */
export function AiReportCard({ fileName, modality, status }: AiReportCardProps) {
  const hasImage = Boolean(fileName)
  const hasRun = status === 'analyzed'
  const modalityLabel = getModality(modality).label

  return (
    <Card>
      <CardHeader
        icon={<DocumentIcon />}
        title="AI Report"
        subtitle="Structured summary of the screening result"
        action={
          <Badge tone={hasRun ? 'brand' : 'muted'}>
            {hasRun ? 'Awaiting generation' : 'Not started'}
          </Badge>
        }
      />

      <CardBody className="space-y-4">
        <Notice
          tone="info"
          icon={<SparkIcon />}
          title="Report generation is not connected"
        >
          The report is composed from the prediction payload on the server. No
          model, inference or language generation runs in this build, so no
          clinical text is produced.
        </Notice>

        <div className="space-y-3">
          <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            Summary
          </p>
          <div className="space-y-2 rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3.5 py-3">
            <div className="space-y-2">
              <Skeleton className="h-2 w-full" />
              <Skeleton className="h-2 w-11/12" />
              <Skeleton className="h-2 w-4/5" />
            </div>
            <p className="text-[11px] leading-relaxed text-ink-400">
              {hasRun
                ? 'Four sentences of narrative summary will appear here once a report is generated from a real result.'
                : 'Run the analysis to unlock the report sections.'}
            </p>
          </div>
        </div>

        <div className="space-y-3">
          {REPORT_SECTIONS.map((section) => (
            <div key={section.key} className="space-y-1.5">
              <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
                {section.label}
              </p>
              <div className="space-y-2 rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-3.5 py-2.5">
                <Skeleton className="h-2 w-10/12" />
                <Skeleton className="h-2 w-7/12" />
              </div>
            </div>
          ))}
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <Placeholder
            label="Referral advice"
            value="-"
            hint="Derived from the predicted grade."
          />
          <Placeholder
            label="Confidence in report"
            value="-"
            hint="Generator self-reported reliability."
          />
        </div>

        <div>
          <p className="mb-2 flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            <CpuIcon className="size-3" />
            Generation metadata
          </p>
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {REPORT_META.map((entry) => (
              <div
                key={entry.label}
                className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line"
              >
                <dt className="truncate text-[11px] text-ink-500">{entry.label}</dt>
                <dd className="mt-0.5 font-mono text-xs font-medium text-ink-700">
                  {entry.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="space-y-2 border-t border-line pt-3">
          <div className="flex items-center justify-between gap-3 text-[11px]">
            <span className="text-ink-500">Source capture</span>
            <span className="truncate font-mono text-ink-400" title={fileName ?? ''}>
              {fileName ? `${fileName} (${modalityLabel})` : '-'}
            </span>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Button variant="secondary" size="sm" block disabled>
              Generate report
            </Button>
            <Button variant="ghost" size="sm" block disabled>
              Export PDF
            </Button>
          </div>
          <p className="text-[11px] leading-relaxed text-ink-400">
            {hasImage
              ? 'Buttons stay disabled until the generator is wired up.'
              : 'Upload an image on the Analyze page to enable reporting.'}
          </p>
        </div>
      </CardBody>
    </Card>
  )
}
