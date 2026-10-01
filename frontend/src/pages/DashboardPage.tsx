import {
  CheckIcon,
  CpuIcon,
  DocumentIcon,
  HistoryIcon,
  LayersIcon,
  SparkIcon,
  UploadIcon,
} from '../components/icons'
import { Badge } from '../components/ui/Badge'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Notice, Placeholder } from '../components/ui/Placeholder'
import { MODALITIES, MODEL_FILE_NAMES } from '../data/clinical'

const STATS = [
  { label: 'Screened today', hint: 'Completed studies' },
  { label: 'Mean confidence', hint: 'Average top-class probability' },
  { label: 'DR-positive rate', hint: 'Grade >= mild' },
  { label: 'Awaiting review', hint: 'Unconfirmed grades' },
] as const

const WORKFLOW = [
  {
    title: 'Choose modality',
    detail: 'CFP or UWF selects which checkpoint is loaded.',
    Icon: LayersIcon,
  },
  {
    title: 'Upload capture',
    detail: 'Local preview only; validated against backend limits.',
    Icon: UploadIcon,
  },
  {
    title: 'Run prediction',
    detail: 'EfficientNet-B0 head returns 5 DR class logits.',
    Icon: SparkIcon,
  },
  {
    title: 'Review evidence',
    detail: 'Grad-CAM highlights the regions driving the grade.',
    Icon: CpuIcon,
  },
  {
    title: 'Generate report',
    detail: 'Structured summary assembled from the result payload.',
    Icon: DocumentIcon,
  },
] as const

/** Landing view: workspace status plus an overview of the screening flow. */
export function DashboardPage() {
  return (
    <>
      <Notice
        tone="info"
        icon={<SparkIcon />}
        title="All figures on this page are placeholders"
      >
        Screening history is not implemented, so the counters below have no
        data source and stay empty rather than showing sample numbers.
      </Notice>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {STATS.map((stat) => (
          <Card key={stat.label} className="p-4">
            <Placeholder label={stat.label} value="-" hint={stat.hint} />
          </Card>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-2 xl:items-start">
        <Card>
          <CardHeader
            icon={<CpuIcon />}
            title="Model Status"
            subtitle="Checkpoints expected by the inference service"
          />
          <CardBody className="space-y-3">
            {MODALITIES.map((modality) => (
              <div
                key={modality.key}
                className="flex items-start justify-between gap-3 rounded-xl bg-ink-50/70 px-3.5 py-3 ring-1 ring-line"
              >
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-ink-800">
                    {modality.label} &middot; EfficientNet-B0
                  </p>
                  <p className="mt-0.5 truncate font-mono text-[11px] text-ink-500">
                    {MODEL_FILE_NAMES[modality.key]}
                  </p>
                </div>
                <Badge tone="muted">Not loaded</Badge>
              </div>
            ))}

            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line">
                <p className="text-ink-500">Device</p>
                <p className="mt-0.5 font-mono font-medium text-ink-400">CPU</p>
              </div>
              <div className="rounded-lg bg-ink-50/70 px-2.5 py-2 ring-1 ring-line">
                <p className="text-ink-500">Classes</p>
                <p className="mt-0.5 font-mono font-medium text-ink-400">5</p>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            icon={<HistoryIcon />}
            title="Screening Flow"
            subtitle="Steps of a single study, from capture to report"
          />
          <CardBody>
            <ol className="space-y-3">
              {WORKFLOW.map((step, index) => (
                <li key={step.title} className="flex items-start gap-3">
                  <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg bg-brand-50 text-brand-700 ring-1 ring-brand-100">
                    <step.Icon className="size-3.5" />
                  </span>
                  <div className="min-w-0 flex-1 border-b border-line pb-3 last:border-0 last:pb-0">
                    <p className="flex items-center gap-2 text-xs font-semibold text-ink-800">
                      <span className="font-mono text-[10px] text-ink-400">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      {step.title}
                    </p>
                    <p className="mt-0.5 text-[11px] leading-relaxed text-ink-500">
                      {step.detail}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          icon={<CheckIcon />}
          title="Recent Studies"
          subtitle="Completed screenings appear here"
        />
        <CardBody>
          <EmptyState
            icon={<HistoryIcon />}
            title="No studies recorded"
            description="Screening history is not implemented yet. Once results are persisted, each study will list its modality, predicted grade, confidence and review status."
          />
        </CardBody>
      </Card>
    </>
  )
}
