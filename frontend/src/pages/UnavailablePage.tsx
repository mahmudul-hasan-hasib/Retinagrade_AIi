import { EmptyState } from '../components/ui/EmptyState'
import { HistoryIcon, SettingsIcon } from '../components/icons'
import { NAV_ITEMS } from '../data/clinical'

export interface UnavailablePageProps {
  itemId: 'history' | 'settings'
}

const ICONS = {
  history: HistoryIcon,
  settings: SettingsIcon,
} as const

const PLANS: Record<UnavailablePageProps['itemId'], readonly string[]> = {
  history: [
    'Persist each study with its modality, predicted grade and confidence.',
    'List past captures with per-study thumbnails and review status.',
    'Export a single study or a batch as PDF.',
  ],
  settings: [
    'Choose which checkpoints to load and when to release them.',
    'Tune Grad-CAM target layer, alpha and upsampling.',
    'Configure the report generator and its prompt version.',
  ],
}

/**
 * Placeholder route for the sections that are still unbuilt. Shown instead of a
 * half-finished screen so nothing here reads as a working feature.
 */
export function UnavailablePage({ itemId }: UnavailablePageProps) {
  const item = NAV_ITEMS.find((entry) => entry.id === itemId)
  const Icon = ICONS[itemId]
  const plans = PLANS[itemId]

  return (
    <EmptyState
      icon={<Icon />}
      title={`${item?.label ?? 'Section'} is not implemented`}
      description="This part of the interface is planned but has no implementation behind it yet, so it is intentionally left empty rather than filled with sample data."
      action={
        <ul className="w-full max-w-md space-y-1.5 text-left">
          {plans.map((plan) => (
            <li
              key={plan}
              className="flex items-start gap-2 rounded-lg bg-ink-50/70 px-3 py-2 text-[11px] leading-relaxed text-ink-500 ring-1 ring-line"
            >
              <span className="mt-1.5 size-1 shrink-0 rounded-full bg-ink-300" />
              {plan}
            </li>
          ))}
        </ul>
      }
    />
  )
}
