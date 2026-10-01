import { MODALITIES } from '../../data/clinical'
import type { ModalityKey } from '../../types'
import { RetinaIcon } from '../icons'

export interface ModalitySelectorProps {
  value: ModalityKey
  onChange: (next: ModalityKey) => void
  disabled?: boolean
}

/**
 * Segmented CFP / UWF selector. The selected modality decides which model will
 * run once inference is connected - CFP for a centred fundus photo, UWF for a
 * wide-field capture.
 */
export function ModalitySelector({
  value,
  onChange,
  disabled = false,
}: ModalitySelectorProps) {
  const active = MODALITIES.find((m) => m.key === value) ?? MODALITIES[0]

  return (
    <fieldset className="space-y-2.5" disabled={disabled}>
      <legend className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
        Imaging modality
      </legend>

      <div
        role="radiogroup"
        aria-label="Imaging modality"
        className="grid gap-2 sm:grid-cols-2"
      >
        {MODALITIES.map((modality) => {
          const isSelected = modality.key === value
          return (
            <button
              key={modality.key}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(modality.key)}
              className={[
                'group flex items-start gap-3 rounded-xl border px-3.5 py-3 text-left transition-all',
                isSelected
                  ? 'border-brand-300 bg-brand-50/70 ring-2 ring-brand-500/25'
                  : 'border-line bg-surface hover:border-ink-300 hover:bg-ink-50/60',
                disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer',
              ].join(' ')}
            >
              <span
                className={[
                  'mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg ring-1 transition-colors',
                  isSelected
                    ? 'bg-brand-600 text-white ring-brand-600'
                    : 'bg-ink-50 text-ink-400 ring-line',
                ].join(' ')}
              >
                <RetinaIcon className="size-4.5" />
              </span>

              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <span
                    className={`text-sm font-semibold ${isSelected ? 'text-brand-800' : 'text-ink-800'}`}
                  >
                    {modality.label}
                  </span>
                  {isSelected ? (
                    <span className="size-1.5 rounded-full bg-brand-600" />
                  ) : null}
                </span>
                <span className="mt-0.5 block text-[11px] leading-relaxed text-ink-500">
                  {modality.fieldOfView}
                </span>
              </span>
            </button>
          )
        })}
      </div>

      <p className="text-xs leading-relaxed text-ink-500">{active.description}</p>
    </fieldset>
  )
}
