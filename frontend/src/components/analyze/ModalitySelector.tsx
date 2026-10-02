import { useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { MODALITIES } from '../../data/clinical'
import type { ModalityKey } from '../../types'
import { CheckIcon, RetinaIcon } from '../icons'

export interface ModalitySelectorProps {
  value: ModalityKey
  onChange: (next: ModalityKey) => void
  disabled?: boolean
}

/**
 * Large CFP / UWF selection cards - never a dropdown.
 *
 * The chosen modality decides which model runs, so the two options are given
 * room and are always visible together.
 *
 * Implemented as an ARIA radiogroup with a roving tab stop: one Tab press moves
 * into the group, arrow keys (and Home / End) move between the two options, and
 * each card is a real button so Space and Enter select it.
 */
export function ModalitySelector({
  value,
  onChange,
  disabled = false,
}: ModalitySelectorProps) {
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const activeIndex = MODALITIES.findIndex((modality) => modality.key === value)

  function focusOption(index: number) {
    const bounded = (index + MODALITIES.length) % MODALITIES.length
    optionRefs.current[bounded]?.focus()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        event.preventDefault()
        focusOption(index + 1)
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        event.preventDefault()
        focusOption(index - 1)
        break
      case 'Home':
        event.preventDefault()
        focusOption(0)
        break
      case 'End':
        event.preventDefault()
        focusOption(MODALITIES.length - 1)
        break
      default:
        break
    }
  }

  return (
    <fieldset className="space-y-3" disabled={disabled}>
      <legend className="sr-only">Imaging modality</legend>

      <div
        role="radiogroup"
        aria-label="Imaging modality"
        className="grid gap-3 sm:grid-cols-2"
      >
        {MODALITIES.map((modality, index) => {
          const isSelected = modality.key === value
          return (
            <button
              key={modality.key}
              ref={(node) => {
                optionRefs.current[index] = node
              }}
              type="button"
              role="radio"
              aria-checked={isSelected}
              tabIndex={index === activeIndex ? 0 : -1}
              disabled={disabled}
              onClick={() => onChange(modality.key)}
              onKeyDown={(event) => handleKeyDown(event, index)}
              className={[
                'flex items-center gap-3.5 rounded-[14px] border px-4 py-4 text-left transition-colors',
                disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer',
                isSelected
                  ? 'border-brand-400 bg-brand-50'
                  : 'border-line bg-surface hover:border-ink-300 hover:bg-ink-50/60',
              ].join(' ')}
            >
              <span
                className={[
                  'grid size-10 shrink-0 place-items-center rounded-xl transition-colors',
                  isSelected
                    ? 'bg-brand-600 text-white'
                    : 'bg-ink-50 text-ink-500 ring-1 ring-line',
                ].join(' ')}
              >
                <RetinaIcon className="size-5" />
              </span>

              <span className="min-w-0 flex-1">
                <span className="block text-[15px] font-semibold text-ink-900">
                  {modality.label}
                </span>
                <span className="mt-0.5 block text-[13px] leading-relaxed text-ink-500">
                  {modality.fullName}
                </span>
              </span>

              <span
                aria-hidden="true"
                className={[
                  'grid size-5 shrink-0 place-items-center rounded-full border transition-colors',
                  isSelected
                    ? 'border-brand-600 bg-brand-600 text-white'
                    : 'border-ink-200 bg-surface text-transparent',
                ].join(' ')}
              >
                <CheckIcon className="size-3" />
              </span>
            </button>
          )
        })}
      </div>
    </fieldset>
  )
}