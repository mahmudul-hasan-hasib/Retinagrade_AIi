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
 * CFP / UWF as two selectable cards - never a dropdown.
 *
 * The chosen modality decides which model runs, so both options are always
 * visible and carry equal weight. Selection is stated three ways - crimson
 * border, tinted fill, filled check - so it never depends on colour alone.
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
        className="grid gap-3.5 sm:grid-cols-2"
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
                'flex items-center gap-4 rounded-2xl border px-5 py-5 text-left transition-colors duration-150',
                disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
                isSelected
                  ? 'border-brand-500/60 bg-brand-500/[0.08]'
                  : 'border-line bg-white/[0.02] hover:border-white/[0.16] hover:bg-white/[0.04]',
              ].join(' ')}
            >
              <span
                className={[
                  'grid size-11 shrink-0 place-items-center rounded-xl transition-colors duration-150',
                  isSelected
                    ? 'bg-brand-600 text-white'
                    : 'bg-white/[0.05] text-ink-400 ring-1 ring-white/10',
                ].join(' ')}
              >
                <RetinaIcon className="size-5" />
              </span>

              <span className="min-w-0 flex-1">
                <span
                  className={[
                    'block text-[16px] leading-tight font-semibold tracking-tight',
                    isSelected ? 'text-ink-900' : 'text-ink-800',
                  ].join(' ')}
                >
                  {modality.label}
                </span>
                <span className="mt-1 block text-[13px] leading-relaxed text-ink-500">
                  {modality.fullName}
                </span>
              </span>

              <span
                aria-hidden="true"
                className={[
                  'grid size-5 shrink-0 place-items-center rounded-full border transition-colors duration-150',
                  isSelected
                    ? 'border-brand-600 bg-brand-600 text-white'
                    : 'border-white/[0.16] text-transparent',
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