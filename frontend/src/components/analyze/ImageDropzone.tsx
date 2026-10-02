import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { ACCEPTED_IMAGE_TYPES } from '../../data/clinical'
import { AlertIcon, UploadIcon } from '../icons'
import { ImagePreview } from './ImagePreview'

export interface ImageDropzoneProps {
  file: File | null
  previewUrl: string | null
  modalityLabel: string
  error: string | null
  onSelect: (file: File | null) => void
  disabled?: boolean
}

const ACCEPT_ATTRIBUTE = ACCEPTED_IMAGE_TYPES.join(',')

/**
 * One upload card: an empty drop target before a capture exists, and that
 * capture's preview afterwards, so the page never shows the same image twice.
 *
 * Selection is local: the file is handed to the parent for a preview and only
 * leaves the browser when "Analyze Image" triggers the `/predict` request.
 */
export function ImageDropzone({
  file,
  previewUrl,
  modalityLabel,
  error,
  onSelect,
  disabled = false,
}: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  function openPicker() {
    if (!disabled) {
      inputRef.current?.click()
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    onSelect(event.target.files?.[0] ?? null)
    // Allow re-selecting the same file after a failed validation.
    event.target.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) {
      return
    }
    onSelect(event.dataTransfer.files?.[0] ?? null)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  const fileInput = (
    <input
      ref={inputRef}
      type="file"
      accept={ACCEPT_ATTRIBUTE}
      className="sr-only"
      onChange={handleInputChange}
      tabIndex={-1}
      aria-hidden="true"
    />
  )

  if (file && previewUrl) {
    return (
      <div>
        <ImagePreview
          previewUrl={previewUrl}
          fileName={file.name}
          modalityLabel={modalityLabel}
          onChangeImage={openPicker}
          disabled={disabled}
        />
        {fileInput}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled || undefined}
        aria-label="Upload retinal image"
        onClick={openPicker}
        onKeyDown={handleKeyDown}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) {
            setIsDragging(true)
          }
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={[
          'flex flex-col items-center justify-center gap-5 rounded-2xl border border-dashed px-6 py-16 text-center transition-colors duration-150 sm:py-20',
          disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
          isDragging
            ? 'border-brand-500/60 bg-brand-500/[0.08]'
            : 'border-white/[0.12] bg-white/[0.02] hover:border-brand-500/40 hover:bg-brand-500/[0.03]',
        ].join(' ')}
      >
        <span className="grid size-14 place-items-center rounded-2xl bg-white/[0.05] text-ink-400 ring-1 ring-white/10">
          <UploadIcon className="size-6" />
        </span>

        <span className="space-y-1.5">
          <span className="block text-[16px] font-semibold tracking-tight text-ink-900">
            Upload retinal image
          </span>
          <span className="block text-[13px] leading-relaxed text-ink-500">
            Drag &amp; drop or{' '}
            <span className="font-medium text-ink-700">browse</span>
          </span>
        </span>
      </div>

      {error ? <UploadError message={error} /> : null}

      {fileInput}
    </div>
  )
}

/** Rejection of a file the browser could not accept, stated once and plainly. */
function UploadError({ message }: { message: string }) {
  return (
    <p
      role="alert"
      className="flex items-start gap-2 rounded-xl border border-brand-500/25 bg-brand-500/[0.07] px-3.5 py-2.5 text-[13px] leading-relaxed text-ink-500"
    >
      <AlertIcon className="mt-px size-3.5 shrink-0 text-brand-700" />
      {message}
    </p>
  )
}