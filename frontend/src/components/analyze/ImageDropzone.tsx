import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { ACCEPTED_IMAGE_TYPES } from '../../data/clinical'
import { AlertIcon, UploadIcon } from '../icons'
import {
  ImagePreview,
  SUPPORTED_FORMAT_LABELS,
  SUPPORTED_SIZE_LABEL,
} from './ImagePreview'

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
 * Drag-and-drop / click-to-browse upload area.
 *
 * Selection is local: the file is handed to the parent for a preview and only
 * leaves the browser when "Analyze Image" triggers the `/predict` request.
 *
 * Once a capture is chosen this same area becomes its preview, so the page never
 * shows the same image twice.
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
      <div className="space-y-2">
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
    <div className="space-y-2">
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
          'flex flex-col items-center justify-center gap-4 rounded-[14px] border border-dashed px-6 py-14 text-center transition-colors',
          disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer',
          isDragging
            ? 'border-brand-400 bg-brand-50/60'
            : 'border-ink-200 bg-ink-50/50 hover:border-brand-300 hover:bg-brand-50/30',
        ].join(' ')}
      >
        <span className="grid size-12 place-items-center rounded-2xl bg-surface text-brand-600 ring-1 ring-line">
          <UploadIcon className="size-5.5" />
        </span>

        <span className="space-y-1.5">
          <span className="block text-[15px] font-semibold text-ink-900">
            Upload retinal image
          </span>
          <span className="block text-[13px] leading-relaxed text-ink-500">
            Drag and drop your image here or{' '}
            <span className="font-medium text-brand-700">browse from your computer</span>
          </span>
        </span>

        <span className="text-[12px] text-ink-400">
          {SUPPORTED_FORMAT_LABELS} &middot; {SUPPORTED_SIZE_LABEL}
        </span>
      </div>

      {error ? (
        <p
          role="alert"
          className="flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-[13px] text-red-700 ring-1 ring-red-200"
        >
          <AlertIcon className="mt-px size-3.5 shrink-0" />
          {error}
        </p>
      ) : null}

      {fileInput}
    </div>
  )
}