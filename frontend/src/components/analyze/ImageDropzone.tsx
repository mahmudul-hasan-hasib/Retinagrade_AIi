import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_BYTES } from '../../data/clinical'
import { AlertIcon, ImageIcon, TrashIcon, UploadIcon } from '../icons'
import { Button } from '../ui/Button'

export interface ImageDropzoneProps {
  file: File | null
  error: string | null
  onSelect: (file: File | null) => void
  onClear: () => void
  accept?: string
}

const ACCEPT_ATTRIBUTE = ACCEPTED_IMAGE_TYPES.join(',')

/**
 * Drag-and-drop / click-to-browse upload area.
 *
 * Selection is local: the file is handed to the parent for a preview and only
 * leaves the browser when "Analyze Image" triggers the `/predict` request.
 */
export function ImageDropzone({
  file,
  error,
  onSelect,
  onClear,
  accept = ACCEPT_ATTRIBUTE,
}: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    onSelect(event.target.files?.[0] ?? null)
    // Allow re-selecting the same file after a failed validation.
    event.target.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    onSelect(event.dataTransfer.files?.[0] ?? null)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      inputRef.current?.click()
    }
  }

  if (file) {
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 rounded-xl border border-line bg-ink-50/60 px-3.5 py-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-surface text-brand-600 ring-1 ring-line">
            <ImageIcon className="size-4.5" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-ink-800" title={file.name}>
              {file.name}
            </p>
            <p className="mt-0.5 font-mono text-[11px] text-ink-500">
              {file.type || 'unknown type'} &middot; {(file.size / 1024).toFixed(0)} KB
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            icon={<TrashIcon />}
            aria-label="Remove selected image"
          >
            Remove
          </Button>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={handleInputChange}
        />
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload retinal image"
        onClick={() => inputRef.current?.click()}
        onKeyDown={handleKeyDown}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={[
          'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors',
          isDragging
            ? 'border-brand-400 bg-brand-50/60'
            : 'border-ink-200 bg-ink-50/40 hover:border-brand-300 hover:bg-brand-50/30',
        ].join(' ')}
      >
        <span className="grid size-12 place-items-center rounded-2xl bg-surface text-brand-600 ring-1 ring-line shadow-sm">
          <UploadIcon className="size-5.5" />
        </span>
        <span className="space-y-1">
          <span className="block text-sm font-medium text-ink-800">
            Drop a retinal image here
          </span>
          <span className="block text-xs text-ink-500">
            or <span className="font-medium text-brand-700">browse files</span>
          </span>
        </span>
        <span className="mt-1 font-mono text-[11px] text-ink-400">
          JPEG &middot; PNG &middot; TIFF &middot; WEBP &middot; BMP &mdash; max{' '}
          {MAX_UPLOAD_BYTES / (1024 * 1024)} MB
        </span>
      </div>

      {error ? (
        <p
          role="alert"
          className="flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 ring-1 ring-red-200"
        >
          <AlertIcon className="mt-px size-3.5 shrink-0" />
          {error}
        </p>
      ) : null}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={handleInputChange}
      />
    </div>
  )
}
