import { useState } from 'react'
import type { SyntheticEvent } from 'react'
import { MAX_UPLOAD_BYTES } from '../../data/clinical'
import { UploadIcon } from '../icons'

export interface ImagePreviewProps {
  previewUrl: string
  fileName: string
  modalityLabel: string
  onChangeImage: () => void
  disabled?: boolean
}

/** Formats offered in the picker, as short labels. */
export const SUPPORTED_FORMAT_LABELS = 'JPG · PNG · TIFF'

/** Upper bound on one upload, matching the server limit. */
export const SUPPORTED_SIZE_LABEL = `Up to ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB`

/**
 * Selected-capture state of the upload area.
 *
 * Shows the image itself plus who it is: file name, chosen modality and - only
 * once the browser has decoded the local preview - its pixel size. No file size,
 * MIME type, checksum or model detail: none of that helps judge a capture.
 */
export function ImagePreview({
  previewUrl,
  fileName,
  modalityLabel,
  onChangeImage,
  disabled = false,
}: ImagePreviewProps) {
  /**
   * Measured size, tagged with the capture it belongs to.
   *
   * Deriving the label from the tag means a newly selected image can never show
   * the previous file's dimensions, without an effect that would reset state
   * after the render that needs it.
   */
  const [measured, setMeasured] = useState<{ url: string; label: string } | null>(
    null,
  )
  const dimensions = measured?.url === previewUrl ? measured.label : null

  function handleLoad(event: SyntheticEvent<HTMLImageElement>) {
    const image = event.currentTarget
    if (!image.naturalWidth || !image.naturalHeight) {
      return
    }
    setMeasured({
      url: previewUrl,
      label: `${image.naturalWidth} × ${image.naturalHeight} px`,
    })
  }

  return (
    <div className="overflow-hidden rounded-[14px] border border-line bg-surface">
      <div className="flex flex-col items-center gap-4 p-5 sm:flex-row sm:items-center">
        <img
          src={previewUrl}
          alt={`Retinal image selected for analysis: ${fileName}`}
          onLoad={handleLoad}
          className="size-28 shrink-0 rounded-xl bg-ink-900 object-contain"
        />

        <div className="min-w-0 flex-1 text-center sm:text-left">
          <p className="text-[11px] font-medium tracking-wide text-ink-400 uppercase">
            Selected image
          </p>
          <p
            className="mt-1 truncate text-[15px] font-semibold text-ink-900"
            title={fileName}
          >
            {fileName}
          </p>

          <dl className="mt-2 flex flex-wrap items-center justify-center gap-x-5 gap-y-1 text-[13px] sm:justify-start">
            <div className="flex items-center gap-1.5">
              <dt className="text-ink-400">Modality</dt>
              <dd className="font-medium text-ink-700">{modalityLabel}</dd>
            </div>
            {dimensions ? (
              <div className="flex items-center gap-1.5">
                <dt className="text-ink-400">Dimensions</dt>
                <dd className="font-medium text-ink-700">{dimensions}</dd>
              </div>
            ) : null}
          </dl>

          <button
            type="button"
            onClick={onChangeImage}
            disabled={disabled}
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg text-[13px] font-medium text-brand-700 transition-colors hover:text-brand-800 disabled:cursor-not-allowed disabled:text-ink-300"
          >
            <UploadIcon className="size-3.5" />
            Change image
          </button>
        </div>
      </div>
    </div>
  )
}