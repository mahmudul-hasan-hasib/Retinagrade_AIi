import { UploadIcon } from '../icons'

export interface ImagePreviewProps {
  previewUrl: string
  fileName: string
  modalityLabel: string
  onChangeImage: () => void
  disabled?: boolean
}

/**
 * Selected-capture state of the upload card.
 *
 * Exactly what someone needs to confirm they picked the right scan: the image,
 * its filename, and the modality it will be graded as. No pixel dimensions, file
 * size, MIME type or checksum - none of it changes the decision being made here.
 */
export function ImagePreview({
  previewUrl,
  fileName,
  modalityLabel,
  onChangeImage,
  disabled = false,
}: ImagePreviewProps) {
  return (
    <div className="rounded-2xl border border-line bg-white/[0.02] p-4 sm:p-5">
      <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center">
        {/*
          The capture is shown on true black with no filter, tint or blend of any
          kind: a fundus image has to stay colour-accurate to be worth reading.
          The frame around it is what adapts to the theme, not the image.
        */}
        <img
          src={previewUrl}
          alt={`Retinal image selected for analysis: ${fileName}`}
          className="size-32 shrink-0 rounded-xl bg-black object-contain ring-1 ring-white/10"
        />

        <div className="min-w-0 flex-1 text-center sm:text-left">
          <p
            className="truncate text-[15px] font-semibold tracking-tight text-ink-900"
            title={fileName}
          >
            {fileName}
          </p>

          <p className="mt-2 flex items-center justify-center gap-2 text-[13px] sm:justify-start">
            <span className="text-ink-500">Modality</span>
            <span className="font-medium text-ink-800">{modalityLabel}</span>
          </p>

          <button
            type="button"
            onClick={onChangeImage}
            disabled={disabled}
            className="mt-4 inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] font-medium text-ink-500 transition-colors duration-150 hover:border-white/[0.16] hover:bg-white/[0.05] hover:text-ink-900 disabled:cursor-not-allowed disabled:text-ink-400"
          >
            <UploadIcon className="size-3.5" />
            Change image
          </button>
        </div>
      </div>
    </div>
  )
}