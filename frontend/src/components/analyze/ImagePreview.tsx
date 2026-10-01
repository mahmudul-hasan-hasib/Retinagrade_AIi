import { EyeIcon, ImageIcon, LayersIcon } from '../icons'
import { Skeleton } from '../ui/Feedback'

export interface ImagePreviewProps {
  previewUrl: string | null
  fileName: string | null
  modalityLabel: string
}

/**
 * Preview surface for the selected fundus image.
 *
 * Renders the local file object URL only - no network access, no decoding to
 * pixels by the app itself.
 */
export function ImagePreview({
  previewUrl,
  fileName,
  modalityLabel,
}: ImagePreviewProps) {
  const hasImage = Boolean(previewUrl)

  return (
    <div className="flex h-full flex-col">
      <div className="relative flex min-h-56 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-ink-900 p-3 sm:min-h-72">
        {hasImage ? (
          <img
            src={previewUrl ?? ''}
            alt={`Uploaded retinal image: ${fileName ?? 'retinal image'}`}
            className="max-h-[26rem] w-auto rounded-lg object-contain"
          />
        ) : (
          <div className="flex w-full max-w-sm flex-col items-center gap-3 py-10 text-center">
            <span className="grid size-12 place-items-center rounded-2xl bg-white/5 text-ink-400 ring-1 ring-white/10">
              <ImageIcon className="size-5.5" />
            </span>
            <p className="text-sm font-medium text-ink-300">No image loaded</p>
            <p className="text-xs leading-relaxed text-ink-500">
              Select or drop a {modalityLabel} capture to preview it here.
            </p>
          </div>
        )}

        {hasImage ? (
          <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur-sm">
            <EyeIcon className="size-3" />
            Local preview
          </span>
        ) : null}
      </div>

      <div className="mt-3 space-y-2.5">
        <div className="flex items-center justify-between gap-3">
          <span className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            Source file
          </span>
          <span className="truncate font-mono text-[11px] text-ink-600" title={fileName ?? ''}>
            {fileName ?? '—'}
          </span>
        </div>

        <div className="space-y-1.5">
          <p className="flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-ink-500 uppercase">
            <LayersIcon className="size-3" />
            Grad-CAM overlay
          </p>
          <div className="grid min-h-24 place-items-center rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-4 text-center">
            <div className="space-y-2">
              <Skeleton className="mx-auto h-2 w-24" />
              <p className="text-[11px] leading-relaxed text-ink-400">
                Heatmap appears here after Grad-CAM is implemented.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
