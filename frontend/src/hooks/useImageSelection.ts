import { useCallback, useEffect, useRef, useState } from 'react'
import { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_BYTES } from '../data/clinical'

export interface ImageSelection {
  file: File | null
  /** Local `blob:` URL for the preview. Revoked automatically on replace/clear. */
  previewUrl: string | null
  error: string | null
  select: (file: File | null) => void
  clear: () => void
}

/**
 * Client-side-only image selection.
 *
 * Validation mirrors the backend limits (accepted MIME types, 25 MB) so the UI
 * can reject obvious mistakes early. Nothing is uploaded: the file is only ever
 * read locally to build a preview URL.
 */
export function useImageSelection(): ImageSelection {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const objectUrlRef = useRef<string | null>(null)

  const releaseObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
  }, [])

  // Free the last preview URL when the component unmounts.
  useEffect(() => releaseObjectUrl, [releaseObjectUrl])

  const select = useCallback(
    (next: File | null) => {
      releaseObjectUrl()

      if (!next) {
        setFile(null)
        setPreviewUrl(null)
        setError(null)
        return
      }

      const isImage = ACCEPTED_IMAGE_TYPES.includes(next.type)
      if (!isImage) {
        setFile(null)
        setPreviewUrl(null)
        setError(
          `Unsupported file type "${next.type || 'unknown'}". Use JPEG, PNG, TIFF, WEBP or BMP.`,
        )
        return
      }

      if (next.size > MAX_UPLOAD_BYTES) {
        setFile(null)
        setPreviewUrl(null)
        setError(
          `File is larger than the ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB limit.`,
        )
        return
      }

      if (next.size === 0) {
        setFile(null)
        setPreviewUrl(null)
        setError('The selected file is empty (0 bytes).')
        return
      }

      const url = URL.createObjectURL(next)
      objectUrlRef.current = url
      setFile(next)
      setPreviewUrl(url)
      setError(null)
    },
    [releaseObjectUrl],
  )

  const clear = useCallback(() => select(null), [select])

  return { file, previewUrl, error, select, clear }
}
