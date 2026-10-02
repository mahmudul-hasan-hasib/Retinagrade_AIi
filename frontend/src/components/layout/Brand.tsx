import { APP_NAME } from '../../data/clinical'
import { RetinaIcon } from '../icons'

export interface BrandProps {
  size?: 'sm' | 'md'
  /** Mark only, for the collapsed sidebar. Keeps the same icon box size. */
  collapsed?: boolean
}

/** Product mark plus wordmark. No tagline: the mark is enough to place it. */
export function Brand({ size = 'md', collapsed = false }: BrandProps) {
  const boxSize = size === 'sm' ? 'size-8' : 'size-9'
  const iconSize = size === 'sm' ? 'size-4.5' : 'size-5'

  return (
    <div
      className={`flex min-w-0 items-center ${collapsed ? 'justify-center' : 'gap-2.5'}`}
    >
      <span
        className={`${boxSize} grid shrink-0 place-items-center rounded-xl bg-brand-600 text-white ring-1 ring-brand-500/30`}
      >
        <RetinaIcon className={iconSize} />
      </span>

      {collapsed ? null : (
        <span className="min-w-0 truncate text-sm leading-tight font-semibold tracking-tight text-ink-900">
          {APP_NAME}
        </span>
      )}
    </div>
  )
}