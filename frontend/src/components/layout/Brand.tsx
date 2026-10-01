import { APP_NAME, APP_TAGLINE } from '../../data/clinical'
import { RetinaIcon } from '../icons'

export interface BrandProps {
  size?: 'sm' | 'md'
  showTagline?: boolean
}

export function Brand({ size = 'md', showTagline = true }: BrandProps) {
  const boxSize = size === 'sm' ? 'size-8' : 'size-10'
  const iconSize = size === 'sm' ? 'size-4.5' : 'size-5.5'

  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span
        className={`${boxSize} grid shrink-0 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-sm shadow-brand-700/25`}
      >
        <RetinaIcon className={iconSize} />
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm leading-tight font-bold tracking-tight text-ink-900">
          {APP_NAME}
        </span>
        {showTagline ? (
          <span className="block truncate text-[11px] leading-tight text-ink-500">
            {APP_TAGLINE}
          </span>
        ) : null}
      </span>
    </div>
  )
}
