import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'sm' | 'md' | 'lg'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  icon?: ReactNode
  iconRight?: ReactNode
  block?: boolean
  children?: ReactNode
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    'bg-brand-600 text-white shadow-sm ring-1 ring-brand-700/20 hover:bg-brand-700 active:bg-brand-800 disabled:bg-ink-300 disabled:ring-0',
  secondary:
    'bg-surface text-ink-700 ring-1 ring-line hover:bg-ink-50 active:bg-ink-100 disabled:text-ink-400',
  ghost:
    'bg-transparent text-ink-600 hover:bg-ink-100 active:bg-ink-200 disabled:text-ink-300',
}

const SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 gap-1.5 px-2.5 text-xs',
  md: 'h-10 gap-2 px-3.5 text-sm',
  lg: 'h-12 gap-2 px-5 text-sm',
}

export function Button({
  variant = 'primary',
  size = 'md',
  icon,
  iconRight,
  block = false,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={[
        'inline-flex items-center justify-center rounded-xl font-medium transition-colors',
        'disabled:cursor-not-allowed',
        VARIANTS[variant],
        SIZES[size],
        block ? 'w-full' : '',
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
      {...rest}
    >
      {icon ? <span className="shrink-0 [&>svg]:size-4">{icon}</span> : null}
      {children ? <span className="truncate">{children}</span> : null}
      {iconRight ? (
        <span className="shrink-0 [&>svg]:size-4">{iconRight}</span>
      ) : null}
    </button>
  )
}
