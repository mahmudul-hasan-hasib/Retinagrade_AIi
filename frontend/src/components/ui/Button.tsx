import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'sm' | 'md' | 'lg' | 'xl'

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
    'bg-brand-600 text-white shadow-sm shadow-brand-900/40 ring-1 ring-brand-500/30 hover:bg-brand-500 active:bg-brand-600 active:brightness-95 disabled:bg-ink-200 disabled:text-ink-500 disabled:shadow-none disabled:ring-0',
  secondary:
    'bg-white/[0.06] text-ink-800 ring-1 ring-white/12 backdrop-blur-sm hover:bg-white/[0.1] hover:text-ink-900 active:bg-white/[0.14] disabled:text-ink-400',
  ghost:
    'bg-transparent text-ink-500 hover:bg-white/[0.06] hover:text-ink-900 active:bg-white/[0.1] disabled:text-ink-400',
}

const SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 gap-1.5 px-2.5 text-xs',
  md: 'h-10 gap-2 px-3.5 text-sm',
  lg: 'h-12 gap-2 px-5 text-sm',
  /** Full-width primary action. */
  xl: 'h-14 gap-2 px-6 text-[15px] font-semibold',
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
