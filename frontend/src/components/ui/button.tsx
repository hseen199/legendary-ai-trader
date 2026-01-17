import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 focus-visible:ring-offset-2 focus-visible:ring-offset-[#08080c] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-to-r from-violet-600 via-violet-500 to-purple-500 text-white shadow-[0_4px_20px_rgba(139,92,246,0.35)] hover:shadow-[0_6px_30px_rgba(139,92,246,0.5)] hover:-translate-y-0.5 active:translate-y-0 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/20 before:to-transparent before:-translate-x-full hover:before:translate-x-full before:transition-transform before:duration-500",
        destructive:
          "bg-gradient-to-r from-red-600 to-red-500 text-white shadow-[0_4px_20px_rgba(239,68,68,0.35)] hover:shadow-[0_6px_30px_rgba(239,68,68,0.5)] hover:-translate-y-0.5",
        outline:
          "border border-violet-500/30 bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 hover:border-violet-500/50 hover:shadow-[0_0_20px_rgba(139,92,246,0.2)]",
        secondary:
          "bg-violet-500/15 text-violet-300 border border-violet-500/20 hover:bg-violet-500/25 hover:border-violet-500/40",
        ghost:
          "text-violet-300 hover:bg-violet-500/10 hover:text-violet-200",
        success:
          "bg-gradient-to-r from-emerald-600 to-emerald-500 text-white shadow-[0_4px_20px_rgba(34,197,94,0.35)] hover:shadow-[0_6px_30px_rgba(34,197,94,0.5)] hover:-translate-y-0.5",
        warning:
          "bg-gradient-to-r from-amber-600 to-amber-500 text-white shadow-[0_4px_20px_rgba(245,158,11,0.35)] hover:shadow-[0_6px_30px_rgba(245,158,11,0.5)] hover:-translate-y-0.5",
        link:
          "text-violet-400 underline-offset-4 hover:underline hover:text-violet-300",
        glow:
          "bg-gradient-to-r from-violet-600 via-violet-500 to-purple-500 text-white shadow-[0_0_20px_rgba(139,92,246,0.4)] hover:shadow-[0_0_40px_rgba(139,92,246,0.6)] animate-pulse-slow",
      },
      size: {
        default: "min-h-10 px-5 py-2.5",
        sm: "min-h-8 rounded-lg px-4 text-xs",
        lg: "min-h-12 rounded-xl px-8 text-base",
        xl: "min-h-14 rounded-2xl px-10 text-lg",
        icon: "h-10 w-10 rounded-xl",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  },
)
Button.displayName = "Button"

export { Button, buttonVariants }
