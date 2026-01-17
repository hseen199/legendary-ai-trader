import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold transition-all duration-300",
  {
    variants: {
      variant: {
        default:
          "border-violet-500/25 bg-violet-500/15 text-violet-300 shadow-[0_0_10px_rgba(139,92,246,0.15)]",
        secondary:
          "border-purple-500/25 bg-purple-500/15 text-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.15)]",
        success:
          "border-emerald-500/25 bg-emerald-500/15 text-emerald-400 shadow-[0_0_10px_rgba(34,197,94,0.15)]",
        warning:
          "border-amber-500/25 bg-amber-500/15 text-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.15)]",
        destructive:
          "border-red-500/25 bg-red-500/15 text-red-400 shadow-[0_0_10px_rgba(239,68,68,0.15)]",
        outline:
          "border-violet-500/40 bg-transparent text-violet-300",
        ghost:
          "border-transparent bg-white/10 text-white/80",
        glow:
          "border-violet-500/30 bg-violet-500/20 text-violet-200 shadow-[0_0_15px_rgba(139,92,246,0.3)] animate-pulse",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants }
