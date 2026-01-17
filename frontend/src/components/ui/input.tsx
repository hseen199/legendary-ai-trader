import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-11 w-full rounded-xl border border-violet-500/15 bg-[rgba(18,18,28,0.6)] px-4 py-2.5 text-base text-white ring-offset-[#08080c] transition-all duration-300 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-white placeholder:text-white/35 focus-visible:outline-none focus-visible:border-violet-500/50 focus-visible:ring-2 focus-visible:ring-violet-500/20 focus-visible:shadow-[0_0_20px_rgba(139,92,246,0.15)] focus-visible:bg-[rgba(18,18,28,0.8)] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
