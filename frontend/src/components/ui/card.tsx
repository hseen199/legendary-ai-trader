import * as React from "react"

import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-2xl border border-violet-500/15 bg-[rgba(18,18,28,0.6)] backdrop-blur-xl text-white shadow-lg transition-all duration-400 hover:border-violet-500/30 hover:shadow-[0_8px_40px_rgba(139,92,246,0.12)] hover:-translate-y-0.5 relative overflow-hidden",
      "before:absolute before:top-0 before:left-0 before:right-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-violet-500/30 before:to-transparent",
      className
    )}
    {...props}
  />
));
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "text-xl font-bold leading-none tracking-tight bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm text-white/50", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

// Premium Glass Card with Glow
const CardGlass = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-2xl border border-violet-500/20 bg-[rgba(18,18,28,0.7)] backdrop-blur-2xl text-white shadow-[0_0_40px_rgba(139,92,246,0.08)] transition-all duration-400 hover:border-violet-500/40 hover:shadow-[0_0_60px_rgba(139,92,246,0.15)] relative overflow-hidden",
      "before:absolute before:top-0 before:left-0 before:right-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-violet-500/40 before:to-transparent",
      className
    )}
    {...props}
  />
));
CardGlass.displayName = "CardGlass"

// Stat Card with Side Accent
const CardStat = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { 
    variant?: 'default' | 'success' | 'warning' | 'danger' | 'purple'
  }
>(({ className, variant = 'default', ...props }, ref) => {
  const accentColors = {
    default: 'before:bg-violet-500',
    success: 'before:bg-emerald-500',
    warning: 'before:bg-amber-500',
    danger: 'before:bg-red-500',
    purple: 'before:bg-gradient-to-b before:from-violet-500 before:to-purple-500',
  }
  
  return (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl border border-violet-500/12 bg-[rgba(18,18,28,0.6)] backdrop-blur-xl p-6 transition-all duration-400 hover:border-violet-500/30 hover:-translate-y-1 hover:shadow-[0_12px_40px_rgba(139,92,246,0.12)] relative overflow-hidden",
        "before:absolute before:top-4 before:bottom-4 before:right-0 before:w-1 before:rounded-full",
        "after:absolute after:top-0 after:right-0 after:w-20 after:h-20 after:bg-[radial-gradient(circle,rgba(139,92,246,0.1)_0%,transparent_70%)] after:rounded-tr-2xl",
        accentColors[variant],
        className
      )}
      {...props}
    />
  )
});
CardStat.displayName = "CardStat"

// Feature Card with Icon
const CardFeature = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "group rounded-2xl border border-violet-500/15 bg-[rgba(18,18,28,0.5)] backdrop-blur-xl p-6 transition-all duration-400 hover:border-violet-500/40 hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)] hover:-translate-y-1 relative overflow-hidden",
      className
    )}
    {...props}
  />
));
CardFeature.displayName = "CardFeature"

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
  CardGlass,
  CardStat,
  CardFeature,
}
