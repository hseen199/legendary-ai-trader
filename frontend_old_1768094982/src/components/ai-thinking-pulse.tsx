import { cn } from "@/lib/utils";

interface AIThinkingPulseProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export function AIThinkingPulse({ className, size = "md" }: AIThinkingPulseProps) {
  const sizeClasses = {
    sm: "w-2 h-2",
    md: "w-3 h-3",
    lg: "w-4 h-4"
  };

  return (
    <div className={cn("flex items-center gap-1", className)}>
      <div className={cn("rounded-full bg-primary animate-pulse", sizeClasses[size])} style={{ animationDelay: "0ms" }} />
      <div className={cn("rounded-full bg-primary animate-pulse", sizeClasses[size])} style={{ animationDelay: "150ms" }} />
      <div className={cn("rounded-full bg-primary animate-pulse", sizeClasses[size])} style={{ animationDelay: "300ms" }} />
    </div>
  );
}

export default AIThinkingPulse;
