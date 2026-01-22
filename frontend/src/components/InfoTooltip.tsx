/**
 * InfoTooltip Component
 * مكون تلميحات المعلومات للأزرار والعناصر
 * يعرض شرح توضيحي عند تمرير الماوس
 */
import React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { HelpCircle, Info } from "lucide-react";
import { cn } from "../lib/utils";

interface InfoTooltipProps {
  content: string;
  children?: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  showIcon?: boolean;
  iconType?: "help" | "info";
  className?: string;
}

export function InfoTooltip({
  content,
  children,
  side = "top",
  showIcon = true,
  iconType = "info",
  className,
}: InfoTooltipProps) {
  const Icon = iconType === "help" ? HelpCircle : Info;

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("inline-flex items-center gap-1 cursor-help", className)}>
            {children}
            {showIcon && (
              <Icon className="h-4 w-4 text-muted-foreground hover:text-primary transition-colors" />
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent 
          side={side} 
          className="max-w-[300px] text-sm bg-popover text-popover-foreground border shadow-lg z-50"
        >
          <p className="leading-relaxed">{content}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * ButtonTooltip Component
 * مكون تلميحات للأزرار - يظهر الشرح عند تمرير الماوس على الزر
 */
interface ButtonTooltipProps {
  content: string;
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
}

export function ButtonTooltip({
  content,
  children,
  side = "top",
}: ButtonTooltipProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          {children}
        </TooltipTrigger>
        <TooltipContent 
          side={side}
          className="max-w-[250px] text-sm bg-popover text-popover-foreground border shadow-lg z-50"
        >
          <p>{content}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export default InfoTooltip;
