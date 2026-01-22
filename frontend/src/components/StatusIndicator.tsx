/**
 * StatusIndicator Component
 * مكون مؤشرات الحالة للعمليات المختلفة
 */
import React from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Loader2,
  ArrowUpCircle,
  ArrowDownCircle,
  RefreshCw,
} from 'lucide-react';
import { cn } from '../lib/utils';

type StatusType = 
  | 'success' 
  | 'error' 
  | 'pending' 
  | 'warning' 
  | 'loading'
  | 'deposit'
  | 'withdraw'
  | 'processing';

interface StatusConfig {
  icon: React.ElementType;
  color: string;
  bgColor: string;
  label: string;
  labelEn: string;
}

const statusConfigs: Record<StatusType, StatusConfig> = {
  success: {
    icon: CheckCircle,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    label: 'مكتمل',
    labelEn: 'Completed',
  },
  error: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    label: 'فشل',
    labelEn: 'Failed',
  },
  pending: {
    icon: Clock,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    label: 'قيد الانتظار',
    labelEn: 'Pending',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    label: 'تحذير',
    labelEn: 'Warning',
  },
  loading: {
    icon: Loader2,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    label: 'جاري التحميل',
    labelEn: 'Loading',
  },
  deposit: {
    icon: ArrowDownCircle,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    label: 'إيداع',
    labelEn: 'Deposit',
  },
  withdraw: {
    icon: ArrowUpCircle,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    label: 'سحب',
    labelEn: 'Withdraw',
  },
  processing: {
    icon: RefreshCw,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    label: 'قيد المعالجة',
    labelEn: 'Processing',
  },
};

interface StatusIndicatorProps {
  status: StatusType;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  language?: 'ar' | 'en';
  className?: string;
  animate?: boolean;
}

export function StatusIndicator({
  status,
  showLabel = true,
  size = 'md',
  language = 'ar',
  className,
  animate = true,
}: StatusIndicatorProps) {
  const config = statusConfigs[status];
  const Icon = config.icon;
  const isRTL = language === 'ar';

  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  };

  const textSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  const isAnimated = animate && (status === 'loading' || status === 'processing');

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <motion.div
        initial={animate ? { scale: 0.8, opacity: 0 } : false}
        animate={{ scale: 1, opacity: 1 }}
        className={cn("p-1 rounded-full", config.bgColor)}
      >
        <Icon 
          className={cn(
            sizeClasses[size], 
            config.color,
            isAnimated && "animate-spin"
          )} 
        />
      </motion.div>
      {showLabel && (
        <span className={cn(textSizes[size], config.color, "font-medium")}>
          {isRTL ? config.label : config.labelEn}
        </span>
      )}
    </div>
  );
}

/**
 * StatusBadge - شارة الحالة المدمجة
 */
interface StatusBadgeProps {
  status: StatusType;
  language?: 'ar' | 'en';
  className?: string;
}

export function StatusBadge({ status, language = 'ar', className }: StatusBadgeProps) {
  const config = statusConfigs[status];
  const Icon = config.icon;
  const isRTL = language === 'ar';

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
        config.bgColor,
        config.color,
        className
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {isRTL ? config.label : config.labelEn}
    </span>
  );
}

/**
 * ProgressStatus - مؤشر التقدم مع الحالة
 */
interface ProgressStatusProps {
  steps: {
    id: string;
    label: string;
    labelEn: string;
    status: 'completed' | 'current' | 'upcoming';
  }[];
  language?: 'ar' | 'en';
  className?: string;
}

export function ProgressStatus({ steps, language = 'ar', className }: ProgressStatusProps) {
  const isRTL = language === 'ar';

  return (
    <div className={cn("flex items-center justify-between", className)} dir={isRTL ? 'rtl' : 'ltr'}>
      {steps.map((step, index) => (
        <React.Fragment key={step.id}>
          <div className="flex flex-col items-center">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium",
                step.status === 'completed' && "bg-green-500 text-white",
                step.status === 'current' && "bg-primary text-primary-foreground",
                step.status === 'upcoming' && "bg-muted text-muted-foreground"
              )}
            >
              {step.status === 'completed' ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                index + 1
              )}
            </motion.div>
            <span className={cn(
              "text-xs mt-2 text-center max-w-[80px]",
              step.status === 'current' ? "text-primary font-medium" : "text-muted-foreground"
            )}>
              {isRTL ? step.label : step.labelEn}
            </span>
          </div>
          
          {index < steps.length - 1 && (
            <div className={cn(
              "flex-1 h-0.5 mx-2",
              step.status === 'completed' ? "bg-green-500" : "bg-muted"
            )} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

/**
 * TransactionStatus - حالة المعاملة المفصلة
 */
interface TransactionStatusProps {
  type: 'deposit' | 'withdraw';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  amount: number;
  date: string;
  txHash?: string;
  language?: 'ar' | 'en';
}

export function TransactionStatus({
  type,
  status,
  amount,
  date,
  txHash,
  language = 'ar',
}: TransactionStatusProps) {
  const isRTL = language === 'ar';
  const typeConfig = statusConfigs[type];
  const statusConfig = statusConfigs[status === 'processing' ? 'processing' : status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'pending'];
  const TypeIcon = typeConfig.icon;

  return (
    <div className="flex items-center justify-between p-4 bg-card rounded-lg border" dir={isRTL ? 'rtl' : 'ltr'}>
      <div className="flex items-center gap-3">
        <div className={cn("p-2 rounded-full", typeConfig.bgColor)}>
          <TypeIcon className={cn("h-5 w-5", typeConfig.color)} />
        </div>
        <div>
          <p className="font-medium">
            {isRTL ? typeConfig.label : typeConfig.labelEn}
          </p>
          <p className="text-sm text-muted-foreground">{date}</p>
        </div>
      </div>
      
      <div className="text-left">
        <p className={cn(
          "font-bold",
          type === 'deposit' ? "text-green-500" : "text-orange-500"
        )}>
          {type === 'deposit' ? '+' : '-'}${amount.toFixed(2)}
        </p>
        <StatusBadge status={status === 'completed' ? 'success' : status === 'failed' ? 'error' : status} language={language} />
      </div>
    </div>
  );
}

export default StatusIndicator;
