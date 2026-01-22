/**
 * ConfirmationDialog Component
 * مكون رسائل التأكيد والتوضيح قبل العمليات المهمة
 */
import React from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./ui/alert-dialog";
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  HelpCircle,
  Wallet,
  ArrowUpDown,
  Shield
} from 'lucide-react';
import { cn } from '../lib/utils';

type DialogType = 'info' | 'warning' | 'success' | 'confirm';

interface ConfirmationDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  description: string;
  confirmText?: string;
  cancelText?: string;
  type?: DialogType;
  details?: string[];
  isLoading?: boolean;
}

const typeConfig = {
  info: {
    icon: Info,
    iconColor: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
  },
  warning: {
    icon: AlertTriangle,
    iconColor: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
  },
  success: {
    icon: CheckCircle,
    iconColor: 'text-green-500',
    bgColor: 'bg-green-500/10',
  },
  confirm: {
    icon: HelpCircle,
    iconColor: 'text-primary',
    bgColor: 'bg-primary/10',
  },
};

export function ConfirmationDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  description,
  confirmText = 'تأكيد',
  cancelText = 'إلغاء',
  type = 'confirm',
  details,
  isLoading = false,
}: ConfirmationDialogProps) {
  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="max-w-md" dir="rtl">
        <AlertDialogHeader>
          <div className="flex items-center gap-4">
            <div className={cn("p-3 rounded-full", config.bgColor)}>
              <Icon className={cn("h-6 w-6", config.iconColor)} />
            </div>
            <AlertDialogTitle className="text-xl">{title}</AlertDialogTitle>
          </div>
          <AlertDialogDescription className="text-base leading-relaxed mt-4">
            {description}
          </AlertDialogDescription>
          
          {/* تفاصيل إضافية */}
          {details && details.length > 0 && (
            <div className="mt-4 p-4 bg-muted rounded-lg space-y-2">
              {details.map((detail, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                  <span>{detail}</span>
                </div>
              ))}
            </div>
          )}
        </AlertDialogHeader>
        
        <AlertDialogFooter className="flex-row-reverse gap-2 mt-6">
          <AlertDialogAction
            onClick={onConfirm}
            disabled={isLoading}
            className="flex-1"
          >
            {isLoading ? 'جاري المعالجة...' : confirmText}
          </AlertDialogAction>
          <AlertDialogCancel onClick={onClose} className="flex-1">
            {cancelText}
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/**
 * DepositConfirmDialog - رسالة تأكيد الإيداع
 */
interface DepositConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  amount: number;
  network: string;
  fee: number;
  isLoading?: boolean;
}

export function DepositConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  amount,
  network,
  fee,
  isLoading,
}: DepositConfirmDialogProps) {
  const netAmount = amount - fee;
  
  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="max-w-md" dir="rtl">
        <AlertDialogHeader>
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-full bg-green-500/10">
              <Wallet className="h-6 w-6 text-green-500" />
            </div>
            <AlertDialogTitle className="text-xl">تأكيد الإيداع</AlertDialogTitle>
          </div>
          
          <AlertDialogDescription className="text-base leading-relaxed mt-4">
            أنت على وشك إنشاء طلب إيداع. يرجى مراجعة التفاصيل التالية:
          </AlertDialogDescription>
          
          <div className="mt-4 p-4 bg-muted rounded-lg space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">المبلغ المطلوب:</span>
              <span className="font-bold">${amount.toFixed(2)} USDC</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">الشبكة:</span>
              <span className="font-medium">{network}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">رسوم المعالجة (1%):</span>
              <span className="text-yellow-500">-${fee.toFixed(2)}</span>
            </div>
            <div className="border-t pt-3 flex justify-between items-center">
              <span className="text-muted-foreground">المبلغ الصافي:</span>
              <span className="font-bold text-green-500">${netAmount.toFixed(2)} USDC</span>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-yellow-500/10 rounded-lg flex items-start gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-yellow-600 dark:text-yellow-400">
              تأكد من إرسال USDC فقط على شبكة {network}. إرسال أي عملة أخرى قد يؤدي لخسارة الأموال.
            </p>
          </div>
        </AlertDialogHeader>
        
        <AlertDialogFooter className="flex-row-reverse gap-2 mt-6">
          <AlertDialogAction
            onClick={onConfirm}
            disabled={isLoading}
            className="flex-1 bg-green-600 hover:bg-green-700"
          >
            {isLoading ? 'جاري الإنشاء...' : 'إنشاء طلب الإيداع'}
          </AlertDialogAction>
          <AlertDialogCancel onClick={onClose} className="flex-1">
            إلغاء
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/**
 * WithdrawConfirmDialog - رسالة تأكيد السحب
 */
interface WithdrawConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  amount: number;
  address: string;
  fee: number;
  isLoading?: boolean;
}

export function WithdrawConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  amount,
  address,
  fee,
  isLoading,
}: WithdrawConfirmDialogProps) {
  const netAmount = amount - fee;
  
  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="max-w-md" dir="rtl">
        <AlertDialogHeader>
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-full bg-orange-500/10">
              <ArrowUpDown className="h-6 w-6 text-orange-500" />
            </div>
            <AlertDialogTitle className="text-xl">تأكيد السحب</AlertDialogTitle>
          </div>
          
          <AlertDialogDescription className="text-base leading-relaxed mt-4">
            أنت على وشك سحب أموال من حسابك. يرجى التأكد من صحة العنوان.
          </AlertDialogDescription>
          
          <div className="mt-4 p-4 bg-muted rounded-lg space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">المبلغ:</span>
              <span className="font-bold">${amount.toFixed(2)} USDC</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">رسوم السحب:</span>
              <span className="text-yellow-500">-${fee.toFixed(2)}</span>
            </div>
            <div className="border-t pt-3 flex justify-between items-center">
              <span className="text-muted-foreground">المبلغ المستلم:</span>
              <span className="font-bold text-green-500">${netAmount.toFixed(2)} USDC</span>
            </div>
            <div className="border-t pt-3">
              <span className="text-muted-foreground text-sm">عنوان المحفظة:</span>
              <p className="font-mono text-xs mt-1 break-all bg-background p-2 rounded">
                {address}
              </p>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-red-500/10 rounded-lg flex items-start gap-2">
            <Shield className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-600 dark:text-red-400">
              تأكد من صحة العنوان! لا يمكن استرداد الأموال المرسلة لعنوان خاطئ.
            </p>
          </div>
        </AlertDialogHeader>
        
        <AlertDialogFooter className="flex-row-reverse gap-2 mt-6">
          <AlertDialogAction
            onClick={onConfirm}
            disabled={isLoading}
            className="flex-1 bg-orange-600 hover:bg-orange-700"
          >
            {isLoading ? 'جاري المعالجة...' : 'تأكيد السحب'}
          </AlertDialogAction>
          <AlertDialogCancel onClick={onClose} className="flex-1">
            إلغاء
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export default ConfirmationDialog;
