/**
 * TermTooltip Component
 * مكون لعرض شرح المصطلحات التقنية عند التمرير أو النقر
 */
import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { cn } from '../lib/utils';

// قاموس المصطلحات التقنية
export const technicalTerms: Record<string, { ar: string; en: string }> = {
  NAV: {
    ar: 'صافي قيمة الأصول - سعر الوحدة الاستثمارية الواحدة. يتغير يومياً بناءً على أداء المحفظة.',
    en: 'Net Asset Value - The price of one investment unit. Changes daily based on portfolio performance.',
  },
  RSI: {
    ar: 'مؤشر القوة النسبية - مؤشر يقيس سرعة وتغير حركة الأسعار. قيمة فوق 70 تعني "ذروة شراء" وتحت 30 تعني "ذروة بيع".',
    en: 'Relative Strength Index - Measures the speed and change of price movements. Above 70 means "overbought", below 30 means "oversold".',
  },
  MACD: {
    ar: 'مؤشر التقارب والتباعد للمتوسطات المتحركة - يساعد في تحديد اتجاه السوق وقوة الزخم.',
    en: 'Moving Average Convergence Divergence - Helps identify market trend and momentum strength.',
  },
  'Stop Loss': {
    ar: 'وقف الخسارة - أمر تلقائي لإغلاق الصفقة عند وصول السعر لمستوى معين لتقليل الخسائر.',
    en: 'An automatic order to close a trade when price reaches a certain level to minimize losses.',
  },
  'Take Profit': {
    ar: 'جني الأرباح - أمر تلقائي لإغلاق الصفقة عند وصول السعر لمستوى ربح معين.',
    en: 'An automatic order to close a trade when price reaches a certain profit level.',
  },
  Leverage: {
    ar: 'الرافعة المالية - تتيح لك التداول بمبلغ أكبر من رأس مالك. مثال: رافعة 10x تعني أن 100$ تتداول بقوة 1000$.',
    en: 'Allows you to trade with more than your capital. Example: 10x leverage means $100 trades with $1000 power.',
  },
  PnL: {
    ar: 'الربح والخسارة - الفرق بين سعر الشراء وسعر البيع.',
    en: 'Profit and Loss - The difference between buy and sell price.',
  },
  'High-Water Mark': {
    ar: 'علامة الذروة المائية - أعلى قيمة وصل إليها رصيدك. لا تُخصم رسوم أداء إلا عند تجاوزها.',
    en: 'The highest value your balance has reached. Performance fees are only charged when exceeded.',
  },
  Units: {
    ar: 'الوحدات الاستثمارية - حصتك في الصندوق. عند الإيداع تحصل على وحدات بسعر NAV الحالي.',
    en: 'Investment units - Your share in the fund. When depositing, you get units at current NAV price.',
  },
  USDC: {
    ar: 'عملة مستقرة مرتبطة بالدولار الأمريكي. 1 USDC = 1 دولار تقريباً.',
    en: 'A stablecoin pegged to the US Dollar. 1 USDC ≈ 1 USD.',
  },
  BEP20: {
    ar: 'شبكة BNB Smart Chain - شبكة بلوكتشين سريعة ورسومها منخفضة.',
    en: 'BNB Smart Chain network - A fast blockchain with low fees.',
  },
  SOL: {
    ar: 'شبكة سولانا - شبكة بلوكتشين سريعة جداً ورسومها منخفضة.',
    en: 'Solana network - A very fast blockchain with low fees.',
  },
};

interface TermTooltipProps {
  term: string;
  language?: 'ar' | 'en';
  className?: string;
  showIcon?: boolean;
  children?: React.ReactNode;
}

export function TermTooltip({
  term,
  language = 'ar',
  className,
  showIcon = true,
  children,
}: TermTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const definition = technicalTerms[term];

  if (!definition) {
    return <span className={className}>{children || term}</span>;
  }

  const tooltipText = language === 'ar' ? definition.ar : definition.en;

  return (
    <span
      className={cn('relative inline-flex items-center gap-1 cursor-help', className)}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      onClick={() => setIsOpen(!isOpen)}
    >
      {children || term}
      {showIcon && (
        <HelpCircle className="w-4 h-4 text-gray-400 hover:text-primary-600 transition-colors" />
      )}

      {/* Tooltip */}
      {isOpen && (
        <div
          className={cn(
            'absolute z-50 w-64 p-3 text-sm rounded-lg shadow-lg',
            'bg-gray-900 dark:bg-gray-800 text-white',
            'bottom-full mb-2 left-1/2 -translate-x-1/2',
            'animate-in fade-in-0 zoom-in-95 duration-200'
          )}
          dir={language === 'ar' ? 'rtl' : 'ltr'}
        >
          <p className="font-semibold text-primary-400 mb-1">{term}</p>
          <p className="text-gray-200 leading-relaxed">{tooltipText}</p>
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-gray-900 dark:border-t-gray-800" />
        </div>
      )}
    </span>
  );
}

// مكون مساعد لعرض النص مع تحويل المصطلحات تلقائياً
interface TextWithTooltipsProps {
  text: string;
  language?: 'ar' | 'en';
  className?: string;
}

export function TextWithTooltips({ text, language = 'ar', className }: TextWithTooltipsProps) {
  const terms = Object.keys(technicalTerms);
  const regex = new RegExp(`\\b(${terms.join('|')})\\b`, 'gi');

  const parts = text.split(regex);

  return (
    <span className={className}>
      {parts.map((part, index) => {
        const matchedTerm = terms.find((t) => t.toLowerCase() === part.toLowerCase());
        if (matchedTerm) {
          return (
            <TermTooltip key={index} term={matchedTerm} language={language}>
              {part}
            </TermTooltip>
          );
        }
        return part;
      })}
    </span>
  );
}

export default TermTooltip;
