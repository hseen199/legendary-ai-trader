/**
 * DailyTips Component
 * نصائح يومية للمستخدمين
 */
import React, { useState, useEffect } from 'react';
import { 
  Lightbulb, 
  X,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  TrendingUp,
  Shield,
  Wallet,
  Target,
  Clock,
  RefreshCw,
} from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface DailyTipsProps {
  language?: 'ar' | 'en';
  variant?: 'card' | 'banner' | 'floating';
  onDismiss?: () => void;
}

interface Tip {
  id: number;
  category: 'trading' | 'security' | 'investment' | 'platform';
  icon: React.ReactNode;
  titleAr: string;
  titleEn: string;
  contentAr: string;
  contentEn: string;
}

const tips: Tip[] = [
  {
    id: 1,
    category: 'investment',
    icon: <TrendingUp className="h-5 w-5" />,
    titleAr: 'تنويع المحفظة',
    titleEn: 'Portfolio Diversification',
    contentAr: 'الوكيل الذكي يقوم بتنويع استثماراتك تلقائياً عبر عدة عملات لتقليل المخاطر وزيادة فرص الربح.',
    contentEn: 'The AI agent automatically diversifies your investments across multiple currencies to reduce risk and increase profit opportunities.',
  },
  {
    id: 2,
    category: 'security',
    icon: <Shield className="h-5 w-5" />,
    titleAr: 'حماية حسابك',
    titleEn: 'Protect Your Account',
    contentAr: 'فعّل المصادقة الثنائية (2FA) لحماية حسابك من الوصول غير المصرح به. هذا يضيف طبقة أمان إضافية.',
    contentEn: 'Enable Two-Factor Authentication (2FA) to protect your account from unauthorized access. This adds an extra layer of security.',
  },
  {
    id: 3,
    category: 'trading',
    icon: <Target className="h-5 w-5" />,
    titleAr: 'الصبر مفتاح النجاح',
    titleEn: 'Patience is Key',
    contentAr: 'الاستثمار الناجح يتطلب الصبر. الوكيل الذكي يعمل على المدى الطويل لتحقيق أفضل النتائج.',
    contentEn: 'Successful investing requires patience. The AI agent works long-term to achieve the best results.',
  },
  {
    id: 4,
    category: 'platform',
    icon: <Wallet className="h-5 w-5" />,
    titleAr: 'فهم NAV',
    titleEn: 'Understanding NAV',
    contentAr: 'NAV (صافي قيمة الأصول) يعكس أداء المحفظة. عندما يرتفع NAV، ترتفع قيمة استثمارك.',
    contentEn: 'NAV (Net Asset Value) reflects portfolio performance. When NAV rises, your investment value increases.',
  },
  {
    id: 5,
    category: 'investment',
    icon: <Clock className="h-5 w-5" />,
    titleAr: 'أفضل وقت للاستثمار',
    titleEn: 'Best Time to Invest',
    contentAr: 'لا تحاول توقيت السوق. الاستثمار المنتظم على فترات زمنية يقلل من تأثير تقلبات السوق.',
    contentEn: 'Don\'t try to time the market. Regular investing over time reduces the impact of market volatility.',
  },
  {
    id: 6,
    category: 'security',
    icon: <Shield className="h-5 w-5" />,
    titleAr: 'راقب نشاط حسابك',
    titleEn: 'Monitor Account Activity',
    contentAr: 'راجع سجل النشاط بانتظام للتأكد من عدم وجود أي نشاط مشبوه في حسابك.',
    contentEn: 'Review your activity log regularly to ensure there\'s no suspicious activity in your account.',
  },
  {
    id: 7,
    category: 'platform',
    icon: <BookOpen className="h-5 w-5" />,
    titleAr: 'استخدم الأسئلة الشائعة',
    titleEn: 'Use the FAQ',
    contentAr: 'لديك سؤال؟ تحقق من صفحة الأسئلة الشائعة أولاً. ستجد إجابات لمعظم الاستفسارات الشائعة.',
    contentEn: 'Have a question? Check the FAQ page first. You\'ll find answers to most common questions.',
  },
  {
    id: 8,
    category: 'investment',
    icon: <TrendingUp className="h-5 w-5" />,
    titleAr: 'لا تستثمر أكثر مما تتحمل خسارته',
    titleEn: 'Never Invest More Than You Can Afford to Lose',
    contentAr: 'استثمر فقط الأموال التي يمكنك تحمل خسارتها. التداول ينطوي على مخاطر.',
    contentEn: 'Only invest money you can afford to lose. Trading involves risks.',
  },
];

const categoryColors: Record<string, string> = {
  trading: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
  security: 'bg-red-500/10 text-red-500 border-red-500/30',
  investment: 'bg-green-500/10 text-green-500 border-green-500/30',
  platform: 'bg-purple-500/10 text-purple-500 border-purple-500/30',
};

const categoryLabels: Record<string, { ar: string; en: string }> = {
  trading: { ar: 'تداول', en: 'Trading' },
  security: { ar: 'أمان', en: 'Security' },
  investment: { ar: 'استثمار', en: 'Investment' },
  platform: { ar: 'المنصة', en: 'Platform' },
};

export function DailyTips({ 
  language = 'ar',
  variant = 'card',
  onDismiss,
}: DailyTipsProps) {
  const isRTL = language === 'ar';
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  // اختيار نصيحة عشوائية عند التحميل
  useEffect(() => {
    const randomIndex = Math.floor(Math.random() * tips.length);
    setCurrentTipIndex(randomIndex);
  }, []);

  const currentTip = tips[currentTipIndex];

  const nextTip = () => {
    setCurrentTipIndex((prev) => (prev + 1) % tips.length);
  };

  const prevTip = () => {
    setCurrentTipIndex((prev) => (prev - 1 + tips.length) % tips.length);
  };

  const handleDismiss = () => {
    setDismissed(true);
    onDismiss?.();
  };

  if (dismissed) return null;

  // Banner variant
  if (variant === 'banner') {
    return (
      <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent border-b border-primary/20 py-2 px-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <Lightbulb className="h-5 w-5 text-primary" />
            <span className="text-sm">
              <strong>{isRTL ? 'نصيحة اليوم:' : 'Tip of the day:'}</strong>{' '}
              {isRTL ? currentTip.contentAr : currentTip.contentEn}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={nextTip}>
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={handleDismiss}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Floating variant
  if (variant === 'floating') {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 50 }}
          className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 z-50"
        >
          <Card className="shadow-lg border-primary/20">
            <CardContent className="p-4">
              <div className="flex items-start justify-between gap-3">
                <div className={cn("p-2 rounded-full", categoryColors[currentTip.category])}>
                  {currentTip.icon}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="outline" className={categoryColors[currentTip.category]}>
                      {isRTL 
                        ? categoryLabels[currentTip.category].ar 
                        : categoryLabels[currentTip.category].en
                      }
                    </Badge>
                  </div>
                  <p className="font-semibold mb-1">
                    {isRTL ? currentTip.titleAr : currentTip.titleEn}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {isRTL ? currentTip.contentAr : currentTip.contentEn}
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={handleDismiss}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>
    );
  }

  // Card variant (default)
  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'} className="overflow-hidden">
      <div className="bg-gradient-to-r from-primary/10 to-transparent p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-primary" />
            <span className="font-semibold">
              {isRTL ? 'نصيحة اليوم' : 'Tip of the Day'}
            </span>
          </div>
          <Badge variant="outline" className={categoryColors[currentTip.category]}>
            {isRTL 
              ? categoryLabels[currentTip.category].ar 
              : categoryLabels[currentTip.category].en
            }
          </Badge>
        </div>
      </div>
      
      <CardContent className="p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentTip.id}
            initial={{ opacity: 0, x: isRTL ? -20 : 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: isRTL ? 20 : -20 }}
            transition={{ duration: 0.2 }}
          >
            <div className="flex items-start gap-3">
              <div className={cn("p-2 rounded-full shrink-0", categoryColors[currentTip.category])}>
                {currentTip.icon}
              </div>
              <div>
                <h4 className="font-semibold mb-1">
                  {isRTL ? currentTip.titleAr : currentTip.titleEn}
                </h4>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? currentTip.contentAr : currentTip.contentEn}
                </p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
        
        {/* Navigation */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t">
          <Button variant="ghost" size="sm" onClick={prevTip}>
            {isRTL ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            {isRTL ? 'السابق' : 'Previous'}
          </Button>
          <span className="text-sm text-muted-foreground">
            {currentTipIndex + 1} / {tips.length}
          </span>
          <Button variant="ghost" size="sm" onClick={nextTip}>
            {isRTL ? 'التالي' : 'Next'}
            {isRTL ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default DailyTips;
