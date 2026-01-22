/**
 * OnboardingProvider Component
 * مزود سياق جولة التعريف للمستخدمين الجدد
 */
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  ChevronLeft, 
  ChevronRight, 
  LayoutDashboard,
  Wallet,
  TrendingUp,
  Settings,
  HelpCircle,
  CheckCircle,
  ArrowRight,
  Sparkles,
  Gift,
  Shield,
  Bot,
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import { cn } from '../lib/utils';

// تعريف خطوات الجولة
interface TourStep {
  id: string;
  title: string;
  titleEn: string;
  description: string;
  descriptionEn: string;
  icon: React.ReactNode;
  tips?: string[];
  tipsEn?: string[];
}

const tourSteps: TourStep[] = [
  {
    id: 'welcome',
    title: 'مرحباً بك في ASINAX!',
    titleEn: 'Welcome to ASINAX!',
    description: 'منصة التداول الذكي التي تعمل من أجلك على مدار الساعة. دعنا نأخذك في جولة سريعة للتعرف على المنصة.',
    descriptionEn: 'The smart trading platform that works for you 24/7. Let us take you on a quick tour to learn about the platform.',
    icon: <Sparkles className="h-10 w-10 text-primary" />,
    tips: [
      'الوكيل الذكي يتداول نيابة عنك',
      'لا تحتاج لخبرة سابقة في التداول',
      'متابعة مباشرة لأداء محفظتك',
    ],
    tipsEn: [
      'AI agent trades on your behalf',
      'No prior trading experience needed',
      'Real-time portfolio tracking',
    ],
  },
  {
    id: 'dashboard',
    title: 'لوحة التحكم',
    titleEn: 'Dashboard',
    description: 'هنا يمكنك متابعة أداء محفظتك بشكل مباشر. ترى قيمة استثمارك الحالية، الأرباح والخسائر، وآخر الصفقات.',
    descriptionEn: 'Here you can track your portfolio performance in real-time. See your current investment value, profits and losses, and latest trades.',
    icon: <LayoutDashboard className="h-10 w-10 text-blue-500" />,
    tips: [
      'قيمة المحفظة تتحدث تلقائياً',
      'رسم بياني يوضح أداء NAV',
      'آخر الصفقات المنفذة',
    ],
    tipsEn: [
      'Portfolio value updates automatically',
      'Chart shows NAV performance',
      'Latest executed trades',
    ],
  },
  {
    id: 'wallet',
    title: 'المحفظة المالية',
    titleEn: 'Wallet',
    description: 'من هنا تدير أموالك - إيداع وسحب. نحن ندعم USDC على شبكات متعددة برسوم منخفضة.',
    descriptionEn: 'From here you manage your funds - deposit and withdraw. We support USDC on multiple networks with low fees.',
    icon: <Wallet className="h-10 w-10 text-green-500" />,
    tips: [
      'الحد الأدنى للإيداع: 100 USDC',
      'رسوم الإيداع: 1% فقط',
      'السحب متاح بعد 7 أيام',
    ],
    tipsEn: [
      'Minimum deposit: 100 USDC',
      'Deposit fee: only 1%',
      'Withdrawal available after 7 days',
    ],
  },
  {
    id: 'trades',
    title: 'الصفقات',
    titleEn: 'Trades',
    description: 'تابع جميع الصفقات التي ينفذها الوكيل الذكي. يمكنك رؤية تفاصيل كل صفقة والأرباح المحققة.',
    descriptionEn: 'Track all trades executed by the AI agent. You can see details of each trade and profits made.',
    icon: <TrendingUp className="h-10 w-10 text-orange-500" />,
    tips: [
      'الصفقات النشطة والمغلقة',
      'تفاصيل كاملة لكل صفقة',
      'نسبة الربح/الخسارة',
    ],
    tipsEn: [
      'Active and closed trades',
      'Full details for each trade',
      'Profit/loss percentage',
    ],
  },
  {
    id: 'ai-agent',
    title: 'الوكيل الذكي',
    titleEn: 'AI Agent',
    description: 'الوكيل الذكي يعمل 24/7 لتحليل السوق واتخاذ قرارات التداول. يستخدم خوارزميات متقدمة لتحقيق أفضل النتائج.',
    descriptionEn: 'The AI agent works 24/7 to analyze the market and make trading decisions. It uses advanced algorithms to achieve the best results.',
    icon: <Bot className="h-10 w-10 text-purple-500" />,
    tips: [
      'تحليل فني متقدم',
      'إدارة مخاطر ذكية',
      'تنفيذ سريع للصفقات',
    ],
    tipsEn: [
      'Advanced technical analysis',
      'Smart risk management',
      'Fast trade execution',
    ],
  },
  {
    id: 'referrals',
    title: 'برنامج الإحالات',
    titleEn: 'Referral Program',
    description: 'شارك رمز الإحالة الخاص بك مع أصدقائك واكسب 10$ عن كل صديق يسجل ويودع!',
    descriptionEn: 'Share your referral code with friends and earn $10 for each friend who signs up and deposits!',
    icon: <Gift className="h-10 w-10 text-pink-500" />,
    tips: [
      'رمز إحالة فريد لك',
      '10$ مكافأة لكل إحالة',
      'بدون حد أقصى للإحالات',
    ],
    tipsEn: [
      'Unique referral code for you',
      '$10 reward per referral',
      'No limit on referrals',
    ],
  },
  {
    id: 'support',
    title: 'الدعم والمساعدة',
    titleEn: 'Support & Help',
    description: 'إذا واجهت أي مشكلة أو لديك سؤال، فريق الدعم جاهز لمساعدتك. يمكنك أيضاً الاطلاع على الأسئلة الشائعة.',
    descriptionEn: 'If you face any problem or have a question, our support team is ready to help. You can also check the FAQ.',
    icon: <HelpCircle className="h-10 w-10 text-cyan-500" />,
    tips: [
      'أسئلة شائعة شاملة',
      'نظام تذاكر الدعم',
      'رد سريع من الفريق',
    ],
    tipsEn: [
      'Comprehensive FAQ',
      'Support ticket system',
      'Quick response from team',
    ],
  },
  {
    id: 'complete',
    title: 'أنت جاهز للبدء!',
    titleEn: 'You are ready to start!',
    description: 'رائع! الآن أنت تعرف كل شيء عن المنصة. ابدأ بإيداع أموالك ودع الوكيل الذكي يعمل من أجلك.',
    descriptionEn: 'Great! Now you know everything about the platform. Start by depositing your funds and let the AI agent work for you.',
    icon: <CheckCircle className="h-10 w-10 text-green-500" />,
    tips: [
      'ابدأ بإيداع 100 USDC أو أكثر',
      'تابع أداء محفظتك يومياً',
      'تواصل معنا إذا احتجت مساعدة',
    ],
    tipsEn: [
      'Start with 100 USDC or more',
      'Track your portfolio daily',
      'Contact us if you need help',
    ],
  },
];

// سياق الجولة
interface OnboardingContextType {
  showTour: boolean;
  currentStep: number;
  startTour: () => void;
  closeTour: () => void;
  completeTour: () => void;
  nextStep: () => void;
  prevStep: () => void;
  goToStep: (step: number) => void;
  hasCompletedTour: boolean;
  resetTour: () => void;
}

const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined);

export function useOnboarding() {
  const context = useContext(OnboardingContext);
  if (!context) {
    throw new Error('useOnboarding must be used within OnboardingProvider');
  }
  return context;
}

interface OnboardingProviderProps {
  children: React.ReactNode;
}

export function OnboardingProvider({ children }: OnboardingProviderProps) {
  const [showTour, setShowTour] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [hasCompletedTour, setHasCompletedTour] = useState(false);
  const [language, setLanguage] = useState<'ar' | 'en'>('ar');

  // تحقق من حالة الجولة عند التحميل
  useEffect(() => {
    const completed = localStorage.getItem('asinax_tour_completed');
    setHasCompletedTour(completed === 'true');
    
    // تحقق من اللغة
    const savedLang = localStorage.getItem('language') || 'ar';
    setLanguage(savedLang as 'ar' | 'en');
  }, []);

  // عرض الجولة للمستخدمين الجدد بعد تأخير قصير
  useEffect(() => {
    if (!hasCompletedTour) {
      const timer = setTimeout(() => {
        setShowTour(true);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [hasCompletedTour]);

  const startTour = useCallback(() => {
    setCurrentStep(0);
    setShowTour(true);
  }, []);

  const closeTour = useCallback(() => {
    setShowTour(false);
  }, []);

  const completeTour = useCallback(() => {
    localStorage.setItem('asinax_tour_completed', 'true');
    setHasCompletedTour(true);
    setShowTour(false);
  }, []);

  const nextStep = useCallback(() => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      completeTour();
    }
  }, [currentStep, completeTour]);

  const prevStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  }, [currentStep]);

  const goToStep = useCallback((step: number) => {
    if (step >= 0 && step < tourSteps.length) {
      setCurrentStep(step);
    }
  }, []);

  const resetTour = useCallback(() => {
    localStorage.removeItem('asinax_tour_completed');
    setHasCompletedTour(false);
    setCurrentStep(0);
    setShowTour(true);
  }, []);

  const isRTL = language === 'ar';
  const progress = ((currentStep + 1) / tourSteps.length) * 100;
  const step = tourSteps[currentStep];

  return (
    <OnboardingContext.Provider
      value={{
        showTour,
        currentStep,
        startTour,
        closeTour,
        completeTour,
        nextStep,
        prevStep,
        goToStep,
        hasCompletedTour,
        resetTour,
      }}
    >
      {children}

      {/* Tour Modal */}
      <AnimatePresence>
        {showTour && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/70 backdrop-blur-sm"
              onClick={closeTour}
            />

            {/* Tour Card */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              transition={{ type: "spring", duration: 0.5 }}
              className="relative z-10 w-full max-w-lg"
              dir={isRTL ? 'rtl' : 'ltr'}
            >
              <Card className="border-2 border-primary/30 shadow-2xl bg-card/95 backdrop-blur">
                <CardContent className="p-6">
                  {/* Close Button */}
                  <button
                    onClick={closeTour}
                    className="absolute top-4 left-4 p-2 rounded-full hover:bg-muted transition-colors"
                    aria-label="Close"
                  >
                    <X className="h-5 w-5 text-muted-foreground" />
                  </button>

                  {/* Progress Bar */}
                  <div className="mb-6">
                    <div className="flex justify-between text-xs text-muted-foreground mb-2">
                      <span>
                        {isRTL ? `الخطوة ${currentStep + 1} من ${tourSteps.length}` : `Step ${currentStep + 1} of ${tourSteps.length}`}
                      </span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>

                  {/* Step Content */}
                  <motion.div
                    key={step.id}
                    initial={{ opacity: 0, x: isRTL ? -20 : 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: isRTL ? 20 : -20 }}
                    transition={{ duration: 0.3 }}
                    className="text-center"
                  >
                    {/* Icon */}
                    <div className="flex justify-center mb-4">
                      <motion.div 
                        className="p-4 rounded-full bg-primary/10"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        {step.icon}
                      </motion.div>
                    </div>

                    {/* Title */}
                    <h3 className="text-2xl font-bold mb-3">
                      {isRTL ? step.title : step.titleEn}
                    </h3>

                    {/* Description */}
                    <p className="text-muted-foreground leading-relaxed mb-6">
                      {isRTL ? step.description : step.descriptionEn}
                    </p>

                    {/* Tips */}
                    {step.tips && step.tips.length > 0 && (
                      <div className="bg-muted/50 rounded-lg p-4 mb-6 text-right">
                        <div className="space-y-2">
                          {(isRTL ? step.tips : step.tipsEn)?.map((tip, index) => (
                            <div key={index} className="flex items-center gap-2 text-sm">
                              <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                              <span>{tip}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>

                  {/* Navigation Buttons */}
                  <div className="flex items-center justify-between gap-3">
                    {currentStep > 0 ? (
                      <Button
                        variant="outline"
                        onClick={prevStep}
                        className="flex items-center gap-2"
                      >
                        {isRTL ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                        {isRTL ? 'السابق' : 'Previous'}
                      </Button>
                    ) : (
                      <Button
                        variant="ghost"
                        onClick={closeTour}
                        className="text-muted-foreground"
                      >
                        {isRTL ? 'تخطي' : 'Skip'}
                      </Button>
                    )}

                    <Button
                      onClick={nextStep}
                      className="flex items-center gap-2 min-w-[120px]"
                    >
                      {currentStep === tourSteps.length - 1 ? (
                        <>
                          {isRTL ? 'ابدأ الآن' : 'Start Now'}
                          <ArrowRight className="h-4 w-4" />
                        </>
                      ) : (
                        <>
                          {isRTL ? 'التالي' : 'Next'}
                          {isRTL ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Step Indicators */}
                  <div className="flex justify-center gap-1.5 mt-6">
                    {tourSteps.map((_, index) => (
                      <button
                        key={index}
                        onClick={() => goToStep(index)}
                        className={cn(
                          "h-2 rounded-full transition-all",
                          index === currentStep
                            ? "bg-primary w-6"
                            : index < currentStep
                            ? "bg-primary/50 w-2"
                            : "bg-muted w-2"
                        )}
                        aria-label={`Go to step ${index + 1}`}
                      />
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </OnboardingContext.Provider>
  );
}

export default OnboardingProvider;
