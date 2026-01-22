/**
 * OnboardingTour Component
 * جولة تعريفية للمستخدمين الجدد
 * تشرح أقسام المنصة الرئيسية خطوة بخطوة
 */
import React, { useState, useEffect } from 'react';
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
  ArrowRight
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Progress } from './ui/progress';
import { cn } from '../lib/utils';

interface TourStep {
  id: string;
  title: string;
  titleEn: string;
  description: string;
  descriptionEn: string;
  icon: React.ReactNode;
  target?: string; // CSS selector for highlighting
}

const tourSteps: TourStep[] = [
  {
    id: 'welcome',
    title: 'مرحباً بك في ASINAX!',
    titleEn: 'Welcome to ASINAX!',
    description: 'دعنا نأخذك في جولة سريعة للتعرف على منصتنا وكيفية استخدامها بسهولة.',
    descriptionEn: 'Let us take you on a quick tour to learn about our platform and how to use it easily.',
    icon: <CheckCircle className="h-8 w-8 text-primary" />,
  },
  {
    id: 'dashboard',
    title: 'لوحة التحكم',
    titleEn: 'Dashboard',
    description: 'هنا يمكنك متابعة أداء محفظتك، رصيدك الحالي، وآخر الصفقات. كل المعلومات المهمة في مكان واحد.',
    descriptionEn: 'Here you can track your portfolio performance, current balance, and latest trades. All important information in one place.',
    icon: <LayoutDashboard className="h-8 w-8 text-blue-500" />,
    target: '[data-tour="dashboard"]',
  },
  {
    id: 'wallet',
    title: 'المحفظة المالية',
    titleEn: 'Wallet',
    description: 'من هنا يمكنك إيداع الأموال أو سحبها. نحن ندعم USDC على شبكات متعددة بأقل رسوم.',
    descriptionEn: 'From here you can deposit or withdraw funds. We support USDC on multiple networks with minimal fees.',
    icon: <Wallet className="h-8 w-8 text-green-500" />,
    target: '[data-tour="wallet"]',
  },
  {
    id: 'trades',
    title: 'الصفقات',
    titleEn: 'Trades',
    description: 'تابع جميع الصفقات التي ينفذها الوكيل الذكي نيابة عنك. يمكنك رؤية تفاصيل كل صفقة والأرباح المحققة.',
    descriptionEn: 'Track all trades executed by the AI agent on your behalf. You can see details of each trade and profits made.',
    icon: <TrendingUp className="h-8 w-8 text-orange-500" />,
    target: '[data-tour="trades"]',
  },
  {
    id: 'support',
    title: 'الدعم والمساعدة',
    titleEn: 'Support & Help',
    description: 'إذا واجهت أي مشكلة أو لديك سؤال، فريق الدعم جاهز لمساعدتك. يمكنك أيضاً الاطلاع على الأسئلة الشائعة.',
    descriptionEn: 'If you face any problem or have a question, our support team is ready to help. You can also check the FAQ.',
    icon: <HelpCircle className="h-8 w-8 text-purple-500" />,
    target: '[data-tour="support"]',
  },
  {
    id: 'settings',
    title: 'الإعدادات',
    titleEn: 'Settings',
    description: 'خصص تجربتك! غيّر اللغة، المظهر، أو إعدادات الأمان من هنا.',
    descriptionEn: 'Customize your experience! Change language, theme, or security settings from here.',
    icon: <Settings className="h-8 w-8 text-gray-500" />,
    target: '[data-tour="settings"]',
  },
  {
    id: 'complete',
    title: 'أنت جاهز!',
    titleEn: 'You are ready!',
    description: 'رائع! الآن أنت تعرف كل شيء عن المنصة. ابدأ بإيداع أموالك ودع الوكيل الذكي يعمل من أجلك.',
    descriptionEn: 'Great! Now you know everything about the platform. Start by depositing your funds and let the AI agent work for you.',
    icon: <CheckCircle className="h-8 w-8 text-green-500" />,
  },
];

interface OnboardingTourProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
  language?: 'ar' | 'en';
}

export function OnboardingTour({ 
  isOpen, 
  onClose, 
  onComplete,
  language = 'ar' 
}: OnboardingTourProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const isRTL = language === 'ar';

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    onClose();
  };

  const progress = ((currentStep + 1) / tourSteps.length) * 100;
  const step = tourSteps[currentStep];

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={handleSkip}
        />

        {/* Tour Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          transition={{ type: "spring", duration: 0.5 }}
          className="relative z-10 w-full max-w-md mx-4"
          dir={isRTL ? 'rtl' : 'ltr'}
        >
          <Card className="border-2 border-primary/20 shadow-2xl">
            <CardContent className="p-6">
              {/* Close Button */}
              <button
                onClick={handleSkip}
                className="absolute top-4 left-4 p-1 rounded-full hover:bg-muted transition-colors"
              >
                <X className="h-5 w-5 text-muted-foreground" />
              </button>

              {/* Progress Bar */}
              <div className="mb-6">
                <div className="flex justify-between text-xs text-muted-foreground mb-2">
                  <span>{currentStep + 1} / {tourSteps.length}</span>
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
                  <div className="p-4 rounded-full bg-primary/10">
                    {step.icon}
                  </div>
                </div>

                {/* Title */}
                <h3 className="text-xl font-bold mb-3">
                  {isRTL ? step.title : step.titleEn}
                </h3>

                {/* Description */}
                <p className="text-muted-foreground leading-relaxed mb-6">
                  {isRTL ? step.description : step.descriptionEn}
                </p>
              </motion.div>

              {/* Navigation Buttons */}
              <div className="flex items-center justify-between gap-3">
                {currentStep > 0 ? (
                  <Button
                    variant="outline"
                    onClick={handlePrev}
                    className="flex items-center gap-2"
                  >
                    {isRTL ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                    {isRTL ? 'السابق' : 'Previous'}
                  </Button>
                ) : (
                  <Button
                    variant="ghost"
                    onClick={handleSkip}
                    className="text-muted-foreground"
                  >
                    {isRTL ? 'تخطي' : 'Skip'}
                  </Button>
                )}

                <Button
                  onClick={handleNext}
                  className="flex items-center gap-2"
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
              <div className="flex justify-center gap-2 mt-6">
                {tourSteps.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentStep(index)}
                    className={cn(
                      "w-2 h-2 rounded-full transition-all",
                      index === currentStep
                        ? "bg-primary w-6"
                        : index < currentStep
                        ? "bg-primary/50"
                        : "bg-muted"
                    )}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

/**
 * Hook to manage onboarding tour state
 */
export function useOnboardingTour() {
  const [showTour, setShowTour] = useState(false);

  useEffect(() => {
    // Check if user has completed the tour
    const hasCompletedTour = localStorage.getItem('asinax_tour_completed');
    if (!hasCompletedTour) {
      // Show tour after a short delay for new users
      const timer = setTimeout(() => {
        setShowTour(true);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, []);

  const completeTour = () => {
    localStorage.setItem('asinax_tour_completed', 'true');
    setShowTour(false);
  };

  const closeTour = () => {
    setShowTour(false);
  };

  const resetTour = () => {
    localStorage.removeItem('asinax_tour_completed');
    setShowTour(true);
  };

  return {
    showTour,
    completeTour,
    closeTour,
    resetTour,
  };
}

export default OnboardingTour;
