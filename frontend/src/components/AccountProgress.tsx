/**
 * AccountProgress Component
 * شريط تقدم إكمال الحساب
 */
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  CheckCircle, 
  Circle,
  User,
  Mail,
  Shield,
  Wallet,
  Bell,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

interface AccountProgressProps {
  language?: 'ar' | 'en';
  variant?: 'full' | 'compact' | 'minimal';
}

interface ProgressStep {
  id: string;
  icon: React.ReactNode;
  titleAr: string;
  titleEn: string;
  descAr: string;
  descEn: string;
  actionAr: string;
  actionEn: string;
  path: string;
  checkKey: string;
}

const progressSteps: ProgressStep[] = [
  {
    id: 'profile',
    icon: <User className="h-5 w-5" />,
    titleAr: 'إكمال الملف الشخصي',
    titleEn: 'Complete Profile',
    descAr: 'أضف معلوماتك الشخصية',
    descEn: 'Add your personal information',
    actionAr: 'إكمال',
    actionEn: 'Complete',
    path: '/settings',
    checkKey: 'profile_completed',
  },
  {
    id: 'email',
    icon: <Mail className="h-5 w-5" />,
    titleAr: 'تأكيد البريد الإلكتروني',
    titleEn: 'Verify Email',
    descAr: 'تأكد من بريدك الإلكتروني',
    descEn: 'Verify your email address',
    actionAr: 'تأكيد',
    actionEn: 'Verify',
    path: '/settings',
    checkKey: 'email_verified',
  },
  {
    id: '2fa',
    icon: <Shield className="h-5 w-5" />,
    titleAr: 'تفعيل المصادقة الثنائية',
    titleEn: 'Enable 2FA',
    descAr: 'حماية إضافية لحسابك',
    descEn: 'Extra security for your account',
    actionAr: 'تفعيل',
    actionEn: 'Enable',
    path: '/settings',
    checkKey: '2fa_enabled',
  },
  {
    id: 'deposit',
    icon: <Wallet className="h-5 w-5" />,
    titleAr: 'أول إيداع',
    titleEn: 'First Deposit',
    descAr: 'أودع أول مبلغ لبدء الاستثمار',
    descEn: 'Make your first deposit to start investing',
    actionAr: 'إيداع',
    actionEn: 'Deposit',
    path: '/wallet',
    checkKey: 'first_deposit',
  },
  {
    id: 'notifications',
    icon: <Bell className="h-5 w-5" />,
    titleAr: 'إعداد الإشعارات',
    titleEn: 'Setup Notifications',
    descAr: 'اختر إشعاراتك المفضلة',
    descEn: 'Choose your preferred notifications',
    actionAr: 'إعداد',
    actionEn: 'Setup',
    path: '/settings',
    checkKey: 'notifications_setup',
  },
];

// بيانات تجريبية
const mockProgressData = {
  profile_completed: true,
  email_verified: true,
  '2fa_enabled': false,
  first_deposit: true,
  notifications_setup: false,
};

export function AccountProgress({ 
  language = 'ar',
  variant = 'full',
}: AccountProgressProps) {
  const isRTL = language === 'ar';
  const navigate = useNavigate();

  // جلب حالة التقدم
  const { data: progressData = mockProgressData } = useQuery({
    queryKey: ['/api/v1/user/progress'],
    queryFn: async () => {
      try {
        const res = await api.get('/user/progress');
        return res.data;
      } catch {
        return mockProgressData;
      }
    },
  });

  const completedSteps = progressSteps.filter(step => progressData[step.checkKey]).length;
  const totalSteps = progressSteps.length;
  const progressPercentage = (completedSteps / totalSteps) * 100;
  const isComplete = completedSteps === totalSteps;

  // Minimal variant - just a progress bar
  if (variant === 'minimal') {
    if (isComplete) return null;
    
    return (
      <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">
            {isRTL ? 'إكمال الحساب' : 'Account Setup'}
          </span>
          <span className="text-sm text-muted-foreground">
            {completedSteps}/{totalSteps}
          </span>
        </div>
        <Progress value={progressPercentage} className="h-2" />
      </div>
    );
  }

  // Compact variant
  if (variant === 'compact') {
    if (isComplete) return null;
    
    const nextStep = progressSteps.find(step => !progressData[step.checkKey]);
    
    return (
      <div className="p-4 rounded-lg bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-primary/20">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="font-medium">
                {isRTL 
                  ? `أكمل حسابك (${completedSteps}/${totalSteps})`
                  : `Complete your account (${completedSteps}/${totalSteps})`
                }
              </p>
              {nextStep && (
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'الخطوة التالية: ' : 'Next: '}
                  {isRTL ? nextStep.titleAr : nextStep.titleEn}
                </p>
              )}
            </div>
          </div>
          {nextStep && (
            <Button size="sm" onClick={() => navigate(nextStep.path)}>
              {isRTL ? nextStep.actionAr : nextStep.actionEn}
              <ArrowRight className="h-4 w-4 mr-1" />
            </Button>
          )}
        </div>
        <Progress value={progressPercentage} className="h-2 mt-3" />
      </div>
    );
  }

  // Full variant
  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            {isRTL ? 'إكمال الحساب' : 'Account Setup'}
          </CardTitle>
          <Badge variant={isComplete ? 'default' : 'secondary'}>
            {completedSteps}/{totalSteps} {isRTL ? 'مكتمل' : 'completed'}
          </Badge>
        </div>
        <Progress value={progressPercentage} className="h-2 mt-2" />
      </CardHeader>

      <CardContent>
        {isComplete ? (
          <div className="text-center py-4">
            <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-3">
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
            <p className="font-semibold text-green-500">
              {isRTL ? 'تهانينا! حسابك مكتمل' : 'Congratulations! Your account is complete'}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {isRTL 
                ? 'استمتع بجميع مزايا المنصة'
                : 'Enjoy all platform features'
              }
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {progressSteps.map((step, index) => {
              const isCompleted = progressData[step.checkKey];
              const isNext = !isCompleted && progressSteps.slice(0, index).every(s => progressData[s.checkKey]);
              
              return (
                <div
                  key={step.id}
                  className={cn(
                    "flex items-center gap-3 p-3 rounded-lg border transition-all",
                    isCompleted 
                      ? "bg-green-500/5 border-green-500/20" 
                      : isNext
                        ? "bg-primary/5 border-primary/20"
                        : "bg-muted/30 border-muted"
                  )}
                >
                  <div className={cn(
                    "p-2 rounded-full",
                    isCompleted 
                      ? "bg-green-500/20 text-green-500" 
                      : isNext
                        ? "bg-primary/20 text-primary"
                        : "bg-muted text-muted-foreground"
                  )}>
                    {isCompleted ? <CheckCircle className="h-5 w-5" /> : step.icon}
                  </div>
                  
                  <div className="flex-1">
                    <p className={cn(
                      "font-medium",
                      isCompleted && "line-through text-muted-foreground"
                    )}>
                      {isRTL ? step.titleAr : step.titleEn}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isRTL ? step.descAr : step.descEn}
                    </p>
                  </div>
                  
                  {!isCompleted && (
                    <Button
                      size="sm"
                      variant={isNext ? 'default' : 'outline'}
                      onClick={() => navigate(step.path)}
                    >
                      {isRTL ? step.actionAr : step.actionEn}
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AccountProgress;
