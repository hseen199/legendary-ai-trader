/**
 * EmailNotificationSettings Component
 * إعدادات إشعارات البريد الإلكتروني
 */
import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  Mail, 
  Bell, 
  Shield, 
  Wallet, 
  TrendingUp,
  FileText,
  CheckCircle,
  Loader2,
  Save,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import toast from 'react-hot-toast';
import api from '../services/api';

interface NotificationSetting {
  key: string;
  enabled: boolean;
}

interface EmailNotificationSettingsProps {
  language?: 'ar' | 'en';
}

const notificationCategories = [
  {
    id: 'security',
    icon: <Shield className="h-5 w-5" />,
    titleAr: 'الأمان',
    titleEn: 'Security',
    color: 'text-red-500 bg-red-500/10',
    settings: [
      {
        key: 'email_login_alert',
        titleAr: 'تنبيهات تسجيل الدخول',
        titleEn: 'Login Alerts',
        descAr: 'إشعار عند كل تسجيل دخول لحسابك',
        descEn: 'Notification for every login to your account',
        recommended: true,
      },
      {
        key: 'email_new_device',
        titleAr: 'تسجيل دخول من جهاز جديد',
        titleEn: 'New Device Login',
        descAr: 'إشعار عند تسجيل الدخول من جهاز جديد',
        descEn: 'Notification when logging in from a new device',
        recommended: true,
      },
      {
        key: 'email_password_change',
        titleAr: 'تغيير كلمة المرور',
        titleEn: 'Password Change',
        descAr: 'إشعار عند تغيير كلمة المرور',
        descEn: 'Notification when password is changed',
        recommended: true,
      },
      {
        key: 'email_2fa_change',
        titleAr: 'تغييرات المصادقة الثنائية',
        titleEn: '2FA Changes',
        descAr: 'إشعار عند تفعيل أو تعطيل المصادقة الثنائية',
        descEn: 'Notification when 2FA is enabled or disabled',
        recommended: true,
      },
    ],
  },
  {
    id: 'transactions',
    icon: <Wallet className="h-5 w-5" />,
    titleAr: 'المعاملات المالية',
    titleEn: 'Financial Transactions',
    color: 'text-green-500 bg-green-500/10',
    settings: [
      {
        key: 'email_deposit_confirmed',
        titleAr: 'تأكيد الإيداع',
        titleEn: 'Deposit Confirmation',
        descAr: 'إشعار عند تأكيد إيداعك',
        descEn: 'Notification when your deposit is confirmed',
        recommended: true,
      },
      {
        key: 'email_withdrawal_requested',
        titleAr: 'طلب السحب',
        titleEn: 'Withdrawal Request',
        descAr: 'إشعار عند تقديم طلب سحب',
        descEn: 'Notification when withdrawal is requested',
        recommended: true,
      },
      {
        key: 'email_withdrawal_completed',
        titleAr: 'إتمام السحب',
        titleEn: 'Withdrawal Completed',
        descAr: 'إشعار عند إتمام عملية السحب',
        descEn: 'Notification when withdrawal is completed',
        recommended: true,
      },
    ],
  },
  {
    id: 'trading',
    icon: <TrendingUp className="h-5 w-5" />,
    titleAr: 'التداول',
    titleEn: 'Trading',
    color: 'text-blue-500 bg-blue-500/10',
    settings: [
      {
        key: 'email_significant_trade',
        titleAr: 'الصفقات الكبيرة',
        titleEn: 'Significant Trades',
        descAr: 'إشعار عند تنفيذ صفقات كبيرة',
        descEn: 'Notification for significant trades',
        recommended: false,
      },
      {
        key: 'email_daily_summary',
        titleAr: 'ملخص يومي',
        titleEn: 'Daily Summary',
        descAr: 'ملخص يومي لنشاط التداول',
        descEn: 'Daily trading activity summary',
        recommended: false,
      },
    ],
  },
  {
    id: 'reports',
    icon: <FileText className="h-5 w-5" />,
    titleAr: 'التقارير',
    titleEn: 'Reports',
    color: 'text-purple-500 bg-purple-500/10',
    settings: [
      {
        key: 'email_weekly_report',
        titleAr: 'التقرير الأسبوعي',
        titleEn: 'Weekly Report',
        descAr: 'تقرير أسبوعي عن أداء محفظتك',
        descEn: 'Weekly report on your portfolio performance',
        recommended: true,
      },
      {
        key: 'email_monthly_report',
        titleAr: 'التقرير الشهري',
        titleEn: 'Monthly Report',
        descAr: 'تقرير شهري مفصل عن أداء محفظتك',
        descEn: 'Detailed monthly report on your portfolio performance',
        recommended: true,
      },
    ],
  },
];

export function EmailNotificationSettings({ language = 'ar' }: EmailNotificationSettingsProps) {
  const isRTL = language === 'ar';
  const queryClient = useQueryClient();
  
  // جلب الإعدادات الحالية
  const { data: currentSettings = {}, isLoading } = useQuery({
    queryKey: ['/api/v1/user/notification-settings'],
    queryFn: async () => {
      try {
        const res = await api.get('/user/notification-settings');
        return res.data;
      } catch {
        // إعدادات افتراضية
        return {
          email_login_alert: true,
          email_new_device: true,
          email_password_change: true,
          email_2fa_change: true,
          email_deposit_confirmed: true,
          email_withdrawal_requested: true,
          email_withdrawal_completed: true,
          email_significant_trade: false,
          email_daily_summary: false,
          email_weekly_report: true,
          email_monthly_report: true,
        };
      }
    },
  });

  const [settings, setSettings] = useState<Record<string, boolean>>(currentSettings);
  const [hasChanges, setHasChanges] = useState(false);

  // تحديث الإعدادات المحلية عند تحميل البيانات
  React.useEffect(() => {
    if (currentSettings && Object.keys(currentSettings).length > 0) {
      setSettings(currentSettings);
    }
  }, [currentSettings]);

  // حفظ الإعدادات
  const saveMutation = useMutation({
    mutationFn: async (newSettings: Record<string, boolean>) => {
      const res = await api.put('/user/notification-settings', newSettings);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/user/notification-settings'] });
      setHasChanges(false);
      toast.success(isRTL ? 'تم حفظ الإعدادات بنجاح' : 'Settings saved successfully');
    },
    onError: () => {
      toast.error(isRTL ? 'فشل في حفظ الإعدادات' : 'Failed to save settings');
    },
  });

  const toggleSetting = (key: string) => {
    setSettings(prev => {
      const newSettings = { ...prev, [key]: !prev[key] };
      setHasChanges(true);
      return newSettings;
    });
  };

  const handleSave = () => {
    saveMutation.mutate(settings);
  };

  const enableAll = () => {
    const allEnabled: Record<string, boolean> = {};
    notificationCategories.forEach(cat => {
      cat.settings.forEach(s => {
        allEnabled[s.key] = true;
      });
    });
    setSettings(allEnabled);
    setHasChanges(true);
  };

  const disableAll = () => {
    const allDisabled: Record<string, boolean> = {};
    notificationCategories.forEach(cat => {
      cat.settings.forEach(s => {
        allDisabled[s.key] = false;
      });
    });
    setSettings(allDisabled);
    setHasChanges(true);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-primary" />
              {isRTL ? 'إشعارات البريد الإلكتروني' : 'Email Notifications'}
            </CardTitle>
            <CardDescription>
              {isRTL 
                ? 'اختر الإشعارات التي تريد استلامها عبر البريد الإلكتروني'
                : 'Choose which notifications you want to receive via email'
              }
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={enableAll}>
              {isRTL ? 'تفعيل الكل' : 'Enable All'}
            </Button>
            <Button variant="outline" size="sm" onClick={disableAll}>
              {isRTL ? 'تعطيل الكل' : 'Disable All'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {notificationCategories.map((category) => (
          <div key={category.id} className="space-y-3">
            <div className="flex items-center gap-2">
              <div className={cn("p-2 rounded-full", category.color)}>
                {category.icon}
              </div>
              <h3 className="font-semibold">
                {isRTL ? category.titleAr : category.titleEn}
              </h3>
            </div>
            
            <div className="space-y-2 mr-9">
              {category.settings.map((setting) => (
                <div
                  key={setting.key}
                  className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-medium">
                        {isRTL ? setting.titleAr : setting.titleEn}
                      </p>
                      {setting.recommended && (
                        <Badge variant="secondary" className="text-xs">
                          {isRTL ? 'موصى به' : 'Recommended'}
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {isRTL ? setting.descAr : setting.descEn}
                    </p>
                  </div>
                  <Switch
                    checked={settings[setting.key] ?? false}
                    onCheckedChange={() => toggleSetting(setting.key)}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* زر الحفظ */}
        <div className="flex justify-end pt-4 border-t">
          <Button 
            onClick={handleSave} 
            disabled={!hasChanges || saveMutation.isPending}
            className="min-w-[120px]"
          >
            {saveMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <Save className="h-4 w-4 ml-2" />
                {isRTL ? 'حفظ الإعدادات' : 'Save Settings'}
              </>
            )}
          </Button>
        </div>

        {/* ملاحظة */}
        <div className="p-3 rounded-lg bg-muted/50 text-sm text-muted-foreground">
          <div className="flex items-start gap-2">
            <Bell className="h-4 w-4 mt-0.5" />
            <p>
              {isRTL 
                ? 'ملاحظة: الإشعارات الأمنية المهمة (مثل تسجيل الدخول من جهاز جديد) يُنصح بإبقائها مفعلة لحماية حسابك.'
                : 'Note: Important security notifications (like new device login) are recommended to keep enabled to protect your account.'
              }
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default EmailNotificationSettings;
