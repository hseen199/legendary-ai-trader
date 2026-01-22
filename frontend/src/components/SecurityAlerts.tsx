/**
 * SecurityAlerts Component
 * تنبيهات الأمان للمستخدم
 */
import React, { useState } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  X,
  Bell,
  Lock,
  Smartphone,
  Mail,
  Eye,
  AlertCircle,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface SecurityAlert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'danger';
  titleAr: string;
  titleEn: string;
  messageAr: string;
  messageEn: string;
  actionAr?: string;
  actionEn?: string;
  onAction?: () => void;
  dismissible?: boolean;
}

interface SecurityAlertsProps {
  language?: 'ar' | 'en';
  alerts?: SecurityAlert[];
  className?: string;
}

// تنبيهات افتراضية
const defaultAlerts: SecurityAlert[] = [
  {
    id: 'new-device',
    type: 'warning',
    titleAr: 'تسجيل دخول من جهاز جديد',
    titleEn: 'Login from new device',
    messageAr: 'تم تسجيل الدخول من جهاز جديد. إذا لم تكن أنت، قم بتغيير كلمة المرور فوراً.',
    messageEn: 'Login detected from a new device. If this was not you, change your password immediately.',
    actionAr: 'تأكيد الجهاز',
    actionEn: 'Confirm Device',
    dismissible: true,
  },
  {
    id: '2fa-reminder',
    type: 'info',
    titleAr: 'فعّل المصادقة الثنائية',
    titleEn: 'Enable Two-Factor Authentication',
    messageAr: 'حسابك غير محمي بالمصادقة الثنائية. فعّلها الآن لحماية أفضل.',
    messageEn: 'Your account is not protected with 2FA. Enable it now for better security.',
    actionAr: 'تفعيل الآن',
    actionEn: 'Enable Now',
    dismissible: true,
  },
];

const alertStyles = {
  warning: {
    bg: 'bg-yellow-500/10 border-yellow-500/30',
    icon: <AlertTriangle className="h-5 w-5 text-yellow-500" />,
    iconBg: 'bg-yellow-500/20',
  },
  info: {
    bg: 'bg-blue-500/10 border-blue-500/30',
    icon: <AlertCircle className="h-5 w-5 text-blue-500" />,
    iconBg: 'bg-blue-500/20',
  },
  success: {
    bg: 'bg-green-500/10 border-green-500/30',
    icon: <CheckCircle className="h-5 w-5 text-green-500" />,
    iconBg: 'bg-green-500/20',
  },
  danger: {
    bg: 'bg-red-500/10 border-red-500/30',
    icon: <AlertTriangle className="h-5 w-5 text-red-500" />,
    iconBg: 'bg-red-500/20',
  },
};

export function SecurityAlerts({ 
  language = 'ar', 
  alerts = defaultAlerts,
  className 
}: SecurityAlertsProps) {
  const isRTL = language === 'ar';
  const [dismissedAlerts, setDismissedAlerts] = useState<string[]>([]);

  const visibleAlerts = alerts.filter(alert => !dismissedAlerts.includes(alert.id));

  const dismissAlert = (id: string) => {
    setDismissedAlerts(prev => [...prev, id]);
  };

  if (visibleAlerts.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-3", className)} dir={isRTL ? 'rtl' : 'ltr'}>
      <AnimatePresence>
        {visibleAlerts.map((alert) => {
          const style = alertStyles[alert.type];
          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: isRTL ? 100 : -100 }}
              className={cn(
                "p-4 rounded-lg border flex items-start gap-3",
                style.bg
              )}
            >
              <div className={cn("p-2 rounded-full", style.iconBg)}>
                {style.icon}
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-semibold">
                  {isRTL ? alert.titleAr : alert.titleEn}
                </h4>
                <p className="text-sm text-muted-foreground mt-1">
                  {isRTL ? alert.messageAr : alert.messageEn}
                </p>
                {(alert.actionAr || alert.actionEn) && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="mt-2"
                    onClick={alert.onAction}
                  >
                    {isRTL ? alert.actionAr : alert.actionEn}
                  </Button>
                )}
              </div>
              {alert.dismissible && (
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="p-1 rounded-full hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// مكون إعدادات تنبيهات الأمان
interface SecurityNotificationSettingsProps {
  language?: 'ar' | 'en';
}

export function SecurityNotificationSettings({ language = 'ar' }: SecurityNotificationSettingsProps) {
  const isRTL = language === 'ar';
  const [settings, setSettings] = useState({
    loginAlerts: true,
    newDeviceAlerts: true,
    withdrawalAlerts: true,
    depositAlerts: true,
    securityUpdates: true,
    weeklyReport: false,
  });

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const notificationOptions = [
    {
      key: 'loginAlerts' as const,
      icon: <Lock className="h-5 w-5" />,
      titleAr: 'تنبيهات تسجيل الدخول',
      titleEn: 'Login Alerts',
      descAr: 'إشعار عند كل تسجيل دخول لحسابك',
      descEn: 'Notification for every login to your account',
    },
    {
      key: 'newDeviceAlerts' as const,
      icon: <Smartphone className="h-5 w-5" />,
      titleAr: 'تنبيهات الأجهزة الجديدة',
      titleEn: 'New Device Alerts',
      descAr: 'إشعار عند تسجيل الدخول من جهاز جديد',
      descEn: 'Notification when logging in from a new device',
    },
    {
      key: 'withdrawalAlerts' as const,
      icon: <AlertTriangle className="h-5 w-5" />,
      titleAr: 'تنبيهات السحب',
      titleEn: 'Withdrawal Alerts',
      descAr: 'إشعار عند طلب سحب من حسابك',
      descEn: 'Notification when withdrawal is requested',
    },
    {
      key: 'depositAlerts' as const,
      icon: <CheckCircle className="h-5 w-5" />,
      titleAr: 'تنبيهات الإيداع',
      titleEn: 'Deposit Alerts',
      descAr: 'إشعار عند تأكيد إيداع في حسابك',
      descEn: 'Notification when deposit is confirmed',
    },
    {
      key: 'securityUpdates' as const,
      icon: <Shield className="h-5 w-5" />,
      titleAr: 'تحديثات الأمان',
      titleEn: 'Security Updates',
      descAr: 'إشعارات حول تحديثات الأمان المهمة',
      descEn: 'Notifications about important security updates',
    },
    {
      key: 'weeklyReport' as const,
      icon: <Mail className="h-5 w-5" />,
      titleAr: 'التقرير الأسبوعي',
      titleEn: 'Weekly Report',
      descAr: 'ملخص أسبوعي لنشاط حسابك',
      descEn: 'Weekly summary of your account activity',
    },
  ];

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="h-5 w-5 text-primary" />
          {isRTL ? 'إعدادات تنبيهات الأمان' : 'Security Alert Settings'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {notificationOptions.map((option) => (
            <div
              key={option.key}
              className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-primary/10 text-primary">
                  {option.icon}
                </div>
                <div>
                  <p className="font-medium">
                    {isRTL ? option.titleAr : option.titleEn}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {isRTL ? option.descAr : option.descEn}
                  </p>
                </div>
              </div>
              <Switch
                checked={settings[option.key]}
                onCheckedChange={() => toggleSetting(option.key)}
              />
            </div>
          ))}
        </div>

        <Button className="w-full mt-4">
          {isRTL ? 'حفظ الإعدادات' : 'Save Settings'}
        </Button>
      </CardContent>
    </Card>
  );
}

export default SecurityAlerts;
