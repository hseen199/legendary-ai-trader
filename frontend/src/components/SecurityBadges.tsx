/**
 * SecurityBadges Component
 * شارات الأمان لعرض مستوى حماية المنصة
 */
import React from 'react';
import { 
  Shield, 
  Lock, 
  CheckCircle, 
  ShieldCheck,
  Key,
  Fingerprint,
  Server,
  Eye,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';

interface SecurityBadge {
  id: string;
  icon: React.ReactNode;
  titleAr: string;
  titleEn: string;
  descriptionAr: string;
  descriptionEn: string;
  status: 'active' | 'verified' | 'protected';
}

const securityBadges: SecurityBadge[] = [
  {
    id: 'ssl',
    icon: <Lock className="h-5 w-5" />,
    titleAr: 'تشفير SSL/TLS',
    titleEn: 'SSL/TLS Encryption',
    descriptionAr: 'جميع الاتصالات مشفرة بتقنية SSL/TLS',
    descriptionEn: 'All connections encrypted with SSL/TLS',
    status: 'active',
  },
  {
    id: 'aes',
    icon: <ShieldCheck className="h-5 w-5" />,
    titleAr: 'تشفير AES-256',
    titleEn: 'AES-256 Encryption',
    descriptionAr: 'بيانات المستخدمين مشفرة بمعيار AES-256',
    descriptionEn: 'User data encrypted with AES-256 standard',
    status: 'protected',
  },
  {
    id: '2fa',
    icon: <Key className="h-5 w-5" />,
    titleAr: 'المصادقة الثنائية',
    titleEn: 'Two-Factor Auth',
    descriptionAr: 'حماية إضافية لحسابك بالمصادقة الثنائية',
    descriptionEn: 'Extra account protection with 2FA',
    status: 'verified',
  },
  {
    id: 'monitoring',
    icon: <Eye className="h-5 w-5" />,
    titleAr: 'مراقبة 24/7',
    titleEn: '24/7 Monitoring',
    descriptionAr: 'مراقبة أمنية مستمرة على مدار الساعة',
    descriptionEn: 'Continuous security monitoring around the clock',
    status: 'active',
  },
  {
    id: 'servers',
    icon: <Server className="h-5 w-5" />,
    titleAr: 'خوادم آمنة',
    titleEn: 'Secure Servers',
    descriptionAr: 'خوادم محمية في مراكز بيانات معتمدة',
    descriptionEn: 'Protected servers in certified data centers',
    status: 'protected',
  },
  {
    id: 'biometric',
    icon: <Fingerprint className="h-5 w-5" />,
    titleAr: 'حماية متقدمة',
    titleEn: 'Advanced Protection',
    descriptionAr: 'أنظمة كشف الاحتيال والنشاط المشبوه',
    descriptionEn: 'Fraud detection and suspicious activity systems',
    status: 'active',
  },
];

interface SecurityBadgesProps {
  language?: 'ar' | 'en';
  variant?: 'compact' | 'full' | 'inline';
  className?: string;
}

export function SecurityBadges({ 
  language = 'ar', 
  variant = 'compact',
  className 
}: SecurityBadgesProps) {
  const isRTL = language === 'ar';

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'verified':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      case 'protected':
        return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getStatusText = (status: string) => {
    const texts: Record<string, { ar: string; en: string }> = {
      active: { ar: 'نشط', en: 'Active' },
      verified: { ar: 'موثق', en: 'Verified' },
      protected: { ar: 'محمي', en: 'Protected' },
    };
    return isRTL ? texts[status]?.ar : texts[status]?.en;
  };

  if (variant === 'inline') {
    return (
      <div className={cn("flex flex-wrap gap-2", className)} dir={isRTL ? 'rtl' : 'ltr'}>
        {securityBadges.slice(0, 4).map((badge) => (
          <Badge
            key={badge.id}
            variant="outline"
            className={cn(
              "flex items-center gap-1.5 px-2 py-1",
              getStatusColor(badge.status)
            )}
          >
            {badge.icon}
            <span className="text-xs">
              {isRTL ? badge.titleAr : badge.titleEn}
            </span>
          </Badge>
        ))}
      </div>
    );
  }

  if (variant === 'compact') {
    return (
      <Card className={cn("border-green-500/20 bg-green-500/5", className)}>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Shield className="h-5 w-5 text-green-500" />
            {isRTL ? 'حماية أمنية متقدمة' : 'Advanced Security Protection'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {securityBadges.map((badge) => (
              <Badge
                key={badge.id}
                variant="outline"
                className={cn(
                  "flex items-center gap-1.5",
                  getStatusColor(badge.status)
                )}
              >
                {badge.icon}
                <span>{isRTL ? badge.titleAr : badge.titleEn}</span>
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Full variant
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-green-500" />
          {isRTL ? 'شارات الأمان' : 'Security Badges'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {securityBadges.map((badge) => (
            <div
              key={badge.id}
              className={cn(
                "p-4 rounded-lg border transition-all hover:shadow-md",
                getStatusColor(badge.status)
              )}
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-full bg-background/50">
                  {badge.icon}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold">
                      {isRTL ? badge.titleAr : badge.titleEn}
                    </h4>
                    <Badge variant="secondary" className="text-xs">
                      {getStatusText(badge.status)}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    {isRTL ? badge.descriptionAr : badge.descriptionEn}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default SecurityBadges;
