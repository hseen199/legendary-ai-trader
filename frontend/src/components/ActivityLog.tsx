/**
 * ActivityLog Component
 * سجل نشاط المستخدم وعمليات تسجيل الدخول
 */
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Activity, 
  LogIn, 
  LogOut, 
  Wallet, 
  ArrowUpCircle,
  ArrowDownCircle,
  Settings,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Monitor,
  Smartphone,
  Globe,
  Filter,
  RefreshCw,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Skeleton } from './ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { cn } from '../lib/utils';
import { format, formatDistanceToNow } from 'date-fns';
import { ar } from 'date-fns/locale';
import api from '../services/api';

interface ActivityItem {
  id: number;
  type: 'login' | 'logout' | 'deposit' | 'withdrawal' | 'settings' | 'security' | 'trade';
  description: string;
  ip_address?: string;
  device?: string;
  location?: string;
  status: 'success' | 'failed' | 'pending' | 'warning';
  created_at: string;
  metadata?: Record<string, any>;
}

interface ActivityLogProps {
  language?: 'ar' | 'en';
  limit?: number;
  showFilters?: boolean;
}

const activityIcons: Record<string, React.ReactNode> = {
  login: <LogIn className="h-4 w-4" />,
  logout: <LogOut className="h-4 w-4" />,
  deposit: <ArrowDownCircle className="h-4 w-4" />,
  withdrawal: <ArrowUpCircle className="h-4 w-4" />,
  settings: <Settings className="h-4 w-4" />,
  security: <Shield className="h-4 w-4" />,
  trade: <Activity className="h-4 w-4" />,
};

const activityColors: Record<string, string> = {
  login: 'text-blue-500 bg-blue-500/10',
  logout: 'text-gray-500 bg-gray-500/10',
  deposit: 'text-green-500 bg-green-500/10',
  withdrawal: 'text-orange-500 bg-orange-500/10',
  settings: 'text-purple-500 bg-purple-500/10',
  security: 'text-red-500 bg-red-500/10',
  trade: 'text-cyan-500 bg-cyan-500/10',
};

const statusColors: Record<string, string> = {
  success: 'bg-green-500/10 text-green-500 border-green-500/20',
  failed: 'bg-red-500/10 text-red-500 border-red-500/20',
  pending: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20',
  warning: 'bg-orange-500/10 text-orange-500 border-orange-500/20',
};

// بيانات تجريبية للعرض
const mockActivities: ActivityItem[] = [
  {
    id: 1,
    type: 'login',
    description: 'تسجيل دخول ناجح',
    ip_address: '192.168.1.xxx',
    device: 'Chrome on Windows',
    location: 'الرياض، السعودية',
    status: 'success',
    created_at: new Date().toISOString(),
  },
  {
    id: 2,
    type: 'deposit',
    description: 'إيداع 100 USDC',
    status: 'success',
    created_at: new Date(Date.now() - 3600000).toISOString(),
  },
  {
    id: 3,
    type: 'settings',
    description: 'تغيير إعدادات الإشعارات',
    status: 'success',
    created_at: new Date(Date.now() - 7200000).toISOString(),
  },
  {
    id: 4,
    type: 'security',
    description: 'تفعيل المصادقة الثنائية',
    status: 'success',
    created_at: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: 5,
    type: 'login',
    description: 'محاولة تسجيل دخول فاشلة',
    ip_address: '10.0.0.xxx',
    device: 'Unknown',
    location: 'غير معروف',
    status: 'failed',
    created_at: new Date(Date.now() - 172800000).toISOString(),
  },
];

export function ActivityLog({ 
  language = 'ar', 
  limit = 20,
  showFilters = true 
}: ActivityLogProps) {
  const isRTL = language === 'ar';
  const [activeFilter, setActiveFilter] = useState<string>('all');
  
  // جلب سجل النشاط من API
  const { data: activities = mockActivities, isLoading, refetch } = useQuery<ActivityItem[]>({
    queryKey: ['/api/v1/user/activity-log', limit],
    queryFn: async () => {
      try {
        const res = await api.get(`/user/activity-log?limit=${limit}`);
        return res.data;
      } catch {
        return mockActivities;
      }
    },
  });

  const filteredActivities = activeFilter === 'all' 
    ? activities 
    : activities.filter(a => a.type === activeFilter);

  const getDeviceIcon = (device?: string) => {
    if (!device) return <Globe className="h-4 w-4" />;
    if (device.toLowerCase().includes('mobile') || device.toLowerCase().includes('iphone') || device.toLowerCase().includes('android')) {
      return <Smartphone className="h-4 w-4" />;
    }
    return <Monitor className="h-4 w-4" />;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-orange-500" />;
    }
  };

  const getStatusText = (status: string) => {
    const texts: Record<string, { ar: string; en: string }> = {
      success: { ar: 'ناجح', en: 'Success' },
      failed: { ar: 'فاشل', en: 'Failed' },
      pending: { ar: 'قيد الانتظار', en: 'Pending' },
      warning: { ar: 'تحذير', en: 'Warning' },
    };
    return isRTL ? texts[status]?.ar : texts[status]?.en;
  };

  const formatTime = (date: string) => {
    return formatDistanceToNow(new Date(date), {
      addSuffix: true,
      locale: isRTL ? ar : undefined,
    });
  };

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              {isRTL ? 'سجل النشاط' : 'Activity Log'}
            </CardTitle>
            <CardDescription>
              {isRTL 
                ? 'جميع الأنشطة وعمليات تسجيل الدخول في حسابك'
                : 'All activities and login operations in your account'
              }
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Filters */}
        {showFilters && (
          <Tabs value={activeFilter} onValueChange={setActiveFilter} className="mb-4">
            <TabsList className="grid grid-cols-4 lg:grid-cols-7 gap-1">
              <TabsTrigger value="all" className="text-xs">
                {isRTL ? 'الكل' : 'All'}
              </TabsTrigger>
              <TabsTrigger value="login" className="text-xs">
                {isRTL ? 'الدخول' : 'Login'}
              </TabsTrigger>
              <TabsTrigger value="deposit" className="text-xs">
                {isRTL ? 'إيداع' : 'Deposit'}
              </TabsTrigger>
              <TabsTrigger value="withdrawal" className="text-xs">
                {isRTL ? 'سحب' : 'Withdraw'}
              </TabsTrigger>
              <TabsTrigger value="settings" className="text-xs hidden lg:block">
                {isRTL ? 'إعدادات' : 'Settings'}
              </TabsTrigger>
              <TabsTrigger value="security" className="text-xs hidden lg:block">
                {isRTL ? 'أمان' : 'Security'}
              </TabsTrigger>
              <TabsTrigger value="trade" className="text-xs hidden lg:block">
                {isRTL ? 'صفقات' : 'Trades'}
              </TabsTrigger>
            </TabsList>
          </Tabs>
        )}

        {/* Activity List */}
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : filteredActivities.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>{isRTL ? 'لا توجد أنشطة' : 'No activities'}</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredActivities.map((activity) => (
              <div
                key={activity.id}
                className={cn(
                  "flex items-start gap-3 p-3 rounded-lg border transition-all hover:bg-muted/50",
                  activity.status === 'failed' && "border-red-500/20 bg-red-500/5"
                )}
              >
                {/* Icon */}
                <div className={cn(
                  "p-2 rounded-full",
                  activityColors[activity.type]
                )}>
                  {activityIcons[activity.type]}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-medium truncate">
                      {activity.description}
                    </p>
                    <Badge 
                      variant="outline" 
                      className={cn("shrink-0", statusColors[activity.status])}
                    >
                      {getStatusIcon(activity.status)}
                      <span className="mr-1">{getStatusText(activity.status)}</span>
                    </Badge>
                  </div>

                  {/* Details */}
                  <div className="flex flex-wrap items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatTime(activity.created_at)}
                    </span>
                    {activity.device && (
                      <span className="flex items-center gap-1">
                        {getDeviceIcon(activity.device)}
                        {activity.device}
                      </span>
                    )}
                    {activity.ip_address && (
                      <span className="flex items-center gap-1">
                        <Globe className="h-3 w-3" />
                        {activity.ip_address}
                      </span>
                    )}
                    {activity.location && (
                      <span>{activity.location}</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Security Notice */}
        <div className="mt-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500 shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-yellow-600">
                {isRTL ? 'نصيحة أمنية' : 'Security Tip'}
              </p>
              <p className="text-muted-foreground">
                {isRTL 
                  ? 'إذا لاحظت أي نشاط مشبوه لم تقم به، قم بتغيير كلمة المرور فوراً وتواصل مع الدعم.'
                  : 'If you notice any suspicious activity you did not perform, change your password immediately and contact support.'
                }
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ActivityLog;
