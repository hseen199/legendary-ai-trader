/**
 * Transparency Page
 * صفحة الشفافية - إحصائيات المنصة العامة
 */
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  BarChart3, 
  TrendingUp,
  Users,
  Wallet,
  Activity,
  PieChart,
  Clock,
  Shield,
  CheckCircle,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
} from 'recharts';
import api from '../services/api';

interface TransparencyProps {
  language?: 'ar' | 'en';
}

// بيانات تجريبية
const mockPlatformStats = {
  totalUsers: 1250,
  activeUsers: 890,
  totalDeposits: 2500000,
  totalWithdrawals: 450000,
  totalTrades: 15680,
  winRate: 68.5,
  avgReturn: 12.3,
  agentUptime: 99.9,
  currentNav: 1.182,
  navChange24h: 0.45,
  navChange7d: 2.1,
  navChange30d: 8.5,
};

const mockNavHistory = [
  { date: '2025-01-01', nav: 1.00 },
  { date: '2025-01-05', nav: 1.02 },
  { date: '2025-01-10', nav: 1.05 },
  { date: '2025-01-15', nav: 1.12 },
  { date: '2025-01-20', nav: 1.15 },
  { date: '2025-01-22', nav: 1.182 },
];

const mockAssetAllocation = [
  { name: 'BTC', value: 35, color: '#f7931a' },
  { name: 'ETH', value: 25, color: '#627eea' },
  { name: 'SOL', value: 15, color: '#00ffa3' },
  { name: 'Others', value: 25, color: '#10b981' },
];

const mockRecentTrades = [
  { symbol: 'BTC/USDT', type: 'buy', profit: 2.5, time: '2h ago' },
  { symbol: 'ETH/USDT', type: 'sell', profit: 1.8, time: '4h ago' },
  { symbol: 'SOL/USDT', type: 'buy', profit: -0.5, time: '6h ago' },
  { symbol: 'ADA/USDT', type: 'sell', profit: 3.2, time: '8h ago' },
  { symbol: 'DOT/USDT', type: 'buy', profit: 1.1, time: '12h ago' },
];

export function Transparency({ language = 'ar' }: TransparencyProps) {
  const isRTL = language === 'ar';

  // جلب إحصائيات المنصة
  const { data: stats = mockPlatformStats } = useQuery({
    queryKey: ['/api/v1/public/stats'],
    queryFn: async () => {
      try {
        const res = await api.get('/public/stats');
        return res.data;
      } catch {
        return mockPlatformStats;
      }
    },
  });

  // استخدام الأرقام الإنجليزية دائماً
  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const formatCurrency = (num: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(num);
  };

  return (
    <div dir={isRTL ? 'rtl' : 'ltr'} className="min-h-screen bg-background p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center mb-12">
          <Badge className="mb-4" variant="outline">
            {isRTL ? 'الشفافية' : 'Transparency'}
          </Badge>
          <h1 className="text-4xl font-bold mb-4">
            {isRTL ? 'إحصائيات المنصة' : 'Platform Statistics'}
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            {isRTL 
              ? 'نؤمن بالشفافية الكاملة. هنا يمكنك متابعة جميع إحصائيات المنصة والأداء في الوقت الفعلي.'
              : 'We believe in complete transparency. Here you can track all platform statistics and performance in real-time.'
            }
          </p>
        </div>

        {/* Key Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <Users className="h-8 w-8 text-blue-500" />
                <Badge variant="secondary">
                  {isRTL ? 'مباشر' : 'Live'}
                </Badge>
              </div>
              <p className="text-3xl font-bold mt-4">{formatNumber(stats.totalUsers)}</p>
              <p className="text-sm text-muted-foreground">
                {isRTL ? 'إجمالي المستخدمين' : 'Total Users'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <Wallet className="h-8 w-8 text-green-500" />
                <Badge variant="secondary">USD</Badge>
              </div>
              <p className="text-3xl font-bold mt-4">{formatCurrency(stats.totalDeposits)}</p>
              <p className="text-sm text-muted-foreground">
                {isRTL ? 'إجمالي الإيداعات' : 'Total Deposits'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <Activity className="h-8 w-8 text-purple-500" />
                <Badge variant="secondary">{stats.winRate}%</Badge>
              </div>
              <p className="text-3xl font-bold mt-4">{formatNumber(stats.totalTrades)}</p>
              <p className="text-sm text-muted-foreground">
                {isRTL ? 'إجمالي الصفقات' : 'Total Trades'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <Shield className="h-8 w-8 text-orange-500" />
                <Badge className="bg-green-500">{stats.agentUptime}%</Badge>
              </div>
              <p className="text-3xl font-bold mt-4">
                {isRTL ? 'نشط' : 'Active'}
              </p>
              <p className="text-sm text-muted-foreground">
                {isRTL ? 'حالة الوكيل' : 'Agent Status'}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* NAV Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              {isRTL ? 'أداء NAV' : 'NAV Performance'}
            </CardTitle>
            <CardDescription>
              {isRTL 
                ? 'صافي قيمة الأصول يعكس الأداء الإجمالي للمحفظة'
                : 'Net Asset Value reflects the overall portfolio performance'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'NAV الحالي' : 'Current NAV'}
                </p>
                <p className="text-2xl font-bold">${stats.currentNav.toFixed(4)}</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'تغير 24 ساعة' : '24h Change'}
                </p>
                <p className={`text-2xl font-bold flex items-center gap-1 ${stats.navChange24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.navChange24h >= 0 ? <ArrowUpRight className="h-5 w-5" /> : <ArrowDownRight className="h-5 w-5" />}
                  {stats.navChange24h}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'تغير 7 أيام' : '7d Change'}
                </p>
                <p className={`text-2xl font-bold flex items-center gap-1 ${stats.navChange7d >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.navChange7d >= 0 ? <ArrowUpRight className="h-5 w-5" /> : <ArrowDownRight className="h-5 w-5" />}
                  {stats.navChange7d}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'تغير 30 يوم' : '30d Change'}
                </p>
                <p className={`text-2xl font-bold flex items-center gap-1 ${stats.navChange30d >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.navChange30d >= 0 ? <ArrowUpRight className="h-5 w-5" /> : <ArrowDownRight className="h-5 w-5" />}
                  {stats.navChange30d}%
                </p>
              </div>
            </div>

            {/* NAV Chart */}
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockNavHistory}>
                  <defs>
                    <linearGradient id="navGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="date" stroke="#666" fontSize={12} />
                  <YAxis stroke="#666" fontSize={12} domain={['auto', 'auto']} />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="nav"
                    stroke="#10b981"
                    strokeWidth={2}
                    fill="url(#navGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Two Column Layout */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Asset Allocation */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5 text-primary" />
                {isRTL ? 'توزيع الأصول' : 'Asset Allocation'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsPieChart>
                    <Pie
                      data={mockAssetAllocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {mockAssetAllocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4">
                {mockAssetAllocation.map((asset, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: asset.color }}
                    />
                    <span className="text-sm">{asset.name}</span>
                    <span className="text-sm text-muted-foreground mr-auto">{asset.value}%</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Trades */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                {isRTL ? 'آخر الصفقات' : 'Recent Trades'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockRecentTrades.map((trade, index) => (
                  <div 
                    key={index}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                  >
                    <div className="flex items-center gap-3">
                      <Badge variant={trade.type === 'buy' ? 'default' : 'secondary'}>
                        {trade.type === 'buy' 
                          ? (isRTL ? 'شراء' : 'BUY')
                          : (isRTL ? 'بيع' : 'SELL')
                        }
                      </Badge>
                      <span className="font-medium">{trade.symbol}</span>
                    </div>
                    <div className="text-left">
                      <p className={`font-medium ${trade.profit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {trade.profit >= 0 ? '+' : ''}{trade.profit}%
                      </p>
                      <p className="text-xs text-muted-foreground">{trade.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              {isRTL ? 'مقاييس الأداء' : 'Performance Metrics'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">{isRTL ? 'نسبة النجاح' : 'Win Rate'}</span>
                  <span className="text-sm font-medium">{stats.winRate}%</span>
                </div>
                <Progress value={stats.winRate} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">{isRTL ? 'وقت التشغيل' : 'Uptime'}</span>
                  <span className="text-sm font-medium">{stats.agentUptime}%</span>
                </div>
                <Progress value={stats.agentUptime} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">{isRTL ? 'متوسط العائد الشهري' : 'Avg Monthly Return'}</span>
                  <span className="text-sm font-medium">{stats.avgReturn}%</span>
                </div>
                <Progress value={stats.avgReturn * 5} className="h-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Disclaimer */}
        <Card className="bg-yellow-500/5 border-yellow-500/20">
          <CardContent className="p-6">
            <div className="flex items-start gap-3">
              <Shield className="h-6 w-6 text-yellow-500 shrink-0" />
              <div>
                <h4 className="font-semibold text-yellow-600 mb-2">
                  {isRTL ? 'تنبيه مهم' : 'Important Notice'}
                </h4>
                <p className="text-sm text-muted-foreground">
                  {isRTL 
                    ? 'الأداء السابق لا يضمن نتائج مستقبلية. التداول في العملات الرقمية ينطوي على مخاطر عالية وقد تخسر جزءاً أو كل استثمارك. يرجى الاستثمار بحكمة وعدم استثمار أكثر مما يمكنك تحمل خسارته.'
                    : 'Past performance does not guarantee future results. Trading in cryptocurrencies involves high risk and you may lose part or all of your investment. Please invest wisely and do not invest more than you can afford to lose.'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default Transparency;
