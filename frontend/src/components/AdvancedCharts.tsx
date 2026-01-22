/**
 * AdvancedCharts Component
 * رسوم بيانية متقدمة لعرض أداء المحفظة
 */
import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3,
  PieChart,
  Calendar,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ComposedChart,
} from 'recharts';

interface AdvancedChartsProps {
  language?: 'ar' | 'en';
  navHistory?: any[];
  tradesData?: any[];
  portfolioData?: any;
}

// ألوان الرسوم البيانية
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

// بيانات تجريبية
const mockNavHistory = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
  nav: 1 + Math.random() * 0.2 - 0.05,
  btc: 40000 + Math.random() * 5000,
}));

const mockTradesData = [
  { name: 'BTC', value: 35, profit: 12.5 },
  { name: 'ETH', value: 25, profit: 8.3 },
  { name: 'SOL', value: 15, profit: -2.1 },
  { name: 'ADA', value: 10, profit: 5.7 },
  { name: 'Others', value: 15, profit: 3.2 },
];

const mockMonthlyPerformance = [
  { month: 'يناير', profit: 5.2, trades: 45 },
  { month: 'فبراير', profit: 3.8, trades: 38 },
  { month: 'مارس', profit: -1.2, trades: 52 },
  { month: 'أبريل', profit: 7.5, trades: 41 },
  { month: 'مايو', profit: 4.1, trades: 35 },
  { month: 'يونيو', profit: 2.9, trades: 48 },
];

export function AdvancedCharts({ 
  language = 'ar',
  navHistory = mockNavHistory,
  tradesData = mockTradesData,
}: AdvancedChartsProps) {
  const isRTL = language === 'ar';
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');
  const [chartType, setChartType] = useState<'nav' | 'comparison' | 'distribution' | 'performance'>('nav');

  // حساب الإحصائيات
  const calculateStats = () => {
    if (navHistory.length < 2) return { change: 0, changePercent: 0, trend: 'neutral' };
    
    const firstNav = navHistory[0].nav;
    const lastNav = navHistory[navHistory.length - 1].nav;
    const change = lastNav - firstNav;
    const changePercent = ((change / firstNav) * 100).toFixed(2);
    const trend = change > 0 ? 'up' : change < 0 ? 'down' : 'neutral';
    
    return { change: change.toFixed(4), changePercent, trend };
  };

  const stats = calculateStats();

  // تنسيق التاريخ
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString(isRTL ? 'ar-SA' : 'en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
  };

  // Custom Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground">{formatDate(label)}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              {isRTL ? 'التحليلات المتقدمة' : 'Advanced Analytics'}
            </CardTitle>
            <CardDescription>
              {isRTL 
                ? 'رسوم بيانية تفصيلية لأداء محفظتك'
                : 'Detailed charts for your portfolio performance'
              }
            </CardDescription>
          </div>
          
          {/* Time Range Selector */}
          <div className="flex gap-1 bg-muted rounded-lg p-1">
            {(['7d', '30d', '90d', '1y'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={cn(
                  "px-3 py-1 text-sm rounded-md transition-colors",
                  timeRange === range 
                    ? "bg-background text-foreground shadow-sm" 
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {range === '7d' && (isRTL ? '7 أيام' : '7D')}
                {range === '30d' && (isRTL ? '30 يوم' : '30D')}
                {range === '90d' && (isRTL ? '90 يوم' : '90D')}
                {range === '1y' && (isRTL ? 'سنة' : '1Y')}
              </button>
            ))}
          </div>
        </div>

        {/* Stats Summary */}
        <div className="flex items-center gap-4 mt-4">
          <div className="flex items-center gap-2">
            {stats.trend === 'up' ? (
              <ArrowUpRight className="h-5 w-5 text-green-500" />
            ) : stats.trend === 'down' ? (
              <ArrowDownRight className="h-5 w-5 text-red-500" />
            ) : (
              <Minus className="h-5 w-5 text-gray-500" />
            )}
            <span className={cn(
              "text-lg font-bold",
              stats.trend === 'up' ? 'text-green-500' : 
              stats.trend === 'down' ? 'text-red-500' : 'text-gray-500'
            )}>
              {stats.trend === 'up' ? '+' : ''}{stats.changePercent}%
            </span>
          </div>
          <Badge variant={stats.trend === 'up' ? 'default' : stats.trend === 'down' ? 'destructive' : 'secondary'}>
            {isRTL 
              ? stats.trend === 'up' ? 'صاعد' : stats.trend === 'down' ? 'هابط' : 'مستقر'
              : stats.trend === 'up' ? 'Bullish' : stats.trend === 'down' ? 'Bearish' : 'Stable'
            }
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={chartType} onValueChange={(v) => setChartType(v as any)}>
          <TabsList className="grid grid-cols-4 mb-4">
            <TabsTrigger value="nav" className="text-xs">
              {isRTL ? 'NAV' : 'NAV'}
            </TabsTrigger>
            <TabsTrigger value="comparison" className="text-xs">
              {isRTL ? 'مقارنة' : 'Compare'}
            </TabsTrigger>
            <TabsTrigger value="distribution" className="text-xs">
              {isRTL ? 'التوزيع' : 'Distribution'}
            </TabsTrigger>
            <TabsTrigger value="performance" className="text-xs">
              {isRTL ? 'الأداء' : 'Performance'}
            </TabsTrigger>
          </TabsList>

          {/* NAV Chart */}
          <TabsContent value="nav">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={navHistory}>
                  <defs>
                    <linearGradient id="navGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                    stroke="#666"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#666"
                    fontSize={12}
                    domain={['auto', 'auto']}
                    tickFormatter={(v) => v.toFixed(2)}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="nav"
                    name={isRTL ? 'قيمة NAV' : 'NAV Value'}
                    stroke="#10b981"
                    strokeWidth={2}
                    fill="url(#navGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          {/* Comparison Chart */}
          <TabsContent value="comparison">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={navHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={formatDate}
                    stroke="#666"
                    fontSize={12}
                  />
                  <YAxis 
                    yAxisId="left"
                    stroke="#10b981"
                    fontSize={12}
                    tickFormatter={(v) => v.toFixed(2)}
                  />
                  <YAxis 
                    yAxisId="right"
                    orientation="right"
                    stroke="#3b82f6"
                    fontSize={12}
                    tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="nav"
                    name={isRTL ? 'NAV' : 'NAV'}
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="btc"
                    name={isRTL ? 'BTC' : 'BTC'}
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-2">
              {isRTL 
                ? 'مقارنة أداء NAV مع سعر البيتكوين'
                : 'Comparing NAV performance with Bitcoin price'
              }
            </p>
          </TabsContent>

          {/* Distribution Chart */}
          <TabsContent value="distribution">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPieChart>
                  <Pie
                    data={tradesData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {tradesData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </RechartsPieChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-2">
              {isRTL 
                ? 'توزيع الصفقات حسب العملة'
                : 'Trade distribution by currency'
              }
            </p>
          </TabsContent>

          {/* Performance Chart */}
          <TabsContent value="performance">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mockMonthlyPerformance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="month" stroke="#666" fontSize={12} />
                  <YAxis stroke="#666" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <Tooltip 
                    formatter={(value: number, name: string) => [
                      name === 'profit' ? `${value}%` : value,
                      name === 'profit' ? (isRTL ? 'الربح' : 'Profit') : (isRTL ? 'الصفقات' : 'Trades')
                    ]}
                  />
                  <Legend />
                  <Bar 
                    dataKey="profit" 
                    name={isRTL ? 'الربح %' : 'Profit %'}
                    fill="#10b981"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-2">
              {isRTL 
                ? 'الأداء الشهري للمحفظة'
                : 'Monthly portfolio performance'
              }
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default AdvancedCharts;
