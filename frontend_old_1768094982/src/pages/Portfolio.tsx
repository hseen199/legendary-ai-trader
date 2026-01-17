import React, { useEffect, useState } from 'react';
import { dashboardAPI } from '../services/api';
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  PieChart,
  BarChart3,
  Calendar,
  Download,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Legend,
} from 'recharts';
import { format } from 'date-fns';
import { ar } from 'date-fns/locale';
import toast from 'react-hot-toast';

interface PortfolioData {
  balance: number;
  units: number;
  current_nav: number;
  total_deposited: number;
  total_withdrawn: number;
  current_value: number;
  profit_loss: number;
  profit_loss_percent: number;
  all_time_high: number;
  all_time_low: number;
  best_day: { date: string; profit: number };
  worst_day: { date: string; loss: number };
}

interface NAVHistory {
  nav_value: number;
  total_assets_usd: number;
  timestamp: string;
}

interface AssetAllocation {
  name: string;
  value: number;
  color: string;
}

const Portfolio: React.FC = () => {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [navHistory, setNavHistory] = useState<NAVHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y' | 'all'>('30d');

  // Mock asset allocation data (in real app, this comes from API)
  const assetAllocation: AssetAllocation[] = [
    { name: 'BTC', value: 35, color: '#F7931A' },
    { name: 'ETH', value: 25, color: '#627EEA' },
    { name: 'BNB', value: 15, color: '#F3BA2F' },
    { name: 'SOL', value: 10, color: '#00FFA3' },
    { name: 'USDC', value: 15, color: '#2775CA' },
  ];

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : timeRange === '90d' ? 90 : timeRange === '1y' ? 365 : 1000;
      const [dashboardRes, navHistoryRes] = await Promise.all([
        dashboardAPI.getDashboard(),
        dashboardAPI.getNAVHistory(days),
      ]);
      
      // Extend dashboard data with portfolio-specific fields
      setData({
        ...dashboardRes.data,
        total_withdrawn: dashboardRes.data.total_withdrawn || 0,
        all_time_high: dashboardRes.data.all_time_high || dashboardRes.data.current_value,
        all_time_low: dashboardRes.data.all_time_low || dashboardRes.data.current_value,
        best_day: dashboardRes.data.best_day || { date: new Date().toISOString(), profit: 0 },
        worst_day: dashboardRes.data.worst_day || { date: new Date().toISOString(), loss: 0 },
      });
      setNavHistory(navHistoryRes.data);
    } catch (error) {
      toast.error('فشل في تحميل البيانات');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  const exportData = () => {
    if (!data || !navHistory.length) return;
    
    const csvContent = [
      ['التاريخ', 'قيمة الوحدة', 'إجمالي الأصول'],
      ...navHistory.map(item => [
        format(new Date(item.timestamp), 'yyyy-MM-dd'),
        item.nav_value.toFixed(4),
        item.total_assets_usd.toFixed(2)
      ])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `portfolio_history_${format(new Date(), 'yyyy-MM-dd')}.csv`;
    link.click();
    toast.success('تم تصدير البيانات بنجاح');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 dark:text-gray-400">فشل في تحميل البيانات</p>
      </div>
    );
  }

  const chartData = navHistory.map((item) => ({
    date: format(new Date(item.timestamp), 'MM/dd'),
    nav: item.nav_value,
    assets: item.total_assets_usd,
  }));

  const timeRanges = [
    { id: '7d', label: '7 أيام' },
    { id: '30d', label: '30 يوم' },
    { id: '90d', label: '3 أشهر' },
    { id: '1y', label: 'سنة' },
    { id: 'all', label: 'الكل' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            محفظتي الاستثمارية
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            نظرة شاملة على أداء استثماراتك
          </p>
        </div>
        <button
          onClick={exportData}
          className="btn-secondary flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          تصدير البيانات
        </button>
      </div>

      {/* Main Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Total Value */}
        <div className="card p-6 bg-gradient-to-br from-primary-500 to-primary-700 text-white">
          <div className="flex items-center justify-between mb-4">
            <span className="text-primary-100">القيمة الإجمالية</span>
            <Wallet className="w-5 h-5 text-primary-200" />
          </div>
          <p className="text-3xl font-bold">
            ${data.current_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
          <p className="text-sm text-primary-200 mt-1">
            {data.units.toFixed(4)} وحدة
          </p>
        </div>

        {/* Profit/Loss */}
        <div className={`card p-6 ${data.profit_loss >= 0 ? 'bg-gradient-to-br from-green-500 to-green-700' : 'bg-gradient-to-br from-red-500 to-red-700'} text-white`}>
          <div className="flex items-center justify-between mb-4">
            <span className={data.profit_loss >= 0 ? 'text-green-100' : 'text-red-100'}>الربح/الخسارة</span>
            {data.profit_loss >= 0 ? (
              <TrendingUp className="w-5 h-5" />
            ) : (
              <TrendingDown className="w-5 h-5" />
            )}
          </div>
          <p className="text-3xl font-bold">
            {data.profit_loss >= 0 ? '+' : ''}
            ${data.profit_loss.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
          <p className="text-sm mt-1">
            {data.profit_loss_percent >= 0 ? '+' : ''}
            {data.profit_loss_percent.toFixed(2)}%
          </p>
        </div>

        {/* Total Deposited */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">إجمالي الإيداعات</span>
            <ArrowDownRight className="w-5 h-5 text-blue-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${data.total_deposited.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>

        {/* Total Withdrawn */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">إجمالي السحوبات</span>
            <ArrowUpRight className="w-5 h-5 text-orange-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${data.total_withdrawn.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Performance Chart */}
        <div className="card p-6 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              أداء المحفظة
            </h2>
            <div className="flex gap-2">
              {timeRanges.map((range) => (
                <button
                  key={range.id}
                  onClick={() => setTimeRange(range.id as any)}
                  className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                    timeRange === range.id
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {range.label}
                </button>
              ))}
            </div>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorNav" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: 'none',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="nav"
                  stroke="#22C55E"
                  strokeWidth={2}
                  fill="url(#colorNav)"
                  name="قيمة الوحدة"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Asset Allocation */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            توزيع الأصول
          </h2>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={assetAllocation}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {assetAllocation.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            {assetAllocation.map((asset) => (
              <div key={asset.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: asset.color }}
                  />
                  <span className="text-gray-600 dark:text-gray-400">{asset.name}</span>
                </div>
                <span className="font-medium text-gray-900 dark:text-white">
                  {asset.value}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* NAV */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
              <BarChart3 className="w-5 h-5 text-primary-600" />
            </div>
            <span className="text-gray-600 dark:text-gray-400">قيمة الوحدة (NAV)</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${data.current_nav.toFixed(4)}
          </p>
        </div>

        {/* All Time High */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <span className="text-gray-600 dark:text-gray-400">أعلى قيمة</span>
          </div>
          <p className="text-2xl font-bold text-green-600">
            ${data.all_time_high.toFixed(2)}
          </p>
        </div>

        {/* All Time Low */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
              <TrendingDown className="w-5 h-5 text-red-600" />
            </div>
            <span className="text-gray-600 dark:text-gray-400">أدنى قيمة</span>
          </div>
          <p className="text-2xl font-bold text-red-600">
            ${data.all_time_low.toFixed(2)}
          </p>
        </div>

        {/* Best Day */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
              <Calendar className="w-5 h-5 text-yellow-600" />
            </div>
            <span className="text-gray-600 dark:text-gray-400">أفضل يوم</span>
          </div>
          <p className="text-2xl font-bold text-green-600">
            +${data.best_day.profit.toFixed(2)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {format(new Date(data.best_day.date), 'dd MMM yyyy', { locale: ar })}
          </p>
        </div>
      </div>

      {/* Investment Summary */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          ملخص الاستثمار
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-gray-600 dark:text-gray-400 mb-1">صافي الاستثمار</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">
              ${(data.total_deposited - data.total_withdrawn).toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </p>
          </div>
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-gray-600 dark:text-gray-400 mb-1">العائد على الاستثمار (ROI)</p>
            <p className={`text-xl font-bold ${data.profit_loss_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {data.profit_loss_percent >= 0 ? '+' : ''}{data.profit_loss_percent.toFixed(2)}%
            </p>
          </div>
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-gray-600 dark:text-gray-400 mb-1">متوسط سعر الشراء</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">
              ${data.units > 0 ? (data.total_deposited / data.units).toFixed(4) : '0.0000'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
