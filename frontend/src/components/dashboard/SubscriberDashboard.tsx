/**
 * SubscriberDashboard.tsx - لوحة معلومات محسّنة للمشتركين
 * يُضاف إلى /opt/asinax/frontend/src/components/dashboard/
 */

import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  PieChart,
  BarChart3,
  Calendar,
  Download,
  Bell,
  Award,
  Target,
  Zap
} from 'lucide-react';

// Types
interface PortfolioData {
  totalValue: number;
  totalDeposited: number;
  totalProfit: number;
  profitPercent: number;
  units: number;
  currentNAV: number;
}

interface PerformanceData {
  daily: number;
  weekly: number;
  monthly: number;
  allTime: number;
}

interface VIPInfo {
  level: string;
  levelName: string;
  icon: string;
  color: string;
  nextLevel: {
    name: string;
    amountNeeded: number;
    progress: number;
  } | null;
  benefits: {
    name: string;
    enabled: boolean;
    icon: string;
  }[];
}

interface TradeStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgProfit: number;
  avgLoss: number;
}

interface ChartDataPoint {
  date: string;
  value: number;
}

// ============ المكونات الفرعية ============

// بطاقة الإحصائية
const StatCard: React.FC<{
  title: string;
  value: string;
  change?: number;
  icon: React.ReactNode;
  color?: string;
}> = ({ title, value, change, icon, color = '#10b981' }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
    <div className="flex items-center justify-between mb-4">
      <span className="text-gray-400 text-sm">{title}</span>
      <div style={{ color }} className="p-2 bg-gray-700/50 rounded-lg">
        {icon}
      </div>
    </div>
    <div className="text-2xl font-bold text-white mb-2">{value}</div>
    {change !== undefined && (
      <div className={`flex items-center text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
        <span className="mr-1">{change >= 0 ? '+' : ''}{change.toFixed(2)}%</span>
      </div>
    )}
  </div>
);

// بطاقة VIP
const VIPCard: React.FC<{ vipInfo: VIPInfo }> = ({ vipInfo }) => (
  <div 
    className="rounded-xl p-6 border"
    style={{ 
      background: `linear-gradient(135deg, ${vipInfo.color}20 0%, transparent 100%)`,
      borderColor: vipInfo.color 
    }}
  >
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-3">
        <span className="text-4xl">{vipInfo.icon}</span>
        <div>
          <h3 className="text-xl font-bold text-white">{vipInfo.levelName}</h3>
          <p className="text-gray-400 text-sm">مستواك الحالي</p>
        </div>
      </div>
      <Award size={32} style={{ color: vipInfo.color }} />
    </div>
    
    {vipInfo.nextLevel && (
      <div className="mt-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">للترقية إلى {vipInfo.nextLevel.name}</span>
          <span className="text-white">${vipInfo.nextLevel.amountNeeded.toLocaleString()}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className="h-2 rounded-full transition-all duration-500"
            style={{ 
              width: `${vipInfo.nextLevel.progress}%`,
              backgroundColor: vipInfo.color 
            }}
          />
        </div>
      </div>
    )}
    
    <div className="mt-4 grid grid-cols-2 gap-2">
      {vipInfo.benefits.slice(0, 4).map((benefit, index) => (
        <div 
          key={index}
          className={`flex items-center gap-2 text-sm ${benefit.enabled ? 'text-green-400' : 'text-gray-500'}`}
        >
          <span>{benefit.icon}</span>
          <span>{benefit.name}</span>
        </div>
      ))}
    </div>
  </div>
);

// رسم بياني للأداء
const PerformanceChart: React.FC<{ data: ChartDataPoint[] }> = ({ data }) => {
  const maxValue = Math.max(...data.map(d => d.value));
  const minValue = Math.min(...data.map(d => d.value));
  const range = maxValue - minValue || 1;
  
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">تطور المحفظة</h3>
        <div className="flex gap-2">
          <button className="px-3 py-1 text-sm bg-emerald-500/20 text-emerald-400 rounded-lg">
            شهر
          </button>
          <button className="px-3 py-1 text-sm bg-gray-700 text-gray-400 rounded-lg hover:bg-gray-600">
            3 أشهر
          </button>
          <button className="px-3 py-1 text-sm bg-gray-700 text-gray-400 rounded-lg hover:bg-gray-600">
            سنة
          </button>
        </div>
      </div>
      
      <div className="h-64 flex items-end gap-1">
        {data.map((point, index) => {
          const height = ((point.value - minValue) / range) * 100;
          const isPositive = index > 0 ? point.value >= data[index - 1].value : true;
          
          return (
            <div 
              key={index}
              className="flex-1 group relative"
            >
              <div 
                className={`w-full rounded-t transition-all duration-300 ${
                  isPositive ? 'bg-emerald-500' : 'bg-red-500'
                } hover:opacity-80`}
                style={{ height: `${Math.max(height, 5)}%` }}
              />
              <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                ${point.value.toLocaleString()}
                <br />
                {point.date}
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="flex justify-between mt-4 text-xs text-gray-500">
        <span>{data[0]?.date}</span>
        <span>{data[data.length - 1]?.date}</span>
      </div>
    </div>
  );
};

// إحصائيات الصفقات
const TradeStatsCard: React.FC<{ stats: TradeStats }> = ({ stats }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
    <div className="flex items-center justify-between mb-6">
      <h3 className="text-lg font-semibold text-white">إحصائيات الصفقات</h3>
      <BarChart3 className="text-emerald-400" size={24} />
    </div>
    
    <div className="grid grid-cols-2 gap-4">
      <div className="text-center p-4 bg-gray-700/30 rounded-lg">
        <div className="text-3xl font-bold text-white">{stats.totalTrades}</div>
        <div className="text-gray-400 text-sm">إجمالي الصفقات</div>
      </div>
      
      <div className="text-center p-4 bg-gray-700/30 rounded-lg">
        <div className="text-3xl font-bold text-emerald-400">{stats.winRate.toFixed(1)}%</div>
        <div className="text-gray-400 text-sm">نسبة النجاح</div>
      </div>
      
      <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/30">
        <div className="text-2xl font-bold text-green-400">{stats.winningTrades}</div>
        <div className="text-gray-400 text-sm">صفقات رابحة</div>
        <div className="text-green-400 text-xs mt-1">+${stats.avgProfit.toFixed(2)} متوسط</div>
      </div>
      
      <div className="text-center p-4 bg-red-500/10 rounded-lg border border-red-500/30">
        <div className="text-2xl font-bold text-red-400">{stats.losingTrades}</div>
        <div className="text-gray-400 text-sm">صفقات خاسرة</div>
        <div className="text-red-400 text-xs mt-1">-${Math.abs(stats.avgLoss).toFixed(2)} متوسط</div>
      </div>
    </div>
  </div>
);

// مقارنة الأداء
const PerformanceComparison: React.FC<{ performance: PerformanceData }> = ({ performance }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
    <div className="flex items-center justify-between mb-6">
      <h3 className="text-lg font-semibold text-white">مقارنة الأداء</h3>
      <Target className="text-emerald-400" size={24} />
    </div>
    
    <div className="space-y-4">
      {[
        { label: 'اليوم', value: performance.daily, benchmark: 0.5 },
        { label: 'الأسبوع', value: performance.weekly, benchmark: 2 },
        { label: 'الشهر', value: performance.monthly, benchmark: 8 },
        { label: 'الإجمالي', value: performance.allTime, benchmark: 20 }
      ].map((item, index) => (
        <div key={index}>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">{item.label}</span>
            <span className={item.value >= 0 ? 'text-green-400' : 'text-red-400'}>
              {item.value >= 0 ? '+' : ''}{item.value.toFixed(2)}%
            </span>
          </div>
          <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className={`absolute h-full rounded-full transition-all duration-500 ${
                item.value >= 0 ? 'bg-emerald-500' : 'bg-red-500'
              }`}
              style={{ 
                width: `${Math.min(Math.abs(item.value) / item.benchmark * 50, 100)}%`,
                left: item.value >= 0 ? '50%' : 'auto',
                right: item.value < 0 ? '50%' : 'auto'
              }}
            />
            <div className="absolute w-0.5 h-full bg-gray-500 left-1/2" />
          </div>
        </div>
      ))}
    </div>
    
    <div className="mt-4 text-center text-xs text-gray-500">
      مقارنة بمعيار السوق
    </div>
  </div>
);

// ============ المكون الرئيسي ============

const SubscriberDashboard: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [vipInfo, setVipInfo] = useState<VIPInfo | null>(null);
  const [tradeStats, setTradeStats] = useState<TradeStats | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchDashboardData();
  }, []);
  
  const fetchDashboardData = async () => {
    try {
      // هنا يتم جلب البيانات من الـ API
      // const response = await api.get('/dashboard/subscriber-data');
      
      // بيانات تجريبية للعرض
      setPortfolio({
        totalValue: 12500.00,
        totalDeposited: 10000.00,
        totalProfit: 2500.00,
        profitPercent: 25.00,
        units: 10.5,
        currentNAV: 1.2381
      });
      
      setPerformance({
        daily: 0.85,
        weekly: 3.2,
        monthly: 12.5,
        allTime: 25.0
      });
      
      setVipInfo({
        level: 'gold',
        levelName: 'ذهبي',
        icon: '🥇',
        color: '#FFD700',
        nextLevel: {
          name: 'بلاتيني',
          amountNeeded: 15000,
          progress: 66.7
        },
        benefits: [
          { name: 'دعم أولوي', enabled: true, icon: '🎧' },
          { name: 'تقارير أسبوعية', enabled: true, icon: '📊' },
          { name: 'تقارير يومية', enabled: true, icon: '📈' },
          { name: 'مدير حساب', enabled: false, icon: '👤' }
        ]
      });
      
      setTradeStats({
        totalTrades: 156,
        winningTrades: 98,
        losingTrades: 58,
        winRate: 62.8,
        avgProfit: 45.50,
        avgLoss: 28.30
      });
      
      // بيانات الرسم البياني
      const chartPoints: ChartDataPoint[] = [];
      let value = 10000;
      for (let i = 30; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        value += (Math.random() - 0.4) * 200;
        chartPoints.push({
          date: date.toLocaleDateString('ar-SA', { month: 'short', day: 'numeric' }),
          value: Math.max(value, 9000)
        });
      }
      setChartData(chartPoints);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };
  
  const handleDownloadReport = async () => {
    try {
      // const response = await api.get('/reports/monthly', { responseType: 'blob' });
      // const url = window.URL.createObjectURL(new Blob([response.data]));
      // const link = document.createElement('a');
      // link.href = url;
      // link.setAttribute('download', 'monthly-report.pdf');
      // document.body.appendChild(link);
      // link.click();
      alert('سيتم تحميل التقرير...');
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-500" />
      </div>
    );
  }
  
  return (
    <div className="space-y-6 p-6">
      {/* الترويسة */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">لوحة المعلومات</h1>
          <p className="text-gray-400">مرحباً بك في ASINAX</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleDownloadReport}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
          >
            <Download size={18} />
            <span>تحميل التقرير</span>
          </button>
          <button className="relative p-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition">
            <Bell size={20} />
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center">
              3
            </span>
          </button>
        </div>
      </div>
      
      {/* البطاقات الإحصائية */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="القيمة الحالية"
          value={`$${portfolio?.totalValue.toLocaleString()}`}
          change={performance?.daily}
          icon={<DollarSign size={20} />}
        />
        <StatCard
          title="إجمالي الربح"
          value={`$${portfolio?.totalProfit.toLocaleString()}`}
          change={portfolio?.profitPercent}
          icon={<TrendingUp size={20} />}
          color={portfolio?.totalProfit && portfolio.totalProfit >= 0 ? '#22c55e' : '#ef4444'}
        />
        <StatCard
          title="الوحدات"
          value={portfolio?.units.toFixed(4) || '0'}
          icon={<PieChart size={20} />}
          color="#8b5cf6"
        />
        <StatCard
          title="NAV الحالي"
          value={`$${portfolio?.currentNAV.toFixed(4)}`}
          change={performance?.daily}
          icon={<Activity size={20} />}
          color="#f59e0b"
        />
      </div>
      
      {/* الصف الثاني */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* بطاقة VIP */}
        {vipInfo && <VIPCard vipInfo={vipInfo} />}
        
        {/* مقارنة الأداء */}
        {performance && <PerformanceComparison performance={performance} />}
        
        {/* إحصائيات الصفقات */}
        {tradeStats && <TradeStatsCard stats={tradeStats} />}
      </div>
      
      {/* الرسم البياني */}
      <PerformanceChart data={chartData} />
      
      {/* الإجراءات السريعة */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button className="flex items-center justify-center gap-3 p-4 bg-emerald-500/20 border border-emerald-500/50 rounded-xl text-emerald-400 hover:bg-emerald-500/30 transition">
          <DollarSign size={24} />
          <span className="font-semibold">إيداع جديد</span>
        </button>
        <button className="flex items-center justify-center gap-3 p-4 bg-blue-500/20 border border-blue-500/50 rounded-xl text-blue-400 hover:bg-blue-500/30 transition">
          <Download size={24} />
          <span className="font-semibold">تحميل التقرير الشهري</span>
        </button>
        <button className="flex items-center justify-center gap-3 p-4 bg-purple-500/20 border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition">
          <Zap size={24} />
          <span className="font-semibold">ترقية VIP</span>
        </button>
      </div>
    </div>
  );
};

export default SubscriberDashboard;
