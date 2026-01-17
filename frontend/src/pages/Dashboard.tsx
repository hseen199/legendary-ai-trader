import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { dashboardAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  RefreshCw,
  Plus,
  Minus,
  BarChart3,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { format } from 'date-fns';
import { ar } from 'date-fns/locale';
import toast from 'react-hot-toast';
import { useLanguage } from '@/lib/i18n';

interface DashboardData {
  balance: number;
  units: number;
  current_nav: number;
  total_deposited: number;
  current_value: number;
  profit_loss: number;
  profit_loss_percent: number;
  can_withdraw: boolean;
  lock_period_ends?: string;
  recent_transactions: any[];
  pending_withdrawals: any[];
}

interface NAVHistory {
  nav_value: number;
  total_assets_usd: number;
  timestamp: string;
}

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const { t, language } = useLanguage();
  const [data, setData] = useState<DashboardData | null>(null);
  const [navHistory, setNavHistory] = useState<NAVHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [dashboardRes, navHistoryRes] = await Promise.all([
        dashboardAPI.getDashboard(),
        dashboardAPI.getNAVHistory(30),
      ]);
      setData(dashboardRes.data);
      setNavHistory(navHistoryRes.data);
    } catch (error) {
      toast.error(t.dashboard.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="relative">
          <RefreshCw className="w-10 h-10 animate-spin text-violet-500" />
          <div className="absolute inset-0 blur-xl bg-violet-500/30 animate-pulse" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <p className="text-white/50">{t.dashboard.loadFailed}</p>
      </div>
    );
  }

  const chartData = navHistory.map((item) => ({
    date: format(new Date(item.timestamp), 'MM/dd'),
    nav: item.nav_value,
  }));

  const isProfit = data.profit_loss >= 0;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
            {t.dashboard.title}
          </h1>
          <p className="text-white/40 text-sm mt-1">
            {t.dashboard.welcome}، {user?.full_name || t.dashboard.investor}
          </p>
        </div>
        <button 
          onClick={fetchData}
          className="p-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20 hover:border-violet-500/40 transition-all duration-300"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Portfolio Value */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(139,92,246,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center justify-between mb-3">
            <span className="text-white/50 text-sm">{t.dashboard.portfolioValue}</span>
            <div className="w-10 h-10 rounded-xl bg-violet-500/15 flex items-center justify-center">
              <Wallet className="w-5 h-5 text-violet-400" />
            </div>
          </div>
          <div className="text-3xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
            ${data.current_value.toFixed(2)}
          </div>
          <div className="text-xs text-white/40 mt-1">
            {data.units.toFixed(4)} {t.dashboard.unit}
          </div>
        </div>

        {/* Total Deposited */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(59,130,246,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center justify-between mb-3">
            <span className="text-white/50 text-sm">{t.dashboard.totalDeposits}</span>
            <div className="w-10 h-10 rounded-xl bg-blue-500/15 flex items-center justify-center">
              <Plus className="w-5 h-5 text-blue-400" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white">
            ${data.total_deposited.toFixed(2)}
          </div>
        </div>

        {/* Profit/Loss */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className={`absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,${isProfit ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'}_0%,transparent_70%)]`} />
          <div className="flex items-center justify-between mb-3">
            <span className="text-white/50 text-sm">{t.dashboard.totalProfit}</span>
            <div className={`w-10 h-10 rounded-xl ${isProfit ? 'bg-emerald-500/15' : 'bg-red-500/15'} flex items-center justify-center`}>
              {isProfit ? (
                <TrendingUp className="w-5 h-5 text-emerald-400" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-400" />
              )}
            </div>
          </div>
          <div className={`text-3xl font-bold ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
            {isProfit ? '+' : ''}${data.profit_loss.toFixed(2)}
          </div>
          <div className={`text-sm mt-1 ${isProfit ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
            ({isProfit ? '+' : ''}{data.profit_loss_percent.toFixed(2)}%)
          </div>
        </div>

        {/* NAV */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(168,85,247,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center justify-between mb-3">
            <span className="text-white/50 text-sm">{t.dashboard.navPrice}</span>
            <div className="w-10 h-10 rounded-xl bg-purple-500/15 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white">
            ${data.current_nav.toFixed(4)}
          </div>
          <div className="inline-flex items-center gap-1 mt-2 px-2 py-1 rounded-full bg-emerald-500/15 text-emerald-400 text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {t.dashboard.active}
          </div>
        </div>
      </div>

      {/* Chart and Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2 rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-6 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-white">{t.dashboard.portfolioPerformance}</h2>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/15 text-emerald-400 text-sm">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              {t.dashboard.live}
            </div>
          </div>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="navGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.1)" />
                <XAxis 
                  dataKey="date" 
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                  domain={['dataMin - 0.01', 'dataMax + 0.01']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(18,18,28,0.95)',
                    border: '1px solid rgba(139,92,246,0.3)',
                    borderRadius: '12px',
                    boxShadow: '0 10px 40px rgba(0,0,0,0.3)',
                  }}
                  labelStyle={{ color: 'rgba(255,255,255,0.7)' }}
                  itemStyle={{ color: '#a78bfa' }}
                />
                <Area
                  type="monotone"
                  dataKey="nav"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  fill="url(#navGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[280px] text-white/40">
              <Clock className="w-6 h-6 mr-2" />
              {t.dashboard.noData}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">{t.dashboard.quickActions}</h2>
          <div className="space-y-3">
            <Link
              to="/wallet"
              className="flex items-center justify-between p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 hover:border-emerald-500/40 transition-all duration-300 group"
            >
              <span className="font-medium text-emerald-400">{t.dashboard.depositFunds}</span>
              <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                <ArrowDownRight className="w-5 h-5 text-emerald-400" />
              </div>
            </Link>
            
            <Link
              to="/wallet"
              className="flex items-center justify-between p-4 rounded-xl bg-violet-500/10 border border-violet-500/20 hover:bg-violet-500/20 hover:border-violet-500/40 transition-all duration-300 group"
            >
              <span className="font-medium text-violet-400">{t.dashboard.withdrawFunds}</span>
              <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                <ArrowUpRight className="w-5 h-5 text-violet-400" />
              </div>
            </Link>
            
            <Link
              to="/trades"
              className="flex items-center justify-between p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 hover:bg-blue-500/20 hover:border-blue-500/40 transition-all duration-300 group"
            >
              <span className="font-medium text-blue-400">{t.dashboard.viewTrades}</span>
              <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                <BarChart3 className="w-5 h-5 text-blue-400" />
              </div>
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Transactions */}
      <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">{t.dashboard.recentActivity}</h2>
        {data.recent_transactions.length > 0 ? (
          <div className="space-y-3">
            {data.recent_transactions.slice(0, 5).map((tx, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    tx.type === 'deposit' ? 'bg-emerald-500/20' : 'bg-violet-500/20'
                  }`}>
                    {tx.type === 'deposit' ? (
                      <ArrowDownRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <ArrowUpRight className="w-5 h-5 text-violet-400" />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-white">
                      {tx.type === 'deposit' ? t.dashboard.deposit : t.dashboard.withdraw}
                    </p>
                    <p className="text-sm text-white/40">
                      {format(new Date(tx.created_at), 'dd/MM/yyyy HH:mm')}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-semibold ${
                    tx.type === 'deposit' ? 'text-emerald-400' : 'text-violet-400'
                  }`}>
                    {tx.type === 'deposit' ? '+' : '-'}${tx.amount.toFixed(2)}
                  </p>
                  <p className={`text-xs px-2 py-0.5 rounded-full ${
                    tx.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
                    tx.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {tx.status === 'completed' ? t.wallet.completed :
                     tx.status === 'pending' ? t.wallet.pending :
                     t.wallet.cancelled}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-white/40">
            <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>{t.dashboard.noActivity}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
