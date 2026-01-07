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
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';
import { ar } from 'date-fns/locale';
import toast from 'react-hot-toast';

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
      toast.error('فشل في تحميل البيانات');
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
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Welcome */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          مرحباً، {user?.full_name || 'مستثمر'}
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          إليك ملخص محفظتك الاستثمارية
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Balance */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">الرصيد الحالي</span>
            <Wallet className="w-5 h-5 text-primary-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${data.current_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {data.units.toFixed(4)} وحدة
          </p>
        </div>

        {/* Profit/Loss */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">الربح/الخسارة</span>
            {data.profit_loss >= 0 ? (
              <TrendingUp className="w-5 h-5 text-green-600" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-600" />
            )}
          </div>
          <p
            className={`text-2xl font-bold ${
              data.profit_loss >= 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {data.profit_loss >= 0 ? '+' : ''}
            ${data.profit_loss.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
          <p
            className={`text-sm mt-1 ${
              data.profit_loss_percent >= 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
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

        {/* NAV */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">قيمة الوحدة (NAV)</span>
            <TrendingUp className="w-5 h-5 text-primary-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${data.current_nav.toFixed(4)}
          </p>
        </div>
      </div>

      {/* Chart & Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Chart */}
        <div className="card p-6 lg:col-span-2">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            أداء المحفظة (آخر 30 يوم)
          </h2>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
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
                <Line
                  type="monotone"
                  dataKey="nav"
                  stroke="#22C55E"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            إجراءات سريعة
          </h2>
          <div className="space-y-3">
            <Link
              to="/deposit"
              className="flex items-center justify-between p-4 bg-green-50 dark:bg-green-900/20 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
            >
              <span className="font-medium text-green-700 dark:text-green-400">
                إيداع أموال
              </span>
              <ArrowDownRight className="w-5 h-5 text-green-600" />
            </Link>
            <Link
              to="/withdraw"
              className={`flex items-center justify-between p-4 rounded-lg transition-colors ${
                data.can_withdraw
                  ? 'bg-blue-50 dark:bg-blue-900/20 hover:bg-blue-100 dark:hover:bg-blue-900/30'
                  : 'bg-gray-100 dark:bg-gray-800 cursor-not-allowed'
              }`}
            >
              <div>
                <span
                  className={`font-medium ${
                    data.can_withdraw
                      ? 'text-blue-700 dark:text-blue-400'
                      : 'text-gray-500'
                  }`}
                >
                  سحب أموال
                </span>
                {!data.can_withdraw && data.lock_period_ends && (
                  <p className="text-xs text-gray-500 mt-1">
                    متاح بعد:{' '}
                    {format(new Date(data.lock_period_ends), 'dd MMM yyyy', {
                      locale: ar,
                    })}
                  </p>
                )}
              </div>
              <ArrowUpRight className="w-5 h-5 text-blue-600" />
            </Link>
            <Link
              to="/trades"
              className="flex items-center justify-between p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
            >
              <span className="font-medium text-purple-700 dark:text-purple-400">
                عرض الصفقات
              </span>
              <TrendingUp className="w-5 h-5 text-purple-600" />
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Transactions */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          آخر العمليات
        </h2>
        {data.recent_transactions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">النوع</th>
                  <th className="pb-3 font-medium">المبلغ</th>
                  <th className="pb-3 font-medium">الوحدات</th>
                  <th className="pb-3 font-medium">الحالة</th>
                  <th className="pb-3 font-medium">التاريخ</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_transactions.map((tx) => (
                  <tr
                    key={tx.id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3">
                      <span
                        className={`badge ${
                          tx.type === 'deposit' ? 'badge-success' : 'badge-info'
                        }`}
                      >
                        {tx.type === 'deposit' ? 'إيداع' : 'سحب'}
                      </span>
                    </td>
                    <td className="py-3">${tx.amount_usd.toFixed(2)}</td>
                    <td className="py-3">{tx.units_transacted?.toFixed(4) || '-'}</td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          tx.status === 'completed'
                            ? 'badge-success'
                            : tx.status === 'pending'
                            ? 'badge-warning'
                            : 'badge-danger'
                        }`}
                      >
                        {tx.status === 'completed'
                          ? 'مكتمل'
                          : tx.status === 'pending'
                          ? 'قيد الانتظار'
                          : 'ملغي'}
                      </span>
                    </td>
                    <td className="py-3 text-gray-600 dark:text-gray-400">
                      {format(new Date(tx.created_at), 'dd/MM/yyyy HH:mm')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center text-gray-600 dark:text-gray-400 py-8">
            لا توجد عمليات حتى الآن
          </p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
