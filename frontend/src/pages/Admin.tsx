import React, { useEffect, useState } from 'react';
import { adminAPI } from '../services/api';
import {
  Users,
  DollarSign,
  TrendingUp,
  Clock,
  AlertTriangle,
  Check,
  X,
  RefreshCw,
  Shield,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

interface AdminStats {
  total_users: number;
  active_users: number;
  total_assets_usd: number;
  total_units: number;
  current_nav: number;
  pending_withdrawals: number;
  total_deposits_today: number;
  total_withdrawals_today: number;
  high_water_mark: number;
  total_fees_collected: number;
  emergency_mode: string;
}

interface PendingWithdrawal {
  id: number;
  user_id: number;
  amount: number;
  units_to_withdraw: number;
  to_address: string;
  network: string;
  coin: string;
  status: string;
  requested_at: string;
}

interface User {
  id: number;
  email: string;
  full_name?: string;
  status: string;
  is_admin: boolean;
  units: number;
  current_value_usd: number;
  total_deposited: number;
  total_withdrawn: number;
  created_at: string;
  last_login?: string;
}

const Admin: React.FC = () => {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [pendingWithdrawals, setPendingWithdrawals] = useState<PendingWithdrawal[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'withdrawals' | 'users'>('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [reviewingId, setReviewingId] = useState<number | null>(null);

  const fetchData = async () => {
    try {
      const [statsRes, withdrawalsRes, usersRes] = await Promise.all([
        adminAPI.getStats(),
        adminAPI.getPendingWithdrawals(),
        adminAPI.getUsers(),
      ]);
      setStats(statsRes.data);
      setPendingWithdrawals(withdrawalsRes.data);
      setUsers(usersRes.data);
    } catch (error) {
      toast.error('فشل في تحميل البيانات');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleWithdrawalReview = async (id: number, action: 'approve' | 'reject', reason?: string) => {
    setReviewingId(id);
    try {
      await adminAPI.reviewWithdrawal(id, action, reason);
      toast.success(action === 'approve' ? 'تمت الموافقة على السحب' : 'تم رفض السحب');
      fetchData();
    } catch (error) {
      toast.error('فشل في معالجة الطلب');
    } finally {
      setReviewingId(null);
    }
  };

  const handleEmergencyToggle = async () => {
    try {
      if (stats?.emergency_mode === 'on') {
        await adminAPI.disableEmergency();
        toast.success('تم إيقاف وضع الطوارئ');
      } else {
        await adminAPI.enableEmergency();
        toast.success('تم تفعيل وضع الطوارئ');
      }
      fetchData();
    } catch (error) {
      toast.error('فشل في تغيير الوضع');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          لوحة تحكم الإدارة
        </h1>
        <button
          onClick={handleEmergencyToggle}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
            stats?.emergency_mode === 'on'
              ? 'bg-red-600 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          <Shield className="w-4 h-4" />
          {stats?.emergency_mode === 'on' ? 'إيقاف الطوارئ' : 'وضع الطوارئ'}
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">إجمالي المستخدمين</span>
            <Users className="w-5 h-5 text-blue-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats?.total_users || 0}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {stats?.active_users || 0} نشط
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">إجمالي الأصول</span>
            <DollarSign className="w-5 h-5 text-green-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${stats?.total_assets_usd.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0'}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {stats?.total_units.toFixed(2) || 0} وحدة
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">قيمة الوحدة (NAV)</span>
            <TrendingUp className="w-5 h-5 text-primary-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${stats?.current_nav.toFixed(4) || '1.0000'}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            HWM: ${stats?.high_water_mark.toFixed(4) || '1.0000'}
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600 dark:text-gray-400">طلبات السحب المعلقة</span>
            <Clock className="w-5 h-5 text-yellow-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats?.pending_withdrawals || 0}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            بانتظار المراجعة
          </p>
        </div>
      </div>

      {/* Today Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            إيداعات اليوم
          </h3>
          <p className="text-3xl font-bold text-green-600">
            ${stats?.total_deposits_today.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0'}
          </p>
        </div>
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            سحوبات اليوم
          </h3>
          <p className="text-3xl font-bold text-blue-600">
            ${stats?.total_withdrawals_today.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0'}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab('overview')}
          className={`pb-3 px-1 border-b-2 transition-colors ${
            activeTab === 'overview'
              ? 'border-primary-600 text-primary-600'
              : 'border-transparent text-gray-600 dark:text-gray-400'
          }`}
        >
          نظرة عامة
        </button>
        <button
          onClick={() => setActiveTab('withdrawals')}
          className={`pb-3 px-1 border-b-2 transition-colors ${
            activeTab === 'withdrawals'
              ? 'border-primary-600 text-primary-600'
              : 'border-transparent text-gray-600 dark:text-gray-400'
          }`}
        >
          طلبات السحب ({pendingWithdrawals.length})
        </button>
        <button
          onClick={() => setActiveTab('users')}
          className={`pb-3 px-1 border-b-2 transition-colors ${
            activeTab === 'users'
              ? 'border-primary-600 text-primary-600'
              : 'border-transparent text-gray-600 dark:text-gray-400'
          }`}
        >
          المستخدمين
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'withdrawals' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            طلبات السحب المعلقة
          </h2>
          {pendingWithdrawals.length > 0 ? (
            <div className="space-y-4">
              {pendingWithdrawals.map((withdrawal) => (
                <div
                  key={withdrawal.id}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {withdrawal.amount} {withdrawal.coin}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        إلى: <span className="font-mono" dir="ltr">{withdrawal.to_address}</span>
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        الشبكة: {withdrawal.network} | 
                        التاريخ: {format(new Date(withdrawal.requested_at), 'dd/MM/yyyy HH:mm')}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleWithdrawalReview(withdrawal.id, 'approve')}
                        disabled={reviewingId === withdrawal.id}
                        className="flex items-center gap-1 px-3 py-1 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                      >
                        <Check className="w-4 h-4" />
                        موافقة
                      </button>
                      <button
                        onClick={() => {
                          const reason = prompt('سبب الرفض:');
                          if (reason) {
                            handleWithdrawalReview(withdrawal.id, 'reject', reason);
                          }
                        }}
                        disabled={reviewingId === withdrawal.id}
                        className="flex items-center gap-1 px-3 py-1 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                      >
                        <X className="w-4 h-4" />
                        رفض
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-gray-600 dark:text-gray-400 py-8">
              لا توجد طلبات سحب معلقة
            </p>
          )}
        </div>
      )}

      {activeTab === 'users' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            قائمة المستخدمين
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">المستخدم</th>
                  <th className="pb-3 font-medium">الرصيد</th>
                  <th className="pb-3 font-medium">إجمالي الإيداعات</th>
                  <th className="pb-3 font-medium">الحالة</th>
                  <th className="pb-3 font-medium">التسجيل</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr
                    key={user.id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3">
                      <p className="font-medium text-gray-900 dark:text-white">
                        {user.full_name || 'بدون اسم'}
                      </p>
                      <p className="text-sm text-gray-500">{user.email}</p>
                    </td>
                    <td className="py-3">
                      ${user.current_value_usd.toFixed(2)}
                    </td>
                    <td className="py-3">
                      ${user.total_deposited.toFixed(2)}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          user.status === 'active'
                            ? 'badge-success'
                            : 'badge-danger'
                        }`}
                      >
                        {user.status === 'active' ? 'نشط' : 'معلق'}
                      </span>
                    </td>
                    <td className="py-3 text-gray-600 dark:text-gray-400">
                      {format(new Date(user.created_at), 'dd/MM/yyyy')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'overview' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            ملخص المنصة
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-gray-600 dark:text-gray-400">إجمالي الرسوم المحصلة</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white mt-1">
                ${stats?.total_fees_collected.toFixed(2) || '0'}
              </p>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-gray-600 dark:text-gray-400">وضع الطوارئ</p>
              <p className={`text-xl font-bold mt-1 ${
                stats?.emergency_mode === 'on' ? 'text-red-600' : 'text-green-600'
              }`}>
                {stats?.emergency_mode === 'on' ? 'مفعّل' : 'غير مفعّل'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Admin;
