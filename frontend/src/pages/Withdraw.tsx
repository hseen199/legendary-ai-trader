import React, { useEffect, useState } from 'react';
import { walletAPI } from '../services/api';
import { AlertCircle, Info, RefreshCw, Lock } from 'lucide-react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';
import { useLanguage } from '@/lib/i18n';

interface Balance {
  units: number;
  current_value_usd: number;
  nav: number;
  can_withdraw: boolean;
  last_deposit_at?: string;
}

interface WithdrawalHistory {
  id: number;
  amount: number;
  units_to_withdraw: number;
  to_address: string;
  network: string;
  coin: string;
  status: string;
  requested_at: string;
  reviewed_at?: string;
  rejection_reason?: string;
  completed_at?: string;
}

const Withdraw: React.FC = () => {
  const { t, language } = useLanguage();
  const [balance, setBalance] = useState<Balance | null>(null);
  const [history, setHistory] = useState<WithdrawalHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [formData, setFormData] = useState({
    amount: '',
    to_address: '',
    network: 'BEP20',
  });

  const networks = [
    { id: 'BEP20', name: 'BEP20 (BSC)' },
    { id: 'SOL', name: 'Solana' },
  ];

  const fetchData = async () => {
    try {
      const [balanceRes, historyRes] = await Promise.all([
        walletAPI.getBalance(),
        walletAPI.getWithdrawalHistory(),
      ]);
      setBalance(balanceRes.data);
      setHistory(historyRes.data);
    } catch (error) {
      toast.error(t.dashboard.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!balance?.can_withdraw) {
      toast.error('السحب غير متاح حالياً');
      return;
    }

    const amount = parseFloat(formData.amount);
    if (isNaN(amount) || amount <= 0) {
      toast.error('يرجى إدخال مبلغ صحيح');
      return;
    }

    if (amount > balance.current_value_usd) {
      toast.error('المبلغ أكبر من الرصيد المتاح');
      return;
    }

    if (!formData.to_address) {
      toast.error('يرجى إدخال عنوان السحب');
      return;
    }

    setIsSubmitting(true);

    try {
      await walletAPI.requestWithdrawal({
        amount,
        to_address: formData.to_address,
        network: formData.network,
        coin: 'USDC',
      });
      toast.success('تم إرسال طلب السحب بنجاح');
      setFormData({ amount: '', to_address: '', network: 'BEP20' });
      fetchData();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'فشل في إرسال الطلب');
    } finally {
      setIsSubmitting(false);
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
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
        سحب الأموال
      </h1>

      {/* Lock Warning */}
      {balance && !balance.can_withdraw && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
          <div className="flex gap-3">
            <Lock className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-800 dark:text-red-200 font-medium">
                السحب مقفل حالياً
              </p>
              <p className="text-red-700 dark:text-red-300 text-sm mt-1">
                يجب انتظار انتهاء فترة القفل (7 أيام من آخر إيداع) قبل السحب.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Balance Card */}
      <div className="card p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-600 dark:text-gray-400">الرصيد المتاح للسحب</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-white mt-1">
              ${balance?.current_value_usd.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0.00'}
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {balance?.units.toFixed(4) || '0'} وحدة × ${balance?.nav.toFixed(4) || '1'} NAV
            </p>
          </div>
          <div
            className={`px-4 py-2 rounded-lg ${
              balance?.can_withdraw
                ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                : 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
            }`}
          >
            {balance?.can_withdraw ? 'السحب متاح' : 'السحب مقفل'}
          </div>
        </div>
      </div>

      {/* Withdrawal Form */}
      <div className="card p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          طلب سحب جديد
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Amount */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              المبلغ (USDC)
            </label>
            <div className="relative">
              <input
                type="number"
                value={formData.amount}
                onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                className="input"
                placeholder="0.00"
                step="0.01"
                min="10"
                max={balance?.current_value_usd}
                disabled={!balance?.can_withdraw}
                dir="ltr"
              />
              <button
                type="button"
                onClick={() =>
                  setFormData({
                    ...formData,
                    amount: balance?.current_value_usd.toString() || '',
                  })
                }
                className="absolute left-2 top-1/2 -translate-y-1/2 text-primary-600 text-sm font-medium"
                disabled={!balance?.can_withdraw}
              >{t.common.all}</button>
            </div>
          </div>

          {/* Network */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              الشبكة
            </label>
            <select
              value={formData.network}
              onChange={(e) => setFormData({ ...formData, network: e.target.value })}
              className="input"
              disabled={!balance?.can_withdraw}
            >
              {networks.map((network) => (
                <option key={network.id} value={network.id}>
                  {network.name}
                </option>
              ))}
            </select>
          </div>

          {/* Address */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">{t.wallet.walletAddress}</label>
            <input
              type="text"
              value={formData.to_address}
              onChange={(e) => setFormData({ ...formData, to_address: e.target.value })}
              className="input font-mono"
              placeholder="أدخل عنوان محفظتك"
              disabled={!balance?.can_withdraw}
              dir="ltr"
            />
          </div>

          {/* Info */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="flex gap-2">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0" />
              <div className="text-sm text-blue-700 dark:text-blue-300">
                <p>• الحد الأدنى للسحب: 10 USDC</p>
                <p>• طلبات السحب تحتاج موافقة الإدارة</p>
                <p>• ستصلك رسالة تأكيد على الإيميل</p>
              </div>
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={!balance?.can_withdraw || isSubmitting}
            className="btn-primary w-full py-3 disabled:opacity-50"
          >
            {isSubmitting ? 'جاري الإرسال...' : 'إرسال طلب السحب'}
          </button>
        </form>
      </div>

      {/* Withdrawal History */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          سجل طلبات السحب
        </h2>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">{t.wallet.amount}</th>
                  <th className="pb-3 font-medium">العنوان</th>
                  <th className="pb-3 font-medium">الحالة</th>
                  <th className="pb-3 font-medium">{t.trades.date}</th>
                </tr>
              </thead>
              <tbody>
                {history.map((withdrawal) => (
                  <tr
                    key={withdrawal.id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3">
                      {withdrawal.amount} {withdrawal.coin}
                    </td>
                    <td className="py-3 font-mono text-sm" dir="ltr">
                      {withdrawal.to_address.slice(0, 8)}...
                      {withdrawal.to_address.slice(-6)}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          withdrawal.status === 'completed'
                            ? 'badge-success'
                            : withdrawal.status === 'rejected'
                            ? 'badge-danger'
                            : 'badge-warning'
                        }`}
                      >
                        {withdrawal.status === 'completed'
                          ? 'مكتمل'
                          : withdrawal.status === 'rejected'
                          ? 'مرفوض'
                          : withdrawal.status === 'pending_approval'
                          ? 'بانتظار الموافقة'
                          : withdrawal.status === 'approved'
                          ? 'تمت الموافقة'
                          : 'قيد المعالجة'}
                      </span>
                      {withdrawal.rejection_reason && (
                        <p className="text-xs text-red-600 mt-1">
                          {withdrawal.rejection_reason}
                        </p>
                      )}
                    </td>
                    <td className="py-3 text-gray-600 dark:text-gray-400">
                      {format(new Date(withdrawal.requested_at), 'dd/MM/yyyy HH:mm')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center text-gray-600 dark:text-gray-400 py-8">
            لا توجد طلبات سحب حتى الآن
          </p>
        )}
      </div>
    </div>
  );
};

export default Withdraw;
