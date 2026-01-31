import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowDownCircle,
  Check,
  X,
  Clock,
  User,
  AlertCircle,
  RefreshCw,
  Search,
  Eye,
  CheckCircle,
  XCircle,
  Loader2
} from 'lucide-react';
import api from '../services/api';

interface Deposit {
  id: number;
  user_id: number;
  user_email: string | null;
  user_name: string | null;
  amount_usd: number;
  coin: string | null;
  network: string | null;
  status: string;
  created_at: string;
  confirmed_at: string | null;
  completed_at: string | null;
  payment_id: string | null;
  pay_address: string | null;
}

const DepositsManagement: React.FC = () => {
  const [deposits, setDeposits] = useState<Deposit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'completed' | 'rejected'>('pending');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDeposit, setSelectedDeposit] = useState<Deposit | null>(null);
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [rejectionReason, setRejectionReason] = useState('');
  const [processing, setProcessing] = useState<number | null>(null);

  const fetchDeposits = async () => {
    try {
      setLoading(true);
      setError(null);
      const endpoint = filter === 'pending' ? '/admin/deposits/pending' : '/admin/deposits';
      const params = filter !== 'pending' && filter !== 'all' ? { status: filter } : {};
      const response = await api.get(endpoint, { params });
      setDeposits(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'فشل في جلب الإيداعات');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDeposits();
  }, [filter]);

  const handleApprove = async (depositId: number) => {
    try {
      setProcessing(depositId);
      setError(null);
      await api.post(`/admin/deposits/${depositId}/approve`);
      // تحديث الإيداع محلياً فوراً قبل إعادة الجلب
      setDeposits(prev => prev.map(d => 
        d.id === depositId ? { ...d, status: 'completed' } : d
      ));
      setSelectedDeposit(null);
      // إعادة جلب البيانات للتأكد من التزامن
      await fetchDeposits();
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'فشل في الموافقة على الإيداع';
      // إذا كان الإيداع تمت معالجته مسبقاً، نحدث القائمة
      if (errorMessage.includes('already processed')) {
        setError('تم معالجة هذا الإيداع مسبقاً. جاري تحديث القائمة...');
        await fetchDeposits();
        setSelectedDeposit(null);
      } else {
        setError(errorMessage);
      }
    } finally {
      setProcessing(null);
    }
  };

  const handleReject = async (depositId: number) => {
    try {
      setProcessing(depositId);
      setError(null);
      await api.post(`/admin/deposits/${depositId}/reject`, null, {
        params: { rejection_reason: rejectionReason }
      });
      // تحديث الإيداع محلياً فوراً قبل إعادة الجلب
      setDeposits(prev => prev.map(d => 
        d.id === depositId ? { ...d, status: 'rejected' } : d
      ));
      setShowRejectModal(false);
      setSelectedDeposit(null);
      setRejectionReason('');
      // إعادة جلب البيانات للتأكد من التزامن
      await fetchDeposits();
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'فشل في رفض الإيداع';
      // إذا كان الإيداع تمت معالجته مسبقاً، نحدث القائمة
      if (errorMessage.includes('already processed')) {
        setError('تم معالجة هذا الإيداع مسبقاً. جاري تحديث القائمة...');
        await fetchDeposits();
        setShowRejectModal(false);
        setSelectedDeposit(null);
        setRejectionReason('');
      } else {
        setError(errorMessage);
      }
    } finally {
      setProcessing(null);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return (
          <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm flex items-center gap-1">
            <Clock className="w-3 h-3" />
            قيد الانتظار
          </span>
        );
      case 'completed':
        return (
          <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm flex items-center gap-1">
            <CheckCircle className="w-3 h-3" />
            مكتمل
          </span>
        );
      case 'rejected':
        return (
          <span className="px-3 py-1 bg-red-500/20 text-red-400 rounded-full text-sm flex items-center gap-1">
            <XCircle className="w-3 h-3" />
            مرفوض
          </span>
        );
      default:
        return (
          <span className="px-3 py-1 bg-gray-500/20 text-gray-400 rounded-full text-sm">
            {status}
          </span>
        );
    }
  };

  const filteredDeposits = deposits.filter(deposit => {
    if (!searchTerm) return true;
    const search = searchTerm.toLowerCase();
    return (
      deposit.user_email?.toLowerCase().includes(search) ||
      deposit.user_name?.toLowerCase().includes(search) ||
      deposit.id.toString().includes(search)
    );
  });

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('ar-SA', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <ArrowDownCircle className="w-7 h-7 text-green-400" />
            إدارة الإيداعات
          </h1>
          <p className="text-gray-400 mt-1">مراجعة والموافقة على طلبات الإيداع</p>
        </div>
        <button onClick={fetchDeposits} className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
          <RefreshCw className={`w-4 h-4 \${loading ? 'animate-spin' : ''}`} />
          تحديث
        </button>
      </div>

      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input type="text" placeholder="البحث بالبريد أو الاسم..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="w-full pr-10 pl-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-purple-500 focus:outline-none" />
        </div>
        <div className="flex gap-2">
          {(['pending', 'all', 'completed', 'rejected'] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)} className={`px-4 py-2 rounded-lg transition-colors \${filter === f ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}>
              {f === 'pending' && 'قيد الانتظار'}
              {f === 'all' && 'الكل'}
              {f === 'completed' && 'مكتمل'}
              {f === 'rejected' && 'مرفوض'}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
          <button onClick={() => setError(null)} className="mr-auto text-red-400 hover:text-red-300"><X className="w-4 h-4" /></button>
        </div>
      )}

      <div className="bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-20"><Loader2 className="w-8 h-8 text-purple-500 animate-spin" /></div>
        ) : filteredDeposits.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-gray-400">
            <ArrowDownCircle className="w-12 h-12 mb-4 opacity-50" />
            <p>لا توجد إيداعات {filter === 'pending' ? 'قيد الانتظار' : ''}</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-900/50">
                <tr>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">المستخدم</th>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">المبلغ</th>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">العملة</th>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">الحالة</th>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">التاريخ</th>
                  <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">الإجراءات</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {filteredDeposits.map((deposit) => (
                  <motion.tr key={deposit.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="hover:bg-gray-800/50 transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-purple-600/20 rounded-full flex items-center justify-center"><User className="w-5 h-5 text-purple-400" /></div>
                        <div>
                          <p className="text-white font-medium">{deposit.user_name || 'غير محدد'}</p>
                          <p className="text-gray-400 text-sm">{deposit.user_email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4"><span className="text-green-400 font-bold text-lg">\${deposit.amount_usd.toFixed(2)}</span></td>
                    <td className="px-6 py-4"><span className="text-gray-300">{deposit.coin || 'USDC'}</span></td>
                    <td className="px-6 py-4">{getStatusBadge(deposit.status)}</td>
                    <td className="px-6 py-4"><span className="text-gray-400 text-sm">{formatDate(deposit.created_at)}</span></td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <button onClick={() => setSelectedDeposit(deposit)} className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors" title="عرض التفاصيل"><Eye className="w-4 h-4 text-gray-300" /></button>
                        {deposit.status === 'pending' && (
                          <>
                            <button onClick={() => handleApprove(deposit.id)} disabled={processing === deposit.id} className="p-2 bg-green-600/20 hover:bg-green-600/40 rounded-lg transition-colors disabled:opacity-50" title="موافقة">
                              {processing === deposit.id ? <Loader2 className="w-4 h-4 text-green-400 animate-spin" /> : <Check className="w-4 h-4 text-green-400" />}
                            </button>
                            <button onClick={() => { setSelectedDeposit(deposit); setShowRejectModal(true); }} disabled={processing === deposit.id} className="p-2 bg-red-600/20 hover:bg-red-600/40 rounded-lg transition-colors disabled:opacity-50" title="رفض"><X className="w-4 h-4 text-red-400" /></button>
                          </>
                        )}
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <AnimatePresence>
        {selectedDeposit && !showRejectModal && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedDeposit(null)}>
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }} className="bg-gray-800 rounded-xl p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white">تفاصيل الإيداع</h3>
                <button onClick={() => setSelectedDeposit(null)} className="p-2 hover:bg-gray-700 rounded-lg transition-colors"><X className="w-5 h-5 text-gray-400" /></button>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg"><span className="text-gray-400">رقم الإيداع</span><span className="text-white font-mono">#{selectedDeposit.id}</span></div>
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg"><span className="text-gray-400">المستخدم</span><span className="text-white">{selectedDeposit.user_email}</span></div>
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg"><span className="text-gray-400">المبلغ</span><span className="text-green-400 font-bold">\${selectedDeposit.amount_usd.toFixed(2)}</span></div>
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg"><span className="text-gray-400">الحالة</span>{getStatusBadge(selectedDeposit.status)}</div>
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg"><span className="text-gray-400">تاريخ الطلب</span><span className="text-white text-sm">{formatDate(selectedDeposit.created_at)}</span></div>
              </div>
              {selectedDeposit.status === 'pending' && (
                <div className="flex gap-3 mt-6">
                  <button onClick={() => handleApprove(selectedDeposit.id)} disabled={processing === selectedDeposit.id} className="flex-1 py-3 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium flex items-center justify-center gap-2 disabled:opacity-50">
                    {processing === selectedDeposit.id ? <Loader2 className="w-5 h-5 animate-spin" /> : <><Check className="w-5 h-5" />موافقة</>}
                  </button>
                  <button onClick={() => setShowRejectModal(true)} disabled={processing === selectedDeposit.id} className="flex-1 py-3 bg-red-600 hover:bg-red-700 rounded-lg text-white font-medium flex items-center justify-center gap-2 disabled:opacity-50"><X className="w-5 h-5" />رفض</button>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showRejectModal && selectedDeposit && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => { setShowRejectModal(false); setRejectionReason(''); }}>
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }} className="bg-gray-800 rounded-xl p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-red-600/20 rounded-full flex items-center justify-center"><XCircle className="w-6 h-6 text-red-400" /></div>
                <div><h3 className="text-xl font-bold text-white">رفض الإيداع</h3><p className="text-gray-400 text-sm">إيداع #{selectedDeposit.id} - \${selectedDeposit.amount_usd.toFixed(2)}</p></div>
              </div>
              <div className="mb-6">
                <label className="block text-gray-400 mb-2">سبب الرفض (اختياري)</label>
                <textarea value={rejectionReason} onChange={(e) => setRejectionReason(e.target.value)} placeholder="أدخل سبب رفض الإيداع..." className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-red-500 focus:outline-none resize-none" rows={3} />
              </div>
              <div className="flex gap-3">
                <button onClick={() => { setShowRejectModal(false); setRejectionReason(''); }} className="flex-1 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-white font-medium">إلغاء</button>
                <button onClick={() => handleReject(selectedDeposit.id)} disabled={processing === selectedDeposit.id} className="flex-1 py-3 bg-red-600 hover:bg-red-700 rounded-lg text-white font-medium flex items-center justify-center gap-2 disabled:opacity-50">
                  {processing === selectedDeposit.id ? <Loader2 className="w-5 h-5 animate-spin" /> : <><X className="w-5 h-5" />تأكيد الرفض</>}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DepositsManagement;
