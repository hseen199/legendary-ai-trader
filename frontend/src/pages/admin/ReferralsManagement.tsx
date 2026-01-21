import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';
import { toast } from 'react-hot-toast';
import { 
  Users, 
  Gift, 
  Search, 
  DollarSign,
  UserPlus,
  ArrowRight,
  CheckCircle,
  Clock
} from 'lucide-react';

interface Referral {
  id: number;
  referrer_id: number;
  referrer_email: string;
  referrer_name: string;
  referred_id: number;
  referred_email: string;
  referred_name: string;
  referred_at: string;
  reward_given: boolean;
  reward_amount: number;
}

interface ReferralStats {
  total_referrals: number;
  total_rewards_given: number;
  pending_rewards: number;
}

export default function ReferralsManagement() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState<'all' | 'rewarded' | 'pending'>('all');
  const [selectedReferral, setSelectedReferral] = useState<Referral | null>(null);
  const [rewardAmount, setRewardAmount] = useState('');
  const queryClient = useQueryClient();

  // جلب قائمة الإحالات
  const { data: referrals, isLoading } = useQuery({
    queryKey: ['admin-referrals'],
    queryFn: async () => {
      const response = await api.get('/api/v1/admin/referrals');
      return response.data;
    }
  });

  // جلب إحصائيات الإحالات
  const { data: stats } = useQuery({
    queryKey: ['admin-referral-stats'],
    queryFn: async () => {
      const response = await api.get('/api/v1/admin/referrals/stats');
      return response.data;
    }
  });

  // إضافة مكافأة للمُحيل
  const giveRewardMutation = useMutation({
    mutationFn: async ({ referrerId, amount }: { referrerId: number; amount: number }) => {
      const response = await api.post('/api/v1/admin/referrals/reward', {
        referrer_id: referrerId,
        amount: amount
      });
      return response.data;
    },
    onSuccess: () => {
      toast.success('تم إضافة المكافأة بنجاح');
      queryClient.invalidateQueries({ queryKey: ['admin-referrals'] });
      queryClient.invalidateQueries({ queryKey: ['admin-referral-stats'] });
      setSelectedReferral(null);
      setRewardAmount('');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'حدث خطأ أثناء إضافة المكافأة');
    }
  });

  // تصفية الإحالات
  const filteredReferrals = referrals?.filter((ref: Referral) => {
    const matchesSearch = 
      ref.referrer_email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ref.referred_email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ref.referrer_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ref.referred_name?.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filter === 'rewarded') return matchesSearch && ref.reward_given;
    if (filter === 'pending') return matchesSearch && !ref.reward_given;
    return matchesSearch;
  }) || [];

  return (
    <div className="min-h-screen bg-[#0a0a1a] text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* العنوان */}
        <div className="flex items-center gap-3 mb-8">
          <Users className="w-8 h-8 text-purple-500" />
          <h1 className="text-2xl font-bold">إدارة الإحالات</h1>
        </div>

        {/* الإحصائيات */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-[#1a1a2e] rounded-xl p-6 border border-purple-500/20">
            <div className="flex items-center gap-3 mb-2">
              <UserPlus className="w-5 h-5 text-blue-400" />
              <span className="text-gray-400">إجمالي الإحالات</span>
            </div>
            <p className="text-3xl font-bold">{stats?.total_referrals || 0}</p>
          </div>
          <div className="bg-[#1a1a2e] rounded-xl p-6 border border-purple-500/20">
            <div className="flex items-center gap-3 mb-2">
              <Gift className="w-5 h-5 text-green-400" />
              <span className="text-gray-400">المكافآت المدفوعة</span>
            </div>
            <p className="text-3xl font-bold text-green-400">${stats?.total_rewards_given || 0}</p>
          </div>
          <div className="bg-[#1a1a2e] rounded-xl p-6 border border-purple-500/20">
            <div className="flex items-center gap-3 mb-2">
              <Clock className="w-5 h-5 text-yellow-400" />
              <span className="text-gray-400">مكافآت معلقة</span>
            </div>
            <p className="text-3xl font-bold text-yellow-400">{stats?.pending_rewards || 0}</p>
          </div>
        </div>

        {/* البحث والفلترة */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="relative flex-1">
            <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="بحث بالاسم أو البريد الإلكتروني..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#1a1a2e] border border-purple-500/20 rounded-lg py-3 pr-10 pl-4 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                filter === 'all' ? 'bg-purple-600 text-white' : 'bg-[#1a1a2e] text-gray-400 hover:bg-[#2a2a3e]'
              }`}
            >
              الكل
            </button>
            <button
              onClick={() => setFilter('rewarded')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                filter === 'rewarded' ? 'bg-green-600 text-white' : 'bg-[#1a1a2e] text-gray-400 hover:bg-[#2a2a3e]'
              }`}
            >
              تم المكافأة
            </button>
            <button
              onClick={() => setFilter('pending')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                filter === 'pending' ? 'bg-yellow-600 text-white' : 'bg-[#1a1a2e] text-gray-400 hover:bg-[#2a2a3e]'
              }`}
            >
              معلق
            </button>
          </div>
        </div>

        {/* جدول الإحالات */}
        <div className="bg-[#1a1a2e] rounded-xl border border-purple-500/20 overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center text-gray-400">جاري التحميل...</div>
          ) : filteredReferrals.length === 0 ? (
            <div className="p-8 text-center text-gray-400">لا توجد إحالات</div>
          ) : (
            <table className="w-full">
              <thead className="bg-[#0a0a1a]">
                <tr>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">المُحيل (من)</th>
                  <th className="text-center py-4 px-6 text-gray-400 font-medium"></th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">المُحال (إلى)</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">تاريخ الإحالة</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">الحالة</th>
                  <th className="text-right py-4 px-6 text-gray-400 font-medium">الإجراءات</th>
                </tr>
              </thead>
              <tbody>
                {filteredReferrals.map((ref: Referral) => (
                  <tr key={ref.id} className="border-t border-purple-500/10 hover:bg-[#2a2a3e]">
                    <td className="py-4 px-6">
                      <div>
                        <p className="font-medium">{ref.referrer_name || 'غير محدد'}</p>
                        <p className="text-sm text-gray-400">{ref.referrer_email}</p>
                      </div>
                    </td>
                    <td className="py-4 px-6 text-center">
                      <ArrowRight className="w-5 h-5 text-purple-400 mx-auto" />
                    </td>
                    <td className="py-4 px-6">
                      <div>
                        <p className="font-medium">{ref.referred_name || 'غير محدد'}</p>
                        <p className="text-sm text-gray-400">{ref.referred_email}</p>
                      </div>
                    </td>
                    <td className="py-4 px-6 text-gray-400">
                      {new Date(ref.referred_at).toLocaleDateString('ar-SA')}
                    </td>
                    <td className="py-4 px-6">
                      {ref.reward_given ? (
                        <span className="flex items-center gap-1 text-green-400">
                          <CheckCircle className="w-4 h-4" />
                          تم المكافأة (${ref.reward_amount})
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-yellow-400">
                          <Clock className="w-4 h-4" />
                          معلق
                        </span>
                      )}
                    </td>
                    <td className="py-4 px-6">
                      {!ref.reward_given && (
                        <button
                          onClick={() => setSelectedReferral(ref)}
                          className="flex items-center gap-1 bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded-lg text-sm transition-colors"
                        >
                          <Gift className="w-4 h-4" />
                          إضافة مكافأة
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* نافذة إضافة المكافأة */}
        {selectedReferral && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-[#1a1a2e] rounded-xl p-6 w-full max-w-md border border-purple-500/20">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Gift className="w-5 h-5 text-green-400" />
                إضافة مكافأة للمُحيل
              </h3>
              
              <div className="mb-4 p-4 bg-[#0a0a1a] rounded-lg">
                <p className="text-gray-400 mb-2">المُحيل:</p>
                <p className="font-medium">{selectedReferral.referrer_name}</p>
                <p className="text-sm text-gray-400">{selectedReferral.referrer_email}</p>
              </div>

              <div className="mb-4 p-4 bg-[#0a0a1a] rounded-lg">
                <p className="text-gray-400 mb-2">أحال:</p>
                <p className="font-medium">{selectedReferral.referred_name}</p>
                <p className="text-sm text-gray-400">{selectedReferral.referred_email}</p>
              </div>

              <div className="mb-6">
                <label className="block text-gray-400 mb-2">مبلغ المكافأة (USD)</label>
                <div className="relative">
                  <DollarSign className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                  <input
                    type="number"
                    value={rewardAmount}
                    onChange={(e) => setRewardAmount(e.target.value)}
                    placeholder="0.00"
                    className="w-full bg-[#0a0a1a] border border-purple-500/20 rounded-lg py-3 pr-10 pl-4 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setSelectedReferral(null);
                    setRewardAmount('');
                  }}
                  className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-3 rounded-lg transition-colors"
                >
                  إلغاء
                </button>
                <button
                  onClick={() => {
                    if (rewardAmount && parseFloat(rewardAmount) > 0) {
                      giveRewardMutation.mutate({
                        referrerId: selectedReferral.referrer_id,
                        amount: parseFloat(rewardAmount)
                      });
                    }
                  }}
                  disabled={!rewardAmount || parseFloat(rewardAmount) <= 0}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 rounded-lg transition-colors"
                >
                  تأكيد المكافأة
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
