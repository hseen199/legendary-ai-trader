import { useState, useEffect } from "react";
import { useLanguage } from "../../lib/i18n";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { 
  Crown, 
  Users, 
  TrendingUp, 
  Gift,
  RefreshCw,
  ChevronUp,
  Star,
  Award,
  Diamond,
  Gem,
} from "lucide-react";
import { Link } from "react-router-dom";

interface VIPLevel {
  key: string;
  name_ar: string;
  name_en: string;
  icon: string;
  color: string;
  min_deposit: number;
  max_deposit: number | null;
  performance_fee: number;
  referral_bonus: number;
  priority_support: boolean;
  weekly_reports: boolean;
  daily_reports: boolean;
  dedicated_manager: boolean;
  early_access: boolean;
}

interface VIPStats {
  total_vip_users: number;
  level_distribution: Record<string, number>;
}

export default function VIPManagement() {
  const { t } = useLanguage();
  const [levels, setLevels] = useState<VIPLevel[]>([]);
  const [stats, setStats] = useState<VIPStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState("");
  const [newLevel, setNewLevel] = useState("");
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    };
  };

  useEffect(() => {
    fetchVIPData();
  }, []);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const fetchVIPData = async () => {
    try {
      setLoading(true);
      
      const levelsRes = await fetch('/api/v1/vip/levels', {
        headers: getAuthHeaders()
      });
      if (levelsRes.ok) {
        const levelsData = await levelsRes.json();
        setLevels(levelsData);
      }

      const statsRes = await fetch('/api/v1/vip/admin/stats', {
        headers: getAuthHeaders()
      });
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
    } catch (error) {
      console.error('Error fetching VIP data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgradeUser = async () => {
    if (!selectedUser || !newLevel) {
      setNotification({ type: 'error', message: 'يرجى اختيار المستخدم والمستوى' });
      return;
    }

    try {
      const response = await fetch('/api/v1/vip/admin/upgrade', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          user_id: parseInt(selectedUser),
          new_level: newLevel
        })
      });

      if (response.ok) {
        setNotification({ type: 'success', message: 'تم ترقية المستخدم بنجاح' });
        fetchVIPData();
        setSelectedUser('');
        setNewLevel('');
      } else {
        const error = await response.json();
        setNotification({ type: 'error', message: error.detail || 'فشل في ترقية المستخدم' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ في ترقية المستخدم' });
    }
  };

  const getLevelIcon = (key: string) => {
    switch (key) {
      case 'bronze': return <Award className="w-6 h-6 text-amber-700" />;
      case 'silver': return <Star className="w-6 h-6 text-gray-400" />;
      case 'gold': return <Crown className="w-6 h-6 text-yellow-400" />;
      case 'platinum': return <Gem className="w-6 h-6 text-purple-400" />;
      case 'diamond': return <Diamond className="w-6 h-6 text-cyan-400" />;
      default: return <Award className="w-6 h-6" />;
    }
  };

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6" dir="rtl">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-yellow-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-amber-500/10 rounded-full blur-[100px]" />
      </div>

      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg ${
          notification.type === 'success' ? 'bg-green-600' : 'bg-red-600'
        } text-white`}>
          {notification.message}
        </div>
      )}

      {/* Header */}
      <div className="relative flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <Link to="/admin" className="text-white/40 hover:text-white transition-colors">
            ← العودة
          </Link>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-yellow-200 via-amber-200 to-orange-200 bg-clip-text text-transparent flex items-center gap-2">
              <Crown className="w-8 h-8 text-yellow-400" />
              إدارة VIP
            </h1>
            <p className="text-white/40 text-sm mt-1">إدارة مستويات العضوية والمزايا</p>
          </div>
        </div>
        <button 
          onClick={fetchVIPData}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 hover:bg-yellow-500/20 hover:border-yellow-500/40 transition-all duration-300"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* VIP Statistics */}
      <div className="relative grid grid-cols-2 md:grid-cols-5 gap-4">
        {levels.map(level => (
          <div 
            key={level.key}
            className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 text-center"
            style={{ borderColor: `${level.color}30` }}
          >
            <div className="text-4xl mb-2">{level.icon}</div>
            <div className="font-bold text-lg" style={{ color: level.color }}>{level.name_ar}</div>
            <div className="text-3xl font-bold text-white mt-2">
              {stats?.level_distribution?.[level.key] || 0}
            </div>
            <div className="text-white/40 text-sm">مشترك</div>
          </div>
        ))}
      </div>

      {/* VIP Levels Table */}
      <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Star className="w-5 h-5 text-yellow-400" />
            مستويات VIP والمزايا
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-white/40 border-b border-white/10">
                  <th className="text-right py-3 px-4">المستوى</th>
                  <th className="text-right py-3 px-4">الحد الأدنى</th>
                  <th className="text-right py-3 px-4">الحد الأقصى</th>
                  <th className="text-right py-3 px-4">رسوم الأداء</th>
                  <th className="text-right py-3 px-4">مكافأة الإحالة</th>
                  <th className="text-right py-3 px-4">المزايا</th>
                </tr>
              </thead>
              <tbody>
                {levels.map(level => (
                  <tr key={level.key} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        {getLevelIcon(level.key)}
                        <span className="font-bold" style={{ color: level.color }}>{level.name_ar}</span>
                      </div>
                    </td>
                    <td className="py-4 px-4 text-white">${level.min_deposit.toLocaleString()}</td>
                    <td className="py-4 px-4 text-white">
                      {level.max_deposit ? `$${level.max_deposit.toLocaleString()}` : '∞'}
                    </td>
                    <td className="py-4 px-4 text-green-400">{level.performance_fee}%</td>
                    <td className="py-4 px-4 text-blue-400">{level.referral_bonus}%</td>
                    <td className="py-4 px-4">
                      <div className="flex gap-1 flex-wrap">
                        {level.priority_support && <Badge className="bg-blue-500/20 text-blue-400">دعم أولوية</Badge>}
                        {level.weekly_reports && <Badge className="bg-green-500/20 text-green-400">تقارير أسبوعية</Badge>}
                        {level.daily_reports && <Badge className="bg-purple-500/20 text-purple-400">تقارير يومية</Badge>}
                        {level.dedicated_manager && <Badge className="bg-yellow-500/20 text-yellow-400">مدير مخصص</Badge>}
                        {level.early_access && <Badge className="bg-pink-500/20 text-pink-400">وصول مبكر</Badge>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Manual Upgrade */}
      <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <ChevronUp className="w-5 h-5 text-green-400" />
            ترقية يدوية
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 items-end flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-white/60 mb-2 text-sm">معرف المستخدم</label>
              <input
                type="number"
                value={selectedUser}
                onChange={(e) => setSelectedUser(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-yellow-500/50 focus:outline-none transition-colors"
                placeholder="أدخل معرف المستخدم"
              />
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-white/60 mb-2 text-sm">المستوى الجديد</label>
              <select
                value={newLevel}
                onChange={(e) => setNewLevel(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-yellow-500/50 focus:outline-none transition-colors"
              >
                <option value="" className="bg-gray-900">اختر المستوى</option>
                {levels.map(level => (
                  <option key={level.key} value={level.key} className="bg-gray-900">
                    {level.icon} {level.name_ar}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={handleUpgradeUser}
              disabled={loading || !selectedUser || !newLevel}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-yellow-500 to-amber-500 text-black font-bold hover:from-yellow-400 hover:to-amber-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ترقية المستخدم
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
