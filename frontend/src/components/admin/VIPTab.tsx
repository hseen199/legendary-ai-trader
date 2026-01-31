// VIP Tab Component for Admin Dashboard
import React, { useState, useEffect } from 'react';

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
  recent_upgrades: Array<{
    user_id: number;
    email: string;
    from_level: string;
    to_level: string;
    upgraded_at: string;
  }>;
}

interface VIPTabProps {
  onNotification: (type: 'success' | 'error', message: string) => void;
}

const VIPTab: React.FC<VIPTabProps> = ({ onNotification }) => {
  const [levels, setLevels] = useState<VIPLevel[]>([]);
  const [stats, setStats] = useState<VIPStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [newLevel, setNewLevel] = useState<string>('');

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

  const fetchVIPData = async () => {
    try {
      setLoading(true);
      
      // Fetch VIP levels
      const levelsRes = await fetch('/api/v1/vip/levels', {
        headers: getAuthHeaders()
      });
      if (levelsRes.ok) {
        const levelsData = await levelsRes.json();
        setLevels(levelsData);
      }

      // Fetch VIP stats
      const statsRes = await fetch('/api/v1/vip/admin/stats', {
        headers: getAuthHeaders()
      });
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
    } catch (error) {
      console.error('Error fetching VIP data:', error);
      onNotification('error', 'فشل في جلب بيانات VIP');
    } finally {
      setLoading(false);
    }
  };

  const handleUpgradeUser = async () => {
    if (!selectedUser || !newLevel) {
      onNotification('error', 'يرجى اختيار المستخدم والمستوى');
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
        onNotification('success', 'تم ترقية المستخدم بنجاح');
        fetchVIPData();
        setSelectedUser('');
        setNewLevel('');
      } else {
        const error = await response.json();
        onNotification('error', error.detail || 'فشل في ترقية المستخدم');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ في ترقية المستخدم');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">جاري تحميل بيانات VIP...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* VIP Statistics */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">📊 إحصائيات VIP</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {levels.map(level => (
            <div 
              key={level.key}
              className="bg-gray-700 rounded-lg p-4 text-center"
              style={{ borderColor: level.color, borderWidth: '2px' }}
            >
              <div className="text-3xl mb-2">{level.icon}</div>
              <div className="font-bold" style={{ color: level.color }}>{level.name_ar}</div>
              <div className="text-2xl font-bold text-white">
                {stats?.level_distribution[level.key] || 0}
              </div>
              <div className="text-gray-400 text-sm">مشترك</div>
            </div>
          ))}
        </div>
      </div>

      {/* VIP Levels Configuration */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">⚙️ مستويات VIP</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
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
                <tr key={level.key} className="border-b border-gray-700 hover:bg-gray-700/50">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{level.icon}</span>
                      <span style={{ color: level.color }}>{level.name_ar}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4">${level.min_deposit.toLocaleString()}</td>
                  <td className="py-3 px-4">
                    {level.max_deposit ? `$${level.max_deposit.toLocaleString()}` : '∞'}
                  </td>
                  <td className="py-3 px-4">{level.performance_fee}%</td>
                  <td className="py-3 px-4">{level.referral_bonus}%</td>
                  <td className="py-3 px-4">
                    <div className="flex gap-1 flex-wrap">
                      {level.priority_support && <span className="bg-blue-600 px-2 py-1 rounded text-xs">دعم أولوية</span>}
                      {level.weekly_reports && <span className="bg-green-600 px-2 py-1 rounded text-xs">تقارير أسبوعية</span>}
                      {level.daily_reports && <span className="bg-purple-600 px-2 py-1 rounded text-xs">تقارير يومية</span>}
                      {level.dedicated_manager && <span className="bg-yellow-600 px-2 py-1 rounded text-xs">مدير مخصص</span>}
                      {level.early_access && <span className="bg-pink-600 px-2 py-1 rounded text-xs">وصول مبكر</span>}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Manual Upgrade Section */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">⬆️ ترقية يدوية</h3>
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-gray-400 mb-2">معرف المستخدم</label>
            <input
              type="number"
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white"
              placeholder="أدخل معرف المستخدم"
            />
          </div>
          <div className="flex-1">
            <label className="block text-gray-400 mb-2">المستوى الجديد</label>
            <select
              value={newLevel}
              onChange={(e) => setNewLevel(e.target.value)}
              className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white"
            >
              <option value="">اختر المستوى</option>
              {levels.map(level => (
                <option key={level.key} value={level.key}>
                  {level.icon} {level.name_ar}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={handleUpgradeUser}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg transition"
          >
            ترقية
          </button>
        </div>
      </div>

      {/* Recent Upgrades */}
      {stats?.recent_upgrades && stats.recent_upgrades.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold mb-4">📈 آخر الترقيات</h3>
          <div className="space-y-2">
            {stats.recent_upgrades.map((upgrade, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-700 rounded-lg p-3">
                <div>
                  <span className="text-white">{upgrade.email}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-400">{upgrade.from_level}</span>
                  <span className="text-green-400">→</span>
                  <span className="text-green-400">{upgrade.to_level}</span>
                </div>
                <div className="text-gray-400 text-sm">
                  {new Date(upgrade.upgraded_at).toLocaleDateString('ar-SA')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VIPTab;
