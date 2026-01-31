// Communication Tab Component for Admin Dashboard
import React, { useState, useEffect } from 'react';

interface CommunicationStats {
  total_notifications: number;
  unread_notifications: number;
  read_rate: number;
  vip_distribution: Record<string, number>;
  recent_broadcasts: Array<{
    id: number;
    type: string;
    title: string;
    created_at: string;
  }>;
}

interface CommunicationTabProps {
  onNotification: (type: 'success' | 'error', message: string) => void;
}

const CommunicationTab: React.FC<CommunicationTabProps> = ({ onNotification }) => {
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<CommunicationStats | null>(null);
  const [broadcastTitle, setBroadcastTitle] = useState('');
  const [broadcastMessage, setBroadcastMessage] = useState('');
  const [messageType, setMessageType] = useState('announcement');
  const [targetAudience, setTargetAudience] = useState('all');
  const [sendEmail, setSendEmail] = useState(true);
  const [sendNotification, setSendNotification] = useState(true);
  const [audienceCount, setAudienceCount] = useState<number | null>(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    };
  };

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    fetchAudienceCount();
  }, [targetAudience]);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/communication/stats', {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching communication stats:', error);
    }
  };

  const fetchAudienceCount = async () => {
    try {
      const response = await fetch(`/api/v1/communication/audience-count?target_audience=${targetAudience}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setAudienceCount(data.count);
      }
    } catch (error) {
      console.error('Error fetching audience count:', error);
    }
  };

  const sendBroadcast = async () => {
    if (!broadcastTitle || !broadcastMessage) {
      onNotification('error', 'يرجى ملء جميع الحقول المطلوبة');
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/v1/communication/broadcast', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          title: broadcastTitle,
          message: broadcastMessage,
          message_type: messageType,
          target_audience: targetAudience,
          send_email: sendEmail,
          send_notification: sendNotification
        })
      });

      if (response.ok) {
        const data = await response.json();
        onNotification('success', `تم إرسال الرسالة إلى ${data.sent_count} مستخدم`);
        setBroadcastTitle('');
        setBroadcastMessage('');
        fetchStats();
      } else {
        const error = await response.json();
        onNotification('error', error.detail || 'فشل في إرسال الرسالة');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ في إرسال الرسالة');
    } finally {
      setLoading(false);
    }
  };

  const sendMaintenanceNotice = async () => {
    const startTime = prompt('أدخل وقت بداية الصيانة (YYYY-MM-DD HH:MM):');
    const endTime = prompt('أدخل وقت نهاية الصيانة (YYYY-MM-DD HH:MM):');
    const description = prompt('وصف الصيانة (اختياري):');

    if (!startTime || !endTime) {
      onNotification('error', 'يرجى إدخال أوقات الصيانة');
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/v1/communication/maintenance', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          start_time: startTime,
          end_time: endTime,
          description: description || undefined
        })
      });

      if (response.ok) {
        const data = await response.json();
        onNotification('success', `تم إرسال إشعار الصيانة إلى ${data.sent_count} مستخدم`);
      } else {
        onNotification('error', 'فشل في إرسال إشعار الصيانة');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ في إرسال إشعار الصيانة');
    } finally {
      setLoading(false);
    }
  };

  const sendMarketUpdate = async (updateType: string) => {
    const summary = prompt('أدخل ملخص تحديث السوق:');
    if (!summary) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/v1/communication/market-update?update_type=${updateType}&summary=${encodeURIComponent(summary)}`, {
        method: 'POST',
        headers: getAuthHeaders()
      });

      if (response.ok) {
        onNotification('success', 'تم إرسال تحديث السوق');
      } else {
        onNotification('error', 'فشل في إرسال تحديث السوق');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Communication Stats */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">📊 إحصائيات التواصل</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-gray-400 text-sm">إجمالي الإشعارات</div>
            <div className="text-2xl font-bold text-white">{stats?.total_notifications || 0}</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-gray-400 text-sm">غير مقروءة</div>
            <div className="text-2xl font-bold text-yellow-400">{stats?.unread_notifications || 0}</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-gray-400 text-sm">نسبة القراءة</div>
            <div className="text-2xl font-bold text-green-400">{(stats?.read_rate || 0).toFixed(1)}%</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-gray-400 text-sm">إجمالي المستخدمين</div>
            <div className="text-2xl font-bold text-blue-400">
              {stats?.vip_distribution ? Object.values(stats.vip_distribution).reduce((a, b) => a + b, 0) : 0}
            </div>
          </div>
        </div>
      </div>

      {/* Broadcast Message */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">📢 إرسال رسالة جماعية</h3>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 mb-2">نوع الرسالة</label>
              <select
                value={messageType}
                onChange={(e) => setMessageType(e.target.value)}
                className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white"
              >
                <option value="announcement">📣 إعلان</option>
                <option value="update">🔄 تحديث</option>
                <option value="alert">⚠️ تنبيه</option>
                <option value="promotion">🎁 عرض ترويجي</option>
                <option value="maintenance">🔧 صيانة</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-400 mb-2">
                الجمهور المستهدف 
                {audienceCount !== null && (
                  <span className="text-blue-400 mr-2">({audienceCount} مستخدم)</span>
                )}
              </label>
              <select
                value={targetAudience}
                onChange={(e) => setTargetAudience(e.target.value)}
                className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white"
              >
                <option value="all">الجميع</option>
                <option value="investors">المستثمرين فقط</option>
                <option value="vip">أعضاء VIP</option>
                <option value="vip_gold_plus">ذهبي وأعلى</option>
                <option value="inactive">غير النشطين</option>
                <option value="new_users">المستخدمين الجدد</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-gray-400 mb-2">عنوان الرسالة</label>
            <input
              type="text"
              value={broadcastTitle}
              onChange={(e) => setBroadcastTitle(e.target.value)}
              className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white"
              placeholder="أدخل عنوان الرسالة"
            />
          </div>

          <div>
            <label className="block text-gray-400 mb-2">محتوى الرسالة</label>
            <textarea
              value={broadcastMessage}
              onChange={(e) => setBroadcastMessage(e.target.value)}
              className="w-full bg-gray-700 rounded-lg px-4 py-2 text-white h-32"
              placeholder="أدخل محتوى الرسالة..."
            />
          </div>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={sendEmail}
                onChange={(e) => setSendEmail(e.target.checked)}
                className="w-5 h-5 rounded"
              />
              <span>إرسال بريد إلكتروني</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={sendNotification}
                onChange={(e) => setSendNotification(e.target.checked)}
                className="w-5 h-5 rounded"
              />
              <span>إشعار داخل المنصة</span>
            </label>
          </div>

          <button
            onClick={sendBroadcast}
            disabled={loading || !broadcastTitle || !broadcastMessage}
            className="w-full bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg transition disabled:opacity-50 font-bold"
          >
            {loading ? 'جاري الإرسال...' : '📤 إرسال الرسالة'}
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">⚡ إجراءات سريعة</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button
            onClick={sendMaintenanceNotice}
            disabled={loading}
            className="bg-yellow-600 hover:bg-yellow-700 px-4 py-3 rounded-lg transition disabled:opacity-50"
          >
            🔧 إشعار صيانة
          </button>
          <button
            onClick={() => sendMarketUpdate('bullish')}
            disabled={loading}
            className="bg-green-600 hover:bg-green-700 px-4 py-3 rounded-lg transition disabled:opacity-50"
          >
            📈 تحديث صعودي
          </button>
          <button
            onClick={() => sendMarketUpdate('bearish')}
            disabled={loading}
            className="bg-red-600 hover:bg-red-700 px-4 py-3 rounded-lg transition disabled:opacity-50"
          >
            📉 تحديث هبوطي
          </button>
          <button
            onClick={() => sendMarketUpdate('volatile')}
            disabled={loading}
            className="bg-purple-600 hover:bg-purple-700 px-4 py-3 rounded-lg transition disabled:opacity-50"
          >
            ⚡ تقلب عالي
          </button>
        </div>
      </div>

      {/* Recent Broadcasts */}
      {stats?.recent_broadcasts && stats.recent_broadcasts.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold mb-4">📜 آخر الرسائل المرسلة</h3>
          <div className="space-y-2">
            {stats.recent_broadcasts.map((broadcast) => (
              <div key={broadcast.id} className="flex items-center justify-between bg-gray-700 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">
                    {broadcast.type === 'announcement' && '📣'}
                    {broadcast.type === 'update' && '🔄'}
                    {broadcast.type === 'maintenance' && '🔧'}
                    {broadcast.type === 'alert' && '⚠️'}
                  </span>
                  <span className="text-white">{broadcast.title}</span>
                </div>
                <div className="text-gray-400 text-sm">
                  {new Date(broadcast.created_at).toLocaleDateString('ar-SA')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CommunicationTab;
