import { useState, useEffect } from "react";
import { useLanguage } from "../../lib/i18n";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { 
  MessageSquare, 
  Send, 
  Users,
  Bell,
  RefreshCw,
  Mail,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Zap,
  Wrench,
} from "lucide-react";
import { Link } from "react-router-dom";

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

export default function CommunicationManagement() {
  const { t } = useLanguage();
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<CommunicationStats | null>(null);
  const [broadcastTitle, setBroadcastTitle] = useState('');
  const [broadcastMessage, setBroadcastMessage] = useState('');
  const [messageType, setMessageType] = useState('announcement');
  const [targetAudience, setTargetAudience] = useState('all');
  const [sendEmail, setSendEmail] = useState(true);
  const [sendNotification, setSendNotification] = useState(true);
  const [audienceCount, setAudienceCount] = useState<number | null>(null);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

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

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

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
      setNotification({ type: 'error', message: 'يرجى ملء جميع الحقول المطلوبة' });
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
        setNotification({ type: 'success', message: `تم إرسال الرسالة إلى ${data.sent_count} مستخدم` });
        setBroadcastTitle('');
        setBroadcastMessage('');
        fetchStats();
      } else {
        const error = await response.json();
        setNotification({ type: 'error', message: error.detail || 'فشل في إرسال الرسالة' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ في إرسال الرسالة' });
    } finally {
      setLoading(false);
    }
  };

  const sendMaintenanceNotice = async () => {
    const startTime = prompt('أدخل وقت بداية الصيانة (YYYY-MM-DD HH:MM):');
    const endTime = prompt('أدخل وقت نهاية الصيانة (YYYY-MM-DD HH:MM):');
    const description = prompt('وصف الصيانة (اختياري):');

    if (!startTime || !endTime) {
      setNotification({ type: 'error', message: 'يرجى إدخال أوقات الصيانة' });
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
        setNotification({ type: 'success', message: `تم إرسال إشعار الصيانة إلى ${data.sent_count} مستخدم` });
      } else {
        setNotification({ type: 'error', message: 'فشل في إرسال إشعار الصيانة' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ في إرسال إشعار الصيانة' });
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
        setNotification({ type: 'success', message: 'تم إرسال تحديث السوق' });
        fetchStats();
      } else {
        setNotification({ type: 'error', message: 'فشل في إرسال تحديث السوق' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ' });
    } finally {
      setLoading(false);
    }
  };

  const totalUsers = stats?.vip_distribution ? Object.values(stats.vip_distribution).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6" dir="rtl">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-green-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-emerald-500/10 rounded-full blur-[100px]" />
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
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-green-200 via-emerald-200 to-teal-200 bg-clip-text text-transparent flex items-center gap-2">
              <MessageSquare className="w-8 h-8 text-green-400" />
              التواصل مع المشتركين
            </h1>
            <p className="text-white/40 text-sm mt-1">إرسال الرسائل والإشعارات للمستخدمين</p>
          </div>
        </div>
        <button 
          onClick={fetchStats}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 hover:border-green-500/40 transition-all duration-300"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Communication Stats */}
      <div className="relative grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <Bell className="w-4 h-4" />
            إجمالي الإشعارات
          </div>
          <div className="text-3xl font-bold text-white">{stats?.total_notifications || 0}</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <Mail className="w-4 h-4 text-yellow-400" />
            غير مقروءة
          </div>
          <div className="text-3xl font-bold text-yellow-400">{stats?.unread_notifications || 0}</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            نسبة القراءة
          </div>
          <div className="text-3xl font-bold text-green-400">{(stats?.read_rate || 0).toFixed(1)}%</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <Users className="w-4 h-4 text-blue-400" />
            إجمالي المستخدمين
          </div>
          <div className="text-3xl font-bold text-blue-400">{totalUsers}</div>
        </div>
      </div>

      {/* Broadcast Message */}
      <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Send className="w-5 h-5 text-green-400" />
            إرسال رسالة جماعية
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-white/60 mb-2 text-sm">نوع الرسالة</label>
              <select
                value={messageType}
                onChange={(e) => setMessageType(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-green-500/50 focus:outline-none"
              >
                <option value="announcement" className="bg-gray-900">📣 إعلان</option>
                <option value="update" className="bg-gray-900">🔄 تحديث</option>
                <option value="alert" className="bg-gray-900">⚠️ تنبيه</option>
                <option value="promotion" className="bg-gray-900">🎁 عرض ترويجي</option>
                <option value="maintenance" className="bg-gray-900">🔧 صيانة</option>
              </select>
            </div>
            <div>
              <label className="block text-white/60 mb-2 text-sm">
                الجمهور المستهدف 
                {audienceCount !== null && (
                  <Badge className="mr-2 bg-blue-500/20 text-blue-400">{audienceCount} مستخدم</Badge>
                )}
              </label>
              <select
                value={targetAudience}
                onChange={(e) => setTargetAudience(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-green-500/50 focus:outline-none"
              >
                <option value="all" className="bg-gray-900">الجميع</option>
                <option value="investors" className="bg-gray-900">المستثمرين فقط</option>
                <option value="vip" className="bg-gray-900">أعضاء VIP</option>
                <option value="vip_gold_plus" className="bg-gray-900">ذهبي وأعلى</option>
                <option value="inactive" className="bg-gray-900">غير النشطين</option>
                <option value="new_users" className="bg-gray-900">المستخدمين الجدد</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-white/60 mb-2 text-sm">عنوان الرسالة</label>
            <input
              type="text"
              value={broadcastTitle}
              onChange={(e) => setBroadcastTitle(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-green-500/50 focus:outline-none"
              placeholder="أدخل عنوان الرسالة"
            />
          </div>

          <div>
            <label className="block text-white/60 mb-2 text-sm">محتوى الرسالة</label>
            <textarea
              value={broadcastMessage}
              onChange={(e) => setBroadcastMessage(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white h-32 focus:border-green-500/50 focus:outline-none resize-none"
              placeholder="أدخل محتوى الرسالة..."
            />
          </div>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer text-white/70 hover:text-white transition-colors">
              <input
                type="checkbox"
                checked={sendEmail}
                onChange={(e) => setSendEmail(e.target.checked)}
                className="w-5 h-5 rounded accent-green-500"
              />
              <Mail className="w-4 h-4" />
              إرسال بريد إلكتروني
            </label>
            <label className="flex items-center gap-2 cursor-pointer text-white/70 hover:text-white transition-colors">
              <input
                type="checkbox"
                checked={sendNotification}
                onChange={(e) => setSendNotification(e.target.checked)}
                className="w-5 h-5 rounded accent-green-500"
              />
              <Bell className="w-4 h-4" />
              إشعار داخل المنصة
            </label>
          </div>

          <button
            onClick={sendBroadcast}
            disabled={loading || !broadcastTitle || !broadcastMessage}
            className="w-full px-6 py-3 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold hover:from-green-400 hover:to-emerald-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Send className="w-5 h-5" />
            {loading ? 'جاري الإرسال...' : 'إرسال الرسالة'}
          </button>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            إجراءات سريعة
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button
              onClick={sendMaintenanceNotice}
              disabled={loading}
              className="flex flex-col items-center gap-2 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 hover:bg-yellow-500/20 transition-all disabled:opacity-50"
            >
              <Wrench className="w-6 h-6" />
              <span>إشعار صيانة</span>
            </button>
            <button
              onClick={() => sendMarketUpdate('bullish')}
              disabled={loading}
              className="flex flex-col items-center gap-2 p-4 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 transition-all disabled:opacity-50"
            >
              <TrendingUp className="w-6 h-6" />
              <span>تحديث صعودي</span>
            </button>
            <button
              onClick={() => sendMarketUpdate('bearish')}
              disabled={loading}
              className="flex flex-col items-center gap-2 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-all disabled:opacity-50"
            >
              <TrendingDown className="w-6 h-6" />
              <span>تحديث هبوطي</span>
            </button>
            <button
              onClick={() => sendMarketUpdate('volatile')}
              disabled={loading}
              className="flex flex-col items-center gap-2 p-4 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20 transition-all disabled:opacity-50"
            >
              <AlertTriangle className="w-6 h-6" />
              <span>تقلب عالي</span>
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Recent Broadcasts */}
      {stats?.recent_broadcasts && stats.recent_broadcasts.length > 0 && (
        <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
          <CardHeader>
            <CardTitle className="text-white">آخر الرسائل المرسلة</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {stats.recent_broadcasts.map((broadcast) => (
                <div key={broadcast.id} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/10">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">
                      {broadcast.type === 'announcement' && '📣'}
                      {broadcast.type === 'update' && '🔄'}
                      {broadcast.type === 'maintenance' && '🔧'}
                      {broadcast.type === 'alert' && '⚠️'}
                    </span>
                    <span className="text-white">{broadcast.title}</span>
                  </div>
                  <div className="text-white/40 text-sm">
                    {new Date(broadcast.created_at).toLocaleDateString('ar-SA')}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
