import { useState, useEffect } from "react";
import { useLanguage } from "../../lib/i18n";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { 
  FileText, 
  Download, 
  Send, 
  Calendar,
  RefreshCw,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Clock,
} from "lucide-react";
import { Link } from "react-router-dom";

interface ReportSummary {
  period: string;
  start_date: string;
  end_date: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  pnl_percent: number;
  portfolio_value: number;
}

export default function ReportsManagement() {
  const { t } = useLanguage();
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<ReportSummary | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    };
  };

  useEffect(() => {
    fetchReportSummary();
  }, [selectedPeriod]);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const fetchReportSummary = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/reports/summary/${selectedPeriod}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      }
    } catch (error) {
      console.error('Error fetching report summary:', error);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (type: string) => {
    try {
      setLoading(true);
      let url = `/api/v1/reports/${type}?language=ar`;
      
      if (type === 'custom' && customStartDate && customEndDate) {
        url = `/api/v1/reports/custom?start_date=${customStartDate}&end_date=${customEndDate}&language=ar`;
      }

      const response = await fetch(url, {
        headers: getAuthHeaders()
      });

      if (response.ok) {
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `ASINAX_Report_${type}_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(downloadUrl);
        setNotification({ type: 'success', message: 'تم تحميل التقرير بنجاح' });
      } else {
        setNotification({ type: 'error', message: 'فشل في تحميل التقرير' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ في تحميل التقرير' });
    } finally {
      setLoading(false);
    }
  };

  const sendReportToAllUsers = async (type: string) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/reports/send-to-all?report_type=${type}`, {
        method: 'POST',
        headers: getAuthHeaders()
      });

      if (response.ok) {
        const data = await response.json();
        setNotification({ type: 'success', message: `تم إرسال التقرير إلى ${data.sent_count || 'جميع'} المستخدمين` });
      } else {
        setNotification({ type: 'error', message: 'فشل في إرسال التقارير' });
      }
    } catch (error) {
      setNotification({ type: 'error', message: 'حدث خطأ في إرسال التقارير' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6" dir="rtl">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-indigo-500/10 rounded-full blur-[100px]" />
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
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-200 via-indigo-200 to-purple-200 bg-clip-text text-transparent flex items-center gap-2">
              <FileText className="w-8 h-8 text-blue-400" />
              إدارة التقارير
            </h1>
            <p className="text-white/40 text-sm mt-1">إنشاء وإرسال تقارير الأداء</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:border-blue-500/50 focus:outline-none"
          >
            <option value="daily" className="bg-gray-900">يومي</option>
            <option value="weekly" className="bg-gray-900">أسبوعي</option>
            <option value="monthly" className="bg-gray-900">شهري</option>
            <option value="yearly" className="bg-gray-900">سنوي</option>
          </select>
          <button 
            onClick={fetchReportSummary}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:bg-blue-500/20 hover:border-blue-500/40 transition-all duration-300"
          >
            <RefreshCw className="w-4 h-4" />
            تحديث
          </button>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="relative grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <BarChart3 className="w-4 h-4" />
            إجمالي الصفقات
          </div>
          <div className="text-3xl font-bold text-white">{summary?.total_trades || 0}</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            الصفقات الرابحة
          </div>
          <div className="text-3xl font-bold text-green-400">{summary?.winning_trades || 0}</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            <TrendingDown className="w-4 h-4 text-red-400" />
            الصفقات الخاسرة
          </div>
          <div className="text-3xl font-bold text-red-400">{summary?.losing_trades || 0}</div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
            نسبة الفوز
          </div>
          <div className="text-3xl font-bold text-blue-400">{(summary?.win_rate || 0).toFixed(1)}%</div>
        </div>
      </div>

      {/* Download Reports */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Calendar className="w-5 h-5 text-green-400" />
              التقرير الأسبوعي
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-white/60 text-sm">ملخص أداء الأسبوع الحالي مع تفاصيل الصفقات</p>
            <div className="flex gap-2">
              <button
                onClick={() => downloadReport('weekly')}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:bg-blue-500/20 transition-all disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                تحميل
              </button>
              <button
                onClick={() => sendReportToAllUsers('weekly')}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 transition-all disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
                إرسال
              </button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Calendar className="w-5 h-5 text-purple-400" />
              التقرير الشهري
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-white/60 text-sm">ملخص أداء الشهر الحالي مع إحصائيات شاملة</p>
            <div className="flex gap-2">
              <button
                onClick={() => downloadReport('monthly')}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:bg-blue-500/20 transition-all disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                تحميل
              </button>
              <button
                onClick={() => sendReportToAllUsers('monthly')}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 transition-all disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
                إرسال
              </button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-amber-400" />
              تقرير مخصص
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <input
                type="date"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:border-amber-500/50 focus:outline-none"
              />
              <input
                type="date"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:border-amber-500/50 focus:outline-none"
              />
            </div>
            <button
              onClick={() => downloadReport('custom')}
              disabled={loading || !customStartDate || !customEndDate}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 hover:bg-amber-500/20 transition-all disabled:opacity-50"
            >
              <Download className="w-4 h-4" />
              تحميل التقرير المخصص
            </button>
          </CardContent>
        </Card>
      </div>

      {/* Scheduled Reports */}
      <Card className="bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Clock className="w-5 h-5 text-violet-400" />
            التقارير المجدولة
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
            <div>
              <h4 className="font-bold text-white">التقرير الأسبوعي التلقائي</h4>
              <p className="text-white/40 text-sm">يُرسل كل يوم أحد الساعة 9 صباحاً</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-500"></div>
            </label>
          </div>
          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
            <div>
              <h4 className="font-bold text-white">التقرير الشهري التلقائي</h4>
              <p className="text-white/40 text-sm">يُرسل أول كل شهر الساعة 9 صباحاً</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-500"></div>
            </label>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
