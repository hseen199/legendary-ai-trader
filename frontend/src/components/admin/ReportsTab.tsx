// Reports Tab Component for Admin Dashboard
import React, { useState, useEffect } from 'react';

interface ReportSummary {
  period: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  total_users: number;
}

interface ReportsTabProps {
  onNotification: (type: 'success' | 'error', message: string) => void;
}

const ReportsTab: React.FC<ReportsTabProps> = ({ onNotification }) => {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<ReportSummary | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');

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
        onNotification('success', 'تم تحميل التقرير بنجاح');
      } else {
        onNotification('error', 'فشل في تحميل التقرير');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ في تحميل التقرير');
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
        onNotification('success', `تم إرسال التقرير إلى ${data.sent_count} مستخدم`);
      } else {
        onNotification('error', 'فشل في إرسال التقارير');
      }
    } catch (error) {
      onNotification('error', 'حدث خطأ في إرسال التقارير');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Report Summary */}
      <div className="bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold">📊 ملخص الأداء</h3>
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-gray-700 rounded-lg px-4 py-2 text-white"
          >
            <option value="daily">يومي</option>
            <option value="weekly">أسبوعي</option>
            <option value="monthly">شهري</option>
            <option value="yearly">سنوي</option>
          </select>
        </div>

        {loading ? (
          <div className="text-center py-8">جاري التحميل...</div>
        ) : summary ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-gray-400 text-sm">إجمالي الصفقات</div>
              <div className="text-2xl font-bold text-white">{summary.total_trades}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-gray-400 text-sm">الصفقات الرابحة</div>
              <div className="text-2xl font-bold text-green-400">{summary.winning_trades}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-gray-400 text-sm">نسبة الفوز</div>
              <div className="text-2xl font-bold text-blue-400">{summary.win_rate.toFixed(1)}%</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-gray-400 text-sm">إجمالي الربح/الخسارة</div>
              <div className={`text-2xl font-bold ${summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${summary.total_pnl.toFixed(2)}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">لا توجد بيانات</div>
        )}
      </div>

      {/* Quick Reports */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">📥 تحميل التقارير</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-bold mb-2">📅 التقرير الأسبوعي</h4>
            <p className="text-gray-400 text-sm mb-4">ملخص أداء الأسبوع الحالي</p>
            <div className="flex gap-2">
              <button
                onClick={() => downloadReport('weekly')}
                disabled={loading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition disabled:opacity-50"
              >
                تحميل PDF
              </button>
              <button
                onClick={() => sendReportToAllUsers('weekly')}
                disabled={loading}
                className="flex-1 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition disabled:opacity-50"
              >
                إرسال للجميع
              </button>
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-bold mb-2">📆 التقرير الشهري</h4>
            <p className="text-gray-400 text-sm mb-4">ملخص أداء الشهر الحالي</p>
            <div className="flex gap-2">
              <button
                onClick={() => downloadReport('monthly')}
                disabled={loading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition disabled:opacity-50"
              >
                تحميل PDF
              </button>
              <button
                onClick={() => sendReportToAllUsers('monthly')}
                disabled={loading}
                className="flex-1 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition disabled:opacity-50"
              >
                إرسال للجميع
              </button>
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-bold mb-2">📊 تقرير مخصص</h4>
            <div className="space-y-2 mb-4">
              <input
                type="date"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
                className="w-full bg-gray-600 rounded px-3 py-1 text-white text-sm"
                placeholder="من تاريخ"
              />
              <input
                type="date"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
                className="w-full bg-gray-600 rounded px-3 py-1 text-white text-sm"
                placeholder="إلى تاريخ"
              />
            </div>
            <button
              onClick={() => downloadReport('custom')}
              disabled={loading || !customStartDate || !customEndDate}
              className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg transition disabled:opacity-50"
            >
              تحميل التقرير المخصص
            </button>
          </div>
        </div>
      </div>

      {/* Scheduled Reports */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">⏰ التقارير المجدولة</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
            <div>
              <h4 className="font-bold">التقرير الأسبوعي التلقائي</h4>
              <p className="text-gray-400 text-sm">يُرسل كل يوم أحد الساعة 9 صباحاً</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
            </label>
          </div>
          <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
            <div>
              <h4 className="font-bold">التقرير الشهري التلقائي</h4>
              <p className="text-gray-400 text-sm">يُرسل أول كل شهر الساعة 9 صباحاً</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsTab;
