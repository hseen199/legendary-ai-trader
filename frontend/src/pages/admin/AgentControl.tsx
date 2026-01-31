import React, { useState, useEffect, useCallback } from 'react';
import { useLanguage } from '@/lib/i18n';

// Types
interface AgentStatus {
  is_running: boolean;
  is_paused: boolean;
  mode: string;
  uptime_seconds?: number;
  current_cycle?: number;
  last_trade_time?: string;
}

interface Position {
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}

interface Portfolio {
  total_value_usd: number;
  available_balance: number;
  positions: Position[];
  positions_count: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}

interface Performance {
  total_pnl: number;
  total_pnl_percent: number;
  win_rate: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
}

interface RiskSettings {
  max_position_size_percent: number;
  max_open_positions: number;
  stop_loss_percent: number;
  take_profit_percent: number;
  min_confidence: number;
  max_daily_loss_percent: number;
}

interface Health {
  status: string;
  binance_connected: boolean;
  database_connected: boolean;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  last_error?: string;
}

interface Decision {
  timestamp: string;
  symbol: string;
  action: string;
  confidence: number;
  reason: string;
}

// API Service
const API_BASE = '/api/v1/admin/agent';

const agentApi = {
  async getStatus(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/status`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getFullStatus(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/full-status`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getPortfolio(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/portfolio`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getPerformance(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/performance`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getHealth(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/health`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getDecisions(limit: number = 20): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/decisions?limit=${limit}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getRiskSettings(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/risk`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async updateRiskSettings(settings: Partial<RiskSettings>): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/risk`, {
      method: 'PUT',
      headers: { 
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(settings)
    });
    return res.json();
  },

  async pauseAgent(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/pause`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async resumeAgent(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/resume`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async stopAgent(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/stop`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async restartAgent(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/restart`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async closePosition(symbol: string): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/close-position`, {
      method: 'POST',
      headers: { 
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ symbol })
    });
    return res.json();
  },

  async closeAllPositions(): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/close-all`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  },

  async getLogs(lines: number = 100): Promise<any> {
    const token = localStorage.getItem('token');
    const res = await fetch(`${API_BASE}/logs?lines=${lines}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
  }
};

// Notification Component
const Notification: React.FC<{ message: string; type: 'success' | 'error' | 'warning'; onClose: () => void }> = 
  ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'success' ? 'bg-green-600' : type === 'error' ? 'bg-red-600' : 'bg-yellow-600';

  return (
    <div className={`fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg ${bgColor} text-white flex items-center gap-3`}>
      <span>{message}</span>
      <button onClick={onClose} className="text-white/80 hover:text-white">&times;</button>
    </div>
  );
};

// Confirmation Modal
const ConfirmModal: React.FC<{
  isOpen: boolean;
  title: string;
  message: string;
  confirmText: string;
  confirmColor?: string;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ isOpen, title, message, confirmText, confirmColor = 'bg-red-600', onConfirm, onCancel }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4">
        <h3 className="text-xl font-bold mb-4">{title}</h3>
        <p className="text-gray-300 mb-6">{message}</p>
        <div className="flex gap-3 justify-end">
          <button 
            onClick={onCancel}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition"
          >
            إلغاء
          </button>
          <button 
            onClick={onConfirm}
            className={`px-4 py-2 ${confirmColor} hover:opacity-90 rounded-lg transition`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Component
const AgentControl: React.FC = () => {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState('status');
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' | 'warning' } | null>(null);
  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    confirmText: string;
    confirmColor?: string;
    onConfirm: () => void;
  } | null>(null);

  // State
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [riskSettings, setRiskSettings] = useState<RiskSettings | null>(null);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      setConnectionError(null);
      
      const [statusRes, portfolioRes, performanceRes, healthRes, riskRes, decisionsRes] = await Promise.all([
        agentApi.getStatus(),
        agentApi.getPortfolio(),
        agentApi.getPerformance(),
        agentApi.getHealth(),
        agentApi.getRiskSettings(),
        agentApi.getDecisions(20)
      ]);

      // Transform status data from agent format
      if (statusRes.success && statusRes.data) {
        const rawStatus = statusRes.data?.data || statusRes.data;
        setStatus({
          is_running: rawStatus.status === 'running' || rawStatus.status === 'healthy' || rawStatus.is_trading === true,
          is_paused: rawStatus.status === 'paused' || rawStatus.is_paused === true,
          mode: rawStatus.mode || 'unknown',
          uptime_seconds: rawStatus.uptime || rawStatus.uptime_seconds || 0,
          current_cycle: rawStatus.current_cycle || 0,
          last_trade_time: rawStatus.last_decision || rawStatus.last_trade_at
        });
      }

      // Transform portfolio data
      if (portfolioRes.success && portfolioRes.data) {
        const rawPortfolio = portfolioRes.data;
        const positions = (rawPortfolio.positions || []).map((pos: any) => ({
          symbol: pos.symbol,
          side: 'buy',
          quantity: pos.quantity || 0,
          entry_price: pos.entry_price || 0,
          current_price: pos.current_price || 0,
          unrealized_pnl: pos.value ? (pos.value - (pos.entry_price * pos.quantity)) : 0,
          unrealized_pnl_percent: pos.entry_price > 0 ? ((pos.current_price - pos.entry_price) / pos.entry_price * 100) : 0
        }));
        
        setPortfolio({
          total_value_usd: rawPortfolio.total_value || 0,
          available_balance: rawPortfolio.available_cash || 0,
          positions: positions,
          positions_count: positions.length,
          unrealized_pnl: positions.reduce((sum: number, p: any) => sum + p.unrealized_pnl, 0),
          unrealized_pnl_percent: 0
        });
      }

      // Transform performance data
      if (performanceRes.success && performanceRes.data) {
        const rawPerf = performanceRes.data?.data || performanceRes.data;
        setPerformance({
          total_pnl: rawPerf.net_profit || 0,
          total_pnl_percent: 0,
          win_rate: rawPerf.win_rate || 0,
          total_trades: rawPerf.total_trades || 0,
          winning_trades: rawPerf.winning_trades || 0,
          losing_trades: rawPerf.losing_trades || 0
        });
      }

      // Transform health data
      if (healthRes.success && healthRes.data) {
        const rawHealth = healthRes.data?.data || healthRes.data;
        setHealth({
          status: rawHealth.overall_health >= 80 ? 'healthy' : rawHealth.overall_health >= 50 ? 'warning' : 'critical',
          binance_connected: rawHealth.binance_connected !== false,
          database_connected: rawHealth.database_connected !== false,
          memory_usage_mb: rawHealth.memory_usage_mb || 0,
          cpu_usage_percent: rawHealth.cpu_usage_percent || 0,
          last_error: rawHealth.last_error || undefined
        });
      }

      // Transform risk settings
      if (riskRes.success && riskRes.data) {
        const rawRisk = riskRes.data;
        setRiskSettings({
          max_position_size_percent: (rawRisk.max_position_size?.value || 0.15) * 100,
          max_open_positions: 5,
          stop_loss_percent: 3,
          take_profit_percent: 5,
          min_confidence: 75,
          max_daily_loss_percent: (rawRisk.max_daily_loss?.value || 0.05) * 100
        });
      }

      // Transform decisions
      if (decisionsRes.success) {
        const rawDecisions = decisionsRes.data?.data || decisionsRes.data || [];
        if (Array.isArray(rawDecisions)) {
          setDecisions(rawDecisions.slice(0, 20).map((d: any) => ({
            timestamp: d.timestamp,
            symbol: d.symbol,
            action: d.action,
            confidence: d.confidence,
            reason: d.reason
          })));
        }
      }

      setLoading(false);
    } catch (error: any) {
      setConnectionError('لا يمكن الاتصال بالوكيل. تأكد من أن سيرفر الوكيل يعمل.');
      setLoading(false);
    }
  }, []);


  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [fetchData]);

  // Actions
  const handlePause = async () => {
    const res = await agentApi.pauseAgent();
    if (res.success) {
      setNotification({ message: 'تم إيقاف الوكيل مؤقتاً', type: 'success' });
      fetchData();
    } else {
      setNotification({ message: res.message || 'فشل إيقاف الوكيل', type: 'error' });
    }
  };

  const handleResume = async () => {
    const res = await agentApi.resumeAgent();
    if (res.success) {
      setNotification({ message: 'تم استئناف الوكيل', type: 'success' });
      fetchData();
    } else {
      setNotification({ message: res.message || 'فشل استئناف الوكيل', type: 'error' });
    }
  };

  const handleStop = () => {
    setConfirmModal({
      isOpen: true,
      title: '⚠️ تأكيد إيقاف الوكيل',
      message: 'هل أنت متأكد من إيقاف الوكيل بالكامل؟ سيتوقف التداول تماماً.',
      confirmText: 'إيقاف الوكيل',
      confirmColor: 'bg-red-600',
      onConfirm: async () => {
        setConfirmModal(null);
        const res = await agentApi.stopAgent();
        if (res.success) {
          setNotification({ message: 'تم إيقاف الوكيل', type: 'success' });
          fetchData();
        } else {
          setNotification({ message: res.message || 'فشل إيقاف الوكيل', type: 'error' });
        }
      }
    });
  };

  const handleRestart = () => {
    setConfirmModal({
      isOpen: true,
      title: '🔄 تأكيد إعادة التشغيل',
      message: 'هل تريد إعادة تشغيل الوكيل؟ قد يستغرق هذا بضع ثوانٍ.',
      confirmText: 'إعادة التشغيل',
      confirmColor: 'bg-blue-600',
      onConfirm: async () => {
        setConfirmModal(null);
        const res = await agentApi.restartAgent();
        if (res.success) {
          setNotification({ message: 'جاري إعادة تشغيل الوكيل...', type: 'success' });
          setTimeout(fetchData, 5000);
        } else {
          setNotification({ message: res.message || 'فشل إعادة التشغيل', type: 'error' });
        }
      }
    });
  };

  const handleClosePosition = (symbol: string) => {
    setConfirmModal({
      isOpen: true,
      title: '📉 إغلاق المركز',
      message: `هل تريد إغلاق مركز ${symbol}؟`,
      confirmText: 'إغلاق المركز',
      confirmColor: 'bg-orange-600',
      onConfirm: async () => {
        setConfirmModal(null);
        const res = await agentApi.closePosition(symbol);
        if (res.success) {
          setNotification({ message: `تم إغلاق مركز ${symbol}`, type: 'success' });
          fetchData();
        } else {
          setNotification({ message: res.message || 'فشل إغلاق المركز', type: 'error' });
        }
      }
    });
  };

  const handleCloseAll = () => {
    setConfirmModal({
      isOpen: true,
      title: '⚠️ إغلاق جميع المراكز',
      message: 'هل أنت متأكد من إغلاق جميع المراكز المفتوحة؟ هذا الإجراء لا يمكن التراجع عنه!',
      confirmText: 'إغلاق الكل',
      confirmColor: 'bg-red-600',
      onConfirm: async () => {
        setConfirmModal(null);
        const res = await agentApi.closeAllPositions();
        if (res.success) {
          setNotification({ message: 'تم إغلاق جميع المراكز', type: 'success' });
          fetchData();
        } else {
          setNotification({ message: res.message || 'فشل إغلاق المراكز', type: 'error' });
        }
      }
    });
  };

  const handleUpdateRisk = async (settings: Partial<RiskSettings>) => {
    const res = await agentApi.updateRiskSettings(settings);
    if (res.success) {
      setNotification({ message: 'تم تحديث إعدادات المخاطر', type: 'success' });
      fetchData();
    } else {
      setNotification({ message: res.message || 'فشل تحديث الإعدادات', type: 'error' });
    }
  };

  // Format uptime
  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'غير متاح';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours} ساعة ${minutes} دقيقة`;
  };

  const tabs = [
    { id: 'status', label: '📊 الحالة', icon: '📊' },
    { id: 'portfolio', label: '💼 المحفظة', icon: '💼' },
    { id: 'performance', label: '📈 الأداء', icon: '📈' },
    { id: 'risk', label: '⚙️ المخاطر', icon: '⚙️' },
    { id: 'decisions', label: '🧠 القرارات', icon: '🧠' },
    { id: 'logs', label: '📝 السجلات', icon: '📝' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <div className="text-white text-xl">جاري الاتصال بالوكيل...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6" dir="rtl">
      {/* Notifications */}
      {notification && (
        <Notification 
          message={notification.message} 
          type={notification.type} 
          onClose={() => setNotification(null)} 
        />
      )}

      {/* Confirm Modal */}
      {confirmModal && (
        <ConfirmModal
          isOpen={confirmModal.isOpen}
          title={confirmModal.title}
          message={confirmModal.message}
          confirmText={confirmModal.confirmText}
          confirmColor={confirmModal.confirmColor}
          onConfirm={confirmModal.onConfirm}
          onCancel={() => setConfirmModal(null)}
        />
      )}

      {/* Connection Error */}
      {connectionError && (
        <div className="bg-red-900/50 border border-red-500 rounded-xl p-4 mb-6">
          <div className="flex items-center gap-3">
            <span className="text-2xl">⚠️</span>
            <div>
              <h3 className="font-bold">خطأ في الاتصال</h3>
              <p className="text-gray-300">{connectionError}</p>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">🤖 التحكم بالوكيل</h1>
          <p className="text-gray-400">إدارة ومراقبة وكيل التداول الذكي</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Status Indicator */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            status?.is_running && !status?.is_paused ? 'bg-green-900/50 border border-green-500' :
            status?.is_paused ? 'bg-yellow-900/50 border border-yellow-500' :
            'bg-red-900/50 border border-red-500'
          }`}>
            <div className={`w-3 h-3 rounded-full ${
              status?.is_running && !status?.is_paused ? 'bg-green-500 animate-pulse' :
              status?.is_paused ? 'bg-yellow-500' :
              'bg-red-500'
            }`} />
            <span>
              {status?.is_running && !status?.is_paused ? 'يعمل' :
               status?.is_paused ? 'متوقف مؤقتاً' : 'متوقف'}
            </span>
          </div>

          {/* Control Buttons */}
          <div className="flex gap-2">
            {status?.is_running && !status?.is_paused && (
              <button onClick={handlePause} className="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded-lg transition">
                ⏸️ إيقاف مؤقت
              </button>
            )}
            {status?.is_paused && (
              <button onClick={handleResume} className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition">
                ▶️ استئناف
              </button>
            )}
            <button onClick={handleRestart} className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition">
              🔄 إعادة تشغيل
            </button>
            <button onClick={handleStop} className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition">
              ⏹️ إيقاف
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition ${
              activeTab === tab.id 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {/* Status Tab */}
        {activeTab === 'status' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Agent Status */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">📊 حالة الوكيل</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">الحالة:</span>
                  <span className={status?.is_running ? 'text-green-400' : 'text-red-400'}>
                    {status?.is_running ? 'يعمل' : 'متوقف'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">الوضع:</span>
                  <span className={status?.mode === 'live' ? 'text-green-400' : 'text-yellow-400'}>
                    {status?.mode === 'live' ? 'حقيقي' : 'تجريبي'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">وقت التشغيل:</span>
                  <span>{formatUptime(status?.uptime_seconds)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">الدورة الحالية:</span>
                  <span>#{status?.current_cycle || 0}</span>
                </div>
              </div>
            </div>

            {/* Health Status */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">🏥 صحة النظام</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Binance:</span>
                  <span className={health?.binance_connected ? 'text-green-400' : 'text-red-400'}>
                    {health?.binance_connected ? '✅ متصل' : '❌ غير متصل'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">قاعدة البيانات:</span>
                  <span className={health?.database_connected ? 'text-green-400' : 'text-red-400'}>
                    {health?.database_connected ? '✅ متصل' : '❌ غير متصل'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">الذاكرة:</span>
                  <span>{health?.memory_usage_mb?.toFixed(0) || 0} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">المعالج:</span>
                  <span>{health?.cpu_usage_percent?.toFixed(1) || 0}%</span>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">📈 إحصائيات سريعة</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">إجمالي المحفظة:</span>
                  <span className="text-green-400">${portfolio?.total_value_usd?.toFixed(2) || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">المراكز المفتوحة:</span>
                  <span>{portfolio?.positions_count || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">الربح غير المحقق:</span>
                  <span className={portfolio?.unrealized_pnl && portfolio.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                    ${portfolio?.unrealized_pnl?.toFixed(2) || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">نسبة الفوز:</span>
                  <span>{performance?.win_rate?.toFixed(1) || 0}%</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Portfolio Tab */}
        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            {/* Portfolio Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="text-gray-400 text-sm">إجمالي القيمة</div>
                <div className="text-2xl font-bold text-green-400">${portfolio?.total_value_usd?.toFixed(2) || 0}</div>
              </div>
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="text-gray-400 text-sm">الرصيد المتاح</div>
                <div className="text-2xl font-bold">${portfolio?.available_balance?.toFixed(2) || 0}</div>
              </div>
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="text-gray-400 text-sm">الربح غير المحقق</div>
                <div className={`text-2xl font-bold ${portfolio?.unrealized_pnl && portfolio.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${portfolio?.unrealized_pnl?.toFixed(2) || 0}
                </div>
              </div>
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="text-gray-400 text-sm">المراكز المفتوحة</div>
                <div className="text-2xl font-bold">{portfolio?.positions_count || 0}</div>
              </div>
            </div>

            {/* Positions Table */}
            <div className="bg-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">المراكز المفتوحة</h3>
                {portfolio?.positions && portfolio.positions.length > 0 && (
                  <button 
                    onClick={handleCloseAll}
                    className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-sm transition"
                  >
                    إغلاق الكل
                  </button>
                )}
              </div>
              
              {portfolio?.positions && portfolio.positions.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-gray-400 border-b border-gray-700">
                        <th className="text-right py-3">الرمز</th>
                        <th className="text-right py-3">الاتجاه</th>
                        <th className="text-right py-3">الكمية</th>
                        <th className="text-right py-3">سعر الدخول</th>
                        <th className="text-right py-3">السعر الحالي</th>
                        <th className="text-right py-3">الربح/الخسارة</th>
                        <th className="text-right py-3">إجراء</th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.positions.map((pos, idx) => (
                        <tr key={idx} className="border-b border-gray-700/50">
                          <td className="py-3 font-medium">{pos.symbol}</td>
                          <td className="py-3">
                            <span className={pos.side === 'LONG' ? 'text-green-400' : 'text-red-400'}>
                              {pos.side === 'LONG' ? '📈 شراء' : '📉 بيع'}
                            </span>
                          </td>
                          <td className="py-3">{pos.quantity}</td>
                          <td className="py-3">${pos.entry_price?.toFixed(4)}</td>
                          <td className="py-3">${pos.current_price?.toFixed(4)}</td>
                          <td className={`py-3 ${pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            ${pos.unrealized_pnl?.toFixed(2)} ({pos.unrealized_pnl_percent?.toFixed(2)}%)
                          </td>
                          <td className="py-3">
                            <button 
                              onClick={() => handleClosePosition(pos.symbol)}
                              className="bg-orange-600 hover:bg-orange-700 px-3 py-1 rounded text-sm transition"
                            >
                              إغلاق
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center text-gray-400 py-8">
                  لا توجد مراكز مفتوحة حالياً
                </div>
              )}
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-gray-800 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">إجمالي الربح/الخسارة</div>
              <div className={`text-3xl font-bold ${performance?.total_pnl && performance.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${performance?.total_pnl?.toFixed(2) || 0}
              </div>
              <div className={`text-sm ${performance?.total_pnl_percent && performance.total_pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {performance?.total_pnl_percent?.toFixed(2) || 0}%
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">نسبة الفوز</div>
              <div className="text-3xl font-bold text-blue-400">
                {performance?.win_rate?.toFixed(1) || 0}%
              </div>
              <div className="text-sm text-gray-400">
                {performance?.winning_trades || 0} رابحة / {performance?.losing_trades || 0} خاسرة
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">إجمالي الصفقات</div>
              <div className="text-3xl font-bold">
                {performance?.total_trades || 0}
              </div>
            </div>
          </div>
        )}

        {/* Risk Settings Tab */}
        {activeTab === 'risk' && riskSettings && (
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-6">⚙️ إعدادات المخاطر</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-400 mb-2">حجم المركز الأقصى (%)</label>
                <input 
                  type="number" 
                  value={riskSettings.max_position_size_percent}
                  onChange={(e) => handleUpdateRisk({ max_position_size_percent: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="1" max="20" step="0.5"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">عدد المراكز المفتوحة الأقصى</label>
                <input 
                  type="number" 
                  value={riskSettings.max_open_positions}
                  onChange={(e) => handleUpdateRisk({ max_open_positions: parseInt(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="1" max="20"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">وقف الخسارة (%)</label>
                <input 
                  type="number" 
                  value={riskSettings.stop_loss_percent}
                  onChange={(e) => handleUpdateRisk({ stop_loss_percent: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="0.5" max="10" step="0.5"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">جني الأرباح (%)</label>
                <input 
                  type="number" 
                  value={riskSettings.take_profit_percent}
                  onChange={(e) => handleUpdateRisk({ take_profit_percent: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="1" max="20" step="0.5"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">الحد الأدنى للثقة (%)</label>
                <input 
                  type="number" 
                  value={riskSettings.min_confidence}
                  onChange={(e) => handleUpdateRisk({ min_confidence: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="50" max="95" step="1"
                />
              </div>
              <div>
                <label className="block text-gray-400 mb-2">الخسارة اليومية القصوى (%)</label>
                <input 
                  type="number" 
                  value={riskSettings.max_daily_loss_percent}
                  onChange={(e) => handleUpdateRisk({ max_daily_loss_percent: parseFloat(e.target.value) })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2"
                  min="5" max="30" step="1"
                />
              </div>
            </div>
          </div>
        )}

        {/* Decisions Tab */}
        {activeTab === 'decisions' && (
          <div className="bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">🧠 آخر القرارات</h3>
            {decisions.length > 0 ? (
              <div className="space-y-3">
                {decisions.map((decision, idx) => (
                  <div key={idx} className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{decision.symbol}</span>
                      <span className={`px-3 py-1 rounded-full text-sm ${
                        decision.action === 'BUY' ? 'bg-green-600' :
                        decision.action === 'SELL' ? 'bg-red-600' :
                        'bg-gray-600'
                      }`}>
                        {decision.action}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-400">
                      <span>الثقة: {decision.confidence?.toFixed(1)}%</span>
                      <span>{new Date(decision.timestamp).toLocaleString('ar-SA')}</span>
                    </div>
                    {decision.reason && (
                      <p className="text-sm text-gray-300 mt-2">{decision.reason}</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                لا توجد قرارات حديثة
              </div>
            )}
          </div>
        )}

        {/* Logs Tab */}
        {activeTab === 'logs' && (
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">📝 سجلات النظام</h3>
              <button 
                onClick={async () => {
                  const res = await agentApi.getLogs(100);
                  if (res.success) setLogs(res.data?.logs || []);
                }}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm transition"
              >
                تحديث
              </button>
            </div>
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
              {logs.length > 0 ? (
                logs.map((log, idx) => (
                  <div key={idx} className="text-gray-300 py-1 border-b border-gray-800">
                    {log}
                  </div>
                ))
              ) : (
                <div className="text-gray-400 text-center py-4">
                  اضغط "تحديث" لجلب السجلات
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentControl;
