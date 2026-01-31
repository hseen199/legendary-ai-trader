import React, { useState, useEffect } from 'react';
import VIPTab from '@/components/admin/VIPTab';
import ReportsTab from '@/components/admin/ReportsTab';
import CommunicationTab from '@/components/admin/CommunicationTab';
import { useLanguage } from '@/lib/i18n';

// Types
interface DashboardStats {
  totalAssets: number;
  totalUsers: number;
  activeUsers: number;
  todayProfit: number;
  profitPercentage: number;
  botStatus: 'running' | 'stopped' | 'paused';
  pendingWithdrawals: number;
  pendingDeposits: number;
  currentNAV: number;
  totalTrades: number;
}

interface User {
  id: number;
  email: string;
  full_name: string;
  current_value_usd: number;
  units: number;
  status: 'active' | 'suspended' | 'pending';
  vipTier: string;
  joinedAt: string;
  lastActivity: string;
  is_active: boolean;
}

interface Withdrawal {
  id: number;
  userId: number;
  userEmail: string;
  userName?: string;
  amount: number;
  amountUsd: number;
  address: string;
  network: string;
  coin: string;
  status: 'pending_approval' | 'approved' | 'rejected' | 'completed';
  createdAt: string;
  rejectionReason?: string;
}

interface SupportTicket {
  id: number;
  ticketNumber: string;
  userId: number;
  userEmail: string;
  subject: string;
  category: string;
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  createdAt: string;
}

interface AuditLog {
  id: number;
  adminId: number;
  action: string;
  targetType: string;
  targetId: number;
  details: any;
  ipAddress: string;
  createdAt: string;
}

// Modal Component
const Modal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}> = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold">{title}</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">&times;</button>
        </div>
        {children}
      </div>
    </div>
  );
};

// Main Component
const AdminDashboard: React.FC = () => {
  const { t, language } = useLanguage();
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [withdrawals, setWithdrawals] = useState<Withdrawal[]>([]);
  const [tickets, setTickets] = useState<SupportTicket[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Auto-hide notification
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
  };

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    };
  };

  // Refresh session token
  const refreshSession = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("/api/v1/auth/refresh-token", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });
      if (response.ok) {
        const data = await response.json();
        localStorage.setItem("token", data.access_token);
        showNotification("success", "تم تحديث الجلسة بنجاح");
        fetchDashboardData();
      } else {
        showNotification("error", "فشل تحديث الجلسة");
      }
    } catch (error) {
      console.error("Error refreshing session:", error);
      showNotification("error", "حدث خطأ في تحديث الجلسة");
    }
  };

  const fetchDashboardData = async () => {
    try {
      const headers = getAuthHeaders();
      const [statsRes, usersRes, withdrawalsRes, ticketsRes, logsRes] = await Promise.all([
        fetch('/api/v1/analytics/dashboard', { headers }),
        fetch('/api/v1/admin/users', { headers }),
        fetch('/api/v1/admin/withdrawals/pending', { headers }),
        fetch('/api/v1/support/admin/tickets?status=open', { headers }),
        fetch('/api/v1/security/audit-logs?limit=20', { headers })
      ]);

      if (statsRes.ok) setStats(await statsRes.json());
      if (usersRes.ok) setUsers(await usersRes.json());
      if (withdrawalsRes.ok) setWithdrawals(await withdrawalsRes.json());
      if (ticketsRes.ok) {
        const data = await ticketsRes.json();
        setTickets(data.tickets || []);
      }
      if (logsRes.ok) {
        const data = await logsRes.json();
        setAuditLogs(data.logs || []);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleWithdrawalAction = async (id: number, action: 'approve' | 'reject', reason?: string) => {
    try {
      const response = await fetch(`/api/v1/admin/withdrawals/${id}/review`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ 
          action, 
          reason: action === 'reject' ? (reason || 'تم الرفض من قبل الإدارة') : undefined 
        })
      });

      if (response.ok) {
        showNotification('success', action === 'approve' ? 'تمت الموافقة على السحب بنجاح' : 'تم رفض طلب السحب');
        setWithdrawals(withdrawals.filter(w => w.id !== id));
        fetchDashboardData();
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'حدث خطأ');
      }
    } catch (error) {
      console.error('Error processing withdrawal:', error);
      showNotification('error', 'حدث خطأ في معالجة الطلب');
    }
  };

  const handleUserStatusToggle = async (userId: number, currentStatus: string) => {
    try {
      const endpoint = currentStatus === 'active' 
        ? `/api/v1/admin/users/${userId}/suspend`
        : `/api/v1/admin/users/${userId}/activate`;
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: getAuthHeaders()
      });

      if (response.ok) {
        showNotification('success', currentStatus === 'active' ? 'تم تعليق المستخدم' : 'تم تفعيل المستخدم');
        fetchDashboardData();
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'حدث خطأ');
      }
    } catch (error) {
      console.error('Error toggling user status:', error);
      showNotification('error', 'حدث خطأ في تغيير حالة المستخدم');
    }
  };

  const handleAdjustBalance = async (userId: number, amount: number, operation: 'add' | 'deduct', reason: string) => {
    try {
      const response = await fetch(`/api/v1/admin/users/${userId}/adjust-balance`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ amount_usd: amount, operation, reason })
      });

      if (response.ok) {
        const result = await response.json();
        showNotification('success', result.message);
        fetchDashboardData();
        return true;
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'حدث خطأ');
        return false;
      }
    } catch (error) {
      console.error('Error adjusting balance:', error);
      showNotification('error', 'حدث خطأ في تعديل الرصيد');
      return false;
    }
  };

  const handleImpersonateUser = async (userId: number) => {
    try {
      const response = await fetch(`/api/v1/admin/users/${userId}/impersonate`, {
        method: 'POST',
        headers: getAuthHeaders()
      });

      if (response.ok) {
        const result = await response.json();
        // Store original admin token
        const adminToken = localStorage.getItem('token');
        localStorage.setItem('admin_token_backup', adminToken || '');
        // Set user token
        localStorage.setItem('token', result.access_token);
        localStorage.setItem('impersonating', 'true');
        localStorage.setItem('impersonated_user', result.user_email);
        // Redirect to user dashboard
        window.location.href = '/dashboard';
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'لا يمكن الدخول على هذا الحساب');
      }
    } catch (error) {
      console.error('Error impersonating user:', error);
      showNotification('error', 'حدث خطأ');
    }
  };

  const handleBotAction = async (action: 'start' | 'stop' | 'pause' | 'resume') => {
    try {
      const response = await fetch(`/api/v1/bot/${action}`, { 
        method: 'POST',
        headers: getAuthHeaders()
      });
      if (response.ok) {
        showNotification('success', `تم ${action === 'start' ? 'تشغيل' : action === 'stop' ? 'إيقاف' : 'تعديل'} الوكيل`);
        fetchDashboardData();
      }
    } catch (error) {
      console.error('Error controlling bot:', error);
    }
  };

  const tabs = [
    { id: 'overview', label: '📊 نظرة عامة', icon: '📊' },
    { id: 'users', label: '👥 المستخدمين', icon: '👥' },
    { id: 'withdrawals', label: '💸 السحوبات', icon: '💸' },
    { id: 'bot', label: '🤖 وكيل التداول', icon: '🤖' },
    { id: 'support', label: '🎫 الدعم', icon: '🎫' },
    { id: 'marketing', label: '📢 التسويق', icon: '📢' },
    { id: 'security', label: '🔐 الأمان', icon: '🔐' },
    { id: 'settings', label: '⚙️ الإعدادات', icon: '⚙️' },
    { id: 'vip', label: '👑 VIP', icon: '👑' },
    { id: 'reports', label: '📄 التقارير', icon: '📄' },
    { id: 'communication', label: '📨 التواصل', icon: '📨' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">جاري التحميل...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white" dir="rtl">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg ${
          notification.type === 'success' ? 'bg-green-600' : 'bg-red-600'
        }`}>
          {notification.message}
        </div>
      )}

      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-blue-400">🎛️ لوحة تحكم الأدمن</h1>
          <div className="flex items-center gap-4">
            <span className="text-gray-400">آخر تحديث: {new Date().toLocaleTimeString('en-US')}</span>
            <button 
              onClick={fetchDashboardData}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition"
            >
              🔄 تحديث
            </button>
            <button 
              onClick={refreshSession}
              className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition"
            >
              🔑 تحديث الجلسة
            </button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 bg-gray-800 min-h-screen border-l border-gray-700">
          <nav className="p-4 space-y-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full text-right px-4 py-3 rounded-lg transition flex items-center gap-3 ${
                  activeTab === tab.id 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-300 hover:bg-gray-700'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label.replace(tab.icon, '').trim()}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {activeTab === 'overview' && <OverviewTab stats={stats} />}
          {activeTab === 'users' && (
            <UsersTab 
              users={users} 
              onRefresh={fetchDashboardData}
              onToggleStatus={handleUserStatusToggle}
              onAdjustBalance={handleAdjustBalance}
              onImpersonate={handleImpersonateUser}
            />
          )}
          {activeTab === 'withdrawals' && (
            <WithdrawalsTab 
              withdrawals={withdrawals} 
              onAction={handleWithdrawalAction}
              onRefresh={fetchDashboardData}
            />
          )}
          {activeTab === 'bot' && <BotTab stats={stats} onAction={handleBotAction} />}
          {activeTab === 'support' && <SupportTab tickets={tickets} onRefresh={fetchDashboardData} />}
          {activeTab === 'marketing' && <MarketingTab />}
          {activeTab === 'security' && <SecurityTab auditLogs={auditLogs} />}
          {activeTab === 'settings' && <SettingsTab />}
          {activeTab === 'vip' && <VIPTab onNotification={showNotification} />}
          {activeTab === 'reports' && <ReportsTab onNotification={showNotification} />}
          {activeTab === 'communication' && <CommunicationTab onNotification={showNotification} />}
        </main>
      </div>
    </div>
  );
};

// Overview Tab
const OverviewTab: React.FC<{ stats: DashboardStats | null }> = ({ stats }) => {
  const { t } = useLanguage();
  if (!stats) return <div>لا توجد بيانات</div>;

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="إجمالي الأصول" 
          value={`$${stats.totalAssets.toLocaleString()}`} 
          icon="💰" 
          color="blue" 
        />
        <StatCard 
          title="المستخدمين النشطين" 
          value={stats.activeUsers.toString()} 
          subtitle={`من ${stats.totalUsers} مستخدم`}
          icon="👥" 
          color="green" 
        />
        <StatCard 
          title="الربح اليوم" 
          value={`${stats.profitPercentage >= 0 ? '+' : ''}${(stats.profitPercentage || 0).toFixed(2)}%`} 
          subtitle={`$${stats.todayProfit.toLocaleString()}`}
          icon="📈" 
          color={stats.profitPercentage >= 0 ? 'green' : 'red'} 
        />
        <StatCard 
          title="حالة وكيل التداول" 
          value={stats.botStatus === 'running' ? 'يعمل' : stats.botStatus === 'paused' ? 'متوقف مؤقتاً' : 'متوقف'} 
          icon="🤖" 
          color={stats.botStatus === 'running' ? 'green' : 'yellow'} 
        />
      </div>

      {/* Secondary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard 
          title="طلبات السحب المعلقة" 
          value={stats.pendingWithdrawals.toString()} 
          icon="⏳" 
          color="yellow" 
        />
        <StatCard 
          title="قيمة الوحدة (NAV)" 
          value={`$${(stats.currentNAV || 0).toFixed(4)}`} 
          icon="📊" 
          color="purple" 
        />
        <StatCard 
          title={t.trades.totalTrades} 
          value={stats.totalTrades.toLocaleString()} 
          icon="📉" 
          color="blue" 
        />
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">⚡ إجراءات سريعة</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <QuickActionButton label="مراجعة السحوبات" icon="💸" count={stats.pendingWithdrawals} />
          <QuickActionButton label="إدارة المستخدمين" icon="👥" />
          <QuickActionButton label="تقارير الأداء" icon="📊" />
          <QuickActionButton label="إعدادات وكيل التداول" icon="🤖" />
        </div>
      </div>
    </div>
  );
};

// Users Tab with Full Functionality
const UsersTab: React.FC<{ 
  users: User[], 
  onRefresh: () => void,
  onToggleStatus: (userId: number, currentStatus: string) => void,
  onAdjustBalance: (userId: number, amount: number, operation: 'add' | 'deduct', reason: string) => Promise<boolean>,
  onImpersonate: (userId: number) => void
}> = ({ users, onRefresh, onToggleStatus, onAdjustBalance, onImpersonate }) => {
  const { t } = useLanguage();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showBalanceModal, setShowBalanceModal] = useState(false);
  const [showUserModal, setShowUserModal] = useState(false);
  const [balanceAmount, setBalanceAmount] = useState('');
  const [balanceOperation, setBalanceOperation] = useState<'add' | 'deduct'>('add');
  const [balanceReason, setBalanceReason] = useState('');
  const [processing, setProcessing] = useState(false);

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.full_name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || user.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  const handleBalanceSubmit = async () => {
    if (!selectedUser || !balanceAmount || !balanceReason) return;
    setProcessing(true);
    const success = await onAdjustBalance(
      selectedUser.id, 
      parseFloat(balanceAmount), 
      balanceOperation, 
      balanceReason
    );
    setProcessing(false);
    if (success) {
      setShowBalanceModal(false);
      setBalanceAmount('');
      setBalanceReason('');
      onRefresh();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">👥 إدارة المستخدمين</h2>
        <div className="flex gap-4">
          <input
            type="text"
            placeholder="بحث..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
          />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
          >
            <option value="all">جميع الحالات</option>
            <option value="active">{t.referrals.active}</option>
            <option value="suspended">معلق</option>
            <option value="pending">{t.wallet.pending}</option>
          </select>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-700">
            <tr>
              <th className="px-4 py-3 text-right">المستخدم</th>
              <th className="px-4 py-3 text-right">الرصيد</th>
              <th className="px-4 py-3 text-right">الوحدات</th>
              <th className="px-4 py-3 text-right">VIP</th>
              <th className="px-4 py-3 text-right">الحالة</th>
              <th className="px-4 py-3 text-right">الإجراءات</th>
            </tr>
          </thead>
          <tbody>
            {filteredUsers.map(user => (
              <tr key={user.id} className="border-t border-gray-700 hover:bg-gray-750">
                <td className="px-4 py-3">
                  <div>
                    <div className="font-medium">{user.full_name}</div>
                    <div className="text-sm text-gray-400">{user.email}</div>
                  </div>
                </td>
                <td className="px-4 py-3">${user.current_value_usd.toLocaleString()}</td>
                <td className="px-4 py-3">{(user.units || 0).toFixed(4)}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    user.vipTier === 'platinum' ? 'bg-purple-600' :
                    user.vipTier === 'gold' ? 'bg-yellow-600' :
                    user.vipTier === 'silver' ? 'bg-gray-500' :
                    'bg-orange-600'
                  }`}>
                    {user.vipTier}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    user.status === 'active' ? 'bg-green-600' :
                    user.status === 'suspended' ? 'bg-red-600' :
                    'bg-yellow-600'
                  }`}>
                    {user.status === 'active' ? 'نشط' : user.status === 'suspended' ? 'معلق' : 'قيد الانتظار'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-2">
                    {/* View User Details */}
                    <button 
                      onClick={() => { setSelectedUser(user); setShowUserModal(true); }}
                      className="text-blue-400 hover:text-blue-300 p-1" 
                      title="عرض التفاصيل"
                    >
                      👁️
                    </button>
                    
                    {/* Adjust Balance */}
                    <button 
                      onClick={() => { setSelectedUser(user); setShowBalanceModal(true); }}
                      className="text-green-400 hover:text-green-300 p-1" 
                      title="تعديل الرصيد"
                    >
                      💰
                    </button>
                    
                    {/* Impersonate User */}
                    <button 
                      onClick={() => onImpersonate(user.id)}
                      className="text-yellow-400 hover:text-yellow-300 p-1" 
                      title="الدخول كمستخدم"
                    >
                      🔑
                    </button>
                    
                    {/* Toggle Status */}
                    <button 
                      onClick={() => onToggleStatus(user.id, user.status)}
                      className={`p-1 ${user.status === 'active' ? 'text-red-400 hover:text-red-300' : 'text-green-400 hover:text-green-300'}`}
                      title={user.status === 'active' ? 'تعليق المستخدم' : 'تفعيل المستخدم'}
                    >
                      {user.status === 'active' ? '🚫' : '✅'}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Balance Adjustment Modal */}
      <Modal 
        isOpen={showBalanceModal} 
        onClose={() => setShowBalanceModal(false)}
        title={`تعديل رصيد: ${selectedUser?.name}`}
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">نوع العملية</label>
            <select
              value={balanceOperation}
              onChange={(e) => setBalanceOperation(e.target.value as 'add' | 'deduct')}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
            >
              <option value="add">إضافة رصيد</option>
              <option value="deduct">خصم رصيد</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">المبلغ (USD)</label>
            <input
              type="number"
              value={balanceAmount}
              onChange={(e) => setBalanceAmount(e.target.value)}
              placeholder="0.00"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">السبب</label>
            <textarea
              value={balanceReason}
              onChange={(e) => setBalanceReason(e.target.value)}
              placeholder="سبب التعديل..."
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 h-24"
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleBalanceSubmit}
              disabled={processing || !balanceAmount || !balanceReason}
              className={`flex-1 py-2 rounded-lg font-medium transition ${
                balanceOperation === 'add' 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-red-600 hover:bg-red-700'
              } disabled:opacity-50`}
            >
              {processing ? 'جاري التنفيذ...' : (balanceOperation === 'add' ? 'إضافة' : 'خصم')}
            </button>
            <button
              onClick={() => setShowBalanceModal(false)}
              className="flex-1 bg-gray-600 hover:bg-gray-700 py-2 rounded-lg font-medium transition"
            >
              إلغاء
            </button>
          </div>
        </div>
      </Modal>

      {/* User Details Modal */}
      <Modal 
        isOpen={showUserModal} 
        onClose={() => setShowUserModal(false)}
        title={`تفاصيل المستخدم: ${selectedUser?.name}`}
      >
        {selectedUser && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-gray-400 text-sm">البريد الإلكتروني</span>
                <p className="font-medium">{selectedUser.email}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">الحالة</span>
                <p className={`font-medium ${selectedUser.status === 'active' ? 'text-green-400' : 'text-red-400'}`}>
                  {selectedUser.status === 'active' ? 'نشط' : 'معلق'}
                </p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">الرصيد</span>
                <p className="font-medium">${selectedUser.current_value_usd.toLocaleString()}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">الوحدات</span>
                <p className="font-medium">{(selectedUser.units || 0).toFixed(4)}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">مستوى VIP</span>
                <p className="font-medium">{selectedUser.vipTier}</p>
              </div>
              <div>
                <span className="text-gray-400 text-sm">تاريخ التسجيل</span>
                <p className="font-medium">{new Date(selectedUser.joinedAt).toLocaleDateString('en-US')}</p>
              </div>
            </div>
            <div className="flex gap-3 pt-4 border-t border-gray-700">
              <button
                onClick={() => { setShowUserModal(false); setShowBalanceModal(true); }}
                className="flex-1 bg-green-600 hover:bg-green-700 py-2 rounded-lg font-medium transition"
              >
                💰 تعديل الرصيد
              </button>
              <button
                onClick={() => { setShowUserModal(false); onImpersonate(selectedUser.id); }}
                className="flex-1 bg-yellow-600 hover:bg-yellow-700 py-2 rounded-lg font-medium transition"
              >
                🔑 الدخول كمستخدم
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

// Withdrawals Tab with Full Functionality
const WithdrawalsTab: React.FC<{ 
  withdrawals: Withdrawal[], 
  onAction: (id: number, action: 'approve' | 'reject', reason?: string) => void,
  onRefresh: () => void
}> = ({ withdrawals, onAction, onRefresh }) => {
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [selectedWithdrawal, setSelectedWithdrawal] = useState<Withdrawal | null>(null);
  const [rejectReason, setRejectReason] = useState('');

  const handleReject = () => {
    if (selectedWithdrawal && rejectReason) {
      onAction(selectedWithdrawal.id, 'reject', rejectReason);
      setShowRejectModal(false);
      setRejectReason('');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">💸 طلبات السحب المعلقة</h2>
        <button 
          onClick={onRefresh}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition"
        >
          🔄 تحديث
        </button>
      </div>

      {withdrawals.length === 0 ? (
        <div className="bg-gray-800 rounded-xl p-8 text-center text-gray-400">
          ✅ لا توجد طلبات سحب معلقة
        </div>
      ) : (
        <div className="space-y-4">
          {withdrawals.map(withdrawal => (
            <div key={withdrawal.id} className="bg-gray-800 rounded-xl p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-medium">{withdrawal.userEmail}</span>
                    <span className="text-gray-400">#{withdrawal.id}</span>
                  </div>
                  <div className="text-3xl font-bold text-yellow-400 mb-3">
                    ${withdrawal.amountUsd?.toLocaleString() || withdrawal.amount?.toLocaleString()}
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">العنوان: </span>
                      <span className="font-mono text-xs">{withdrawal.address}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">الشبكة: </span>
                      <span>{withdrawal.network || 'BSC'}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">العملة: </span>
                      <span>{withdrawal.coin || 'USDC'}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">التاريخ: </span>
                      <span>{new Date(withdrawal.createdAt).toLocaleString('en-US')}</span>
                    </div>
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <button
                    onClick={() => onAction(withdrawal.id, 'approve')}
                    className="bg-green-600 hover:bg-green-700 px-6 py-2 rounded-lg font-medium transition"
                  >
                    ✅ موافقة
                  </button>
                  <button
                    onClick={() => { setSelectedWithdrawal(withdrawal); setShowRejectModal(true); }}
                    className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded-lg font-medium transition"
                  >
                    ❌ رفض
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Reject Modal */}
      <Modal 
        isOpen={showRejectModal} 
        onClose={() => setShowRejectModal(false)}
        title="رفض طلب السحب"
      >
        <div className="space-y-4">
          <p className="text-gray-400">
            سيتم إرسال إشعار للمستخدم برفض طلب السحب مع السبب
          </p>
          <div>
            <label className="block text-sm text-gray-400 mb-2">سبب الرفض</label>
            <textarea
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="اكتب سبب الرفض..."
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 h-24"
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleReject}
              disabled={!rejectReason}
              className="flex-1 bg-red-600 hover:bg-red-700 py-2 rounded-lg font-medium transition disabled:opacity-50"
            >
              تأكيد الرفض
            </button>
            <button
              onClick={() => setShowRejectModal(false)}
              className="flex-1 bg-gray-600 hover:bg-gray-700 py-2 rounded-lg font-medium transition"
            >
              إلغاء
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

// Bot Tab
const BotTab: React.FC<{ 
  stats: DashboardStats | null, 
  onAction: (action: 'start' | 'stop' | 'pause' | 'resume') => void 
}> = ({ stats, onAction }) => {
  const { t } = useLanguage();
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">🤖 وكيل التداول</h2>

      {/* Bot Status Card */}
      <div className="bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className={`w-4 h-4 rounded-full ${
              stats?.botStatus === 'running' ? 'bg-green-500 animate-pulse' :
              stats?.botStatus === 'paused' ? 'bg-yellow-500' :
              'bg-red-500'
            }`} />
            <span className="text-xl font-bold">
              {stats?.botStatus === 'running' ? 'يعمل' : 
               stats?.botStatus === 'paused' ? 'متوقف مؤقتاً' : 'متوقف'}
            </span>
          </div>
          <div className="flex gap-3">
            {stats?.botStatus === 'running' ? (
              <>
                <button 
                  onClick={() => onAction('pause')}
                  className="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded-lg transition"
                >
                  ⏸️ إيقاف مؤقت
                </button>
                <button 
                  onClick={() => onAction('stop')}
                  className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition"
                >
                  ⏹️ إيقاف
                </button>
              </>
            ) : stats?.botStatus === 'paused' ? (
              <>
                <button 
                  onClick={() => onAction('resume')}
                  className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition"
                >
                  ▶️ استئناف
                </button>
                <button 
                  onClick={() => onAction('stop')}
                  className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition"
                >
                  ⏹️ إيقاف
                </button>
              </>
            ) : (
              <button 
                onClick={() => onAction('start')}
                className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition"
              >
                ▶️ تشغيل
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-medium mb-3">⚙️ الإعدادات الحالية</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">وضع التداول:</span>
                <span className="text-green-400">حقيقي</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">الحد الأقصى للمخاطرة:</span>
                <span>1%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">العملة الأساسية:</span>
                <span>USDC</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-medium mb-3">📊 إحصائيات اليوم</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">عدد الصفقات:</span>
                <span>{stats?.totalTrades || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">الربح/الخسارة:</span>
                <span className={stats?.todayProfit && stats.todayProfit >= 0 ? 'text-green-400' : 'text-red-400'}>
                  ${stats?.todayProfit?.toLocaleString() || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">نسبة النجاح:</span>
                <span>67%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Agents Status */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">🧠 حالة الوكلاء</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {['Trend Agent', 'Momentum Agent', 'Volatility Agent', 'Mean Reversion Agent', 'Breakout Agent', 'Sentiment Agent', 'Arbitrage Agent', 'ML Agent', 'Risk Agent'].map((agent, index) => (
            <div key={index} className="bg-gray-700 rounded-lg p-3 flex items-center justify-between">
              <span>{agent}</span>
              <span className="text-green-400">🟢 نشط</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Support Tab
const SupportTab: React.FC<{ tickets: SupportTicket[], onRefresh: () => void }> = ({ tickets, onRefresh }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">🎫 تذاكر الدعم</h2>
        <div className="flex gap-4">
          <span className="bg-red-600 px-3 py-1 rounded-full text-sm">
            {tickets.filter(t => t.priority === 'urgent').length} عاجل
          </span>
          <span className="bg-yellow-600 px-3 py-1 rounded-full text-sm">
            {tickets.filter(t => t.status === 'open').length} مفتوح
          </span>
        </div>
      </div>

      {tickets.length === 0 ? (
        <div className="bg-gray-800 rounded-xl p-8 text-center text-gray-400">
          ✅ لا توجد تذاكر مفتوحة
        </div>
      ) : (
        <div className="space-y-4">
          {tickets.map(ticket => (
            <div key={ticket.id} className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-3">
                    <span className="text-gray-400">#{ticket.ticketNumber}</span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      ticket.priority === 'urgent' ? 'bg-red-600' :
                      ticket.priority === 'high' ? 'bg-orange-600' :
                      ticket.priority === 'medium' ? 'bg-yellow-600' :
                      'bg-gray-600'
                    }`}>
                      {ticket.priority}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      ticket.status === 'open' ? 'bg-blue-600' :
                      ticket.status === 'in_progress' ? 'bg-yellow-600' :
                      'bg-green-600'
                    }`}>
                      {ticket.status}
                    </span>
                  </div>
                  <h4 className="font-medium mt-2">{ticket.subject}</h4>
                  <div className="text-sm text-gray-400 mt-1">
                    {ticket.userEmail} • {ticket.category} • {new Date(ticket.createdAt).toLocaleString('en-US')}
                  </div>
                </div>
                <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition">
                  فتح
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Marketing Tab
const MarketingTab: React.FC = () => {
  const { t } = useLanguage();
  const [activeSection, setActiveSection] = useState('referrals');

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">📢 التسويق</h2>

      <div className="flex gap-4 border-b border-gray-700 pb-4">
        <button
          onClick={() => setActiveSection('referrals')}
          className={`px-4 py-2 rounded-lg transition ${activeSection === 'referrals' ? 'bg-blue-600' : 'bg-gray-700'}`}
        >
          🔗 الإحالات
        </button>
        <button
          onClick={() => setActiveSection('vip')}
          className={`px-4 py-2 rounded-lg transition ${activeSection === 'vip' ? 'bg-blue-600' : 'bg-gray-700'}`}
        >
          ⭐ VIP
        </button>
        <button
          onClick={() => setActiveSection('coupons')}
          className={`px-4 py-2 rounded-lg transition ${activeSection === 'coupons' ? 'bg-blue-600' : 'bg-gray-700'}`}
        >
          🎟️ الكوبونات
        </button>
      </div>

      {activeSection === 'referrals' && (
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">📊 إحصائيات الإحالات</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard title={t.referrals.totalReferrals} value="0" icon="👥" color="blue" />
            <StatCard title="الإحالات الناجحة" value="0" icon="✅" color="green" />
            <StatCard title="العمولات المدفوعة" value="$0" icon="💰" color="yellow" />
          </div>
        </div>
      )}

      {activeSection === 'vip' && (
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">⭐ مستويات VIP</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-orange-900/30 border border-orange-600 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🥉</div>
              <div className="font-bold">Bronze</div>
              <div className="text-2xl font-bold mt-2">0</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-gray-600/30 border border-gray-400 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🥈</div>
              <div className="font-bold">Silver</div>
              <div className="text-2xl font-bold mt-2">0</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-yellow-900/30 border border-yellow-600 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🥇</div>
              <div className="font-bold">Gold</div>
              <div className="text-2xl font-bold mt-2">0</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-purple-900/30 border border-purple-600 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">💎</div>
              <div className="font-bold">Platinum</div>
              <div className="text-2xl font-bold mt-2">0</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
          </div>
        </div>
      )}

      {activeSection === 'coupons' && (
        <div className="bg-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">🎟️ الكوبونات</h3>
            <button className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition">
              + إنشاء كوبون
            </button>
          </div>
          <div className="text-center text-gray-400 py-8">
            لا توجد كوبونات نشطة
          </div>
        </div>
      )}
    </div>
  );
};

// Security Tab
const SecurityTab: React.FC<{ auditLogs: AuditLog[] }> = ({ auditLogs }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">🔐 الأمان والمراقبة</h2>

      {/* Security Alerts */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">⚠️ تنبيهات الأمان</h3>
        <div className="text-center text-green-400 py-4">
          ✅ لا توجد تنبيهات أمنية
        </div>
      </div>

      {/* Audit Logs */}
      <div className="bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">📋 سجل المراقبة</h3>
        {auditLogs.length === 0 ? (
          <div className="text-center text-gray-400 py-4">
            لا توجد سجلات
          </div>
        ) : (
          <div className="space-y-3">
            {auditLogs.map(log => (
              <div key={log.id} className="bg-gray-700 rounded-lg p-3 flex items-center justify-between">
                <div>
                  <div className="font-medium">{log.action}</div>
                  <div className="text-sm text-gray-400">
                    {log.targetType} #{log.targetId} • {log.ipAddress}
                  </div>
                </div>
                <div className="text-sm text-gray-400">
                  {new Date(log.createdAt).toLocaleString('en-US')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* IP Whitelist */}
      <div className="bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">🌐 قائمة IPs المسموح بها</h3>
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition">
            + إضافة IP
          </button>
        </div>
        <div className="text-center text-gray-400 py-4">
          لم يتم تفعيل قائمة IPs
        </div>
      </div>
    </div>
  );
};

// Settings Tab
const SettingsTab: React.FC = () => {
  const { t } = useLanguage();
  const [saving, setSaving] = useState(false);
  const [settings, setSettings] = useState({
    platformName: 'ASINAX',
    minDeposit: 100,
    withdrawalFee: 1,
    baseCurrency: 'USDC',
    maxRisk: 2,
    navPeriod: 'hourly'
  });

  const handleSave = async () => {
    setSaving(true);
    // TODO: Implement settings save
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSaving(false);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">⚙️ الإعدادات</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Platform Settings */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">🏢 إعدادات المنصة</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">اسم المنصة</label>
              <input
                type="text"
                value={settings.platformName}
                onChange={(e) => setSettings({...settings, platformName: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">الحد الأدنى للإيداع</label>
              <input
                type="number"
                value={settings.minDeposit}
                onChange={(e) => setSettings({...settings, minDeposit: parseInt(e.target.value)})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">رسوم السحب (%)</label>
              <input
                type="number"
                value={settings.withdrawalFee}
                onChange={(e) => setSettings({...settings, withdrawalFee: parseInt(e.target.value)})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
          </div>
        </div>

        {/* Trading Settings */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">📈 إعدادات التداول</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">العملة الأساسية</label>
              <select 
                value={settings.baseCurrency}
                onChange={(e) => setSettings({...settings, baseCurrency: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              >
                <option value="USDC">USDC</option>
                <option value="USDT">USDT</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">الحد الأقصى للمخاطرة (%)</label>
              <input
                type="number"
                value={settings.maxRisk}
                onChange={(e) => setSettings({...settings, maxRisk: parseInt(e.target.value)})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">فترة حساب NAV</label>
              <select 
                value={settings.navPeriod}
                onChange={(e) => setSettings({...settings, navPeriod: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              >
                <option value="hourly">كل ساعة</option>
                <option value="daily">يومياً</option>
              </select>
            </div>
          </div>
        </div>

        {/* Email Settings */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">📧 إعدادات البريد</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">SMTP Server</label>
              <input
                type="text"
                placeholder="smtp.gmail.com"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">{t.settings.email}</label>
              <input
                type="email"
                placeholder="noreply@example.com"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
          </div>
        </div>

        {/* Binance Settings */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">🔗 إعدادات Binance</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">API Key</label>
              <input
                type="password"
                placeholder="••••••••••••••••"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">API Secret</label>
              <input
                type="password"
                placeholder="••••••••••••••••"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition w-full">
              اختبار الاتصال
            </button>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button 
          onClick={handleSave}
          disabled={saving}
          className="bg-green-600 hover:bg-green-700 px-8 py-3 rounded-lg font-medium transition disabled:opacity-50"
        >
          {saving ? '⏳ جاري الحفظ...' : '💾 حفظ الإعدادات'}
        </button>
      </div>
    </div>
  );
};

// Helper Components
const StatCard: React.FC<{
  title: string;
  value: string;
  subtitle?: string;
  icon: string;
  color: string;
}> = ({ title, value, subtitle, icon, color }) => {
  const colorClasses = {
    blue: 'bg-blue-900/30 border-blue-600',
    green: 'bg-green-900/30 border-green-600',
    red: 'bg-red-900/30 border-red-600',
    yellow: 'bg-yellow-900/30 border-yellow-600',
    purple: 'bg-purple-900/30 border-purple-600'
  };

  return (
    <div className={`rounded-xl p-4 border ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}`}>
      <div className="flex items-center justify-between">
        <span className="text-2xl">{icon}</span>
      </div>
      <div className="mt-2">
        <div className="text-sm text-gray-400">{title}</div>
        <div className="text-2xl font-bold">{value}</div>
        {subtitle && <div className="text-sm text-gray-400">{subtitle}</div>}
      </div>
    </div>
  );
};

const QuickActionButton: React.FC<{
  label: string;
  icon: string;
  count?: number;
}> = ({ label, icon, count }) => {
  return (
    <button className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 text-center transition relative">
      <div className="text-2xl mb-2">{icon}</div>
      <div className="text-sm">{label}</div>
      {count !== undefined && count > 0 && (
        <span className="absolute -top-2 -right-2 bg-red-600 text-white text-xs rounded-full w-6 h-6 flex items-center justify-center">
          {count}
        </span>
      )}
    </button>
  );
};

export default AdminDashboard;
