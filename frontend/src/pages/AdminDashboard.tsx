import React, { useState, useEffect } from 'react';
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
  name: string;
  balance: number;
  units: number;
  status: 'active' | 'suspended' | 'pending';
  vipTier: string;
  joinedAt: string;
  lastActivity: string;
}

interface Withdrawal {
  id: number;
  userId: number;
  userEmail: string;
  amount: number;
  address: string;
  status: 'pending' | 'approved' | 'rejected';
  createdAt: string;
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

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch all dashboard data
      const [statsRes, usersRes, withdrawalsRes, ticketsRes, logsRes] = await Promise.all([
        fetch('/api/v1/analytics/dashboard'),
        fetch('/api/v1/admin/users'),
        fetch('/api/v1/admin/withdrawals/pending'),
        fetch('/api/v1/support/admin/tickets?status=open'),
        fetch('/api/v1/security/audit-logs?limit=20')
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

  const handleWithdrawalAction = async (id: number, action: 'approve' | 'reject') => {
    try {
      const response = await fetch(`/api/v1/admin/withdrawals/${id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, reason: action === 'reject' ? 'رفض من الأدمن' : undefined })
      });

      if (response.ok) {
        setWithdrawals(withdrawals.filter(w => w.id !== id));
        fetchDashboardData();
      }
    } catch (error) {
      console.error('Error processing withdrawal:', error);
    }
  };

  const handleBotAction = async (action: 'start' | 'stop' | 'pause' | 'resume') => {
    try {
      const response = await fetch(`/api/v1/bot/${action}`, { method: 'POST' });
      if (response.ok) {
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
    { id: 'settings', label: '⚙️ الإعدادات', icon: '⚙️' }
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
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-blue-400">🎛️ لوحة تحكم الأدمن</h1>
          <div className="flex items-center gap-4">
            <span className="text-gray-400">آخر تحديث: {new Date().toLocaleTimeString('ar-SA')}</span>
            <button 
              onClick={fetchDashboardData}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition"
            >
              🔄 تحديث
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
          {activeTab === 'users' && <UsersTab users={users} onRefresh={fetchDashboardData} />}
          {activeTab === 'withdrawals' && <WithdrawalsTab withdrawals={withdrawals} onAction={handleWithdrawalAction} />}
          {activeTab === 'bot' && <BotTab stats={stats} onAction={handleBotAction} />}
          {activeTab === 'support' && <SupportTab tickets={tickets} onRefresh={fetchDashboardData} />}
          {activeTab === 'marketing' && <MarketingTab />}
          {activeTab === 'security' && <SecurityTab auditLogs={auditLogs} />}
          {activeTab === 'settings' && <SettingsTab />}
        </main>
      </div>
    </div>
  );
};

// Overview Tab
const OverviewTab: React.FC<{ stats: DashboardStats | null }> = ({ stats }) => {
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
          value={`${stats.profitPercentage >= 0 ? '+' : ''}${stats.profitPercentage.toFixed(2)}%`} 
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
          value={`$${stats.currentNAV.toFixed(4)}`} 
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

// Users Tab
const UsersTab: React.FC<{ users: User[], onRefresh: () => void }> = ({ users, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || user.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

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
                    <div className="font-medium">{user.name}</div>
                    <div className="text-sm text-gray-400">{user.email}</div>
                  </div>
                </td>
                <td className="px-4 py-3">${user.balance.toLocaleString()}</td>
                <td className="px-4 py-3">{user.units.toFixed(4)}</td>
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
                    <button className="text-blue-400 hover:text-blue-300">👁️</button>
                    <button className="text-yellow-400 hover:text-yellow-300">✏️</button>
                    <button className="text-red-400 hover:text-red-300">🚫</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Withdrawals Tab
const WithdrawalsTab: React.FC<{ 
  withdrawals: Withdrawal[], 
  onAction: (id: number, action: 'approve' | 'reject') => void 
}> = ({ withdrawals, onAction }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">💸 طلبات السحب المعلقة</h2>

      {withdrawals.length === 0 ? (
        <div className="bg-gray-800 rounded-xl p-8 text-center text-gray-400">
          لا توجد طلبات سحب معلقة
        </div>
      ) : (
        <div className="space-y-4">
          {withdrawals.map(withdrawal => (
            <div key={withdrawal.id} className="bg-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{withdrawal.userEmail}</div>
                  <div className="text-2xl font-bold text-yellow-400">${withdrawal.amount.toLocaleString()}</div>
                  <div className="text-sm text-gray-400 mt-2">
                    العنوان: <span className="font-mono">{withdrawal.address}</span>
                  </div>
                  <div className="text-sm text-gray-400">
                    التاريخ: {new Date(withdrawal.createdAt).toLocaleString('ar-SA')}
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => onAction(withdrawal.id, 'approve')}
                    className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-medium transition"
                  >
                    ✅ موافقة
                  </button>
                  <button
                    onClick={() => onAction(withdrawal.id, 'reject')}
                    className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-medium transition"
                  >
                    ❌ رفض
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Bot Tab
const BotTab: React.FC<{ 
  stats: DashboardStats | null, 
  onAction: (action: 'start' | 'stop' | 'pause' | 'resume') => void 
}> = ({ stats, onAction }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">🤖 التحكم بوكيل التداول</h2>

      {/* Bot Status Card */}
      <div className="bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">حالة وكيل التداول</h3>
            <div className={`text-3xl font-bold mt-2 ${
              stats?.botStatus === 'running' ? 'text-green-400' :
              stats?.botStatus === 'paused' ? 'text-yellow-400' :
              'text-red-400'
            }`}>
              {stats?.botStatus === 'running' ? '🟢 يعمل' :
               stats?.botStatus === 'paused' ? '🟡 متوقف مؤقتاً' :
               '🔴 متوقف'}
            </div>
          </div>
          <div className="flex gap-3">
            {stats?.botStatus === 'stopped' && (
              <button
                onClick={() => onAction('start')}
                className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-medium transition"
              >
                ▶️ تشغيل
              </button>
            )}
            {stats?.botStatus === 'running' && (
              <>
                <button
                  onClick={() => onAction('pause')}
                  className="bg-yellow-600 hover:bg-yellow-700 px-6 py-3 rounded-lg font-medium transition"
                >
                  ⏸️ إيقاف مؤقت
                </button>
                <button
                  onClick={() => onAction('stop')}
                  className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-medium transition"
                >
                  ⏹️ إيقاف
                </button>
              </>
            )}
            {stats?.botStatus === 'paused' && (
              <>
                <button
                  onClick={() => onAction('resume')}
                  className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-medium transition"
                >
                  ▶️ استئناف
                </button>
                <button
                  onClick={() => onAction('stop')}
                  className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-medium transition"
                >
                  ⏹️ إيقاف
                </button>
              </>
            )}
          </div>
        </div>

        {/* Bot Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="font-medium mb-3">⚙️ إعدادات التداول</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">وضع التداول:</span>
                <span className="text-green-400">حقيقي</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">الحد الأقصى للمخاطرة:</span>
                <span>2%</span>
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
                  {ticket.userEmail} • {ticket.category} • {new Date(ticket.createdAt).toLocaleString('ar-SA')}
                </div>
              </div>
              <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition">
                فتح
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Marketing Tab
const MarketingTab: React.FC = () => {
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
            <StatCard title={t.referrals.totalReferrals} value="1,234" icon="👥" color="blue" />
            <StatCard title="الإحالات الناجحة" value="567" icon="✅" color="green" />
            <StatCard title="العمولات المدفوعة" value="$12,345" icon="💰" color="yellow" />
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
              <div className="text-2xl font-bold mt-2">45</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-gray-600/30 border border-gray-400 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🥈</div>
              <div className="font-bold">Silver</div>
              <div className="text-2xl font-bold mt-2">23</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-yellow-900/30 border border-yellow-600 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🥇</div>
              <div className="font-bold">Gold</div>
              <div className="text-2xl font-bold mt-2">12</div>
              <div className="text-sm text-gray-400">مستخدم</div>
            </div>
            <div className="bg-purple-900/30 border border-purple-600 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">💎</div>
              <div className="font-bold">Platinum</div>
              <div className="text-2xl font-bold mt-2">5</div>
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
                {new Date(log.createdAt).toLocaleString('ar-SA')}
              </div>
            </div>
          ))}
        </div>
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
                defaultValue="Legendary AI Trader"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">الحد الأدنى للإيداع</label>
              <input
                type="number"
                defaultValue="100"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">رسوم السحب (%)</label>
              <input
                type="number"
                defaultValue="1"
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
              <select className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2">
                <option value="USDC">USDC</option>
                <option value="USDT">USDT</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">الحد الأقصى للمخاطرة (%)</label>
              <input
                type="number"
                defaultValue="2"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">فترة حساب NAV</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2">
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
        <button className="bg-green-600 hover:bg-green-700 px-8 py-3 rounded-lg font-medium transition">
          💾 حفظ الإعدادات
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
