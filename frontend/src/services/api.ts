import axios from 'axios';

const API_BASE_URL = '/api/v1';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============ Auth API ============
export const authAPI = {
  register: (data: { email: string; password: string; full_name?: string; phone?: string }) =>
    api.post('/auth/register', data),
  
  login: (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
  },
  
  getMe: () => api.get('/auth/me'),
  
  changePassword: (currentPassword: string, newPassword: string) =>
    api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    }),
};

// ============ Wallet API ============
export const walletAPI = {
  getBalance: () => api.get('/wallet/balance'),
  
  getDepositAddress: (network: string = 'TRC20', coin: string = 'USDT') =>
    api.get(`/wallet/deposit/address?network=${network}&coin=${coin}`),
  
  getDepositHistory: () => api.get('/wallet/deposit/history'),
  
  
  requestWithdrawal: (data: { amount: number; to_address: string; network: string; coin?: string }) =>
    api.post('/wallet/withdraw/request', data),
  
  getWithdrawalHistory: () => api.get('/wallet/withdraw/history'),
  
  getTrustedAddresses: () => api.get('/wallet/trusted-addresses'),
  
  addTrustedAddress: (data: { address: string; network: string; label?: string }) =>
    api.post('/wallet/trusted-addresses', data),
  
  getTransactions: (limit: number = 50) => api.get(`/wallet/transactions?limit=${limit}`),
};

// ============ Dashboard API ============
export const dashboardAPI = {
  getDashboard: () => api.get('/dashboard/'),
  
  getNAV: () => api.get('/dashboard/nav'),
  
  getNAVHistory: (days: number = 30) => api.get(`/dashboard/nav/history?days=${days}`),
  
  getTrades: (limit: number = 50) => api.get(`/dashboard/trades?limit=${limit}`),
};

// ============ Deposits API (NOWPayments) ============
export const depositsAPI = {
  getCurrencies: () => api.get('/deposits/currencies'),
  
  getMinimum: (currency: string) => api.get(`/deposits/minimum/${currency}`),
  
  createDeposit: (data: { amount: number; currency: string }) =>
    api.post('/deposits/create', data),
  
  getDepositStatus: (paymentId: number) => api.get(`/deposits/status/${paymentId}`),
  
  getDepositHistory: () => api.get('/deposits/history'),
  
  getCurrentNAV: () => api.get("/dashboard/nav"),
};

// ============ Admin API ============
export const adminAPI = {
  getStats: () => api.get('/admin/stats'),
  
  getUsers: (skip: number = 0, limit: number = 100) =>
    api.get(`/admin/users?skip=${skip}&limit=${limit}`),
  
  suspendUser: (userId: number) => api.post(`/admin/users/${userId}/suspend`),
  
  activateUser: (userId: number) => api.post(`/admin/users/${userId}/activate`),
  
  // Withdrawals Management
  getPendingWithdrawals: () => api.get('/admin/withdrawals/pending'),
  
  getWithdrawals: (skip: number = 0, limit: number = 500) =>
    api.get(`/admin/withdrawals?skip=${skip}&limit=${limit}`),
  
  reviewWithdrawal: (withdrawalId: number, action: 'approve' | 'reject', reason?: string) =>
    api.post(`/admin/withdrawals/${withdrawalId}/review`, { action, reason }),
  
  approveWithdrawal: (withdrawalId: number) =>
    api.post(`/admin/withdrawals/${withdrawalId}/review`, { action: 'approve' }),
  
  rejectWithdrawal: (withdrawalId: number, reason?: string) =>
    api.post(`/admin/withdrawals/${withdrawalId}/review`, { action: 'reject', reason }),
  
  // Mark withdrawal as completed (after manual transfer)
  completeWithdrawal: (withdrawalId: number, txHash?: string) =>
    api.post(`/admin/withdrawals/${withdrawalId}/complete`, { tx_hash: txHash }),
  
  getTrades: (limit: number = 100) => api.get(`/admin/trades?limit=${limit}`),
  
  createNAVSnapshot: () => api.post('/admin/nav/snapshot'),
  
  enableEmergency: () => api.post('/admin/emergency/enable'),
  
  disableEmergency: () => api.post('/admin/emergency/disable'),
  
  // Impersonation
  impersonateUser: (userId: number) => api.post(`/admin/users/${userId}/impersonate`),
  
  // Balance Management
  adjustUserBalance: (userId: number, data: { amount_usd: number; reason: string; operation: 'add' | 'deduct' }) =>
    api.post(`/admin/users/${userId}/adjust-balance`, data),
  
  // Platform Settings
  getSettings: () => api.get('/admin/settings'),
  
  updateSettings: (data: {
    min_deposit?: number;
    min_withdrawal?: number;
    withdrawal_fee_percent?: number;
    deposit_fee_percent?: number;
    referral_bonus_percent?: number;
    maintenance_mode?: boolean;
    max_daily_withdrawal?: number;
    withdrawal_cooldown_hours?: number;
    auto_approve_withdrawals?: boolean;
    auto_approve_max_amount?: number;
  }) => api.put('/admin/settings', data),
  
  // Security Settings
  getSecuritySettings: () => api.get('/admin/security'),
  
  updateSecuritySettings: (data: {
    two_factor_required?: boolean;
    session_timeout_minutes?: number;
    max_login_attempts?: number;
    lockout_duration_minutes?: number;
    ip_whitelist_enabled?: boolean;
    ip_whitelist?: string[];
    admin_notification_email?: string;
    suspicious_activity_alerts?: boolean;
    withdrawal_confirmation_required?: boolean;
    large_withdrawal_threshold?: number;
  }) => api.put('/admin/security', data),
  
  // System Health
  getSystemHealth: () => api.get('/admin/system-health'),
  
  // Activity Logs
  getActivityLogs: (limit: number = 100) => api.get(`/admin/activity-logs?limit=${limit}`),
  
  // Bulk Notifications
  sendBulkNotification: (data: { title: string; message: string; user_ids?: number[] }) =>
    api.post('/admin/notifications/bulk', data),
};

export default api;
