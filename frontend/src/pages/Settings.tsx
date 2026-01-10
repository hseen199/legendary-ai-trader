import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  User,
  Mail,
  Phone,
  Lock,
  Bell,
  Globe,
  Shield,
  Save,
  Eye,
  EyeOff,
  Check,
  AlertCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';

interface UserProfile {
  full_name: string;
  email: string;
  phone?: string;
  country?: string;
  language: string;
  timezone: string;
}

interface NotificationSettings {
  email_deposits: boolean;
  email_withdrawals: boolean;
  email_trades: boolean;
  email_weekly_report: boolean;
  email_security_alerts: boolean;
  email_marketing: boolean;
}

const Settings: React.FC = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<'profile' | 'password' | 'notifications' | 'preferences'>('profile');
  const [isLoading, setIsLoading] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);

  // Profile state
  const [profile, setProfile] = useState<UserProfile>({
    full_name: user?.full_name || '',
    email: user?.email || '',
    phone: '',
    country: '',
    language: 'ar',
    timezone: 'Asia/Riyadh',
  });

  // Password state
  const [passwords, setPasswords] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  });

  // Notification settings state
  const [notifications, setNotifications] = useState<NotificationSettings>({
    email_deposits: true,
    email_withdrawals: true,
    email_trades: false,
    email_weekly_report: true,
    email_security_alerts: true,
    email_marketing: false,
  });

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      // API call to update profile
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulated API call
      toast.success('تم تحديث الملف الشخصي بنجاح');
    } catch (error) {
      toast.error('فشل في تحديث الملف الشخصي');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (passwords.new_password !== passwords.confirm_password) {
      toast.error('كلمات المرور غير متطابقة');
      return;
    }

    if (passwords.new_password.length < 8) {
      toast.error('كلمة المرور يجب أن تكون 8 أحرف على الأقل');
      return;
    }

    setIsLoading(true);
    try {
      // API call to change password
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulated API call
      toast.success('تم تغيير كلمة المرور بنجاح');
      setPasswords({ current_password: '', new_password: '', confirm_password: '' });
    } catch (error) {
      toast.error('فشل في تغيير كلمة المرور');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNotificationUpdate = async () => {
    setIsLoading(true);
    try {
      // API call to update notifications
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulated API call
      toast.success('تم تحديث إعدادات الإشعارات');
    } catch (error) {
      toast.error('فشل في تحديث الإعدادات');
    } finally {
      setIsLoading(false);
    }
  };

  const tabs = [
    { id: 'profile', label: 'الملف الشخصي', icon: User },
    { id: 'password', label: 'كلمة المرور', icon: Lock },
    { id: 'notifications', label: 'الإشعارات', icon: Bell },
    { id: 'preferences', label: 'التفضيلات', icon: Globe },
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
        الإعدادات
      </h1>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 mb-8 border-b border-gray-200 dark:border-gray-700 pb-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            معلومات الملف الشخصي
          </h2>
          <form onSubmit={handleProfileUpdate} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Full Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  الاسم الكامل
                </label>
                <div className="relative">
                  <User className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={profile.full_name}
                    onChange={(e) => setProfile({ ...profile, full_name: e.target.value })}
                    className="input pr-10"
                    placeholder="أدخل اسمك الكامل"
                  />
                </div>
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  البريد الإلكتروني
                </label>
                <div className="relative">
                  <Mail className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="email"
                    value={profile.email}
                    onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                    className="input pr-10"
                    placeholder="example@email.com"
                    dir="ltr"
                  />
                </div>
              </div>

              {/* Phone */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  رقم الهاتف (اختياري)
                </label>
                <div className="relative">
                  <Phone className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="tel"
                    value={profile.phone}
                    onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                    className="input pr-10"
                    placeholder="+966 5XX XXX XXXX"
                    dir="ltr"
                  />
                </div>
              </div>

              {/* Country */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  الدولة
                </label>
                <select
                  value={profile.country}
                  onChange={(e) => setProfile({ ...profile, country: e.target.value })}
                  className="input"
                >
                  <option value="">اختر الدولة</option>
                  <option value="SA">السعودية</option>
                  <option value="AE">الإمارات</option>
                  <option value="KW">الكويت</option>
                  <option value="QA">قطر</option>
                  <option value="BH">البحرين</option>
                  <option value="OM">عُمان</option>
                  <option value="EG">مصر</option>
                  <option value="JO">الأردن</option>
                  <option value="LB">لبنان</option>
                  <option value="OTHER">أخرى</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={isLoading}
                className="btn-primary flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                {isLoading ? 'جاري الحفظ...' : 'حفظ التغييرات'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Password Tab */}
      {activeTab === 'password' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            تغيير كلمة المرور
          </h2>
          <form onSubmit={handlePasswordChange} className="space-y-6 max-w-md">
            {/* Current Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                كلمة المرور الحالية
              </label>
              <div className="relative">
                <Lock className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showCurrentPassword ? 'text' : 'password'}
                  value={passwords.current_password}
                  onChange={(e) => setPasswords({ ...passwords, current_password: e.target.value })}
                  className="input pr-10 pl-10"
                  placeholder="أدخل كلمة المرور الحالية"
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                كلمة المرور الجديدة
              </label>
              <div className="relative">
                <Lock className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={passwords.new_password}
                  onChange={(e) => setPasswords({ ...passwords, new_password: e.target.value })}
                  className="input pr-10 pl-10"
                  placeholder="أدخل كلمة المرور الجديدة"
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                يجب أن تكون 8 أحرف على الأقل
              </p>
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                تأكيد كلمة المرور الجديدة
              </label>
              <div className="relative">
                <Lock className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="password"
                  value={passwords.confirm_password}
                  onChange={(e) => setPasswords({ ...passwords, confirm_password: e.target.value })}
                  className="input pr-10"
                  placeholder="أعد إدخال كلمة المرور الجديدة"
                />
              </div>
              {passwords.new_password && passwords.confirm_password && (
                <p className={`text-xs mt-1 ${passwords.new_password === passwords.confirm_password ? 'text-green-600' : 'text-red-600'}`}>
                  {passwords.new_password === passwords.confirm_password ? '✓ كلمات المرور متطابقة' : '✗ كلمات المرور غير متطابقة'}
                </p>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading || !passwords.current_password || !passwords.new_password || !passwords.confirm_password}
              className="btn-primary flex items-center gap-2"
            >
              <Shield className="w-4 h-4" />
              {isLoading ? 'جاري التغيير...' : 'تغيير كلمة المرور'}
            </button>
          </form>
        </div>
      )}

      {/* Notifications Tab */}
      {activeTab === 'notifications' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            إعدادات الإشعارات
          </h2>
          <div className="space-y-6">
            {/* Email Notifications */}
            <div>
              <h3 className="text-md font-medium text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Mail className="w-4 h-4" />
                إشعارات البريد الإلكتروني
              </h3>
              <div className="space-y-4">
                <NotificationToggle
                  label="إشعارات الإيداع"
                  description="استلام بريد عند إتمام الإيداع"
                  checked={notifications.email_deposits}
                  onChange={(checked) => setNotifications({ ...notifications, email_deposits: checked })}
                />
                <NotificationToggle
                  label="إشعارات السحب"
                  description="استلام بريد عند تحديث حالة السحب"
                  checked={notifications.email_withdrawals}
                  onChange={(checked) => setNotifications({ ...notifications, email_withdrawals: checked })}
                />
                <NotificationToggle
                  label="إشعارات الصفقات"
                  description="استلام بريد عند تنفيذ صفقة جديدة"
                  checked={notifications.email_trades}
                  onChange={(checked) => setNotifications({ ...notifications, email_trades: checked })}
                />
                <NotificationToggle
                  label="التقرير الأسبوعي"
                  description="استلام ملخص أسبوعي لأداء المحفظة"
                  checked={notifications.email_weekly_report}
                  onChange={(checked) => setNotifications({ ...notifications, email_weekly_report: checked })}
                />
                <NotificationToggle
                  label="تنبيهات الأمان"
                  description="استلام تنبيهات عند تسجيل دخول جديد"
                  checked={notifications.email_security_alerts}
                  onChange={(checked) => setNotifications({ ...notifications, email_security_alerts: checked })}
                />
                <NotificationToggle
                  label="العروض والأخبار"
                  description="استلام أخبار المنصة والعروض الخاصة"
                  checked={notifications.email_marketing}
                  onChange={(checked) => setNotifications({ ...notifications, email_marketing: checked })}
                />
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={handleNotificationUpdate}
                disabled={isLoading}
                className="btn-primary flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                {isLoading ? 'جاري الحفظ...' : 'حفظ الإعدادات'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Preferences Tab */}
      {activeTab === 'preferences' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            التفضيلات
          </h2>
          <div className="space-y-6 max-w-md">
            {/* Language */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                اللغة
              </label>
              <select
                value={profile.language}
                onChange={(e) => setProfile({ ...profile, language: e.target.value })}
                className="input"
              >
                <option value="ar">العربية</option>
                <option value="en">English</option>
              </select>
            </div>

            {/* Timezone */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                المنطقة الزمنية
              </label>
              <select
                value={profile.timezone}
                onChange={(e) => setProfile({ ...profile, timezone: e.target.value })}
                className="input"
              >
                <option value="Asia/Riyadh">الرياض (GMT+3)</option>
                <option value="Asia/Dubai">دبي (GMT+4)</option>
                <option value="Asia/Kuwait">الكويت (GMT+3)</option>
                <option value="Africa/Cairo">القاهرة (GMT+2)</option>
                <option value="Europe/London">لندن (GMT+0)</option>
                <option value="America/New_York">نيويورك (GMT-5)</option>
              </select>
            </div>

            {/* Currency Display */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                عرض العملة
              </label>
              <select className="input">
                <option value="USD">دولار أمريكي ($)</option>
                <option value="SAR">ريال سعودي (﷼)</option>
                <option value="AED">درهم إماراتي (د.إ)</option>
              </select>
            </div>

            <div className="flex justify-end pt-4">
              <button
                onClick={handleProfileUpdate}
                disabled={isLoading}
                className="btn-primary flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                {isLoading ? 'جاري الحفظ...' : 'حفظ التفضيلات'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Notification Toggle Component
const NotificationToggle: React.FC<{
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}> = ({ label, description, checked, onChange }) => {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div>
        <p className="font-medium text-gray-900 dark:text-white">{label}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          checked ? 'bg-primary-600' : 'bg-gray-300 dark:bg-gray-600'
        }`}
      >
        <span
          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
            checked ? 'right-1' : 'left-1'
          }`}
        />
      </button>
    </div>
  );
};

export default Settings;
