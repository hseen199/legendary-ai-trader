import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../lib/i18n";
import { authAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { 
  User, 
  Lock, 
  Bell, 
  Mail, 
  Phone, 
  Shield, 
  Save,
  Eye,
  EyeOff,
  CheckCircle,
  XCircle,
  History,
  PlayCircle,
  Smartphone,
  QrCode,
  Loader2,
} from "lucide-react";
import { cn } from "../lib/utils";
import toast from "react-hot-toast";

import { SecurityBadges } from "../components/SecurityBadges";
import { useOnboarding } from "../components/OnboardingProvider";

export default function SettingsNew() {
  const { t, language, setLanguage } = useLanguage();
  const { user, refreshUser } = useAuth();
  const queryClient = useQueryClient();
  const { startTour } = useOnboarding();

  // Profile form state
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [phone, setPhone] = useState(user?.phone_number || user?.phone || "");
  const [isSavingProfile, setIsSavingProfile] = useState(false);

  // Password form state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);

  // Notifications state
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [depositNotifications, setDepositNotifications] = useState(true);
  const [withdrawNotifications, setWithdrawNotifications] = useState(true);
  const [tradeNotifications, setTradeNotifications] = useState(true);

  // 2FA state
  const [show2FASetup, setShow2FASetup] = useState(false);
  const [qrCodeUri, setQrCodeUri] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [is2FAEnabled, setIs2FAEnabled] = useState(user?.two_factor_enabled || false);
  const [isLoading2FA, setIsLoading2FA] = useState(false);

  // Update state when user changes
  useEffect(() => {
    if (user) {
      setFullName(user.full_name || "");
      setPhone(user.phone_number || user.phone || "");
      setIs2FAEnabled(user.two_factor_enabled || false);
    }
  }, [user]);

  // Save profile function
  const handleSaveProfile = async () => {
    setIsSavingProfile(true);
    try {
      const response = await fetch('/api/v1/user/profile', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          full_name: fullName || null,
          phone_number: phone || null
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        toast.success(language === 'ar' ? 'تم حفظ التغييرات بنجاح' : 'Changes saved successfully');
        refreshUser();
      } else {
        toast.error(data.detail || (language === 'ar' ? 'فشل في حفظ التغييرات' : 'Failed to save changes'));
      }
    } catch (error) {
      toast.error(language === 'ar' ? 'حدث خطأ أثناء الحفظ' : 'An error occurred while saving');
    }
    setIsSavingProfile(false);
  };

  // Change password mutation
  const changePasswordMutation = useMutation({
    mutationFn: () => authAPI.changePassword(currentPassword, newPassword),
    onSuccess: () => {
      toast.success("تم تغيير كلمة المرور بنجاح");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || "فشل في تغيير كلمة المرور");
    },
  });

  const handleChangePassword = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error("يرجى ملء جميع الحقول");
      return;
    }

    if (newPassword.length < 8) {
      toast.error("كلمة المرور الجديدة يجب أن تكون 8 أحرف على الأقل");
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error("كلمتا المرور غير متطابقتين");
      return;
    }

    changePasswordMutation.mutate();
  };

  // 2FA functions
  const handleEnable2FA = async () => {
    setIsLoading2FA(true);
    try {
      const response = await fetch('/api/v1/security/2fa/generate', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      const data = await response.json();
      if (data.qr_uri) {
        setQrCodeUri(data.qr_uri);
        setSecretKey(data.secret);
        setShow2FASetup(true);
      }
    } catch (error) {
      toast.error('فشل في إنشاء رمز المصادقة الثنائية');
    }
    setIsLoading2FA(false);
  };

  const handleVerify2FA = async () => {
    if (verificationCode.length !== 6) {
      toast.error('يرجى إدخال رمز مكون من 6 أرقام');
      return;
    }
    setIsLoading2FA(true);
    try {
      const response = await fetch('/api/v1/security/2fa/enable', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ code: verificationCode })
      });
      const data = await response.json();
      if (response.ok) {
        setIs2FAEnabled(true);
        setShow2FASetup(false);
        setVerificationCode('');
        toast.success('تم تفعيل المصادقة الثنائية بنجاح');
        refreshUser();
      } else {
        toast.error(data.detail || 'رمز التحقق غير صحيح');
      }
    } catch (error) {
      toast.error('فشل في تفعيل المصادقة الثنائية');
    }
    setIsLoading2FA(false);
  };

  const handleDisable2FA = async () => {
    const code = prompt('أدخل رمز المصادقة الثنائية لتعطيلها:');
    if (!code) return;
    
    setIsLoading2FA(true);
    try {
      const response = await fetch('/api/v1/security/2fa/disable', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ code })
      });
      if (response.ok) {
        setIs2FAEnabled(false);
        toast.success('تم تعطيل المصادقة الثنائية');
        refreshUser();
      } else {
        const data = await response.json();
        toast.error(data.detail || 'رمز التحقق غير صحيح');
      }
    } catch (error) {
      toast.error('فشل في تعطيل المصادقة الثنائية');
    }
    setIsLoading2FA(false);
  };

  const handleSaveNotifications = () => {
    // Save to localStorage for now
    localStorage.setItem("notifications_settings", JSON.stringify({
      email: emailNotifications,
      deposit: depositNotifications,
      withdraw: withdrawNotifications,
      trade: tradeNotifications,
    }));
    toast.success("تم حفظ إعدادات الإشعارات");
  };

  const handleRestartTour = () => {
    startTour();
    toast.success(language === 'ar' ? 'تم بدء الجولة التعريفية' : 'Tour started');
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{t.settings.title}</h1>
        <p className="text-muted-foreground text-sm">{t.settings.subtitle}</p>
      </div>

      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 lg:w-[500px]">
          <TabsTrigger value="profile" className="gap-2">
            <User className="w-4 h-4" />
            <span className="hidden sm:inline">{language === 'ar' ? 'الملف الشخصي' : 'Profile'}</span>
          </TabsTrigger>
          <TabsTrigger value="security" className="gap-2">
            <Lock className="w-4 h-4" />
            <span className="hidden sm:inline">{language === 'ar' ? 'الأمان' : 'Security'}</span>
          </TabsTrigger>
          <TabsTrigger value="notifications" className="gap-2">
            <Bell className="w-4 h-4" />
            <span className="hidden sm:inline">{t.notifications.title}</span>
          </TabsTrigger>


        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <User className="w-5 h-5" />{t.settings.accountInfo}</CardTitle>
              <CardDescription>{language === 'ar' ? 'معلومات حسابك الأساسية' : 'Your basic account information'}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Email (Read Only) */}
              <div>
                <label className="block text-sm font-medium mb-2">{t.settings.email}</label>
                <div className="flex items-center gap-3">
                  <div className="flex-1 px-4 py-3 bg-muted rounded-lg flex items-center gap-2">
                    <Mail className="w-4 h-4 text-muted-foreground" />
                    <span dir="ltr">{user?.email}</span>
                  </div>
                  <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                    <CheckCircle className="w-3 h-3 ml-1" />
                    {language === 'ar' ? 'مُفعّل' : 'Verified'}
                  </Badge>
                </div>
              </div>

              {/* Full Name */}
              <div>
                <label className="block text-sm font-medium mb-2">{t.settings.fullName}</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none placeholder:text-gray-400"
                  placeholder={language === 'ar' ? 'أدخل اسمك الكامل' : 'Enter your full name'}
                />
              </div>

              {/* Phone */}
              <div>
                <label className="block text-sm font-medium mb-2">{t.settings.phone}</label>
                <div className="relative">
                  <Phone className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    className="w-full pr-10 px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none placeholder:text-gray-400"
                    placeholder="+966 5XX XXX XXXX"
                    dir="ltr"
                  />
                </div>
              </div>

              {/* Account Status */}
              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{language === 'ar' ? 'حالة الحساب' : 'Account Status'}</p>
                    <p className="text-sm text-muted-foreground">
                      {language === 'ar' ? 'تاريخ التسجيل:' : 'Registration date:'} {user?.created_at ? new Date(user.created_at).toLocaleDateString(language === 'ar' ? "ar-SA" : "en-US") : "-"}
                    </p>
                  </div>
                  <Badge variant={user ? "default" : "destructive"}>
                    {user ? (language === 'ar' ? "نشط" : "Active") : (language === 'ar' ? "معلق" : "Suspended")}
                  </Badge>
                </div>
              </div>

              <button
                onClick={handleSaveProfile}
                disabled={isSavingProfile}
                className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isSavingProfile ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                {t.settings.saveChanges}
              </button>
            </CardContent>
          </Card>

          {/* Language Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">{language === 'ar' ? 'إعدادات اللغة' : 'Language Settings'}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <button
                  onClick={() => setLanguage("ar")}
                  className={cn(
                    "flex-1 p-4 rounded-lg border-2 transition-colors",
                    language === "ar" 
                      ? "border-primary bg-primary/10" 
                      : "border-border hover:border-muted-foreground"
                  )}
                >
                  <p className="font-medium">العربية</p>
                  <p className="text-sm text-muted-foreground">Arabic</p>
                </button>
                <button
                  onClick={() => setLanguage("en")}
                  className={cn(
                    "flex-1 p-4 rounded-lg border-2 transition-colors",
                    language === "en" 
                      ? "border-primary bg-primary/10" 
                      : "border-border hover:border-muted-foreground"
                  )}
                >
                  <p className="font-medium">English</p>
                  <p className="text-sm text-muted-foreground">الإنجليزية</p>
                </button>
              </div>
            </CardContent>
          </Card>

          {/* Restart Tour */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <PlayCircle className="w-5 h-5" />
                {language === 'ar' ? 'الجولة التعريفية' : 'Platform Tour'}
              </CardTitle>
              <CardDescription>{language === 'ar' ? 'أعد مشاهدة الجولة التعريفية للمنصة' : 'Restart the platform introduction tour'}</CardDescription>
            </CardHeader>
            <CardContent>
              <button
                onClick={handleRestartTour}
                className="w-full py-3 bg-gradient-to-r from-primary/20 to-purple-500/20 text-primary rounded-lg font-medium hover:from-primary/30 hover:to-purple-500/30 transition-colors flex items-center justify-center gap-2 border border-primary/30"
              >
                <PlayCircle className="w-4 h-4" />
                {language === 'ar' ? 'بدء الجولة' : 'Start Tour'}
              </button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6">
          {/* Security Badges */}
          <SecurityBadges />
          
          {/* 2FA Section */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Smartphone className="w-5 h-5" />
                {language === 'ar' ? 'المصادقة الثنائية (2FA)' : 'Two-Factor Authentication (2FA)'}
              </CardTitle>
              <CardDescription>
                {language === 'ar' 
                  ? 'أضف طبقة حماية إضافية لحسابك باستخدام تطبيق المصادقة' 
                  : 'Add an extra layer of security to your account using an authenticator app'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {is2FAEnabled ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                    <CheckCircle className="w-6 h-6 text-green-500" />
                    <div>
                      <p className="font-medium text-green-500">
                        {language === 'ar' ? 'المصادقة الثنائية مفعّلة' : '2FA is enabled'}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {language === 'ar' ? 'حسابك محمي بطبقة أمان إضافية' : 'Your account is protected with an extra layer of security'}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleDisable2FA}
                    disabled={isLoading2FA}
                    className="w-full py-3 bg-red-500/10 text-red-500 rounded-lg font-medium hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2 border border-red-500/20"
                  >
                    {isLoading2FA ? <Loader2 className="w-4 h-4 animate-spin" /> : <XCircle className="w-4 h-4" />}
                    {language === 'ar' ? 'تعطيل المصادقة الثنائية' : 'Disable 2FA'}
                  </button>
                </div>
              ) : show2FASetup ? (
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-4">
                      {language === 'ar' 
                        ? 'امسح رمز QR باستخدام تطبيق المصادقة (Google Authenticator أو Authy)' 
                        : 'Scan the QR code using your authenticator app (Google Authenticator or Authy)'}
                    </p>
                    {qrCodeUri && (
                      <div className="flex justify-center mb-4">
                        <img 
                          src={`https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(qrCodeUri)}`} 
                          alt="2FA QR Code" 
                          className="rounded-lg"
                        />
                      </div>
                    )}
                    <div className="p-3 bg-muted rounded-lg mb-4">
                      <p className="text-xs text-muted-foreground mb-1">
                        {language === 'ar' ? 'أو أدخل هذا الرمز يدوياً:' : 'Or enter this code manually:'}
                      </p>
                      <code className="text-sm font-mono">{secretKey}</code>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      {language === 'ar' ? 'أدخل رمز التحقق' : 'Enter verification code'}
                    </label>
                    <input
                      type="text"
                      value={verificationCode}
                      onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                      className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none text-center text-2xl tracking-widest"
                      placeholder="000000"
                      maxLength={6}
                    />
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setShow2FASetup(false)}
                      className="flex-1 py-3 bg-muted text-muted-foreground rounded-lg font-medium hover:bg-muted/80 transition-colors"
                    >
                      {language === 'ar' ? 'إلغاء' : 'Cancel'}
                    </button>
                    <button
                      onClick={handleVerify2FA}
                      disabled={isLoading2FA || verificationCode.length !== 6}
                      className="flex-1 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {isLoading2FA ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                      {language === 'ar' ? 'تفعيل' : 'Enable'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
                    <Shield className="w-6 h-6 text-yellow-500" />
                    <div>
                      <p className="font-medium text-yellow-500">
                        {language === 'ar' ? 'المصادقة الثنائية غير مفعّلة' : '2FA is not enabled'}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {language === 'ar' ? 'ننصح بتفعيلها لحماية حسابك' : 'We recommend enabling it to protect your account'}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleEnable2FA}
                    disabled={isLoading2FA}
                    className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
                  >
                    {isLoading2FA ? <Loader2 className="w-4 h-4 animate-spin" /> : <QrCode className="w-4 h-4" />}
                    {language === 'ar' ? 'تفعيل المصادقة الثنائية' : 'Enable 2FA'}
                  </button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Change Password */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Lock className="w-5 h-5" />{t.settings.changePassword}</CardTitle>
              <CardDescription>{language === 'ar' ? 'قم بتغيير كلمة المرور الخاصة بك' : 'Change your password'}</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleChangePassword} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">{t.settings.currentPassword}</label>
                  <div className="relative">
                    <input
                      type={showCurrentPassword ? "text" : "password"}
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none"
                      placeholder="••••••••"
                    />
                    <button
                      type="button"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">{t.settings.newPassword}</label>
                  <div className="relative">
                    <input
                      type={showNewPassword ? "text" : "password"}
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none"
                      placeholder="••••••••"
                    />
                    <button
                      type="button"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">{t.settings.confirmPassword}</label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none"
                    placeholder="••••••••"
                  />
                </div>

                <button
                  type="submit"
                  disabled={changePasswordMutation.isPending}
                  className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {changePasswordMutation.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Lock className="w-4 h-4" />
                  )}
                  {t.settings.changePassword}
                </button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Bell className="w-5 h-5" />{t.notifications.title}</CardTitle>
              <CardDescription>{language === 'ar' ? 'تحكم في الإشعارات التي تتلقاها' : 'Control the notifications you receive'}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Email Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Mail className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{language === 'ar' ? 'إشعارات البريد الإلكتروني' : 'Email Notifications'}</p>
                    <p className="text-sm text-muted-foreground">{language === 'ar' ? 'تلقي الإشعارات عبر البريد' : 'Receive notifications via email'}</p>
                  </div>
                </div>
                <button
                  onClick={() => setEmailNotifications(!emailNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    emailNotifications ? "bg-primary" : "bg-muted"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                      emailNotifications ? "right-1" : "left-1"
                    )}
                  />
                </button>
              </div>

              {/* Deposit Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <div>
                    <p className="font-medium">{language === 'ar' ? 'إشعارات الإيداع' : 'Deposit Notifications'}</p>
                    <p className="text-sm text-muted-foreground">{language === 'ar' ? 'عند تأكيد الإيداعات' : 'When deposits are confirmed'}</p>
                  </div>
                </div>
                <button
                  onClick={() => setDepositNotifications(!depositNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    depositNotifications ? "bg-primary" : "bg-muted"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                      depositNotifications ? "right-1" : "left-1"
                    )}
                  />
                </button>
              </div>

              {/* Withdraw Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <XCircle className="w-5 h-5 text-red-500" />
                  <div>
                    <p className="font-medium">{language === 'ar' ? 'إشعارات السحب' : 'Withdrawal Notifications'}</p>
                    <p className="text-sm text-muted-foreground">{language === 'ar' ? 'عند معالجة السحوبات' : 'When withdrawals are processed'}</p>
                  </div>
                </div>
                <button
                  onClick={() => setWithdrawNotifications(!withdrawNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    withdrawNotifications ? "bg-primary" : "bg-muted"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                      withdrawNotifications ? "right-1" : "left-1"
                    )}
                  />
                </button>
              </div>

              {/* Trade Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <History className="w-5 h-5 text-blue-500" />
                  <div>
                    <p className="font-medium">{language === 'ar' ? 'إشعارات التداول' : 'Trade Notifications'}</p>
                    <p className="text-sm text-muted-foreground">{language === 'ar' ? 'عند تنفيذ الصفقات' : 'When trades are executed'}</p>
                  </div>
                </div>
                <button
                  onClick={() => setTradeNotifications(!tradeNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    tradeNotifications ? "bg-primary" : "bg-muted"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                      tradeNotifications ? "right-1" : "left-1"
                    )}
                  />
                </button>
              </div>

              <button
                onClick={handleSaveNotifications}
                className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
              >
                <Save className="w-4 h-4" />
                {language === 'ar' ? 'حفظ الإعدادات' : 'Save Settings'}
              </button>
            </CardContent>
          </Card>
        </TabsContent>



      </Tabs>
    </div>
  );
}
