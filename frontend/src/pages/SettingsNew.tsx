import { useState } from "react";
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
} from "lucide-react";
import { cn } from "../lib/utils";
import toast from "react-hot-toast";

export default function SettingsNew() {
  const { user, refreshUser } = useAuth();
  const { t, language, setLanguage } = useLanguage();
  const queryClient = useQueryClient();

  // Profile form state
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [phone, setPhone] = useState(user?.phone || "");

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

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{t.settings.title}</h1>
        <p className="text-muted-foreground text-sm">{t.settings.subtitle}</p>
      </div>

      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 lg:w-[400px]">
          <TabsTrigger value="profile" className="gap-2">
            <User className="w-4 h-4" />
            الملف الشخصي
          </TabsTrigger>
          <TabsTrigger value="security" className="gap-2">
            <Lock className="w-4 h-4" />
            الأمان
          </TabsTrigger>
          <TabsTrigger value="notifications" className="gap-2">
            <Bell className="w-4 h-4" />
            الإشعارات
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <User className="w-5 h-5" />
                معلومات الحساب
              </CardTitle>
              <CardDescription>معلومات حسابك الأساسية</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Email (Read Only) */}
              <div>
                <label className="block text-sm font-medium mb-2">البريد الإلكتروني</label>
                <div className="flex items-center gap-3">
                  <div className="flex-1 px-4 py-3 bg-muted rounded-lg flex items-center gap-2">
                    <Mail className="w-4 h-4 text-muted-foreground" />
                    <span dir="ltr">{user?.email}</span>
                  </div>
                  <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                    <CheckCircle className="w-3 h-3 ml-1" />
                    مُفعّل
                  </Badge>
                </div>
              </div>

              {/* Full Name */}
              <div>
                <label className="block text-sm font-medium mb-2">الاسم الكامل</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                  placeholder="أدخل اسمك الكامل"
                />
              </div>

              {/* Phone */}
              <div>
                <label className="block text-sm font-medium mb-2">رقم الهاتف</label>
                <div className="relative">
                  <Phone className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    className="w-full pr-10 px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                    placeholder="+966 5XX XXX XXXX"
                    dir="ltr"
                  />
                </div>
              </div>

              {/* Account Status */}
              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">حالة الحساب</p>
                    <p className="text-sm text-muted-foreground">
                      تاريخ التسجيل: {user?.created_at ? new Date(user.created_at).toLocaleDateString("ar-SA") : "-"}
                    </p>
                  </div>
                  <Badge variant={user?.is_active ? "default" : "destructive"}>
                    {user?.is_active ? "نشط" : "معلق"}
                  </Badge>
                </div>
              </div>

              <button
                className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
              >
                <Save className="w-4 h-4" />
                حفظ التغييرات
              </button>
            </CardContent>
          </Card>

          {/* Language Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">إعدادات اللغة</CardTitle>
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
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Lock className="w-5 h-5" />
                تغيير كلمة المرور
              </CardTitle>
              <CardDescription>قم بتغيير كلمة المرور الخاصة بك</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleChangePassword} className="space-y-4">
                {/* Current Password */}
                <div>
                  <label className="block text-sm font-medium mb-2">كلمة المرور الحالية</label>
                  <div className="relative">
                    <input
                      type={showCurrentPassword ? "text" : "password"}
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                      placeholder="أدخل كلمة المرور الحالية"
                    />
                    <button
                      type="button"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                    >
                      {showCurrentPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {/* New Password */}
                <div>
                  <label className="block text-sm font-medium mb-2">كلمة المرور الجديدة</label>
                  <div className="relative">
                    <input
                      type={showNewPassword ? "text" : "password"}
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                      placeholder="أدخل كلمة المرور الجديدة"
                    />
                    <button
                      type="button"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                    >
                      {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    يجب أن تكون 8 أحرف على الأقل
                  </p>
                </div>

                {/* Confirm Password */}
                <div>
                  <label className="block text-sm font-medium mb-2">تأكيد كلمة المرور</label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                    placeholder="أعد إدخال كلمة المرور الجديدة"
                  />
                </div>

                <button
                  type="submit"
                  disabled={changePasswordMutation.isPending}
                  className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {changePasswordMutation.isPending ? "جاري التغيير..." : "تغيير كلمة المرور"}
                </button>
              </form>
            </CardContent>
          </Card>

          {/* Security Info */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Shield className="w-5 h-5" />
                معلومات الأمان
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div>
                  <p className="font-medium">آخر تسجيل دخول</p>
                  <p className="text-sm text-muted-foreground">
                    {user?.last_login ? new Date(user.last_login).toLocaleString("ar-SA") : "غير متاح"}
                  </p>
                </div>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Bell className="w-5 h-5" />
                إعدادات الإشعارات
              </CardTitle>
              <CardDescription>تحكم في الإشعارات التي تصلك</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Email Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Mail className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">إشعارات البريد الإلكتروني</p>
                    <p className="text-sm text-muted-foreground">استلام الإشعارات عبر الإيميل</p>
                  </div>
                </div>
                <button
                  onClick={() => setEmailNotifications(!emailNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    emailNotifications ? "bg-primary" : "bg-muted-foreground/30"
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all",
                    emailNotifications ? "left-0.5" : "right-0.5"
                  )} />
                </button>
              </div>

              {/* Deposit Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div>
                  <p className="font-medium">إشعارات الإيداع</p>
                  <p className="text-sm text-muted-foreground">إشعار عند تأكيد الإيداعات</p>
                </div>
                <button
                  onClick={() => setDepositNotifications(!depositNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    depositNotifications ? "bg-primary" : "bg-muted-foreground/30"
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all",
                    depositNotifications ? "left-0.5" : "right-0.5"
                  )} />
                </button>
              </div>

              {/* Withdrawal Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div>
                  <p className="font-medium">إشعارات السحب</p>
                  <p className="text-sm text-muted-foreground">إشعار عند معالجة طلبات السحب</p>
                </div>
                <button
                  onClick={() => setWithdrawNotifications(!withdrawNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    withdrawNotifications ? "bg-primary" : "bg-muted-foreground/30"
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all",
                    withdrawNotifications ? "left-0.5" : "right-0.5"
                  )} />
                </button>
              </div>

              {/* Trade Notifications */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div>
                  <p className="font-medium">إشعارات الصفقات</p>
                  <p className="text-sm text-muted-foreground">إشعار عند تنفيذ صفقات جديدة</p>
                </div>
                <button
                  onClick={() => setTradeNotifications(!tradeNotifications)}
                  className={cn(
                    "w-12 h-6 rounded-full transition-colors relative",
                    tradeNotifications ? "bg-primary" : "bg-muted-foreground/30"
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all",
                    tradeNotifications ? "left-0.5" : "right-0.5"
                  )} />
                </button>
              </div>

              <button
                onClick={handleSaveNotifications}
                className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
              >
                <Save className="w-4 h-4" />
                حفظ الإعدادات
              </button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
