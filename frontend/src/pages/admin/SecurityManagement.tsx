import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { adminAPI } from "../../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Switch } from "../../components/ui/switch";
import { Skeleton } from "../../components/ui/skeleton";
import toast from "react-hot-toast";
import {
  Shield,
  Lock,
  Clock,
  Mail,
  Globe,
  AlertTriangle,
  Save,
  Plus,
  X,
  Key,
  UserX,
  Eye,
  Bell,
} from "lucide-react";

interface SecuritySettings {
  two_factor_required: boolean;
  session_timeout_minutes: number;
  max_login_attempts: number;
  lockout_duration_minutes: number;
  ip_whitelist_enabled: boolean;
  ip_whitelist: string[];
  admin_notification_email: string | null;
  suspicious_activity_alerts: boolean;
  withdrawal_confirmation_required: boolean;
  large_withdrawal_threshold: number;
}

export default function SecurityManagement() {
  const queryClient = useQueryClient();
  const [editedSettings, setEditedSettings] = useState<Partial<SecuritySettings>>({});
  const [newIP, setNewIP] = useState("");

  // Fetch security settings
  const { data: settings, isLoading } = useQuery({
    queryKey: ["/api/v1/admin/security"],
    queryFn: () => adminAPI.getSecuritySettings().then((res) => res.data),
  });

  // Update settings mutation
  const updateMutation = useMutation({
    mutationFn: (data: Partial<SecuritySettings>) => adminAPI.updateSecuritySettings(data),
    onSuccess: () => {
      toast.success("تم تحديث إعدادات الأمان بنجاح");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/security"] });
      setEditedSettings({});
    },
    onError: () => {
      toast.error("فشل في تحديث إعدادات الأمان");
    },
  });

  const handleSettingChange = (key: keyof SecuritySettings, value: any) => {
    setEditedSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    if (Object.keys(editedSettings).length > 0) {
      updateMutation.mutate(editedSettings);
    }
  };

  const addIPToWhitelist = () => {
    if (!newIP.trim()) return;
    
    // Simple IP validation
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
    if (!ipRegex.test(newIP.trim())) {
      toast.error("عنوان IP غير صالح");
      return;
    }

    const currentList = editedSettings.ip_whitelist || settings?.ip_whitelist || [];
    if (currentList.includes(newIP.trim())) {
      toast.error("عنوان IP موجود بالفعل");
      return;
    }

    handleSettingChange("ip_whitelist", [...currentList, newIP.trim()]);
    setNewIP("");
  };

  const removeIPFromWhitelist = (ip: string) => {
    const currentList = editedSettings.ip_whitelist || settings?.ip_whitelist || [];
    handleSettingChange("ip_whitelist", currentList.filter((i: string) => i !== ip));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6">
        <Skeleton className="h-10 w-64 bg-white/5" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Skeleton className="h-96 bg-white/5" />
          <Skeleton className="h-96 bg-white/5" />
        </div>
      </div>
    );
  }

  const currentSettings = { ...settings, ...editedSettings };

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-violet-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-purple-500/10 rounded-full blur-[100px]" />
      </div>

      {/* Header */}
      <div className="relative flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white via-violet-200 to-purple-200 bg-clip-text text-transparent">
            إعدادات الأمان
          </h1>
          <p className="text-white/40 text-sm mt-1">إدارة أمان المنصة والحماية</p>
        </div>
        <Button
          onClick={handleSave}
          disabled={Object.keys(editedSettings).length === 0 || updateMutation.isPending}
          className="bg-violet-500 hover:bg-violet-600 text-white"
        >
          <Save className="w-4 h-4 ml-2" />
          حفظ التغييرات
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Authentication Settings */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Key className="w-5 h-5 text-violet-400" />
              إعدادات المصادقة
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">المصادقة الثنائية إلزامية</p>
                <p className="text-white/50 text-sm">إجبار جميع المستخدمين على تفعيل 2FA</p>
              </div>
              <Switch
                checked={currentSettings?.two_factor_required || false}
                onCheckedChange={(checked) => handleSettingChange("two_factor_required", checked)}
              />
            </div>

            <div className="space-y-2">
              <Label className="text-white/70">مهلة انتهاء الجلسة (دقائق)</Label>
              <Input
                type="number"
                value={currentSettings?.session_timeout_minutes || 60}
                onChange={(e) => handleSettingChange("session_timeout_minutes", parseInt(e.target.value))}
                className="bg-[#1a1a2e] border-violet-500/20 text-white"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-white/70">محاولات تسجيل الدخول</Label>
                <Input
                  type="number"
                  value={currentSettings?.max_login_attempts || 5}
                  onChange={(e) => handleSettingChange("max_login_attempts", parseInt(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-white/70">مدة الحظر (دقائق)</Label>
                <Input
                  type="number"
                  value={currentSettings?.lockout_duration_minutes || 30}
                  onChange={(e) => handleSettingChange("lockout_duration_minutes", parseInt(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* IP Whitelist */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Globe className="w-5 h-5 text-violet-400" />
              قائمة IP البيضاء
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">تفعيل قائمة IP البيضاء</p>
                <p className="text-white/50 text-sm">السماح فقط لعناوين IP محددة</p>
              </div>
              <Switch
                checked={currentSettings?.ip_whitelist_enabled || false}
                onCheckedChange={(checked) => handleSettingChange("ip_whitelist_enabled", checked)}
              />
            </div>

            {currentSettings?.ip_whitelist_enabled && (
              <>
                <div className="flex gap-2">
                  <Input
                    placeholder="أدخل عنوان IP (مثال: 192.168.1.1)"
                    value={newIP}
                    onChange={(e) => setNewIP(e.target.value)}
                    className="bg-[#1a1a2e] border-violet-500/20 text-white"
                  />
                  <Button onClick={addIPToWhitelist} className="bg-violet-500 hover:bg-violet-600">
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {(currentSettings?.ip_whitelist || []).map((ip: string, index: number) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a2e] border border-violet-500/10"
                    >
                      <span className="text-white font-mono">{ip}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeIPFromWhitelist(ip)}
                        className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                  {(currentSettings?.ip_whitelist || []).length === 0 && (
                    <p className="text-white/50 text-center py-4">لا توجد عناوين IP في القائمة</p>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Withdrawal Security */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Lock className="w-5 h-5 text-violet-400" />
              أمان السحب
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">تأكيد السحب بالبريد</p>
                <p className="text-white/50 text-sm">إرسال رابط تأكيد للبريد الإلكتروني</p>
              </div>
              <Switch
                checked={currentSettings?.withdrawal_confirmation_required || true}
                onCheckedChange={(checked) => handleSettingChange("withdrawal_confirmation_required", checked)}
              />
            </div>

            <div className="space-y-2">
              <Label className="text-white/70">حد السحب الكبير ($)</Label>
              <Input
                type="number"
                value={currentSettings?.large_withdrawal_threshold || 5000}
                onChange={(e) => handleSettingChange("large_withdrawal_threshold", parseFloat(e.target.value))}
                className="bg-[#1a1a2e] border-violet-500/20 text-white"
              />
              <p className="text-white/40 text-xs">
                السحوبات التي تتجاوز هذا المبلغ تتطلب مراجعة إضافية
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Notifications & Alerts */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Bell className="w-5 h-5 text-violet-400" />
              الإشعارات والتنبيهات
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label className="text-white/70">بريد إشعارات الأدمن</Label>
              <Input
                type="email"
                placeholder="admin@example.com"
                value={currentSettings?.admin_notification_email || ""}
                onChange={(e) => handleSettingChange("admin_notification_email", e.target.value)}
                className="bg-[#1a1a2e] border-violet-500/20 text-white"
              />
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">تنبيهات النشاط المشبوه</p>
                <p className="text-white/50 text-sm">إرسال تنبيهات عند اكتشاف نشاط غير عادي</p>
              </div>
              <Switch
                checked={currentSettings?.suspicious_activity_alerts || true}
                onCheckedChange={(checked) => handleSettingChange("suspicious_activity_alerts", checked)}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Security Tips */}
      <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-yellow-500/15">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-yellow-400">
            <AlertTriangle className="w-5 h-5" />
            نصائح أمنية
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-white/70">
            <li className="flex items-start gap-2">
              <Shield className="w-4 h-4 mt-1 text-violet-400" />
              <span>قم بتفعيل المصادقة الثنائية لجميع حسابات الأدمن</span>
            </li>
            <li className="flex items-start gap-2">
              <Shield className="w-4 h-4 mt-1 text-violet-400" />
              <span>راجع قائمة IP البيضاء بانتظام وأزل العناوين غير المستخدمة</span>
            </li>
            <li className="flex items-start gap-2">
              <Shield className="w-4 h-4 mt-1 text-violet-400" />
              <span>حدد حد السحب الكبير بناءً على حجم المعاملات المعتاد</span>
            </li>
            <li className="flex items-start gap-2">
              <Shield className="w-4 h-4 mt-1 text-violet-400" />
              <span>فعّل تنبيهات النشاط المشبوه لمراقبة الحسابات</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
