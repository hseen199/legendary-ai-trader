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
  Settings,
  DollarSign,
  Percent,
  Clock,
  Shield,
  Bell,
  Save,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Server,
  Activity,
  Database,
  Wifi,
  Bot,
} from "lucide-react";
import { cn } from "../../lib/utils";

interface PlatformSettings {
  min_deposit: number;
  min_withdrawal: number;
  withdrawal_fee_percent: number;
  deposit_fee_percent: number;
  referral_bonus_percent: number;
  emergency_mode: string;
  maintenance_mode: boolean;
  max_daily_withdrawal: number;
  withdrawal_cooldown_hours: number;
  auto_approve_withdrawals: boolean;
  auto_approve_max_amount: number;
}

interface SystemHealth {
  status: string;
  database: string;
  redis: string;
  binance_api: string;
  nowpayments_api: string;
  bot_status: string;
  last_nav_update: string | null;
  last_trade: string | null;
  uptime_hours: number;
}

export default function SettingsManagement() {
  const queryClient = useQueryClient();
  const [editedSettings, setEditedSettings] = useState<Partial<PlatformSettings>>({});

  // Fetch platform settings
  const { data: settings, isLoading: loadingSettings } = useQuery({
    queryKey: ["/api/v1/admin/settings"],
    queryFn: () => adminAPI.getSettings().then((res) => res.data),
  });

  // Fetch system health
  const { data: health, isLoading: loadingHealth, refetch: refetchHealth } = useQuery({
    queryKey: ["/api/v1/admin/system-health"],
    queryFn: () => adminAPI.getSystemHealth().then((res) => res.data),
    refetchInterval: 30000,
  });

  // Update settings mutation
  const updateMutation = useMutation({
    mutationFn: (data: Partial<PlatformSettings>) => adminAPI.updateSettings(data),
    onSuccess: () => {
      toast.success("تم تحديث الإعدادات بنجاح");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/settings"] });
      setEditedSettings({});
    },
    onError: () => {
      toast.error("فشل في تحديث الإعدادات");
    },
  });

  // Emergency mode mutations
  const enableEmergencyMutation = useMutation({
    mutationFn: () => adminAPI.enableEmergency(),
    onSuccess: () => {
      toast.success("تم تفعيل وضع الطوارئ");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/settings"] });
    },
  });

  const disableEmergencyMutation = useMutation({
    mutationFn: () => adminAPI.disableEmergency(),
    onSuccess: () => {
      toast.success("تم إلغاء وضع الطوارئ");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/settings"] });
    },
  });

  const handleSettingChange = (key: keyof PlatformSettings, value: number | boolean) => {
    setEditedSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    if (Object.keys(editedSettings).length > 0) {
      updateMutation.mutate(editedSettings);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "text-green-400";
      case "degraded":
        return "text-yellow-400";
      case "error":
        return "text-red-400";
      default:
        return "text-white/50";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case "degraded":
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case "error":
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default:
        return <Activity className="w-4 h-4 text-white/50" />;
    }
  };

  if (loadingSettings) {
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
            إعدادات المنصة
          </h1>
          <p className="text-white/40 text-sm mt-1">إدارة إعدادات المنصة والنظام</p>
        </div>
        <div className="flex gap-3">
          <Button
            onClick={handleSave}
            disabled={Object.keys(editedSettings).length === 0 || updateMutation.isPending}
            className="bg-violet-500 hover:bg-violet-600 text-white"
          >
            <Save className="w-4 h-4 ml-2" />
            حفظ التغييرات
          </Button>
        </div>
      </div>

      {/* Emergency Mode Alert */}
      {settings?.emergency_mode === "on" && (
        <div className="relative p-4 rounded-xl bg-red-500/10 border border-red-500/30">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-red-400" />
            <div className="flex-1">
              <p className="text-red-400 font-medium">وضع الطوارئ مفعل</p>
              <p className="text-red-400/70 text-sm">جميع عمليات السحب متوقفة حالياً</p>
            </div>
            <Button
              onClick={() => disableEmergencyMutation.mutate()}
              disabled={disableEmergencyMutation.isPending}
              variant="outline"
              className="border-red-500/30 text-red-400 hover:bg-red-500/10"
            >
              إلغاء وضع الطوارئ
            </Button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Financial Settings */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <DollarSign className="w-5 h-5 text-violet-400" />
              الإعدادات المالية
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-white/70">الحد الأدنى للإيداع ($)</Label>
                <Input
                  type="number"
                  value={currentSettings?.min_deposit || 100}
                  onChange={(e) => handleSettingChange("min_deposit", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-white/70">الحد الأدنى للسحب ($)</Label>
                <Input
                  type="number"
                  value={currentSettings?.min_withdrawal || 50}
                  onChange={(e) => handleSettingChange("min_withdrawal", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-white/70">رسوم الإيداع (%)</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={currentSettings?.deposit_fee_percent || 1}
                  onChange={(e) => handleSettingChange("deposit_fee_percent", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-white/70">رسوم السحب (%)</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={currentSettings?.withdrawal_fee_percent || 0.5}
                  onChange={(e) => handleSettingChange("withdrawal_fee_percent", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-white/70">الحد الأقصى للسحب اليومي ($)</Label>
                <Input
                  type="number"
                  value={currentSettings?.max_daily_withdrawal || 10000}
                  onChange={(e) => handleSettingChange("max_daily_withdrawal", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-white/70">مكافأة الإحالة (%)</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={currentSettings?.referral_bonus_percent || 5}
                  onChange={(e) => handleSettingChange("referral_bonus_percent", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Withdrawal Settings */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Clock className="w-5 h-5 text-violet-400" />
              إعدادات السحب
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label className="text-white/70">فترة الانتظار بين السحوبات (ساعات)</Label>
              <Input
                type="number"
                value={currentSettings?.withdrawal_cooldown_hours || 24}
                onChange={(e) => handleSettingChange("withdrawal_cooldown_hours", parseInt(e.target.value))}
                className="bg-[#1a1a2e] border-violet-500/20 text-white"
              />
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">الموافقة التلقائية على السحب</p>
                <p className="text-white/50 text-sm">موافقة تلقائية على السحوبات الصغيرة</p>
              </div>
              <Switch
                checked={currentSettings?.auto_approve_withdrawals || false}
                onCheckedChange={(checked) => handleSettingChange("auto_approve_withdrawals", checked)}
              />
            </div>

            {currentSettings?.auto_approve_withdrawals && (
              <div className="space-y-2">
                <Label className="text-white/70">الحد الأقصى للموافقة التلقائية ($)</Label>
                <Input
                  type="number"
                  value={currentSettings?.auto_approve_max_amount || 500}
                  onChange={(e) => handleSettingChange("auto_approve_max_amount", parseFloat(e.target.value))}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            )}

            <div className="flex items-center justify-between p-4 rounded-xl bg-[#1a1a2e] border border-violet-500/20">
              <div>
                <p className="text-white font-medium">وضع الصيانة</p>
                <p className="text-white/50 text-sm">إيقاف جميع العمليات مؤقتاً</p>
              </div>
              <Switch
                checked={currentSettings?.maintenance_mode || false}
                onCheckedChange={(checked) => handleSettingChange("maintenance_mode", checked)}
              />
            </div>
          </CardContent>
        </Card>

        {/* System Health */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-violet-500/15">
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-white">
              <div className="flex items-center gap-2">
                <Server className="w-5 h-5 text-violet-400" />
                صحة النظام
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => refetchHealth()}
                className="text-violet-400 hover:text-violet-300"
              >
                <RefreshCw className={cn("w-4 h-4", loadingHealth && "animate-spin")} />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {loadingHealth ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-12 bg-white/5" />
                ))}
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a2e]">
                  <div className="flex items-center gap-3">
                    <Database className="w-5 h-5 text-violet-400" />
                    <span className="text-white">قاعدة البيانات</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(health?.database)}
                    <span className={getStatusColor(health?.database)}>{health?.database}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a2e]">
                  <div className="flex items-center gap-3">
                    <Wifi className="w-5 h-5 text-violet-400" />
                    <span className="text-white">Binance API</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(health?.binance_api)}
                    <span className={getStatusColor(health?.binance_api)}>{health?.binance_api}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a2e]">
                  <div className="flex items-center gap-3">
                    <DollarSign className="w-5 h-5 text-violet-400" />
                    <span className="text-white">NOWPayments API</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(health?.nowpayments_api)}
                    <span className={getStatusColor(health?.nowpayments_api)}>{health?.nowpayments_api}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a2e]">
                  <div className="flex items-center gap-3">
                    <Bot className="w-5 h-5 text-violet-400" />
                    <span className="text-white">وكيل التداول</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(health?.bot_status === "active" ? "healthy" : "degraded")}
                    <span className={health?.bot_status === "active" ? "text-green-400" : "text-yellow-400"}>
                      {health?.bot_status || "غير معروف"}
                    </span>
                  </div>
                </div>

                {health?.last_nav_update && (
                  <div className="p-3 rounded-lg bg-[#1a1a2e] text-white/50 text-sm">
                    آخر تحديث NAV: {new Date(health.last_nav_update).toLocaleString("ar-SA")}
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Emergency Controls */}
        <Card className="relative bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border-red-500/15">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              التحكم في الطوارئ
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
              <p className="text-red-400 font-medium mb-2">تحذير</p>
              <p className="text-white/70 text-sm mb-4">
                تفعيل وضع الطوارئ سيوقف جميع عمليات السحب فوراً. استخدم هذا الخيار فقط في حالات الطوارئ.
              </p>
              {settings?.emergency_mode === "on" ? (
                <Button
                  onClick={() => disableEmergencyMutation.mutate()}
                  disabled={disableEmergencyMutation.isPending}
                  className="w-full bg-green-500 hover:bg-green-600 text-white"
                >
                  <CheckCircle className="w-4 h-4 ml-2" />
                  إلغاء وضع الطوارئ
                </Button>
              ) : (
                <Button
                  onClick={() => enableEmergencyMutation.mutate()}
                  disabled={enableEmergencyMutation.isPending}
                  className="w-full bg-red-500 hover:bg-red-600 text-white"
                >
                  <AlertTriangle className="w-4 h-4 ml-2" />
                  تفعيل وضع الطوارئ
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
