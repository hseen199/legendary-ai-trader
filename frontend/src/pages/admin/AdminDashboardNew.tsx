import { useQuery } from "@tanstack/react-query";
import { useLanguage } from "../../lib/i18n";
import { adminAPI } from "../../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { Skeleton } from "../../components/ui/skeleton";
import { 
  Users, 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  ArrowUpCircle, 
  ArrowDownCircle, 
  Activity,
  Bot,
  BarChart3,
  Clock,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  Percent,
  Shield,
  Settings,
  Database,
} from "lucide-react";
import { cn } from "../../lib/utils";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { format } from "date-fns";
import { Link } from "wouter";

export default function AdminDashboardNew() {
  const { t } = useLanguage();

  // Fetch admin stats
  const { data: stats, isLoading: loadingStats, refetch } = useQuery({
    queryKey: ["/api/v1/admin/stats"],
    queryFn: () => adminAPI.getStats().then(res => res.data),
    refetchInterval: 60000,
  });

  // Fetch pending withdrawals
  const { data: pendingWithdrawals = [], isLoading: loadingWithdrawals } = useQuery({
    queryKey: ["/api/v1/admin/withdrawals/pending"],
    queryFn: () => adminAPI.getPendingWithdrawals().then(res => res.data),
  });

  // Fetch recent trades
  const { data: recentTrades = [], isLoading: loadingTrades } = useQuery({
    queryKey: ["/api/v1/admin/trades"],
    queryFn: () => adminAPI.getTrades(10).then(res => res.data),
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

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
            لوحة تحكم الأدمن
          </h1>
          <p className="text-white/40 text-sm mt-1">نظرة عامة على المنصة</p>
        </div>
        <button 
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20 hover:border-violet-500/40 transition-all duration-300"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Quick Actions */}
      <div className="relative grid grid-cols-2 md:grid-cols-4 gap-3">
        <Link href="/admin/users" className="group">
          <div className="p-4 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.15)]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-blue-500/15 group-hover:bg-blue-500/25 transition-colors">
                <Users className="w-5 h-5 text-blue-400" />
              </div>
              <span className="text-white/70 group-hover:text-white transition-colors">إدارة المستخدمين</span>
            </div>
          </div>
        </Link>
        <Link href="/admin/withdrawals" className="group">
          <div className="p-4 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.15)]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-amber-500/15 group-hover:bg-amber-500/25 transition-colors">
                <ArrowUpCircle className="w-5 h-5 text-amber-400" />
              </div>
              <span className="text-white/70 group-hover:text-white transition-colors">طلبات السحب</span>
            </div>
          </div>
        </Link>
        <Link href="/admin/settings" className="group">
          <div className="p-4 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.15)]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-violet-500/15 group-hover:bg-violet-500/25 transition-colors">
                <Settings className="w-5 h-5 text-violet-400" />
              </div>
              <span className="text-white/70 group-hover:text-white transition-colors">الإعدادات</span>
            </div>
          </div>
        </Link>
        <Link href="/admin/security" className="group">
          <div className="p-4 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.15)]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-emerald-500/15 group-hover:bg-emerald-500/25 transition-colors">
                <Shield className="w-5 h-5 text-emerald-400" />
              </div>
              <span className="text-white/70 group-hover:text-white transition-colors">الأمان</span>
            </div>
          </div>
        </Link>
      </div>

      {/* Main Stats */}
      <div className="relative grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Users */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(59,130,246,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-blue-500/15">
              <Users className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي المستخدمين</p>
              {loadingStats ? (
                <Skeleton className="h-8 w-16 mt-1 bg-white/10" />
              ) : (
                <p className="text-2xl font-bold text-white">{stats?.total_users || 0}</p>
              )}
            </div>
          </div>
        </div>

        {/* Total Assets */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(34,197,94,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-emerald-500/15">
              <DollarSign className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي الأصول</p>
              {loadingStats ? (
                <Skeleton className="h-8 w-24 mt-1 bg-white/10" />
              ) : (
                <p className="text-2xl font-bold text-emerald-400" dir="ltr">{formatCurrency(stats?.total_assets || 0)}</p>
              )}
            </div>
          </div>
        </div>

        {/* Total Deposits */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(139,92,246,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-violet-500/15">
              <ArrowDownCircle className="w-6 h-6 text-violet-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي الإيداعات</p>
              {loadingStats ? (
                <Skeleton className="h-8 w-24 mt-1 bg-white/10" />
              ) : (
                <p className="text-2xl font-bold text-violet-400" dir="ltr">{formatCurrency(stats?.total_deposits || 0)}</p>
              )}
            </div>
          </div>
        </div>

        {/* Total Withdrawals */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 hover:border-violet-500/30 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(139,92,246,0.12)] relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-[radial-gradient(circle,rgba(239,68,68,0.15)_0%,transparent_70%)]" />
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-red-500/15">
              <ArrowUpCircle className="w-6 h-6 text-red-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي السحوبات</p>
              {loadingStats ? (
                <Skeleton className="h-8 w-24 mt-1 bg-white/10" />
              ) : (
                <p className="text-2xl font-bold text-red-400" dir="ltr">{formatCurrency(stats?.total_withdrawals || 0)}</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Secondary Stats */}
      <div className="relative grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* NAV Info */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-lg bg-purple-500/15">
              <BarChart3 className="w-5 h-5 text-purple-400" />
            </div>
            <h3 className="text-white font-semibold">معلومات NAV</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">سعر الوحدة الحالي</span>
              {loadingStats ? (
                <Skeleton className="h-5 w-20 bg-white/10" />
              ) : (
                <span className="text-white font-medium" dir="ltr">${stats?.current_nav?.toFixed(4) || "1.0000"}</span>
              )}
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">إجمالي الوحدات</span>
              {loadingStats ? (
                <Skeleton className="h-5 w-20 bg-white/10" />
              ) : (
                <span className="text-white font-medium">{stats?.total_units?.toFixed(4) || "0"}</span>
              )}
            </div>
          </div>
        </div>

        {/* Bot Status */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-lg bg-cyan-500/15">
              <Bot className="w-5 h-5 text-cyan-400" />
            </div>
            <h3 className="text-white font-semibold">حالة البوت</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">الحالة</span>
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/15 text-emerald-400 text-xs font-semibold border border-emerald-500/25">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                نشط
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">آخر صفقة</span>
              <span className="text-white/70 text-sm">
                {recentTrades[0]?.executed_at 
                  ? format(new Date(recentTrades[0].executed_at), "HH:mm dd/MM")
                  : "-"
                }
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">صفقات اليوم</span>
              <span className="text-white font-medium">{stats?.trades_today || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-white/50 text-sm">ربح اليوم</span>
              <span className={cn(
                "font-medium",
                (stats?.profit_today || 0) >= 0 ? "text-emerald-400" : "text-red-400"
              )} dir="ltr">
                {(stats?.profit_today || 0) >= 0 ? "+" : ""}{formatCurrency(stats?.profit_today || 0)}
              </span>
            </div>
          </div>
        </div>

        {/* Pending Withdrawals */}
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-lg bg-amber-500/15">
              <Clock className="w-5 h-5 text-amber-400" />
            </div>
            <h3 className="text-white font-semibold">طلبات السحب المعلقة</h3>
          </div>
          {loadingWithdrawals ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full bg-white/10" />
              <Skeleton className="h-4 w-3/4 bg-white/10" />
            </div>
          ) : pendingWithdrawals.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-3xl font-bold text-amber-400">{pendingWithdrawals.length}</span>
                <Link href="/admin/withdrawals">
                  <span className="text-violet-400 hover:text-violet-300 text-sm transition-colors">عرض الكل ←</span>
                </Link>
              </div>
              <p className="text-white/50 text-sm">طلب بانتظار الموافقة</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-4">
              <CheckCircle className="w-10 h-10 text-emerald-400/50 mb-2" />
              <p className="text-white/50 text-sm">لا توجد طلبات معلقة</p>
            </div>
          )}
        </div>
      </div>

      {/* Recent Trades */}
      <div className="relative rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
        <div className="p-5 border-b border-violet-500/10">
          <h3 className="text-lg font-semibold text-white">آخر الصفقات</h3>
        </div>
        <div className="p-5">
          {loadingTrades ? (
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-12 w-full bg-white/10" />
              ))}
            </div>
          ) : recentTrades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-right text-white/50 border-b border-violet-500/10">
                    <th className="px-4 py-3 font-medium text-sm">التاريخ</th>
                    <th className="px-4 py-3 font-medium text-sm">الزوج</th>
                    <th className="px-4 py-3 font-medium text-sm">النوع</th>
                    <th className="px-4 py-3 font-medium text-sm">الكمية</th>
                    <th className="px-4 py-3 font-medium text-sm">السعر</th>
                    <th className="px-4 py-3 font-medium text-sm">الإجمالي</th>
                    <th className="px-4 py-3 font-medium text-sm">الربح</th>
                  </tr>
                </thead>
                <tbody>
                  {recentTrades.map((trade: any) => (
                    <tr
                      key={trade.id}
                      className="border-t border-white/5 hover:bg-violet-500/5 transition-colors"
                    >
                      <td className="px-4 py-3 text-white/50 text-sm">
                        {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                      </td>
                      <td className="px-4 py-3 font-medium text-white">{trade.symbol}</td>
                      <td className="px-4 py-3">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold border",
                            trade.side === "BUY"
                              ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                              : "bg-red-500/15 text-red-400 border-red-500/25"
                          )}
                        >
                          {trade.side === "BUY" ? (
                            <>
                              <TrendingUp className="w-3 h-3" />
                              شراء
                            </>
                          ) : (
                            <>
                              <TrendingDown className="w-3 h-3" />
                              بيع
                            </>
                          )}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-white/70" dir="ltr">{trade.quantity?.toFixed(6)}</td>
                      <td className="px-4 py-3 text-white/70" dir="ltr">${trade.price?.toFixed(2)}</td>
                      <td className="px-4 py-3 font-medium text-white" dir="ltr">${trade.total_value?.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {trade.pnl !== undefined ? (
                          <span
                            className={cn(
                              "font-medium",
                              trade.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                            )}
                            dir="ltr"
                          >
                            {trade.pnl >= 0 ? "+" : ""}${trade.pnl?.toFixed(2)}
                          </span>
                        ) : (
                          <span className="text-white/30">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12">
              <Activity className="w-12 h-12 mx-auto mb-3 text-white/20" />
              <p className="text-white/40">لا توجد صفقات</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
