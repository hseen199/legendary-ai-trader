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

export default function AdminDashboardNew() {
  const { t } = useLanguage();

  // Fetch admin stats
  const { data: stats, isLoading: loadingStats, refetch } = useQuery({
    queryKey: ["/api/v1/admin/stats"],
    queryFn: () => adminAPI.getStats().then(res => res.data),
    refetchInterval: 60000, // Refresh every minute
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
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">لوحة تحكم الأدمن</h1>
          <p className="text-muted-foreground text-sm">نظرة عامة على المنصة</p>
        </div>
        <button 
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Main Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Users */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-blue-500/10">
                <Users className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">إجمالي المستخدمين</p>
                {loadingStats ? (
                  <Skeleton className="h-7 w-16 mt-1" />
                ) : (
                  <p className="text-xl font-bold">{stats?.total_users || 0}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Total Assets */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-green-500/10">
                <DollarSign className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">إجمالي الأصول</p>
                {loadingStats ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(stats?.total_assets || 0)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Total Deposits */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-primary/10">
                <ArrowDownCircle className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">إجمالي الإيداعات</p>
                {loadingStats ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(stats?.total_deposits || 0)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Total Withdrawals */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-destructive/10">
                <ArrowUpCircle className="w-5 h-5 text-destructive" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">إجمالي السحوبات</p>
                {loadingStats ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(stats?.total_withdrawals || 0)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Secondary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">NAV الحالي</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-20" />
            ) : (
              <p className="text-xl font-bold" dir="ltr">${stats?.current_nav?.toFixed(4) || "1.0000"}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-green-500" />
              <span className="text-sm text-muted-foreground">إجمالي الصفقات</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold">{stats?.total_trades || 0}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Percent className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-muted-foreground">نسبة النجاح</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-green-500">{stats?.win_rate?.toFixed(1) || 0}%</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-orange-500" />
              <span className="text-sm text-muted-foreground">سحوبات معلقة</span>
            </div>
            {loadingWithdrawals ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-orange-500">{pendingWithdrawals.length}</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pending Withdrawals */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  طلبات السحب المعلقة
                </CardTitle>
                {pendingWithdrawals.length > 0 && (
                  <Badge variant="destructive">{pendingWithdrawals.length} طلب</Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {loadingWithdrawals ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : pendingWithdrawals.length > 0 ? (
                <div className="space-y-3 max-h-[300px] overflow-y-auto">
                  {pendingWithdrawals.map((withdrawal: any) => (
                    <div
                      key={withdrawal.id}
                      className="flex items-center justify-between p-4 bg-muted/50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-orange-500/10">
                          <ArrowUpCircle className="w-4 h-4 text-orange-500" />
                        </div>
                        <div>
                          <p className="font-medium">{withdrawal.user?.email || "مستخدم"}</p>
                          <p className="text-xs text-muted-foreground">
                            {withdrawal.created_at ? format(new Date(withdrawal.created_at), "dd/MM/yyyy HH:mm") : "-"}
                          </p>
                        </div>
                      </div>
                      <div className="text-left">
                        <p className="font-bold" dir="ltr">{formatCurrency(withdrawal.amount)}</p>
                        <Badge variant="outline" className="bg-orange-500/10 text-orange-500 border-orange-500/20">
                          <Clock className="w-3 h-3 ml-1" />
                          قيد الانتظار
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                  <p>لا توجد طلبات سحب معلقة</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Bot Status */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Bot className="w-5 h-5" />
              حالة البوت
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-500/10 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                <div>
                  <p className="font-medium">البوت نشط</p>
                  <p className="text-sm text-muted-foreground">يعمل بشكل طبيعي</p>
                </div>
              </div>
              <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                <Activity className="w-3 h-3 ml-1" />
                نشط
              </Badge>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">آخر صفقة</span>
                <span>
                  {recentTrades[0]?.executed_at 
                    ? format(new Date(recentTrades[0].executed_at), "HH:mm dd/MM")
                    : "-"
                  }
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">صفقات اليوم</span>
                <span>{stats?.trades_today || 0}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">ربح اليوم</span>
                <span className={cn(
                  (stats?.profit_today || 0) >= 0 ? "text-green-500" : "text-destructive"
                )} dir="ltr">
                  {(stats?.profit_today || 0) >= 0 ? "+" : ""}{formatCurrency(stats?.profit_today || 0)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Trades */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">آخر الصفقات</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingTrades ? (
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : recentTrades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted">
                  <tr className="text-right text-muted-foreground">
                    <th className="px-4 py-3 font-medium">التاريخ</th>
                    <th className="px-4 py-3 font-medium">الزوج</th>
                    <th className="px-4 py-3 font-medium">النوع</th>
                    <th className="px-4 py-3 font-medium">الكمية</th>
                    <th className="px-4 py-3 font-medium">السعر</th>
                    <th className="px-4 py-3 font-medium">الإجمالي</th>
                    <th className="px-4 py-3 font-medium">الربح</th>
                  </tr>
                </thead>
                <tbody>
                  {recentTrades.map((trade: any) => (
                    <tr
                      key={trade.id}
                      className="border-t border-border hover:bg-muted/50 transition-colors"
                    >
                      <td className="px-4 py-3 text-muted-foreground">
                        {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                      </td>
                      <td className="px-4 py-3 font-medium">{trade.symbol}</td>
                      <td className="px-4 py-3">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium",
                            trade.side === "BUY"
                              ? "bg-green-500/10 text-green-500"
                              : "bg-destructive/10 text-destructive"
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
                      <td className="px-4 py-3" dir="ltr">{trade.quantity?.toFixed(6)}</td>
                      <td className="px-4 py-3" dir="ltr">${trade.price?.toFixed(2)}</td>
                      <td className="px-4 py-3 font-medium" dir="ltr">${trade.total_value?.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {trade.pnl !== undefined ? (
                          <span
                            className={cn(
                              "font-medium",
                              trade.pnl >= 0 ? "text-green-500" : "text-destructive"
                            )}
                            dir="ltr"
                          >
                            {trade.pnl >= 0 ? "+" : ""}${trade.pnl?.toFixed(2)}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>لا توجد صفقات</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
