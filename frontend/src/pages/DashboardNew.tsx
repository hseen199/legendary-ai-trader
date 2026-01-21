import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../lib/i18n";
import { dashboardAPI, walletAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  ArrowUpCircle, 
  ArrowDownCircle,
  Activity,
  Bot,
  BarChart3,
  Clock,
  RefreshCw
} from "lucide-react";
import { cn } from "../lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { format } from "date-fns";
import { ar } from "date-fns/locale";

// دالة للحصول على شعار العملة
const getCryptoLogo = (symbol: string) => {
  const coin = symbol.replace('USDC', '').replace('USDT', '').toLowerCase();
  return `https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c25fa5/128/color/${coin}.png`;
};

export default function DashboardNew() {
  const { user } = useAuth();
  const { t, language } = useLanguage();

  // Fetch dashboard data
  const { data: dashboardData, isLoading: loadingDashboard, refetch: refetchDashboard } = useQuery({
    queryKey: ["/api/v1/dashboard/"],
    queryFn: () => dashboardAPI.getDashboard().then(res => res.data),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch NAV history
  const { data: navHistory = [], isLoading: loadingNav } = useQuery({
    queryKey: ["/api/v1/dashboard/nav/history"],
    queryFn: () => dashboardAPI.getNAVHistory(30).then(res => res.data),
  });

  // Fetch recent trades
  const { data: trades = [], isLoading: loadingTrades } = useQuery({
    queryKey: ["/api/v1/dashboard/trades"],
    queryFn: () => dashboardAPI.getTrades(10).then(res => res.data),
  });

  // Fetch balance
  const { data: balance, isLoading: loadingBalance } = useQuery({
    queryKey: ["/api/v1/wallet/balance"],
    queryFn: () => walletAPI.getBalance().then(res => res.data),
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  const currentValue = dashboardData?.current_value || balance?.current_value_usd || 0;
  const totalDeposited = dashboardData?.total_deposited || 0;
  const profitLoss = dashboardData?.profit_loss || 0;
  const profitLossPercent = dashboardData?.profit_loss_percent || 0;
  const units = dashboardData?.units || balance?.units || 0;
  const nav = dashboardData?.current_nav || balance?.nav || 1;

  // Prepare chart data
  const chartData = navHistory.map((item: any) => ({
    date: format(new Date(item.timestamp), "MM/dd"),
    nav: item.nav_value,
    value: item.total_assets_usd,
  }));

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 mb-2">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent animate-gradient-shift bg-[length:200%_auto]">
            {t.dashboard.title}
          </h1>
          <p className="text-muted-foreground text-sm">{t.dashboard.subtitle}</p>
        </div>
        <button 
          onClick={() => refetchDashboard()}
          className="p-2.5 rounded-xl bg-gradient-to-r from-primary/10 to-purple-500/10 hover:from-primary/20 hover:to-purple-500/20 transition-all duration-300 hover:scale-105 border border-primary/20"
        >
          <RefreshCw className="w-5 h-5 text-primary" />
        </button>
      </div>

      {/* Stats Cards - Enhanced with gradients and animations */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Current Value */}
        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-primary/5 via-transparent to-purple-500/5 hover:from-primary/10 hover:to-purple-500/10 transition-all duration-500 hover:scale-[1.02] hover:shadow-xl hover:shadow-primary/10">
          <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-5 md:p-6 relative">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-purple-500/20 shadow-lg shadow-primary/20 group-hover:shadow-primary/40 transition-all duration-300">
                <Wallet className="w-5 h-5 md:w-6 md:h-6 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs md:text-sm text-muted-foreground truncate">{t.dashboard.portfolioValue}</p>
                {loadingBalance ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-lg md:text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent" dir="ltr">
                    {formatCurrency(currentValue)}
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Total Deposited */}
        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-blue-500/5 via-transparent to-cyan-500/5 hover:from-blue-500/10 hover:to-cyan-500/10 transition-all duration-500 hover:scale-[1.02] hover:shadow-xl hover:shadow-blue-500/10">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/5 to-blue-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-5 md:p-6 relative">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 shadow-lg shadow-blue-500/20 group-hover:shadow-blue-500/40 transition-all duration-300">
                <ArrowDownCircle className="w-5 h-5 md:w-6 md:h-6 text-blue-500" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs md:text-sm text-muted-foreground truncate">{t.wallet.totalDeposits}</p>
                {loadingDashboard ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-lg md:text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent" dir="ltr">
                    {formatCurrency(totalDeposited)}
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Profit/Loss */}
        <Card className={cn(
          "group relative overflow-hidden border-0 transition-all duration-500 hover:scale-[1.02]",
          profitLoss >= 0 
            ? "bg-gradient-to-br from-green-500/5 via-transparent to-emerald-500/5 hover:from-green-500/10 hover:to-emerald-500/10 hover:shadow-xl hover:shadow-green-500/10"
            : "bg-gradient-to-br from-red-500/5 via-transparent to-rose-500/5 hover:from-red-500/10 hover:to-rose-500/10 hover:shadow-xl hover:shadow-red-500/10"
        )}>
          <div className={cn(
            "absolute inset-0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000",
            profitLoss >= 0 ? "bg-gradient-to-r from-green-500/0 via-green-500/5 to-green-500/0" : "bg-gradient-to-r from-red-500/0 via-red-500/5 to-red-500/0"
          )} />
          <CardContent className="p-5 md:p-6 relative">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-3 rounded-xl shadow-lg transition-all duration-300",
                profitLoss >= 0 
                  ? "bg-gradient-to-br from-green-500/20 to-emerald-500/20 shadow-green-500/20 group-hover:shadow-green-500/40" 
                  : "bg-gradient-to-br from-red-500/20 to-rose-500/20 shadow-red-500/20 group-hover:shadow-red-500/40"
              )}>
                {profitLoss >= 0 ? (
                  <TrendingUp className="w-5 h-5 md:w-6 md:h-6 text-green-500" />
                ) : (
                  <TrendingDown className="w-5 h-5 md:w-6 md:h-6 text-red-500" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs md:text-sm text-muted-foreground truncate">{t.dashboard.totalProfit}</p>
                {loadingDashboard ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <div dir="ltr">
                    <p className={cn(
                      "text-lg md:text-2xl font-bold bg-clip-text text-transparent",
                      profitLoss >= 0 
                        ? "bg-gradient-to-r from-green-400 to-emerald-400" 
                        : "bg-gradient-to-r from-red-400 to-rose-400"
                    )}>
                      {profitLoss >= 0 ? "+" : ""}{formatCurrency(profitLoss)}
                    </p>
                    <p className={cn(
                      "text-xs md:text-sm font-medium",
                      profitLoss >= 0 ? "text-green-500/80" : "text-red-500/80"
                    )}>
                      ({profitLossPercent >= 0 ? "+" : ""}{profitLossPercent.toFixed(2)}%)
                    </p>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* NAV & Units */}
        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-purple-500/5 via-transparent to-pink-500/5 hover:from-purple-500/10 hover:to-pink-500/10 transition-all duration-500 hover:scale-[1.02] hover:shadow-xl hover:shadow-purple-500/10">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-purple-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-5 md:p-6 relative">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 shadow-lg shadow-purple-500/20 group-hover:shadow-purple-500/40 transition-all duration-300">
                <BarChart3 className="w-5 h-5 md:w-6 md:h-6 text-purple-500" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs md:text-sm text-muted-foreground truncate">{t.wallet.shares}</p>
                {loadingBalance ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <div dir="ltr">
                    <p className="text-lg md:text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      {units.toFixed(4)}
                    </p>
                    <p className="text-xs md:text-sm text-purple-400/80">
                      NAV: ${nav.toFixed(4)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2">
          <Card className="border-0 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                {t.dashboard.portfolioHistory}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loadingNav ? (
                <Skeleton className="h-[300px] w-full rounded-xl" />
              ) : chartData.length > 0 ? (
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="colorNav" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                          <stop offset="50%" stopColor="#8b5cf6" stopOpacity={0.1} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="strokeGradient" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="#8b5cf6" />
                          <stop offset="50%" stopColor="#a855f7" />
                          <stop offset="100%" stopColor="#ec4899" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                      <XAxis 
                        dataKey="date" 
                        stroke="hsl(var(--muted-foreground))"
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))"
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => `$${value.toFixed(2)}`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(15, 15, 25, 0.95)",
                          border: "1px solid rgba(139, 92, 246, 0.3)",
                          borderRadius: "12px",
                          boxShadow: "0 8px 32px rgba(139, 92, 246, 0.2)",
                        }}
                        formatter={(value: number) => [`$${value.toFixed(4)}`, "NAV"]}
                      />
                      <Area
                        type="monotone"
                        dataKey="nav"
                        stroke="url(#strokeGradient)"
                        strokeWidth={3}
                        fill="url(#colorNav)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  {t.common.noData}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Bot Status & Quick Actions */}
        <div className="space-y-6">
          {/* Bot Status */}
          <Card className="border-0 bg-gradient-to-br from-green-500/5 to-emerald-500/5 overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Bot className="w-5 h-5 text-green-500" />
                {t.dashboard.botStatus}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-xl border border-green-500/20">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                    <div className="absolute inset-0 w-3 h-3 rounded-full bg-green-500 animate-ping opacity-75" />
                  </div>
                  <div>
                    <p className="font-medium text-green-400">{t.dashboard.botActive}</p>
                    <p className="text-xs text-muted-foreground">
                      {t.settings.botActiveTrading}
                    </p>
                  </div>
                </div>
                <Badge className="bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30">
                  <Activity className="w-3 h-3 ml-1 animate-pulse" />
                  {t.dashboard.botActive}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Recent Trades */}
          <Card className="border-0 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                {t.dashboard.recentTrades}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loadingTrades ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-14 w-full rounded-xl" />
                  ))}
                </div>
              ) : trades.length > 0 ? (
                <div className="space-y-3">
                  {trades.slice(0, 5).map((trade: any, index: number) => (
                    <div
                      key={trade.id}
                      className={cn(
                        "flex items-center justify-between p-3 rounded-xl transition-all duration-300 hover:scale-[1.02] cursor-pointer",
                        trade.side === "BUY" 
                          ? "bg-gradient-to-r from-green-500/10 to-transparent hover:from-green-500/20 border border-green-500/10" 
                          : "bg-gradient-to-r from-red-500/10 to-transparent hover:from-red-500/20 border border-red-500/10"
                      )}
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <div className="flex items-center gap-3">
                        {/* شعار العملة */}
                        <div className="relative">
                          <img 
                            src={getCryptoLogo(trade.symbol)} 
                            alt={trade.symbol}
                            className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-white/10 p-0.5"
                            onError={(e) => {
                              (e.target as HTMLImageElement).style.display = 'none';
                              (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                            }}
                          />
                          <div className={cn(
                            "hidden w-8 h-8 md:w-10 md:h-10 rounded-full items-center justify-center text-xs font-bold",
                            trade.side === "BUY" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                          )}>
                            {trade.symbol.replace('USDC', '').slice(0, 3)}
                          </div>
                          <div className={cn(
                            "absolute -bottom-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center",
                            trade.side === "BUY" ? "bg-green-500" : "bg-red-500"
                          )}>
                            {trade.side === "BUY" ? (
                              <TrendingUp className="w-2.5 h-2.5 text-white" />
                            ) : (
                              <TrendingDown className="w-2.5 h-2.5 text-white" />
                            )}
                          </div>
                        </div>
                        <div>
                          <p className="font-medium text-sm">{trade.symbol.replace('USDC', '')}</p>
                          <p className="text-xs text-muted-foreground">
                            {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                          </p>
                        </div>
                      </div>
                      <div className="text-left">
                        <p className="font-semibold text-sm" dir="ltr">
                          ${trade.total_value?.toFixed(2) || "0.00"}
                        </p>
                        {trade.pnl !== undefined && trade.pnl !== null && (
                          <p className={cn(
                            "text-xs font-medium",
                            trade.pnl >= 0 ? "text-green-400" : "text-red-400"
                          )} dir="ltr">
                            {trade.pnl >= 0 ? "+" : ""}{trade.pnl?.toFixed(2) || "0.00"}
                            {trade.pnl_percent && (
                              <span className="opacity-70"> ({trade.pnl_percent >= 0 ? "+" : ""}{trade.pnl_percent?.toFixed(1)}%)</span>
                            )}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">{t.common.noData}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
