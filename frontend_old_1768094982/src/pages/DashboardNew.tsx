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

  const currentValue = balance?.current_value_usd || 0;
  const totalDeposited = dashboardData?.total_deposited || 0;
  const profitLoss = dashboardData?.profit_loss || 0;
  const profitLossPercent = dashboardData?.profit_loss_percent || 0;
  const units = balance?.units || 0;
  const nav = balance?.nav || 1;

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
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            {t.dashboard.title}
          </h1>
          <p className="text-muted-foreground text-sm">{t.dashboard.subtitle}</p>
        </div>
        <button 
          onClick={() => refetchDashboard()}
          className="p-2 rounded-lg hover:bg-muted transition-colors"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Current Value */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-primary/10">
                <Wallet className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.dashboard.portfolioValue}</p>
                {loadingBalance ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(currentValue)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Total Deposited */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-blue-500/10">
                <ArrowDownCircle className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.wallet.totalDeposits}</p>
                {loadingDashboard ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(totalDeposited)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Profit/Loss */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-3 rounded-lg",
                profitLoss >= 0 ? "bg-green-500/10" : "bg-red-500/10"
              )}>
                {profitLoss >= 0 ? (
                  <TrendingUp className="w-5 h-5 text-green-500" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-red-500" />
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.dashboard.totalProfit}</p>
                {loadingDashboard ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className={cn(
                    "text-xl font-bold",
                    profitLoss >= 0 ? "text-green-500" : "text-red-500"
                  )} dir="ltr">
                    {profitLoss >= 0 ? "+" : ""}{formatCurrency(profitLoss)}
                    <span className="text-sm font-normal mr-1">
                      ({profitLossPercent >= 0 ? "+" : ""}{profitLossPercent.toFixed(2)}%)
                    </span>
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Units */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-purple-500/10">
                <BarChart3 className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.wallet.shares}</p>
                {loadingBalance ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <div>
                    <p className="text-xl font-bold" dir="ltr">{units.toFixed(4)}</p>
                    <p className="text-xs text-muted-foreground" dir="ltr">
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
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">{t.dashboard.portfolioHistory}</CardTitle>
            </CardHeader>
            <CardContent>
              {loadingNav ? (
                <Skeleton className="h-[300px] w-full" />
              ) : chartData.length > 0 ? (
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="colorNav" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis 
                        dataKey="date" 
                        stroke="hsl(var(--muted-foreground))"
                        fontSize={12}
                      />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))"
                        fontSize={12}
                        tickFormatter={(value) => `$${value.toFixed(2)}`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--popover))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                        formatter={(value: number) => [`$${value.toFixed(4)}`, "NAV"]}
                      />
                      <Area
                        type="monotone"
                        dataKey="nav"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
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
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Bot className="w-5 h-5" />
                {t.dashboard.botStatus}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <div>
                    <p className="font-medium">{t.dashboard.botActive}</p>
                    <p className="text-sm text-muted-foreground">
                      {t.settings.botActiveTrading}
                    </p>
                  </div>
                </div>
                <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                  <Activity className="w-3 h-3 ml-1" />
                  {t.dashboard.botActive}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Recent Trades */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">{t.dashboard.recentTrades}</CardTitle>
            </CardHeader>
            <CardContent>
              {loadingTrades ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : trades.length > 0 ? (
                <div className="space-y-3">
                  {trades.slice(0, 5).map((trade: any) => (
                    <div
                      key={trade.id}
                      className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "p-2 rounded-md",
                          trade.side === "BUY" ? "bg-green-500/10" : "bg-red-500/10"
                        )}>
                          {trade.side === "BUY" ? (
                            <TrendingUp className="w-4 h-4 text-green-500" />
                          ) : (
                            <TrendingDown className="w-4 h-4 text-red-500" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{trade.symbol}</p>
                          <p className="text-xs text-muted-foreground">
                            {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                          </p>
                        </div>
                      </div>
                      <div className="text-left">
                        <p className="font-medium text-sm" dir="ltr">
                          ${trade.total_value?.toFixed(2) || "0.00"}
                        </p>
                        {trade.pnl !== undefined && (
                          <p className={cn(
                            "text-xs",
                            trade.pnl >= 0 ? "text-green-500" : "text-red-500"
                          )} dir="ltr">
                            {trade.pnl >= 0 ? "+" : ""}{trade.pnl?.toFixed(2) || "0.00"}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>{t.common.noData}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
