import { useQuery } from "@tanstack/react-query";
import { useLanguage } from "../lib/i18n";
import { dashboardAPI, walletAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  ArrowUpCircle, 
  ArrowDownCircle, 
  BarChart3,
  Download
} from "lucide-react";
import { cn } from "../lib/utils";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { format } from "date-fns";
import toast from "react-hot-toast";

const CHART_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

export default function PortfolioNew() {
  const { t } = useLanguage();

  // Fetch balance
  const { data: balance, isLoading: loadingBalance } = useQuery({
    queryKey: ["/api/v1/wallet/balance"],
    queryFn: () => walletAPI.getBalance().then(res => res.data),
  });

  // Fetch dashboard data
  const { data: dashboardData, isLoading: loadingDashboard } = useQuery({
    queryKey: ["/api/v1/dashboard/"],
    queryFn: () => dashboardAPI.getDashboard().then(res => res.data),
  });

  // Fetch NAV history
  const { data: navHistory = [], isLoading: loadingNav } = useQuery({
    queryKey: ["/api/v1/dashboard/nav/history"],
    queryFn: () => dashboardAPI.getNAVHistory(90).then(res => res.data),
  });

  // Fetch transactions
  const { data: transactions = [], isLoading: loadingTx } = useQuery({
    queryKey: ["/api/v1/wallet/transactions"],
    queryFn: () => walletAPI.getTransactions(50).then(res => res.data),
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

  // Calculate totals from transactions
  const totalDeposits = transactions
    .filter((t: any) => t.type === "deposit" && t.status === "confirmed")
    .reduce((sum: number, t: any) => sum + parseFloat(t.amount || 0), 0);

  const totalWithdrawals = transactions
    .filter((t: any) => t.type === "withdrawal" && t.status === "completed")
    .reduce((sum: number, t: any) => sum + parseFloat(t.amount || 0), 0);

  // Prepare chart data
  const chartData = navHistory.map((item: any) => ({
    date: format(new Date(item.timestamp), "MM/dd"),
    nav: item.nav_value,
    value: item.total_assets_usd,
  }));

  // Export data
  const exportData = () => {
    if (!navHistory.length) return;
    
    const csvContent = [
      ["التاريخ", "قيمة الوحدة", "إجمالي الأصول"],
      ...navHistory.map((item: any) => [
        format(new Date(item.timestamp), "yyyy-MM-dd"),
        item.nav_value.toFixed(4),
        item.total_assets_usd.toFixed(2)
      ])
    ].map(row => row.join(",")).join("\n");
    
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `portfolio_history_${format(new Date(), "yyyy-MM-dd")}.csv`;
    link.click();
    toast.success(t.portfolio.exportSuccess);
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 mb-2">
        <div>
          <h1 className="text-2xl font-bold">{t.portfolio.title}</h1>
          <p className="text-muted-foreground text-sm">{t.portfolio.subtitle}</p>
        </div>
        <button
          onClick={exportData}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          <Download className="w-4 h-4" />{t.portfolio.exportData}</button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-md bg-primary/10">
                <ArrowUpCircle className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.portfolio.totalDeposits}</p>
                {loadingTx ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(totalDeposits)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-md bg-destructive/10">
                <ArrowDownCircle className="w-5 h-5 text-destructive" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.portfolio.totalWithdrawals}</p>
                {loadingTx ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(totalWithdrawals)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-md bg-muted">
                <Wallet className="w-5 h-5 text-foreground" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.portfolio.currentValue}</p>
                {loadingBalance ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className="text-xl font-bold" dir="ltr">{formatCurrency(currentValue)}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-3 rounded-md",
                profitLoss >= 0 ? "bg-green-500/10" : "bg-destructive/10"
              )}>
                {profitLoss >= 0 ? (
                  <TrendingUp className="w-5 h-5 text-green-500" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-destructive" />
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{t.portfolio.profitLoss}</p>
                {loadingDashboard ? (
                  <Skeleton className="h-7 w-24 mt-1" />
                ) : (
                  <p className={cn(
                    "text-xl font-bold",
                    profitLoss >= 0 ? "text-green-500" : "text-destructive"
                  )} dir="ltr">
                    {profitLoss >= 0 && "+"}{formatCurrency(profitLoss)}
                    <span className="text-sm font-normal mr-1">({profitLossPercent.toFixed(2)}%)</span>
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* NAV Info */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            معلومات صافي قيمة الأصول (NAV)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-muted/50 rounded-md">
              <p className="text-xs text-muted-foreground mb-1">{t.portfolio.unitPrice}</p>
              {loadingBalance ? (
                <Skeleton className="h-6 w-20 mx-auto" />
              ) : (
                <p className="text-lg font-bold" dir="ltr">{formatCurrency(nav)}</p>
              )}
            </div>
            <div className="text-center p-4 bg-muted/50 rounded-md">
              <p className="text-xs text-muted-foreground mb-1">وحداتك</p>
              {loadingBalance ? (
                <Skeleton className="h-6 w-20 mx-auto" />
              ) : (
                <p className="text-lg font-bold" dir="ltr">{units.toFixed(4)}</p>
              )}
            </div>
            <div className="text-center p-4 bg-muted/50 rounded-md">
              <p className="text-xs text-muted-foreground mb-1">{t.portfolio.netInvestment}</p>
              {loadingTx ? (
                <Skeleton className="h-6 w-20 mx-auto" />
              ) : (
                <p className="text-lg font-bold" dir="ltr">{formatCurrency(totalDeposits - totalWithdrawals)}</p>
              )}
            </div>
            <div className="text-center p-4 bg-muted/50 rounded-md">
              <p className="text-xs text-muted-foreground mb-1">{t.portfolio.roi}</p>
              {loadingDashboard ? (
                <Skeleton className="h-6 w-20 mx-auto" />
              ) : (
                <p className={cn(
                  "text-lg font-bold",
                  profitLossPercent >= 0 ? "text-green-500" : "text-destructive"
                )} dir="ltr">
                  {profitLossPercent >= 0 ? "+" : ""}{profitLossPercent.toFixed(2)}%
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Chart */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">{t.portfolio.portfolioPerformance}</CardTitle>
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
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">{t.portfolio.noData}</div>
          )}
        </CardContent>
      </Card>

      {/* Recent Transactions */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">آخر المعاملات</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingTx ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : transactions.length > 0 ? (
            <div className="space-y-3">
              {transactions.slice(0, 10).map((tx: any) => (
                <div
                  key={tx.id}
                  className="flex items-center justify-between p-4 rounded-md bg-muted/50"
                >
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "p-2 rounded-md",
                      tx.type === "deposit" ? "bg-green-500/10" : "bg-destructive/10"
                    )}>
                      {tx.type === "deposit" ? (
                        <ArrowUpCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <ArrowDownCircle className="w-4 h-4 text-destructive" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium">{tx.type === "deposit" ? "إيداع" : "سحب"}</p>
                      <p className="text-xs text-muted-foreground">
                        {tx.created_at ? format(new Date(tx.created_at), "dd/MM/yyyy HH:mm") : "-"}
                      </p>
                    </div>
                  </div>
                  <div className="text-left">
                    <p className="font-bold" dir="ltr">{formatCurrency(parseFloat(tx.amount || 0))}</p>
                    <span className={cn(
                      "text-xs px-2 py-0.5 rounded-full",
                      tx.status === "confirmed" || tx.status === "completed" 
                        ? "bg-green-500/10 text-green-500"
                        : tx.status === "pending" 
                        ? "bg-yellow-500/10 text-yellow-500"
                        : "bg-destructive/10 text-destructive"
                    )}>
                      {tx.status === "confirmed" || tx.status === "completed" ? "مكتمل" 
                        : tx.status === "pending" ? "قيد الانتظار" 
                        : "ملغي"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">{t.wallet.noTransactions}</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
