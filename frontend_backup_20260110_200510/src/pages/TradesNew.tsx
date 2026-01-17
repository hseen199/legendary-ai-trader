import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLanguage } from "../lib/i18n";
import { dashboardAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { 
  Search,
  Filter,
  Download,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Calendar,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Target,
  Percent,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";

interface Trade {
  id: number;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  price: number;
  total_value: number;
  pnl?: number;
  pnl_percent?: number;
  executed_at: string;
}

export default function TradesNew() {
  const { t } = useLanguage();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterSide, setFilterSide] = useState<"all" | "BUY" | "SELL">("all");
  const [sortBy, setSortBy] = useState<"date" | "profit" | "amount">("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Fetch trades
  const { data: trades = [], isLoading, refetch } = useQuery({
    queryKey: ["/api/v1/dashboard/trades"],
    queryFn: () => dashboardAPI.getTrades(500).then(res => res.data),
  });

  // Calculate stats
  const winningTrades = trades.filter((t: Trade) => (t.pnl || 0) > 0);
  const losingTrades = trades.filter((t: Trade) => (t.pnl || 0) < 0);
  const totalProfit = winningTrades.reduce((sum: number, t: Trade) => sum + (t.pnl || 0), 0);
  const totalLoss = Math.abs(losingTrades.reduce((sum: number, t: Trade) => sum + (t.pnl || 0), 0));
  const netProfit = totalProfit - totalLoss;
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;
  const bestTrade = Math.max(...trades.map((t: Trade) => t.pnl || 0), 0);

  // Filter and sort trades
  const filteredTrades = trades
    .filter((trade: Trade) => {
      const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesSide = filterSide === "all" || trade.side === filterSide;
      return matchesSearch && matchesSide;
    })
    .sort((a: Trade, b: Trade) => {
      let comparison = 0;
      if (sortBy === "date") {
        comparison = new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime();
      } else if (sortBy === "profit") {
        comparison = (a.pnl || 0) - (b.pnl || 0);
      } else if (sortBy === "amount") {
        comparison = a.total_value - b.total_value;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / itemsPerPage);
  const paginatedTrades = filteredTrades.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const exportTrades = () => {
    const csvContent = [
      ["التاريخ", "الزوج", "النوع", "الكمية", "السعر", "الإجمالي", "الربح/الخسارة"],
      ...filteredTrades.map((trade: Trade) => [
        format(new Date(trade.executed_at), "yyyy-MM-dd HH:mm"),
        trade.symbol,
        trade.side === "BUY" ? "شراء" : "بيع",
        trade.quantity.toString(),
        trade.price.toString(),
        trade.total_value.toString(),
        (trade.pnl || 0).toString(),
      ]),
    ]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `trades_${format(new Date(), "yyyy-MM-dd")}.csv`;
    link.click();
    toast.success("تم تصدير الصفقات بنجاح");
  };

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
          <h1 className="text-2xl font-bold">{t.nav.trades}</h1>
          <p className="text-muted-foreground text-sm">جميع صفقات التداول التي نفذها البوت</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => refetch()}
            className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            تحديث
          </button>
          <button
            onClick={exportTrades}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Download className="w-4 h-4" />
            تصدير CSV
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">إجمالي الصفقات</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold">{trades.length}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-green-500" />
              <span className="text-sm text-muted-foreground">نسبة النجاح</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-green-500">{winRate.toFixed(1)}%</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-sm text-muted-foreground">صفقات رابحة</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-green-500">{winningTrades.length}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-destructive" />
              <span className="text-sm text-muted-foreground">صفقات خاسرة</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-destructive">{losingTrades.length}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Percent className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">صافي الربح</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-20" />
            ) : (
              <p className={cn(
                "text-xl font-bold",
                netProfit >= 0 ? "text-green-500" : "text-destructive"
              )}>
                {formatCurrency(netProfit)}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-muted-foreground">أفضل صفقة</span>
            </div>
            {isLoading ? (
              <Skeleton className="h-7 w-20" />
            ) : (
              <p className="text-xl font-bold text-green-500">+{formatCurrency(bestTrade)}</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-4 items-center">
            {/* Search */}
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="بحث عن زوج..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pr-10 px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
              />
            </div>

            {/* Side Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <select
                value={filterSide}
                onChange={(e) => setFilterSide(e.target.value as any)}
                className="px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
              >
                <option value="all">جميع الأنواع</option>
                <option value="BUY">شراء</option>
                <option value="SELL">بيع</option>
              </select>
            </div>

            {/* Sort */}
            <div className="flex items-center gap-2">
              <ArrowUpDown className="w-4 h-4 text-muted-foreground" />
              <select
                value={`${sortBy}-${sortOrder}`}
                onChange={(e) => {
                  const [by, order] = e.target.value.split("-");
                  setSortBy(by as any);
                  setSortOrder(order as any);
                }}
                className="px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
              >
                <option value="date-desc">الأحدث أولاً</option>
                <option value="date-asc">الأقدم أولاً</option>
                <option value="profit-desc">الأعلى ربحاً</option>
                <option value="profit-asc">الأعلى خسارة</option>
                <option value="amount-desc">الأعلى قيمة</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trades Table */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-6 space-y-3">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : paginatedTrades.length > 0 ? (
            <>
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
                      <th className="px-4 py-3 font-medium">الربح/الخسارة</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedTrades.map((trade: Trade) => (
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
                        <td className="px-4 py-3" dir="ltr">{trade.quantity.toFixed(6)}</td>
                        <td className="px-4 py-3" dir="ltr">${trade.price.toFixed(2)}</td>
                        <td className="px-4 py-3 font-medium" dir="ltr">${trade.total_value.toFixed(2)}</td>
                        <td className="px-4 py-3">
                          {trade.pnl !== undefined ? (
                            <span
                              className={cn(
                                "font-medium",
                                trade.pnl >= 0 ? "text-green-500" : "text-destructive"
                              )}
                              dir="ltr"
                            >
                              {trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(2)}
                              {trade.pnl_percent !== undefined && (
                                <span className="text-xs mr-1">
                                  ({trade.pnl_percent >= 0 ? "+" : ""}{trade.pnl_percent.toFixed(1)}%)
                                </span>
                              )}
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

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-border">
                  <p className="text-sm text-muted-foreground">
                    عرض {(currentPage - 1) * itemsPerPage + 1} إلى{" "}
                    {Math.min(currentPage * itemsPerPage, filteredTrades.length)} من{" "}
                    {filteredTrades.length} صفقة
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="p-2 rounded-lg hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-muted-foreground">
                      صفحة {currentPage} من {totalPages}
                    </span>
                    <button
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="p-2 rounded-lg hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">لا توجد صفقات</p>
              <p className="text-sm text-muted-foreground mt-1">
                سيتم عرض الصفقات هنا عندما يبدأ البوت بالتداول
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
