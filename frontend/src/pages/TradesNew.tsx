import { useState, useEffect } from "react";
import { dashboardAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { 
  RefreshCw,
  TrendingUp,
  TrendingDown,
  BarChart3,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import { useLanguage } from '@/lib/i18n';

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
  const { t, language } = useLanguage();
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterSide, setFilterSide] = useState<"all" | "BUY" | "SELL">("all");

  const fetchTrades = async () => {
    setIsLoading(true);
    try {
      const response = await dashboardAPI.getTrades(500);
      setTrades(response.data || []);
    } catch (error) {
      console.error("Error fetching trades:", error);
      toast.error(t.trades.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  // Calculate stats
  const winningTrades = trades.filter((t) => (t.pnl || 0) > 0);
  const losingTrades = trades.filter((t) => (t.pnl || 0) < 0);
  const totalProfit = winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0));
  const netProfit = totalProfit - totalLoss;
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  // Filter trades
  const filteredTrades = trades.filter((trade) => {
    const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSide = filterSide === "all" || trade.side === filterSide;
    return matchesSearch && matchesSide;
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-violet-500" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">{t.trades.title}</h1>
        <button
          onClick={fetchTrades}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-violet-500/20 text-violet-400 hover:bg-violet-500/30 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />{t.common.refresh}</button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
          <CardContent className="p-4">
            <p className="text-white/50 text-sm">{t.trades.totalTrades}</p>
            <p className="text-2xl font-bold text-white mt-1">{trades.length}</p>
          </CardContent>
        </Card>
        <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
          <CardContent className="p-4">
            <p className="text-white/50 text-sm">{t.trades.winRate}</p>
            <p className="text-2xl font-bold text-emerald-400 mt-1">{winRate.toFixed(1)}%</p>
          </CardContent>
        </Card>
        <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
          <CardContent className="p-4">
            <p className="text-white/50 text-sm">{t.trades.winningTrades}</p>
            <p className="text-2xl font-bold text-emerald-400 mt-1">{winningTrades.length}</p>
          </CardContent>
        </Card>
        <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
          <CardContent className="p-4">
            <p className="text-white/50 text-sm">{t.trades.losingTrades}</p>
            <p className="text-2xl font-bold text-red-400 mt-1">{losingTrades.length}</p>
          </CardContent>
        </Card>
      </div>

      {/* Net Profit Card */}
      <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white/50">{t.trades.netProfitLoss}</p>
              <p className={cn(
                "text-3xl font-bold mt-1",
                netProfit >= 0 ? "text-emerald-400" : "text-red-400"
              )}>
                {netProfit >= 0 ? "+" : ""}${netProfit.toFixed(2)}
              </p>
            </div>
            <div className={cn(
              "w-14 h-14 rounded-xl flex items-center justify-center",
              netProfit >= 0 ? "bg-emerald-500/20" : "bg-red-500/20"
            )}>
              {netProfit >= 0 ? (
                <TrendingUp className="w-8 h-8 text-emerald-400" />
              ) : (
                <TrendingDown className="w-8 h-8 text-red-400" />
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        <input
          type="text"
          placeholder="بحث عن زوج..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="px-4 py-2 rounded-xl bg-[rgba(15,15,25,0.8)] border border-violet-500/20 text-white placeholder-white/30 focus:border-violet-500/50 focus:outline-none"
        />
        <select
          value={filterSide}
          onChange={(e) => setFilterSide(e.target.value as "all" | "BUY" | "SELL")}
          className="px-4 py-2 rounded-xl bg-[rgba(15,15,25,0.8)] border border-violet-500/20 text-white focus:border-violet-500/50 focus:outline-none"
        >
          <option value="all">{t.trades.allTrades}</option>
          <option value="BUY">شراء فقط</option>
          <option value="SELL">بيع فقط</option>
        </select>
      </div>

      {/* Trades Table */}
      <Card className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border-violet-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-violet-400" />{t.trades.allTrades}</CardTitle>
        </CardHeader>
        <CardContent>
          {filteredTrades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-right text-white/50 border-b border-violet-500/20">
                    <th className="pb-3 font-medium">{t.trades.date}</th>
                    <th className="pb-3 font-medium">{t.trades.pair}</th>
                    <th className="pb-3 font-medium">{t.trades.type}</th>
                    <th className="pb-3 font-medium">{t.trades.quantity}</th>
                    <th className="pb-3 font-medium">{t.trades.price}</th>
                    <th className="pb-3 font-medium">الإجمالي</th>
                    <th className="pb-3 font-medium">{t.portfolio.profitLoss}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.map((trade) => (
                    <tr
                      key={trade.id}
                      className="border-b border-violet-500/10 hover:bg-violet-500/5 transition-colors"
                    >
                      <td className="py-3 text-white/50">
                        {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                      </td>
                      <td className="py-3 font-medium text-white">{trade.symbol}</td>
                      <td className="py-3">
                        <span className={cn(
                          "px-2 py-1 rounded-lg text-xs font-medium",
                          trade.side === "BUY"
                            ? "bg-emerald-500/20 text-emerald-400"
                            : "bg-red-500/20 text-red-400"
                        )}>
                          {trade.side === "BUY" ? "شراء" : "بيع"}
                        </span>
                      </td>
                      <td className="py-3 text-white/70" dir="ltr">{trade.quantity.toFixed(6)}</td>
                      <td className="py-3 text-white/70" dir="ltr">${trade.price.toFixed(2)}</td>
                      <td className="py-3 text-white/70" dir="ltr">${trade.total_value.toFixed(2)}</td>
                      <td className="py-3">
                        {trade.pnl !== undefined ? (
                          <span className={cn(
                            "font-medium",
                            trade.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                          )} dir="ltr">
                            {trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(2)}
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
              <BarChart3 className="w-16 h-16 text-violet-500/30 mx-auto mb-4" />
              <p className="text-white/50">{t.trades.noTrades}</p>
              <p className="text-white/30 text-sm mt-2">سيتم عرض الصفقات هنا عندما يبدأ وكيل التداول الذكي بالتداول</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
