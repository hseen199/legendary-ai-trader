import { useState, useEffect } from "react";
import { dashboardAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { 
  RefreshCw,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Search,
  Filter,
  Shield,
  Clock,
  Activity,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import { useLanguage } from '@/lib/i18n';

// واجهة الصفقة المفلترة (بدون معلومات حساسة)
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

// دالة للحصول على شعار العملة
const getCryptoLogo = (symbol: string) => {
  const coin = symbol.replace('USDC', '').replace('USDT', '').toLowerCase();
  return `https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c25fa5/128/color/${coin}.png`;
};

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

  // Calculate stats (بدون المبالغ الحساسة)
  const winningTrades = trades.filter((t) => (t.pnl_percent || 0) > 0);
  const losingTrades = trades.filter((t) => (t.pnl_percent || 0) < 0);
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
        <div className="relative">
          <RefreshCw className="w-10 h-10 animate-spin text-primary" />
          <div className="absolute inset-0 w-10 h-10 rounded-full border-2 border-primary/20 animate-ping" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header with Smart Transparency Badge */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent">
            {t.trades.title}
          </h1>
          {/* Smart Transparency Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-emerald-500/20 to-teal-500/20 border border-emerald-500/30">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">
              {language === 'ar' ? 'الشفافية الذكية' : 'Smart Transparency'}
            </span>
          </div>
        </div>
        <button
          onClick={fetchTrades}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-primary/10 to-purple-500/10 hover:from-primary/20 hover:to-purple-500/20 text-primary border border-primary/20 transition-all duration-300 hover:scale-105"
        >
          <RefreshCw className="w-4 h-4" />
          {t.common.refresh}
        </button>
      </div>

      {/* Smart Transparency Info Banner */}
      <Card className="border-0 bg-gradient-to-r from-emerald-500/10 via-teal-500/5 to-cyan-500/10 border border-emerald-500/20">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
              <Shield className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h3 className="font-semibold text-emerald-400 mb-1">
                {language === 'ar' ? 'نظام الشفافية الذكية' : 'Smart Transparency System'}
              </h3>
              <p className="text-sm text-muted-foreground">
                {language === 'ar' 
                  ? 'لحماية استراتيجية التداول وأمان المنصة، يتم عرض معلومات الصفقات بشكل محدود. الكميات والقيم الإجمالية مخفية للحفاظ على سرية حجم المحفظة.'
                  : 'To protect trading strategy and platform security, trade information is displayed in a limited way. Quantities and total values are hidden to maintain portfolio size confidentiality.'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards - بدون مبالغ حساسة */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-primary/5 via-transparent to-purple-500/5 hover:from-primary/10 hover:to-purple-500/10 transition-all duration-500 hover:scale-[1.02]">
          <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-4 md:p-5 relative">
            <p className="text-xs md:text-sm text-muted-foreground">{t.trades.totalTrades}</p>
            <p className="text-xl md:text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent mt-1">
              {trades.length}
            </p>
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-green-500/5 via-transparent to-emerald-500/5 hover:from-green-500/10 hover:to-emerald-500/10 transition-all duration-500 hover:scale-[1.02]">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/0 via-green-500/5 to-green-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-4 md:p-5 relative">
            <p className="text-xs md:text-sm text-muted-foreground">{t.trades.winRate}</p>
            <p className="text-xl md:text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mt-1">
              {winRate.toFixed(1)}%
            </p>
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-green-500/5 via-transparent to-emerald-500/5 hover:from-green-500/10 hover:to-emerald-500/10 transition-all duration-500 hover:scale-[1.02]">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/0 via-green-500/5 to-green-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-4 md:p-5 relative">
            <p className="text-xs md:text-sm text-muted-foreground">{t.trades.winningTrades}</p>
            <p className="text-xl md:text-3xl font-bold text-green-400 mt-1">{winningTrades.length}</p>
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-0 bg-gradient-to-br from-red-500/5 via-transparent to-rose-500/5 hover:from-red-500/10 hover:to-rose-500/10 transition-all duration-500 hover:scale-[1.02]">
          <div className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/5 to-red-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
          <CardContent className="p-4 md:p-5 relative">
            <p className="text-xs md:text-sm text-muted-foreground">{t.trades.losingTrades}</p>
            <p className="text-xl md:text-3xl font-bold text-red-400 mt-1">{losingTrades.length}</p>
          </CardContent>
        </Card>
      </div>

      {/* Activity Pulse - بديل عن Net Profit */}
      <Card className="relative overflow-hidden border-0 bg-gradient-to-br from-violet-500/10 via-purple-500/5 to-transparent">
        <CardContent className="p-5 md:p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                {language === 'ar' ? 'نبض النشاط' : 'Activity Pulse'}
              </p>
              <div className="flex items-center gap-3 mt-2">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-lg font-semibold text-green-400">
                    {language === 'ar' ? 'الوكيل نشط' : 'Agent Active'}
                  </span>
                </div>
                <span className="text-muted-foreground">•</span>
                <span className="text-sm text-muted-foreground">
                  {trades.length > 0 
                    ? `${language === 'ar' ? 'آخر صفقة:' : 'Last trade:'} ${format(new Date(trades[0]?.executed_at || new Date()), "dd/MM HH:mm")}`
                    : language === 'ar' ? 'لا توجد صفقات بعد' : 'No trades yet'}
                </span>
              </div>
            </div>
            <div className="w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-violet-500/30 to-purple-500/30 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <Activity className="w-7 h-7 md:w-8 md:h-8 text-violet-400" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder={language === 'ar' ? 'بحث عن زوج...' : 'Search pair...'}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pr-10 pl-4 py-2.5 rounded-xl bg-card/50 border border-border/50 text-foreground placeholder-muted-foreground focus:border-primary/50 focus:ring-2 focus:ring-primary/20 focus:outline-none transition-all duration-300"
          />
        </div>
        <div className="relative">
          <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <select
            value={filterSide}
            onChange={(e) => setFilterSide(e.target.value as "all" | "BUY" | "SELL")}
            className="pr-10 pl-4 py-2.5 rounded-xl bg-card/50 border border-border/50 text-foreground focus:border-primary/50 focus:ring-2 focus:ring-primary/20 focus:outline-none appearance-none cursor-pointer transition-all duration-300"
          >
            <option value="all">{t.trades.allTrades}</option>
            <option value="BUY">{language === 'ar' ? 'شراء فقط' : 'Buy Only'}</option>
            <option value="SELL">{language === 'ar' ? 'بيع فقط' : 'Sell Only'}</option>
          </select>
        </div>
      </div>

      {/* Trades List - بدون معلومات حساسة */}
      <Card className="border-0 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            {t.trades.allTrades}
            <span className="text-sm font-normal text-muted-foreground">({filteredTrades.length})</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0 md:p-4">
          {filteredTrades.length > 0 ? (
            <>
              {/* Desktop Table - بدون أعمدة الكمية والإجمالي */}
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-right text-muted-foreground border-b border-border/50">
                      <th className="pb-3 px-4 font-medium">{t.trades.date}</th>
                      <th className="pb-3 px-4 font-medium">{t.trades.pair}</th>
                      <th className="pb-3 px-4 font-medium">{t.trades.type}</th>
                      <th className="pb-3 px-4 font-medium">{t.trades.price}</th>
                      <th className="pb-3 px-4 font-medium">
                        {language === 'ar' ? 'نسبة الربح/الخسارة' : 'P/L %'}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTrades.map((trade, index) => (
                      <tr
                        key={trade.id}
                        className={cn(
                          "border-b border-border/30 transition-all duration-300 hover:scale-[1.01]",
                          trade.side === "BUY" 
                            ? "hover:bg-green-500/5" 
                            : "hover:bg-red-500/5"
                        )}
                        style={{ animationDelay: `${index * 50}ms` }}
                      >
                        <td className="py-3 px-4 text-muted-foreground text-sm">
                          {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <img 
                              src={getCryptoLogo(trade.symbol)} 
                              alt={trade.symbol}
                              className="w-6 h-6 rounded-full bg-white/10"
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                            <span className="font-medium">{trade.symbol.replace('USDC', '')}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span className={cn(
                            "px-3 py-1 rounded-lg text-xs font-semibold",
                            trade.side === "BUY"
                              ? "bg-green-500/20 text-green-400 border border-green-500/30"
                              : "bg-red-500/20 text-red-400 border border-red-500/30"
                          )}>
                            {trade.side === "BUY" ? (language === 'ar' ? "شراء" : "Buy") : (language === 'ar' ? "بيع" : "Sell")}
                          </span>
                        </td>
                        <td className="py-3 px-4" dir="ltr">${(trade.price || 0).toFixed(2)}</td>
                        <td className="py-3 px-4">
                          {trade.pnl_percent !== undefined && trade.pnl_percent !== null && trade.pnl_percent !== 0 ? (
                            <span className={cn(
                              "font-semibold",
                              trade.pnl_percent >= 0 ? "text-green-400" : "text-red-400"
                            )} dir="ltr">
                              {trade.pnl_percent >= 0 ? "+" : ""}{trade.pnl_percent.toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-muted-foreground/50">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Mobile Cards - بدون معلومات حساسة */}
              <div className="md:hidden space-y-3 p-4">
                {filteredTrades.map((trade, index) => (
                  <div
                    key={trade.id}
                    className={cn(
                      "p-4 rounded-xl transition-all duration-300",
                      trade.side === "BUY" 
                        ? "bg-gradient-to-r from-green-500/10 to-transparent border border-green-500/20" 
                        : "bg-gradient-to-r from-red-500/10 to-transparent border border-red-500/20"
                    )}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <img 
                          src={getCryptoLogo(trade.symbol)} 
                          alt={trade.symbol}
                          className="w-10 h-10 rounded-full bg-white/10 p-0.5"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                        <div>
                          <p className="font-semibold">{trade.symbol.replace('USDC', '')}</p>
                          <p className="text-xs text-muted-foreground">
                            {format(new Date(trade.executed_at), "dd/MM HH:mm")}
                          </p>
                        </div>
                      </div>
                      <span className={cn(
                        "px-3 py-1 rounded-lg text-xs font-semibold",
                        trade.side === "BUY"
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                      )}>
                        {trade.side === "BUY" ? (language === 'ar' ? "شراء" : "Buy") : (language === 'ar' ? "بيع" : "Sell")}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground text-xs">{t.trades.price}</p>
                        <p dir="ltr">${(trade.price || 0).toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">
                          {language === 'ar' ? 'نسبة الربح/الخسارة' : 'P/L %'}
                        </p>
                        {trade.pnl_percent !== undefined && trade.pnl_percent !== null && trade.pnl_percent !== 0 ? (
                          <p className={cn(
                            "font-semibold",
                            trade.pnl_percent >= 0 ? "text-green-400" : "text-red-400"
                          )} dir="ltr">
                            {trade.pnl_percent >= 0 ? "+" : ""}{trade.pnl_percent.toFixed(2)}%
                          </p>
                        ) : (
                          <p className="text-muted-foreground/50">-</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-16">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <BarChart3 className="w-10 h-10 text-primary/50" />
              </div>
              <p className="text-muted-foreground">{t.trades.noTrades}</p>
              <p className="text-muted-foreground/50 text-sm mt-2">
                {language === 'ar' 
                  ? 'سيتم عرض الصفقات هنا عندما يبدأ وكيل التداول الذكي بالتداول'
                  : 'Trades will appear here when the AI trading agent starts trading'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
