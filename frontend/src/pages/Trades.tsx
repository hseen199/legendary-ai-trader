import React, { useEffect, useState } from 'react';
import { dashboardAPI } from '../services/api';
import { TrendingUp, TrendingDown, RefreshCw, BarChart3, Search, Filter } from 'lucide-react';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import { useLanguage } from '@/lib/i18n';

interface Trade {
  id: number;
  symbol: string;
  side: string;
  order_type: string;
  price: number;
  quantity: number;
  total_value: number;
  pnl?: number;
  pnl_percent?: number;
  executed_at: string;
}

const Trades: React.FC = () => {
  const { t, language } = useLanguage();
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  const fetchTrades = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await dashboardAPI.getTrades(100);
      setTrades(response.data || []);
    } catch (err: any) {
      console.error('Error fetching trades:', err);
      setError(t.trades.loadFailed);
      toast.error(t.trades.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  // Filter trades
  const filteredTrades = trades.filter((trade) => {
    const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || 
      (filterType === 'buy' && trade.side === 'BUY') ||
      (filterType === 'sell' && trade.side === 'SELL');
    return matchesSearch && matchesFilter;
  });

  // Calculate stats
  const totalTrades = trades.length;
  const winningTrades = trades.filter((t) => (t.pnl || 0) > 0).length;
  const losingTrades = trades.filter((t) => (t.pnl || 0) < 0).length;
  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-violet-500" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
          {t.trades.title}
        </h1>
        <button
          onClick={fetchTrades}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-violet-500/20 text-violet-400 hover:bg-violet-500/30 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          {t.common.refresh}
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-4 hover:border-violet-500/40 transition-all">
          <p className="text-white/50 text-sm">{t.trades.totalTrades}</p>
          <p className="text-2xl font-bold text-white mt-1">
            {totalTrades}
          </p>
        </div>
        <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-4 hover:border-violet-500/40 transition-all">
          <p className="text-white/50 text-sm">{t.trades.winRate}</p>
          <p className="text-2xl font-bold text-emerald-400 mt-1">
            {winRate.toFixed(1)}%
          </p>
        </div>
        <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-4 hover:border-violet-500/40 transition-all">
          <p className="text-white/50 text-sm">{t.trades.winningTrades}</p>
          <p className="text-2xl font-bold text-emerald-400 mt-1">{winningTrades}</p>
        </div>
        <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-4 hover:border-violet-500/40 transition-all">
          <p className="text-white/50 text-sm">{t.trades.losingTrades}</p>
          <p className="text-2xl font-bold text-red-400 mt-1">{losingTrades}</p>
        </div>
      </div>

      {/* Total PnL */}
      <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-6 mb-8">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white/50">{t.trades.netProfitLoss}</p>
            <p
              className={`text-3xl font-bold mt-1 ${
                totalPnL >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </p>
          </div>
          {totalPnL >= 0 ? (
            <div className="w-14 h-14 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <TrendingUp className="w-8 h-8 text-emerald-400" />
            </div>
          ) : (
            <div className="w-14 h-14 rounded-xl bg-red-500/20 flex items-center justify-center">
              <TrendingDown className="w-8 h-8 text-red-400" />
            </div>
          )}
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
          <input
            type="text"
            placeholder={t.trades.searchPair}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 rounded-xl bg-[rgba(15,15,25,0.8)] border border-violet-500/20 text-white placeholder-white/40 focus:border-violet-500/50 focus:outline-none transition-colors"
          />
        </div>
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value)}
          className="px-4 py-3 rounded-xl bg-[rgba(15,15,25,0.8)] border border-violet-500/20 text-white focus:border-violet-500/50 focus:outline-none transition-colors"
        >
          <option value="all">{t.trades.allTrades}</option>
          <option value="buy">{t.trades.buy}</option>
          <option value="sell">{t.trades.sell}</option>
        </select>
      </div>

      {/* Trades Table */}
      <div className="rounded-2xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-violet-400" />
          {t.trades.allTrades}
        </h2>
        
        {error ? (
          <div className="text-center py-12">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={fetchTrades}
              className="px-4 py-2 rounded-xl bg-violet-500/20 text-violet-400 hover:bg-violet-500/30 transition-colors"
            >
              {t.trades.retry}
            </button>
          </div>
        ) : filteredTrades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className={`text-white/50 border-b border-violet-500/20 ${language === 'ar' ? 'text-right' : 'text-left'}`}>
                  <th className="pb-3 font-medium">{t.trades.pair}</th>
                  <th className="pb-3 font-medium">{t.trades.type}</th>
                  <th className="pb-3 font-medium">{t.trades.price}</th>
                  <th className="pb-3 font-medium">{t.trades.quantity}</th>
                  <th className="pb-3 font-medium">{t.trades.total}</th>
                  <th className="pb-3 font-medium">{t.trades.profitLoss}</th>
                  <th className="pb-3 font-medium">{t.trades.date}</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrades.map((trade) => (
                  <tr
                    key={trade.id}
                    className="border-b border-violet-500/10 hover:bg-violet-500/5 transition-colors"
                  >
                    <td className="py-3 font-medium text-white">
                      {trade.symbol}
                    </td>
                    <td className="py-3">
                      <span
                        className={`px-2 py-1 rounded-lg text-xs font-medium ${
                          trade.side === 'BUY' 
                            ? 'bg-emerald-500/20 text-emerald-400' 
                            : 'bg-red-500/20 text-red-400'
                        }`}
                      >
                        {trade.side === 'BUY' ? t.trades.buy : t.trades.sell}
                      </span>
                    </td>
                    <td className="py-3 text-white/70">${trade.price.toFixed(4)}</td>
                    <td className="py-3 text-white/70">{trade.quantity.toFixed(4)}</td>
                    <td className="py-3 text-white/70">${trade.total_value.toFixed(2)}</td>
                    <td className="py-3">
                      {trade.pnl !== undefined ? (
                        <span
                          className={
                            trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }
                        >
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                          {trade.pnl_percent !== undefined && (
                            <span className="text-xs opacity-70 mx-1">
                              ({trade.pnl_percent >= 0 ? '+' : ''}
                              {trade.pnl_percent.toFixed(2)}%)
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="text-white/30">-</span>
                      )}
                    </td>
                    <td className="py-3 text-white/50">
                      {format(new Date(trade.executed_at), 'dd/MM/yyyy HH:mm')}
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
            <p className="text-white/30 text-sm mt-2">{t.trades.noTradesDesc}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Trades;
