import React, { useEffect, useState } from 'react';
import { dashboardAPI } from '../services/api';
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
} from 'lucide-react';
import { format } from 'date-fns';
import { ar } from 'date-fns/locale';
import toast from 'react-hot-toast';
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
  status?: string;
  executed_at: string;
  agent_name?: string;
}

interface TradeStats {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_profit: number;
  total_loss: number;
  net_profit: number;
  average_profit: number;
  average_loss: number;
  best_trade: number;
  worst_trade: number;
}

const TradesAdvanced: React.FC = () => {
  const { t, language } = useLanguage();
  const [trades, setTrades] = useState<Trade[]>([]);
  const [stats, setStats] = useState<TradeStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSide, setFilterSide] = useState<'all' | 'BUY' | 'SELL'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'profit' | 'amount'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [dateRange, setDateRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const itemsPerPage = 20;

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const days = dateRange === '7d' ? 7 : dateRange === '30d' ? 30 : dateRange === '90d' ? 90 : 1000;
      const response = await dashboardAPI.getTrades(days * 10); // Get more trades
      const tradesData = response.data || [];
      setTrades(tradesData);
      
      // Calculate stats from trades
      const winningTrades = tradesData.filter((t: Trade) => (t.pnl || 0) > 0);
      const losingTrades = tradesData.filter((t: Trade) => (t.pnl || 0) < 0);
      
      const totalProfit = winningTrades.reduce((sum: number, t: Trade) => sum + (t.pnl || 0), 0);
      const totalLoss = Math.abs(losingTrades.reduce((sum: number, t: Trade) => sum + (t.pnl || 0), 0));
      
      setStats({
        total_trades: tradesData.length,
        winning_trades: winningTrades.length,
        losing_trades: losingTrades.length,
        win_rate: tradesData.length > 0 ? (winningTrades.length / tradesData.length) * 100 : 0,
        total_profit: totalProfit,
        total_loss: totalLoss,
        net_profit: totalProfit - totalLoss,
        average_profit: winningTrades.length > 0 ? totalProfit / winningTrades.length : 0,
        average_loss: losingTrades.length > 0 ? totalLoss / losingTrades.length : 0,
        best_trade: Math.max(...tradesData.map((t: Trade) => t.pnl || 0), 0),
        worst_trade: Math.min(...tradesData.map((t: Trade) => t.pnl || 0), 0),
      });
    } catch (error) {
      toast.error(t.trades.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [dateRange]);

  // Filter and sort trades
  const filteredTrades = trades
    .filter((trade) => {
      const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesSide = filterSide === 'all' || trade.side === filterSide;
      return matchesSearch && matchesSide;
    })
    .sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'date') {
        comparison = new Date(a.executed_at).getTime() - new Date(b.executed_at).getTime();
      } else if (sortBy === 'profit') {
        comparison = (a.pnl || 0) - (b.pnl || 0);
      } else if (sortBy === 'amount') {
        comparison = a.total_value - b.total_value;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / itemsPerPage);
  const paginatedTrades = filteredTrades.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const exportTrades = () => {
    const csvContent = [
      ['التاريخ', 'الزوج', 'النوع', 'الكمية', 'السعر', 'الإجمالي', 'الربح/الخسارة'],
      ...filteredTrades.map((trade) => [
        format(new Date(trade.executed_at), 'yyyy-MM-dd HH:mm'),
        trade.symbol,
        trade.side === 'BUY' ? 'شراء' : 'بيع',
        trade.quantity.toString(),
        trade.price.toString(),
        trade.total_value.toString(),
        (trade.pnl || 0).toString(),
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `trades_${format(new Date(), 'yyyy-MM-dd')}.csv`;
    link.click();
    toast.success('تم تصدير الصفقات بنجاح');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{t.trades.title}</h1>
          <p className="text-gray-600 dark:text-gray-400">
            جميع صفقات التداول التي نفذها وكيل التداول الذكي
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={fetchData}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />{t.common.refresh}</button>
          <button
            onClick={exportTrades}
            className="btn-primary flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            تصدير CSV
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-primary-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">{t.trades.totalTrades}</span>
            </div>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{stats.total_trades}</p>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-green-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">نسبة النجاح</span>
            </div>
            <p className="text-xl font-bold text-green-600">{stats.win_rate.toFixed(1)}%</p>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">{t.trades.winningTrades}</span>
            </div>
            <p className="text-xl font-bold text-green-600">{stats.winning_trades}</p>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-red-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">{t.trades.losingTrades}</span>
            </div>
            <p className="text-xl font-bold text-red-600">{stats.losing_trades}</p>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Percent className="w-4 h-4 text-primary-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">صافي الربح</span>
            </div>
            <p className={`text-xl font-bold ${stats.net_profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${stats.net_profit.toFixed(2)}
            </p>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-yellow-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">أفضل صفقة</span>
            </div>
            <p className="text-xl font-bold text-green-600">+${stats.best_trade.toFixed(2)}</p>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="card p-4 mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="بحث عن زوج..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input pr-10"
            />
          </div>

          {/* Date Range */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-400" />
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value as any)}
              className="input py-2"
            >
              <option value="7d">آخر 7 أيام</option>
              <option value="30d">آخر 30 يوم</option>
              <option value="90d">آخر 3 أشهر</option>
              <option value="all">{t.common.all}</option>
            </select>
          </div>

          {/* Side Filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={filterSide}
              onChange={(e) => setFilterSide(e.target.value as any)}
              className="input py-2"
            >
              <option value="all">جميع الأنواع</option>
              <option value="BUY">{t.trades.buy}</option>
              <option value="SELL">{t.trades.sell}</option>
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <ArrowUpDown className="w-4 h-4 text-gray-400" />
            <select
              value={`${sortBy}-${sortOrder}`}
              onChange={(e) => {
                const [by, order] = e.target.value.split('-');
                setSortBy(by as any);
                setSortOrder(order as any);
              }}
              className="input py-2"
            >
              <option value="date-desc">الأحدث أولاً</option>
              <option value="date-asc">الأقدم أولاً</option>
              <option value="profit-desc">الأعلى ربحاً</option>
              <option value="profit-asc">الأعلى خسارة</option>
              <option value="amount-desc">الأعلى قيمة</option>
            </select>
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="card overflow-hidden">
        {paginatedTrades.length > 0 ? (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr className="text-right text-gray-600 dark:text-gray-400">
                    <th className="px-4 py-3 font-medium">{t.trades.date}</th>
                    <th className="px-4 py-3 font-medium">{t.trades.pair}</th>
                    <th className="px-4 py-3 font-medium">{t.trades.type}</th>
                    <th className="px-4 py-3 font-medium">{t.trades.quantity}</th>
                    <th className="px-4 py-3 font-medium">{t.trades.price}</th>
                    <th className="px-4 py-3 font-medium">الإجمالي</th>
                    <th className="px-4 py-3 font-medium">{t.portfolio.profitLoss}</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedTrades.map((trade) => (
                    <tr
                      key={trade.id}
                      className="border-t border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                    >
                      <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                        {format(new Date(trade.executed_at), 'dd/MM HH:mm')}
                      </td>
                      <td className="px-4 py-3 font-medium text-gray-900 dark:text-white">
                        {trade.symbol}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
                            trade.side === 'BUY'
                              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                          }`}
                        >
                          {trade.side === 'BUY' ? (
                            <>
                              <TrendingUp className="w-3 h-3" />{t.trades.buy}</>
                          ) : (
                            <>
                              <TrendingDown className="w-3 h-3" />{t.trades.sell}</>
                          )}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-900 dark:text-white">
                        {trade.quantity.toFixed(6)}
                      </td>
                      <td className="px-4 py-3 text-gray-900 dark:text-white">
                        ${trade.price.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 font-medium text-gray-900 dark:text-white">
                        ${trade.total_value.toFixed(2)}
                      </td>
                      <td className="px-4 py-3">
                        {trade.pnl !== undefined ? (
                          <span
                            className={`font-medium ${
                              trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}
                          >
                            {trade.pnl >= 0 ? '+' : ''}
                            ${trade.pnl.toFixed(2)}
                            {trade.pnl_percent !== undefined && (
                              <span className="text-xs mr-1">
                                ({trade.pnl_percent >= 0 ? '+' : ''}
                                {trade.pnl_percent.toFixed(1)}%)
                              </span>
                            )}
                          </span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100 dark:border-gray-800">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  عرض {(currentPage - 1) * itemsPerPage + 1} إلى{' '}
                  {Math.min(currentPage * itemsPerPage, filteredTrades.length)} من{' '}
                  {filteredTrades.length} صفقة
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    صفحة {currentPage} من {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-12">
            <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">{t.trades.noTrades}</p>
            <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
              سيتم عرض الصفقات هنا عندما يبدأ وكيل التداول الذكي بالتداول
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradesAdvanced;
