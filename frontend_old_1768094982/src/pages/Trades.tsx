import React, { useEffect, useState } from 'react';
import { dashboardAPI } from '../services/api';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

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
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchTrades = async () => {
    try {
      const response = await dashboardAPI.getTrades(100);
      setTrades(response.data);
    } catch (error) {
      toast.error('فشل في تحميل الصفقات');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  // Calculate stats
  const totalTrades = trades.length;
  const winningTrades = trades.filter((t) => (t.pnl || 0) > 0).length;
  const losingTrades = trades.filter((t) => (t.pnl || 0) < 0).length;
  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
        سجل الصفقات
      </h1>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card p-4">
          <p className="text-gray-600 dark:text-gray-400 text-sm">إجمالي الصفقات</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {totalTrades}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-gray-600 dark:text-gray-400 text-sm">نسبة الربح</p>
          <p className="text-2xl font-bold text-green-600">
            {winRate.toFixed(1)}%
          </p>
        </div>
        <div className="card p-4">
          <p className="text-gray-600 dark:text-gray-400 text-sm">صفقات رابحة</p>
          <p className="text-2xl font-bold text-green-600">{winningTrades}</p>
        </div>
        <div className="card p-4">
          <p className="text-gray-600 dark:text-gray-400 text-sm">صفقات خاسرة</p>
          <p className="text-2xl font-bold text-red-600">{losingTrades}</p>
        </div>
      </div>

      {/* Total PnL */}
      <div className="card p-6 mb-8">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-600 dark:text-gray-400">إجمالي الربح/الخسارة</p>
            <p
              className={`text-3xl font-bold ${
                totalPnL >= 0 ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </p>
          </div>
          {totalPnL >= 0 ? (
            <TrendingUp className="w-12 h-12 text-green-600" />
          ) : (
            <TrendingDown className="w-12 h-12 text-red-600" />
          )}
        </div>
      </div>

      {/* Trades Table */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          جميع الصفقات
        </h2>

        {trades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">الزوج</th>
                  <th className="pb-3 font-medium">النوع</th>
                  <th className="pb-3 font-medium">السعر</th>
                  <th className="pb-3 font-medium">الكمية</th>
                  <th className="pb-3 font-medium">القيمة</th>
                  <th className="pb-3 font-medium">الربح/الخسارة</th>
                  <th className="pb-3 font-medium">التاريخ</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade) => (
                  <tr
                    key={trade.id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3 font-medium text-gray-900 dark:text-white">
                      {trade.symbol}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          trade.side === 'BUY' ? 'badge-success' : 'badge-danger'
                        }`}
                      >
                        {trade.side === 'BUY' ? 'شراء' : 'بيع'}
                      </span>
                    </td>
                    <td className="py-3">${trade.price.toFixed(4)}</td>
                    <td className="py-3">{trade.quantity.toFixed(4)}</td>
                    <td className="py-3">${trade.total_value.toFixed(2)}</td>
                    <td className="py-3">
                      {trade.pnl !== undefined ? (
                        <span
                          className={
                            trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                          }
                        >
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                          {trade.pnl_percent !== undefined && (
                            <span className="text-xs ml-1">
                              ({trade.pnl_percent >= 0 ? '+' : ''}
                              {trade.pnl_percent.toFixed(2)}%)
                            </span>
                          )}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="py-3 text-gray-600 dark:text-gray-400">
                      {format(new Date(trade.executed_at), 'dd/MM/yyyy HH:mm')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center text-gray-600 dark:text-gray-400 py-8">
            لا توجد صفقات حتى الآن
          </p>
        )}
      </div>
    </div>
  );
};

export default Trades;
