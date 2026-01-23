import React, { useEffect, useState } from 'react';
import { walletAPI } from '../services/api';
import { Copy, Check, RefreshCw, AlertCircle, Info } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import toast from 'react-hot-toast';
import { format } from 'date-fns';
import { useLanguage } from '@/lib/i18n';

interface DepositAddress {
  address: string;
  network: string;
  coin: string;
}

interface DepositHistory {
  id: number;
  amount: number;
  coin: string;
  network: string;
  status: string;
  tx_hash?: string;
  units_received?: number;
  nav_at_deposit?: number;
  created_at: string;
  completed_at?: string;
}

const Deposit: React.FC = () => {
  const { t, language } = useLanguage();
  const [address, setAddress] = useState<DepositAddress | null>(null);
  const [history, setHistory] = useState<DepositHistory[]>([]);
  const [selectedNetwork, setSelectedNetwork] = useState('BEP20');
  const [isLoading, setIsLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  const networks = [
    { id: 'BEP20', name: 'BEP20 (BSC)', fee: 'رسوم منخفضة' },
    { id: 'SOL', name: 'Solana', fee: 'رسوم منخفضة' },
  ];

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [addressRes, historyRes] = await Promise.all([
        walletAPI.getDepositAddress(selectedNetwork),
        walletAPI.getDepositHistory(),
      ]);
      setAddress(addressRes.data);
      setHistory(historyRes.data);
    } catch (error) {
      toast.error(t.dashboard.loadFailed);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedNetwork]);

  const copyAddress = () => {
    if (address) {
      navigator.clipboard.writeText(address.address);
      setCopied(true);
      toast.success('تم نسخ العنوان');
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
        إيداع الأموال
      </h1>

      {/* Warning */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-6">
        <div className="flex gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-yellow-800 dark:text-yellow-200 font-medium">
              تنبيه مهم
            </p>
            <p className="text-yellow-700 dark:text-yellow-300 text-sm mt-1">
              أرسل فقط USDC إلى هذا العنوان. إرسال أي عملة أخرى قد يؤدي إلى فقدان الأموال.
            </p>
          </div>
        </div>
      </div>

      {/* Network Selection */}
      <div className="card p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{t.wallet.selectNetwork}</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {networks.map((network) => (
            <button
              key={network.id}
              onClick={() => setSelectedNetwork(network.id)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedNetwork === network.id
                  ? 'border-primary-600 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
            >
              <p className="font-medium text-gray-900 dark:text-white">
                {network.name}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {network.fee}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Deposit Address */}
      <div className="card p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          عنوان الإيداع
        </h2>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
          </div>
        ) : address ? (
          <div className="flex flex-col md:flex-row gap-6 items-center">
            {/* QR Code */}
            <div className="bg-white p-4 rounded-lg">
              <QRCodeSVG value={address.address} size={180} />
            </div>

            {/* Address */}
            <div className="flex-1 w-full">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                عنوان {address.coin} ({address.network})
              </p>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={address.address}
                  readOnly
                  className="input flex-1 font-mono text-sm"
                  dir="ltr"
                />
                <button
                  onClick={copyAddress}
                  className="btn-secondary flex items-center gap-2"
                >
                  {copied ? (
                    <Check className="w-4 h-4" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                  نسخ
                </button>
              </div>

              {/* Fee Info - رسالة واضحة عن الرسوم */}
              <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                <div className="flex gap-2">
                  <Info className="w-5 h-5 text-green-600 flex-shrink-0" />
                  <div className="text-sm text-green-700 dark:text-green-300">
                    <p className="font-semibold mb-1">💰 رسوم الإيداع: 1% فقط</p>
                    <p>مثال: إذا أودعت 100 USDC، سيتم خصم 1 USDC كرسوم وستحصل على 99 USDC صافي.</p>
                  </div>
                </div>
              </div>

              {/* Info */}
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex gap-2">
                  <Info className="w-5 h-5 text-blue-600 flex-shrink-0" />
                  <div className="text-sm text-blue-700 dark:text-blue-300">
                    <p>• الحد الأدنى للإيداع: 100 USDC</p>
                    <p>• سيتم إضافة الرصيد بعد 6 تأكيدات</p>
                    <p>• فترة القفل: 7 أيام من آخر إيداع</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-center text-gray-600 dark:text-gray-400 py-8">
            فشل في تحميل العنوان
          </p>
        )}
      </div>

      {/* Deposit History */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          سجل الإيداعات
        </h2>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">{t.wallet.amount}</th>
                  <th className="pb-3 font-medium">الشبكة</th>
                  <th className="pb-3 font-medium">الوحدات</th>
                  <th className="pb-3 font-medium">الحالة</th>
                  <th className="pb-3 font-medium">{t.trades.date}</th>
                </tr>
              </thead>
              <tbody>
                {history.map((deposit) => (
                  <tr
                    key={deposit.id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3">
                      {deposit.amount} {deposit.coin}
                    </td>
                    <td className="py-3">{deposit.network}</td>
                    <td className="py-3">
                      {deposit.units_received?.toFixed(4) || '-'}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          deposit.status === 'completed'
                            ? 'badge-success'
                            : deposit.status === 'pending'
                            ? 'badge-warning'
                            : 'badge-danger'
                        }`}
                      >
                        {deposit.status === 'completed'
                          ? 'مكتمل'
                          : deposit.status === 'pending'
                          ? 'قيد الانتظار'
                          : 'ملغي'}
                      </span>
                    </td>
                    <td className="py-3 text-gray-600 dark:text-gray-400">
                      {format(new Date(deposit.created_at), 'dd/MM/yyyy HH:mm')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center text-gray-600 dark:text-gray-400 py-8">
            لا توجد إيداعات حتى الآن
          </p>
        )}
      </div>
    </div>
  );
};

export default Deposit;
