import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { format } from "date-fns";
import { 
  Wallet, 
  ArrowDownCircle, 
  ArrowUpCircle, 
  Copy, 
  Check, 
  RefreshCw,
  Eye,
  EyeOff,
  Bell,
  AlertCircle,
  Clock,
  TrendingUp,
  X,
  QrCode,
  Info
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import api from "@/services/api";

// Constants
const MIN_DEPOSIT = 100;
const MIN_WITHDRAWAL = 50;
const DEPOSIT_FEE_PERCENT = 1; // رسوم الإيداع 1%

const networks = [
  { id: "bsc", name: "BNB Smart Chain (BEP20)", coin: "usdcbsc" },
  { id: "sol", name: "Solana", coin: "usdcsol" },
];

// نوع بيانات الإيداع
interface DepositData {
  payment_id: number;
  payment_status: string;
  pay_address: string;
  pay_amount: number;
  pay_currency: string;
  price_amount: number;
  price_currency: string;
  order_id: string;
  created_at: string;
  expiration_estimate_date?: string;
  // معلومات الرسوم
  original_amount?: number;  // المبلغ الأصلي بدون رسوم
  fee_amount?: number;  // مبلغ الرسوم
  fee_percentage?: number;  // نسبة الرسوم (1%)
}

export default function WalletNew() {
  const queryClient = useQueryClient();
  const [language, setLanguage] = useState<"ar" | "en">("ar");
  const [depositAmount, setDepositAmount] = useState("");
  const [selectedNetwork, setSelectedNetwork] = useState("bsc");
  const [withdrawAmount, setWithdrawAmount] = useState("");
  const [withdrawNetwork, setWithdrawNetwork] = useState("bsc");
  const [withdrawAddress, setWithdrawAddress] = useState("");
  const [copied, setCopied] = useState(false);
  
  // حالة نافذة عنوان الإيداع
  const [showDepositModal, setShowDepositModal] = useState(false);
  const [depositData, setDepositData] = useState<DepositData | null>(null);
  const [addressCopied, setAddressCopied] = useState(false);
  
  // ميزة إظهار/إخفاء الرصيد
  const [showBalance, setShowBalance] = useState(() => {
    const saved = localStorage.getItem("showBalance");
    return saved !== null ? JSON.parse(saved) : true;
  });

  // حفظ تفضيل إظهار/إخفاء الرصيد
  useEffect(() => {
    localStorage.setItem("showBalance", JSON.stringify(showBalance));
  }, [showBalance]);

  // Detect language
  useEffect(() => {
    const savedLang = localStorage.getItem("language") || "ar";
    setLanguage(savedLang as "ar" | "en");
  }, []);

  // Fetch balance - تحديث تلقائي كل 30 ثانية
  const { data: balance, isLoading: loadingBalance, refetch: refetchBalance } = useQuery({
    queryKey: ["balance"],
    queryFn: async () => {
      const res = await api.get("/wallet/balance");
      return res.data;
    },
    refetchInterval: 30000, // تحديث كل 30 ثانية
    staleTime: 10000,
  });

  // Fetch NAV history
  const { data: navData } = useQuery({
    queryKey: ["nav-current"],
    queryFn: async () => {
      const res = await api.get("/dashboard/nav/current");
      return res.data;
    },
    refetchInterval: 60000,
  });

  // Fetch transactions
  const { data: transactions = [], isLoading: loadingTx } = useQuery({
    queryKey: ["transactions"],
    queryFn: async () => {
      const res = await api.get("/wallet/transactions?limit=20");
      return res.data;
    },
    refetchInterval: 30000,
  });

  // Fetch pending withdrawals
  const { data: pendingWithdrawals = [] } = useQuery({
    queryKey: ["pending-withdrawals"],
    queryFn: async () => {
      const res = await api.get("/wallet/withdraw/history");
      return res.data.filter((w: any) => w.status === "pending_approval" || w.status === "approved");
    },
    refetchInterval: 30000,
  });

  // Deposit mutation - محدث لعرض النافذة المنبثقة
  const depositMutation = useMutation({
    mutationFn: async (data: { amount: number; currency: string }) => {
      const res = await api.post("/deposits/create", data);
      return res.data;
    },
    onSuccess: (data: DepositData) => {
      // حفظ بيانات الإيداع وعرض النافذة
      setDepositData(data);
      setShowDepositModal(true);
      setDepositAmount("");
      queryClient.invalidateQueries({ queryKey: ["transactions"] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || (language === "ar" ? "فشل إنشاء طلب الإيداع" : "Failed to create deposit"));
    },
  });

  // Withdraw mutation
  const withdrawMutation = useMutation({
    mutationFn: async (data: { amount: number; network: string; to_address: string; coin: string }) => {
      const res = await api.post("/wallet/withdraw", data);
      return res.data;
    },
    onSuccess: () => {
      toast.success(language === "ar" ? "تم إرسال طلب السحب للمراجعة" : "Withdrawal request submitted for review");
      setWithdrawAmount("");
      setWithdrawAddress("");
      queryClient.invalidateQueries({ queryKey: ["pending-withdrawals"] });
      queryClient.invalidateQueries({ queryKey: ["transactions"] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || (language === "ar" ? "فشل إرسال طلب السحب" : "Failed to submit withdrawal"));
    },
  });

  const handleDeposit = (e: React.FormEvent) => {
    e.preventDefault();
    const amount = parseFloat(depositAmount);
    if (amount < MIN_DEPOSIT) {
      toast.error(language === "ar" ? `الحد الأدنى للإيداع ${MIN_DEPOSIT} USDC` : `Minimum deposit is ${MIN_DEPOSIT} USDC`);
      return;
    }
    const network = networks.find(n => n.id === selectedNetwork);
    depositMutation.mutate({ 
      amount, 
      currency: network?.coin || "usdcbsc"
    });
  };

  const handleWithdraw = (e: React.FormEvent) => {
    e.preventDefault();
    const amount = parseFloat(withdrawAmount);
    if (amount < MIN_WITHDRAWAL) {
      toast.error(language === "ar" ? `الحد الأدنى للسحب ${MIN_WITHDRAWAL} USDC` : `Minimum withdrawal is ${MIN_WITHDRAWAL} USDC`);
      return;
    }
    if (!withdrawAddress) {
      toast.error(language === "ar" ? "الرجاء إدخال عنوان المحفظة" : "Please enter wallet address");
      return;
    }
    const network = networks.find(n => n.id === withdrawNetwork);
    withdrawMutation.mutate({ 
      amount, 
      network: withdrawNetwork,
      to_address: withdrawAddress,
      coin: network?.coin || "usdcbsc"
    });
  };

  // نسخ عنوان الإيداع
  const copyDepositAddress = () => {
    if (depositData?.pay_address) {
      navigator.clipboard.writeText(depositData.pay_address);
      setAddressCopied(true);
      setTimeout(() => setAddressCopied(false), 2000);
      toast.success(language === "ar" ? "تم نسخ العنوان" : "Address copied");
    }
  };

  // إغلاق نافذة الإيداع
  const closeDepositModal = () => {
    setShowDepositModal(false);
    setDepositData(null);
    setAddressCopied(false);
  };

  const formatCurrency = (value: number) => {
    if (!showBalance) return "****";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatUnits = (value: number) => {
    if (!showBalance) return "****";
    return (value || 0).toFixed(4);
  };

  const getStatusBadge = (status: string) => {
    const statusConfig: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
      pending: { label: language === "ar" ? "قيد الانتظار" : "Pending", variant: "secondary" },
      pending_approval: { label: language === "ar" ? "بانتظار الموافقة" : "Pending Approval", variant: "secondary" },
      approved: { label: language === "ar" ? "تمت الموافقة" : "Approved", variant: "default" },
      processing: { label: language === "ar" ? "قيد المعالجة" : "Processing", variant: "default" },
      completed: { label: language === "ar" ? "مكتمل" : "Completed", variant: "default" },
      rejected: { label: language === "ar" ? "مرفوض" : "Rejected", variant: "destructive" },
      failed: { label: language === "ar" ? "فشل" : "Failed", variant: "destructive" },
    };
    const config = statusConfig[status] || { label: status, variant: "outline" as const };
    return <Badge variant={config.variant} className="text-xs">{config.label}</Badge>;
  };

  // تنسيق تاريخ انتهاء الصلاحية
  const formatExpiration = (dateStr?: string) => {
    if (!dateStr) return null;
    try {
      const date = new Date(dateStr);
      return format(date, "dd/MM/yyyy HH:mm");
    } catch {
      return null;
    }
  };

  const currentBalance = balance?.current_value_usd || balance?.balance_usd || 0;
  const canWithdraw = balance && currentBalance >= MIN_WITHDRAWAL;
  // Use NAV from balance API (more reliable) or fallback to navData
  const currentNAV = balance?.nav || navData?.nav_value || 1.0;

  return (
    <div className="container mx-auto p-4 sm:p-6 space-y-6">
      {/* نافذة عنوان الإيداع المنبثقة - محسنة */}
      {showDepositModal && depositData && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <div className="bg-[#1a1a2e] rounded-2xl p-4 sm:p-6 max-w-md w-full border border-purple-500/30 shadow-2xl max-h-[90vh] overflow-y-auto">
            {/* رأس النافذة */}
            <div className="flex items-center justify-between mb-4 sm:mb-6">
              <div className="flex items-center gap-2 sm:gap-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 flex items-center justify-center">
                  <Wallet className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400" />
                </div>
                <h2 className="text-lg sm:text-xl font-bold text-white">
                  {language === "ar" ? "عنوان الإيداع" : "Deposit Address"}
                </h2>
              </div>
              <button 
                onClick={closeDepositModal}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* معلومات الإيداع */}
            <div className="space-y-3 sm:space-y-4">
              {/* المبلغ المطلوب إرساله */}
              <div className="p-3 sm:p-4 bg-purple-500/10 rounded-xl border border-purple-500/20">
                <p className="text-xs sm:text-sm text-gray-400 mb-1">
                  {language === "ar" ? "المبلغ المطلوب إرساله" : "Amount to Send"}
                </p>
                <p className="text-xl sm:text-2xl font-bold text-purple-400" dir="ltr">
                  {depositData.pay_amount.toFixed(6)} {depositData.pay_currency.toUpperCase()}
                </p>
                <p className="text-xs text-gray-500 mt-1" dir="ltr">
                  ≈ ${depositData.price_amount.toFixed(2)} USD
                </p>
              </div>

              {/* تفاصيل الرسوم المحسنة */}
              <div className="p-3 sm:p-4 bg-blue-500/10 rounded-xl border border-blue-500/20">
                <div className="flex items-start gap-2 mb-3">
                  <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                  <p className="text-sm font-medium text-blue-300">
                    {language === "ar" ? "تفاصيل الرسوم:" : "Fee Details:"}
                  </p>
                </div>
                <div className="space-y-2 text-xs sm:text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">
                      {language === "ar" ? "المبلغ الأصلي:" : "Original Amount:"}
                    </span>
                    <span className="text-white font-medium" dir="ltr">
                      ${depositData.original_amount?.toFixed(2) || (depositData.price_amount / 1.01).toFixed(2)} USD
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">
                      {language === "ar" ? `رسوم المنصة + الشبكة (${depositData.fee_percentage || DEPOSIT_FEE_PERCENT}%):` : `Platform + Network Fees (${depositData.fee_percentage || DEPOSIT_FEE_PERCENT}%):`}
                    </span>
                    <span className="text-yellow-400 font-medium" dir="ltr">
                      ${depositData.fee_amount?.toFixed(2) || (depositData.price_amount - depositData.price_amount / 1.01).toFixed(2)} USD
                    </span>
                  </div>
                  <div className="border-t border-blue-500/30 pt-2 mt-2">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-medium">
                        {language === "ar" ? "الإجمالي المطلوب:" : "Total Required:"}
                      </span>
                      <span className="text-green-400 font-bold" dir="ltr">
                        ${depositData.price_amount.toFixed(2)} USD
                      </span>
                    </div>
                  </div>
                  <div className="bg-green-500/10 p-2 rounded-lg mt-2 border border-green-500/20">
                    <p className="text-green-400 text-center text-xs">
                      {language === "ar" 
                        ? `سيُضاف لرصيدك: $${depositData.original_amount?.toFixed(2) || (depositData.price_amount / 1.01).toFixed(2)} USD`
                        : `Will be added to your balance: $${depositData.original_amount?.toFixed(2) || (depositData.price_amount / 1.01).toFixed(2)} USD`
                      }
                    </p>
                  </div>
                </div>
              </div>

              {/* عنوان الإيداع */}
              <div className="p-3 sm:p-4 bg-[#252542] rounded-xl border border-gray-700">
                <p className="text-xs sm:text-sm text-gray-400 mb-2">
                  {language === "ar" ? "أرسل إلى هذا العنوان" : "Send to this address"}
                </p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-xs sm:text-sm font-mono bg-[#1a1a2e] text-green-400 p-2 sm:p-3 rounded-lg border border-gray-700 break-all select-all" dir="ltr">
                    {depositData.pay_address}
                  </code>
                  <button
                    onClick={copyDepositAddress}
                    className="p-2 sm:p-3 rounded-lg bg-purple-500 text-white hover:bg-purple-600 transition-colors shrink-0"
                    title={language === "ar" ? "نسخ العنوان" : "Copy address"}
                  >
                    {addressCopied ? <Check className="w-4 h-4 sm:w-5 sm:h-5" /> : <Copy className="w-4 h-4 sm:w-5 sm:h-5" />}
                  </button>
                </div>
              </div>

              {/* الشبكة */}
              <div className="flex items-center justify-between p-3 bg-[#252542] rounded-lg">
                <span className="text-xs sm:text-sm text-gray-400">
                  {language === "ar" ? "الشبكة" : "Network"}
                </span>
                <Badge variant="outline" className="bg-purple-500/20 text-purple-300 border-purple-500/30 text-xs">
                  {depositData.pay_currency === "usdcbsc" ? "BNB Smart Chain (BEP20)" : 
                   depositData.pay_currency === "usdcsol" ? "Solana" : 
                   depositData.pay_currency.toUpperCase()}
                </Badge>
              </div>

              {/* تاريخ انتهاء الصلاحية */}
              {depositData.expiration_estimate_date && (
                <div className="flex items-center justify-between p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
                  <span className="text-xs sm:text-sm text-yellow-500">
                    {language === "ar" ? "ينتهي في" : "Expires at"}
                  </span>
                  <span className="text-xs sm:text-sm font-medium text-yellow-500" dir="ltr">
                    {formatExpiration(depositData.expiration_estimate_date)}
                  </span>
                </div>
              )}

              {/* تحذيرات */}
              <div className="p-3 sm:p-4 bg-red-500/10 rounded-xl border border-red-500/20">
                <div className="flex items-start gap-2 sm:gap-3">
                  <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 text-red-400 shrink-0 mt-0.5" />
                  <div className="text-xs sm:text-sm text-red-300 space-y-1">
                    <p className="font-medium">
                      {language === "ar" ? "تنبيهات مهمة:" : "Important warnings:"}
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>
                        {language === "ar" 
                          ? "أرسل فقط USDC على الشبكة المحددة" 
                          : "Only send USDC on the specified network"}
                      </li>
                      <li>
                        {language === "ar" 
                          ? "إرسال عملة أخرى قد يؤدي لفقدان الأموال" 
                          : "Sending other tokens may result in loss of funds"}
                      </li>
                      <li>
                        {language === "ar" 
                          ? "سيتم إضافة الرصيد تلقائياً بعد التأكيد" 
                          : "Balance will be added automatically after confirmation"}
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* زر إغلاق */}
              <button
                onClick={closeDepositModal}
                className="w-full py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
              >
                {language === "ar" ? "تم، سأقوم بالإرسال" : "Done, I will send"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header with Balance Toggle */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl sm:text-2xl font-bold">
          {language === "ar" ? "المحفظة" : "Wallet"}
        </h1>
        <div className="flex items-center gap-2 sm:gap-3">
          {/* زر تحديث الرصيد */}
          <button
            onClick={() => refetchBalance()}
            className="p-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
            title={language === "ar" ? "تحديث الرصيد" : "Refresh Balance"}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          {/* زر إظهار/إخفاء الرصيد */}
          <button
            onClick={() => setShowBalance(!showBalance)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
          >
            {showBalance ? (
              <>
                <EyeOff className="w-4 h-4" />
                <span className="text-sm hidden sm:inline">{language === "ar" ? "إخفاء" : "Hide"}</span>
              </>
            ) : (
              <>
                <Eye className="w-4 h-4" />
                <span className="text-sm hidden sm:inline">{language === "ar" ? "إظهار" : "Show"}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Pending Withdrawals Alert */}
      {pendingWithdrawals.length > 0 && (
        <Card className="border-yellow-500/50 bg-yellow-500/10">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-yellow-500" />
              <div>
                <p className="font-medium text-yellow-500">
                  {language === "ar" 
                    ? `لديك ${pendingWithdrawals.length} طلب سحب قيد المراجعة`
                    : `You have ${pendingWithdrawals.length} pending withdrawal request(s)`
                  }
                </p>
                <p className="text-sm text-muted-foreground">
                  {language === "ar" 
                    ? "سيتم إشعارك عند الموافقة أو الرفض"
                    : "You will be notified when approved or rejected"
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Balance Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
        {/* الرصيد الحالي */}
        <Card className="bg-gradient-to-br from-primary/10 to-primary/5">
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-muted-foreground">
                {language === "ar" ? "الرصيد الحالي" : "Current Balance"}
              </span>
              <Wallet className="w-4 h-4 text-primary" />
            </div>
            {loadingBalance ? (
              <Skeleton className="h-6 sm:h-8 w-20 sm:w-24" />
            ) : (
              <p className="text-lg sm:text-2xl font-bold" dir="ltr">
                {formatCurrency(currentBalance)}
              </p>
            )}
          </CardContent>
        </Card>

        {/* الوحدات */}
        <Card>
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-muted-foreground">
                {language === "ar" ? "الوحدات" : "Units"}
              </span>
              <TrendingUp className="w-4 h-4 text-green-500" />
            </div>
            {loadingBalance ? (
              <Skeleton className="h-6 sm:h-8 w-20 sm:w-24" />
            ) : (
              <p className="text-lg sm:text-2xl font-bold" dir="ltr">
                {formatUnits(balance?.units || 0)}
              </p>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              NAV: ${(currentNAV || 0).toFixed(4)}
            </p>
          </CardContent>
        </Card>

        {/* إجمالي الإيداعات */}
        <Card>
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-muted-foreground">
                {language === "ar" ? "إجمالي الإيداعات" : "Total Deposited"}
              </span>
              <ArrowDownCircle className="w-4 h-4 text-green-500" />
            </div>
            {loadingBalance ? (
              <Skeleton className="h-6 sm:h-8 w-20 sm:w-24" />
            ) : (
              <p className="text-lg sm:text-2xl font-bold text-green-500" dir="ltr">
                {formatCurrency(balance?.total_deposited || 0)}
              </p>
            )}
          </CardContent>
        </Card>

        {/* إجمالي السحوبات */}
        <Card>
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs sm:text-sm text-muted-foreground">
                {language === "ar" ? "إجمالي السحوبات" : "Total Withdrawn"}
              </span>
              <ArrowUpCircle className="w-4 h-4 text-destructive" />
            </div>
            {loadingBalance ? (
              <Skeleton className="h-6 sm:h-8 w-20 sm:w-24" />
            ) : (
              <p className="text-lg sm:text-2xl font-bold text-destructive" dir="ltr">
                {formatCurrency(balance?.total_withdrawn || 0)}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Deposit/Withdraw Tabs */}
        <div>
          <Card>
            <Tabs defaultValue="deposit">
              <CardHeader className="pb-0">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="deposit" className="gap-2">
                    <ArrowDownCircle className="w-4 h-4" />
                    {language === "ar" ? "إيداع" : "Deposit"}
                  </TabsTrigger>
                  <TabsTrigger value="withdraw" className="gap-2">
                    <ArrowUpCircle className="w-4 h-4" />
                    {language === "ar" ? "سحب" : "Withdraw"}
                  </TabsTrigger>
                </TabsList>
              </CardHeader>
              <CardContent className="pt-6">
                {/* Deposit Tab */}
                <TabsContent value="deposit" className="space-y-4 mt-0">
                  <form onSubmit={handleDeposit} className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        {language === "ar" ? "المبلغ (USDC)" : "Amount (USDC)"}
                      </label>
                      <input
                        type="number"
                        value={depositAmount}
                        onChange={(e) => setDepositAmount(e.target.value)}
                        className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none placeholder-gray-500"
                        placeholder={`${language === "ar" ? "الحد الأدنى" : "Minimum"} ${MIN_DEPOSIT}`}
                        min={MIN_DEPOSIT}
                        dir="ltr"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        {language === "ar" ? "الشبكة" : "Network"}
                      </label>
                      <select
                        value={selectedNetwork}
                        onChange={(e) => setSelectedNetwork(e.target.value)}
                        className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
                      >
                        {networks.map((network) => (
                          <option key={network.id} value={network.id} className="bg-[#1a1a2e] text-white">
                            {network.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="p-4 bg-[#252542] rounded-lg text-sm space-y-2">
                      <p className="text-gray-300">• {language === "ar" ? "الحد الأدنى للإيداع:" : "Minimum deposit:"} {MIN_DEPOSIT} USDC</p>
                      <p className="text-blue-400">• {language === "ar" ? `رسوم الإيداع: ${DEPOSIT_FEE_PERCENT}% (رسوم المنصة + الشبكة)` : `Deposit fee: ${DEPOSIT_FEE_PERCENT}% (platform + network fees)`}</p>
                      <p className="text-gray-300">• {language === "ar" ? "سيتم إضافة الرصيد تلقائياً بعد التأكيد" : "Balance will be added automatically after confirmation"}</p>
                      <p className="text-gray-300">• {language === "ar" ? "ستحصل على وحدات بناءً على قيمة NAV الحالية" : "You will receive units based on current NAV"}</p>
                    </div>
                    <button
                      type="submit"
                      disabled={depositMutation.isPending}
                      className="w-full py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors disabled:opacity-50"
                    >
                      {depositMutation.isPending 
                        ? (language === "ar" ? "جاري الإنشاء..." : "Creating...") 
                        : (language === "ar" ? "إنشاء طلب إيداع" : "Create Deposit Request")
                      }
                    </button>
                  </form>
                </TabsContent>

                {/* Withdraw Tab */}
                <TabsContent value="withdraw" className="space-y-4 mt-0">
                  {!canWithdraw && (
                    <div className="p-4 bg-destructive/10 rounded-lg flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 text-destructive" />
                      <p className="text-sm text-destructive">
                        {language === "ar" 
                          ? `الرصيد غير كافٍ. الحد الأدنى للسحب ${MIN_WITHDRAWAL} USDC`
                          : `Insufficient balance. Minimum withdrawal is ${MIN_WITHDRAWAL} USDC`
                        }
                      </p>
                    </div>
                  )}
                  <form onSubmit={handleWithdraw} className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        {language === "ar" ? "المبلغ (USDC)" : "Amount (USDC)"}
                      </label>
                      <input
                        type="number"
                        value={withdrawAmount}
                        onChange={(e) => setWithdrawAmount(e.target.value)}
                        className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none placeholder-gray-500"
                        placeholder={`${language === "ar" ? "الحد الأدنى" : "Minimum"} ${MIN_WITHDRAWAL}`}
                        min={MIN_WITHDRAWAL}
                        max={currentBalance}
                        disabled={!canWithdraw}
                        dir="ltr"
                      />
                      {canWithdraw && (
                        <button
                          type="button"
                          onClick={() => setWithdrawAmount(String(currentBalance))}
                          className="text-xs text-purple-400 mt-1 hover:underline"
                        >
                          {language === "ar" ? "سحب الكل" : "Withdraw All"}
                        </button>
                      )}
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        {language === "ar" ? "الشبكة" : "Network"}
                      </label>
                      <select
                        value={withdrawNetwork}
                        onChange={(e) => setWithdrawNetwork(e.target.value)}
                        className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
                        disabled={!canWithdraw}
                      >
                        {networks.map((network) => (
                          <option key={network.id} value={network.id} className="bg-[#1a1a2e] text-white">
                            {network.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        {language === "ar" ? "عنوان المحفظة" : "Wallet Address"}
                      </label>
                      <input
                        type="text"
                        value={withdrawAddress}
                        onChange={(e) => setWithdrawAddress(e.target.value)}
                        className="w-full px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none font-mono placeholder-gray-500"
                        placeholder={language === "ar" ? "أدخل عنوان محفظتك" : "Enter your wallet address"}
                        disabled={!canWithdraw}
                        dir="ltr"
                      />
                    </div>
                    <div className="p-4 bg-[#252542] rounded-lg text-sm space-y-1">
                      <p className="text-gray-300">• {language === "ar" ? "الحد الأدنى للسحب:" : "Minimum withdrawal:"} {MIN_WITHDRAWAL} USDC</p>
                      <p className="text-gray-300">• {language === "ar" ? "طلبات السحب تحتاج موافقة الإدارة" : "Withdrawal requests require admin approval"}</p>
                      <p className="text-gray-300">• {language === "ar" ? "ستصلك رسالة تأكيد على الإيميل" : "You will receive a confirmation email"}</p>
                      <p className="text-gray-300">• {language === "ar" ? "سيتم إشعارك عند الموافقة أو الرفض" : "You will be notified on approval/rejection"}</p>
                    </div>
                    <button
                      type="submit"
                      disabled={!canWithdraw || withdrawMutation.isPending}
                      className="w-full py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors disabled:opacity-50"
                    >
                      {withdrawMutation.isPending 
                        ? (language === "ar" ? "جاري الإرسال..." : "Sending...") 
                        : (language === "ar" ? "إرسال طلب السحب" : "Submit Withdrawal Request")
                      }
                    </button>
                  </form>
                </TabsContent>
              </CardContent>
            </Tabs>
          </Card>
        </div>

        {/* Transaction History */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                <span>{language === "ar" ? "سجل المعاملات" : "Transaction History"}</span>
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["transactions"] })}
                  className="p-1 rounded hover:bg-muted"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loadingTx ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : transactions.length > 0 ? (
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                  {transactions.map((tx: any) => (
                    <div
                      key={tx.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "p-2 rounded-md",
                          tx.type === "deposit" ? "bg-green-500/10" : "bg-destructive/10"
                        )}>
                          {tx.type === "deposit" ? (
                            <ArrowDownCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <ArrowUpCircle className="w-4 h-4 text-destructive" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-sm">
                            {tx.type === "deposit" 
                              ? (language === "ar" ? "إيداع" : "Deposit") 
                              : (language === "ar" ? "سحب" : "Withdraw")
                            }
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {tx.created_at ? format(new Date(tx.created_at), "dd/MM/yyyy HH:mm") : "-"}
                          </p>
                        </div>
                      </div>
                      <div className="text-left">
                        <p className="font-medium text-sm" dir="ltr">
                          {showBalance 
                            ? formatCurrency(parseFloat(tx.amount_usd || tx.amount || 0))
                            : "****"
                          }
                        </p>
                        {getStatusBadge(tx.status)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Wallet className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>{language === "ar" ? "لا توجد معاملات" : "No transactions"}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
