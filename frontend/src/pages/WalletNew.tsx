import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useLanguage } from "../lib/i18n";
import { walletAPI } from "../services/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { 
  Wallet, 
  ArrowUpCircle, 
  ArrowDownCircle, 
  Copy, 
  CheckCircle, 
  Clock, 
  XCircle, 
  RefreshCw, 
  AlertTriangle,
  TrendingUp,
  TrendingDown
} from "lucide-react";
import { cn } from "../lib/utils";
import { QRCodeSVG } from "qrcode.react";
import { format } from "date-fns";
import toast from "react-hot-toast";

const MIN_DEPOSIT = 100;
const MIN_WITHDRAWAL = 10;

export default function WalletNew() {
  const { t, language } = useLanguage();
  const queryClient = useQueryClient();
  const [copiedAddress, setCopiedAddress] = useState(false);
  const [activeTab, setActiveTab] = useState("deposit");
  
  // Form states
  const [depositAmount, setDepositAmount] = useState("");
  const [depositTxHash, setDepositTxHash] = useState("");
  const [withdrawAddress, setWithdrawAddress] = useState("");
  const [withdrawAmount, setWithdrawAmount] = useState("");
  const [withdrawNetwork, setWithdrawNetwork] = useState("TRC20");

  // Fetch balance
  const { data: balance, isLoading: loadingBalance, refetch: refetchBalance } = useQuery({
    queryKey: ["/api/v1/wallet/balance"],
    queryFn: () => walletAPI.getBalance().then(res => res.data),
  });

  // Fetch deposit address
  const { data: depositAddress, isLoading: loadingAddress } = useQuery({
    queryKey: ["/api/v1/wallet/deposit/address"],
    queryFn: () => walletAPI.getDepositAddress("TRC20", "USDC").then(res => res.data),
  });

  // Fetch transactions
  const { data: transactions = [], isLoading: loadingTx } = useQuery({
    queryKey: ["/api/v1/wallet/transactions"],
    queryFn: () => walletAPI.getTransactions(50).then(res => res.data),
  });

  // Fetch deposit history
  const { data: depositHistory = [], isLoading: loadingDeposits } = useQuery({
    queryKey: ["/api/v1/wallet/deposit/history"],
    queryFn: () => walletAPI.getDepositHistory().then(res => res.data),
  });

  // Fetch withdrawal history
  const { data: withdrawalHistory = [], isLoading: loadingWithdrawals } = useQuery({
    queryKey: ["/api/v1/wallet/withdraw/history"],
    queryFn: () => walletAPI.getWithdrawalHistory().then(res => res.data),
  });

  // Withdrawal mutation
  const withdrawMutation = useMutation({
    mutationFn: (data: { amount: number; to_address: string; network: string; coin: string }) =>
      walletAPI.requestWithdrawal(data),
    onSuccess: () => {
      toast.success(language === "ar" ? "تم إرسال طلب السحب بنجاح" : "Withdrawal request sent successfully");
      setWithdrawAddress("");
      setWithdrawAmount("");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/wallet/withdraw/history"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/wallet/balance"] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || (language === "ar" ? "فشل في إرسال طلب السحب" : "Failed to send withdrawal request"));
    },
  });

  const userBalance = balance?.current_value_usd || 0;
  const canWithdraw = balance?.can_withdraw || false;

  const copyToClipboard = () => {
    if (depositAddress?.address) {
      navigator.clipboard.writeText(depositAddress.address);
      setCopiedAddress(true);
      setTimeout(() => setCopiedAddress(false), 2000);
      toast.success(language === "ar" ? "تم نسخ العنوان" : "Address copied");
    }
  };

  const handleWithdraw = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!canWithdraw) {
      toast.error(language === "ar" ? "السحب غير متاح حالياً - يجب انتظار انتهاء فترة القفل" : "Withdrawal not available - please wait for lock period to end");
      return;
    }

    const amount = parseFloat(withdrawAmount);
    if (isNaN(amount) || amount < MIN_WITHDRAWAL) {
      toast.error(language === "ar" ? `الحد الأدنى للسحب هو ${MIN_WITHDRAWAL} USDC` : `Minimum withdrawal is ${MIN_WITHDRAWAL} USDC`);
      return;
    }

    if (amount > userBalance) {
      toast.error(language === "ar" ? `الرصيد غير كافٍ. المتاح: $${userBalance.toFixed(2)}` : `Insufficient balance. Available: $${userBalance.toFixed(2)}`);
      return;
    }

    if (!withdrawAddress || withdrawAddress.length < 30) {
      toast.error(language === "ar" ? "يرجى إدخال عنوان محفظة صحيح" : "Please enter a valid wallet address");
      return;
    }

    withdrawMutation.mutate({
      amount,
      to_address: withdrawAddress,
      network: withdrawNetwork,
      coin: "USDC",
    });
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "confirmed":
      case "completed":
        return (
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
            <CheckCircle className="w-3 h-3 ml-1" />
            {language === "ar" ? "مكتمل" : "Completed"}
          </Badge>
        );
      case "pending":
      case "pending_approval":
        return (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
            <Clock className="w-3 h-3 ml-1" />
            {language === "ar" ? "قيد الانتظار" : "Pending"}
          </Badge>
        );
      case "failed":
      case "rejected":
      case "cancelled":
        return (
          <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/20">
            <XCircle className="w-3 h-3 ml-1" />
            {status === "rejected" ? language === "ar" ? "مرفوض" : "Rejected" : language === "ar" ? "ملغي" : "Cancelled"}
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const networks = [
    { id: "BEP20", name: "BEP20 (BNB Smart Chain)" },
    { id: "SOL", name: "Solana" },
    { id: "POLYGON", name: "Polygon" },
  ];

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">{t.wallet.title}</h1>
          <p className="text-muted-foreground text-sm">{t.wallet.subtitle}</p>
        </div>
        <button
          onClick={() => refetchBalance()}
          className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          {language === "ar" ? "تحديث" : "Refresh"}
        </button>
      </div>

      {/* Balance Card */}
      <Card className="bg-gradient-to-br from-primary/10 to-primary/5">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-muted-foreground mb-1">{ language === "ar" ? "الرصيد المتاح" : "Available Balance" }</p>
              {loadingBalance ? (
                <Skeleton className="h-10 w-32" />
              ) : (
                <p className="text-3xl font-bold" dir="ltr">{formatCurrency(userBalance)}</p>
              )}
              {balance && (
                <p className="text-sm text-muted-foreground mt-1" dir="ltr">
                  {balance.units?.toFixed(4) || "0"} {language === "ar" ? "وحدة" : "units"} × ${balance.nav?.toFixed(4) || "1"} NAV
                </p>
              )}
            </div>
            <div className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium",
              canWithdraw 
                ? "bg-green-500/10 text-green-500" 
                : "bg-destructive/10 text-destructive"
            )}>
              {canWithdraw ? (language === "ar" ? "السحب متاح" : "Withdrawal Available") : (language === "ar" ? "السحب مقفل" : "Withdrawal Locked")}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Deposit/Withdraw Forms */}
        <div className="lg:col-span-2">
          <Card>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
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
                <TabsContent value="deposit" className="space-y-6 mt-0">
                  <div>
                    <h3 className="font-medium mb-2">{ language === "ar" ? "عنوان الإيداع (USDC - BEP20/Solana)" : "Deposit Address (USDC - BEP20/Solana)" }</h3>
                    <div className="p-4 bg-muted rounded-lg">
                      {loadingAddress ? (
                        <Skeleton className="h-[150px] w-[150px] mx-auto" />
                      ) : depositAddress?.address ? (
                        <div className="flex flex-col items-center gap-4">
                          <QRCodeSVG value={depositAddress.address} size={150} />
                          <div className="w-full">
                            <div className="flex items-center gap-2 p-3 bg-background rounded-lg">
                              <code className="flex-1 text-sm break-all" dir="ltr">
                                {depositAddress.address}
                              </code>
                              <button
                                onClick={copyToClipboard}
                                className="p-2 hover:bg-muted rounded-lg transition-colors"
                              >
                                {copiedAddress ? (
                                  <CheckCircle className="w-4 h-4 text-green-500" />
                                ) : (
                                  <Copy className="w-4 h-4" />
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <p className="text-center text-muted-foreground">{ language === "ar" ? "لا يوجد عنوان متاح" : "No address available" }</p>
                      )}
                    </div>
                  </div>

                  <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                    <div className="flex gap-3">
                      <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
                      <div className="text-sm">
                        <p className="font-medium text-yellow-500">{ language === "ar" ? "تنبيه هام" : "Important Notice" }</p>
                        <ul className="text-muted-foreground mt-1 space-y-1">
                          <li>• {language === "ar" ? "أرسل USDC عبر شبكة BEP20 أو Solana" : "Send USDC via BEP20 or Solana network"}</li>
                          <li>• {language === "ar" ? "الحد الأدنى للإيداع:" : "Minimum deposit:"} {MIN_DEPOSIT} USDC</li>
                          <li>• {language === "ar" ? "سيتم تأكيد الإيداع خلال 10-30 دقيقة" : "Deposit will be confirmed within 10-30 minutes"}</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Withdraw Tab */}
                <TabsContent value="withdraw" className="space-y-6 mt-0">
                  {!canWithdraw && (
                    <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                      <div className="flex gap-3">
                        <XCircle className="w-5 h-5 text-destructive flex-shrink-0" />
                        <div>
                          <p className="font-medium text-destructive">{language === "ar" ? "السحب مقفل" : "Withdrawal Locked"}</p>
                          <p className="text-sm text-muted-foreground mt-1">
                            {language === "ar" ? "يجب انتظار 7 أيام من آخر إيداع قبل السحب" : "You must wait 7 days from last deposit before withdrawal"}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  <form onSubmit={handleWithdraw} className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">{ language === "ar" ? "المبلغ (USDC)" : "Amount (USDC)" }</label>
                      <div className="relative">
                        <input
                          type="number"
                          value={withdrawAmount}
                          onChange={(e) => setWithdrawAmount(e.target.value)}
                          className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                          placeholder="0.00"
                          step="0.01"
                          min={MIN_WITHDRAWAL}
                          max={userBalance}
                          disabled={!canWithdraw}
                          dir="ltr"
                        />
                        <button
                          type="button"
                          onClick={() => setWithdrawAmount(userBalance.toString())}
                          className="absolute left-3 top-1/2 -translate-y-1/2 text-primary text-sm font-medium"
                          disabled={!canWithdraw}
                        >
                          {language === "ar" ? "الكل" : "All"}
                        </button>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2">{language === "ar" ? "الشبكة" : "Network"}</label>
                      <select
                        value={withdrawNetwork}
                        onChange={(e) => setWithdrawNetwork(e.target.value)}
                        className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                        disabled={!canWithdraw}
                      >
                        {networks.map((network) => (
                          <option key={network.id} value={network.id}>
                            {network.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2">{ language === "ar" ? "عنوان المحفظة" : "Wallet Address" }</label>
                      <input
                        type="text"
                        value={withdrawAddress}
                        onChange={(e) => setWithdrawAddress(e.target.value)}
                        className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none font-mono"
                        placeholder={language === "ar" ? "أدخل عنوان محفظتك" : "Enter your wallet address"}
                        disabled={!canWithdraw}
                        dir="ltr"
                      />
                    </div>

                    <div className="p-4 bg-muted rounded-lg text-sm">
                      <p>• {language === "ar" ? "الحد الأدنى للسحب:" : "Minimum withdrawal:"} {MIN_WITHDRAWAL} USDC</p>
                      <p>• {language === "ar" ? "طلبات السحب تحتاج موافقة الإدارة" : "Withdrawal requests require admin approval"}</p>
                      <p>• {language === "ar" ? "ستصلك رسالة تأكيد على الإيميل" : "You will receive a confirmation email"}</p>
                    </div>

                    <button
                      type="submit"
                      disabled={!canWithdraw || withdrawMutation.isPending}
                      className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                    >
                      {withdrawMutation.isPending ? language === "ar" ? "جاري الإرسال..." : "Sending..." : language === "ar" ? "إرسال طلب السحب" : "Submit Withdrawal Request"}
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
              <CardTitle className="text-lg">{ language === "ar" ? "سجل المعاملات" : "Transaction History" }</CardTitle>
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
                  {transactions.slice(0, 15).map((tx: any) => (
                    <div
                      key={tx.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
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
                            {tx.type === "deposit" ? language === "ar" ? "إيداع" : "Deposit" : language === "ar" ? "سحب" : "Withdraw"}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {tx.created_at ? format(new Date(tx.created_at), "dd/MM HH:mm") : "-"}
                          </p>
                        </div>
                      </div>
                      <div className="text-left">
                        <p className="font-medium text-sm" dir="ltr">
                          {formatCurrency(parseFloat(tx.amount || 0))}
                        </p>
                        {getStatusBadge(tx.status)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Wallet className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>{ language === "ar" ? "لا توجد معاملات" : "No transactions" }</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
