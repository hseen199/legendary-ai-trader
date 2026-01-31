import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { adminAPI } from "../../services/api";
import { Skeleton } from "../../components/ui/skeleton";
import { 
  ArrowUpCircle,
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  XCircle,
  Clock,
  ArrowLeft,
  DollarSign,
  AlertTriangle,
  Send,
  Copy,
  ExternalLink,
  Mail,
} from "lucide-react";
import { format } from "date-fns";
import toast from "react-hot-toast";
import { Link } from "wouter";
import { cn } from "../../lib/utils";

interface Withdrawal {
  id: number;
  user_id: number;
  user_email?: string;
  user_name?: string;
  amount: number;
  amount_usd: number;
  wallet_address: string;
  to_address: string;
  network: string;
  coin: string;
  status: "pending_approval" | "approved" | "rejected" | "completed" | "processing";
  created_at: string;
  requested_at: string;
  reviewed_at?: string;
  completed_at?: string;
  tx_hash?: string;
  rejection_reason?: string;
}

export default function WithdrawalsManagement() {
  const queryClient = useQueryClient();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState<"all" | "pending_approval" | "approved" | "rejected" | "completed">("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [selectedWithdrawal, setSelectedWithdrawal] = useState<Withdrawal | null>(null);
  const [txHash, setTxHash] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const itemsPerPage = 20;

  // Fetch withdrawals
  const { data: withdrawals = [], isLoading, refetch } = useQuery({
    queryKey: ["/api/v1/admin/withdrawals"],
    queryFn: () => adminAPI.getWithdrawals(0, 500).then(res => res.data),
  });

  // Approve withdrawal mutation
  const approveMutation = useMutation({
    mutationFn: (withdrawalId: number) => adminAPI.approveWithdrawal(withdrawalId),
    onSuccess: () => {
      toast.success("✅ تمت الموافقة على السحب وتم إرسال إيميل للمستخدم");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || 'فشل في الموافقة على السحب';
      if (errorMessage.includes('already')) {
        toast.error('تم معالجة هذا الطلب مسبقاً. جاري تحديث القائمة...');
        queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
      } else {
        toast.error(errorMessage);
      }
    },
  });

  // Reject withdrawal mutation
  const rejectMutation = useMutation({
    mutationFn: ({ withdrawalId, reason }: { withdrawalId: number; reason: string }) => 
      adminAPI.rejectWithdrawal(withdrawalId, reason),
    onSuccess: () => {
      toast.success("❌ تم رفض السحب وتم إرسال إيميل للمستخدم بالسبب");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
      setShowRejectModal(false);
      setSelectedWithdrawal(null);
      setRejectReason("");
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || 'فشل في رفض السحب';
      if (errorMessage.includes('already')) {
        toast.error('تم معالجة هذا الطلب مسبقاً. جاري تحديث القائمة...');
        queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
        setShowRejectModal(false);
        setSelectedWithdrawal(null);
        setRejectReason("");
      } else {
        toast.error(errorMessage);
      }
    },
  });

  // Complete withdrawal mutation (after manual transfer)
  const completeMutation = useMutation({
    mutationFn: ({ withdrawalId, txHash }: { withdrawalId: number; txHash?: string }) => 
      adminAPI.completeWithdrawal(withdrawalId, txHash),
    onSuccess: () => {
      toast.success("✅ تم تأكيد إتمام السحب وتم إرسال إيميل للمستخدم");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
      setShowCompleteModal(false);
      setSelectedWithdrawal(null);
      setTxHash("");
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || 'فشل في تأكيد إتمام السحب';
      if (errorMessage.includes('already')) {
        toast.error('تم معالجة هذا الطلب مسبقاً. جاري تحديث القائمة...');
        queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals"] });
        setShowCompleteModal(false);
        setSelectedWithdrawal(null);
        setTxHash("");
      } else {
        toast.error(errorMessage);
      }
    },
  });

  // Filter withdrawals
  const filteredWithdrawals = withdrawals.filter((w: Withdrawal) => {
    const matchesSearch = 
      (w.user_email?.toLowerCase() || "").includes(searchTerm.toLowerCase()) ||
      (w.user_name?.toLowerCase() || "").includes(searchTerm.toLowerCase()) ||
      (w.wallet_address || w.to_address || "").toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = 
      filterStatus === "all" || w.status === filterStatus;
    
    return matchesSearch && matchesStatus;
  });

  // Pagination
  const totalPages = Math.ceil(filteredWithdrawals.length / itemsPerPage);
  const paginatedWithdrawals = filteredWithdrawals.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("تم نسخ العنوان");
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      pending_approval: "bg-amber-500/15 text-amber-400 border-amber-500/25",
      approved: "bg-blue-500/15 text-blue-400 border-blue-500/25",
      rejected: "bg-red-500/15 text-red-400 border-red-500/25",
      completed: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
      processing: "bg-violet-500/15 text-violet-400 border-violet-500/25",
    };
    const labels: Record<string, string> = {
      pending_approval: "بانتظار الموافقة",
      approved: "تمت الموافقة",
      rejected: "مرفوض",
      completed: "مكتمل",
      processing: "قيد المعالجة",
    };
    return (
      <span className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border",
        styles[status] || styles.pending_approval
      )}>
        {status === "pending_approval" && <Clock className="w-3 h-3" />}
        {status === "approved" && <CheckCircle className="w-3 h-3" />}
        {status === "rejected" && <XCircle className="w-3 h-3" />}
        {status === "completed" && <CheckCircle className="w-3 h-3" />}
        {status === "processing" && <RefreshCw className="w-3 h-3 animate-spin" />}
        {labels[status] || status}
      </span>
    );
  };

  const pendingCount = withdrawals.filter((w: Withdrawal) => w.status === "pending_approval").length;
  const approvedCount = withdrawals.filter((w: Withdrawal) => w.status === "approved").length;
  const totalPending = withdrawals
    .filter((w: Withdrawal) => w.status === "pending_approval")
    .reduce((sum: number, w: Withdrawal) => sum + (w.amount_usd || w.amount || 0), 0);
  const totalApproved = withdrawals
    .filter((w: Withdrawal) => w.status === "approved")
    .reduce((sum: number, w: Withdrawal) => sum + (w.amount_usd || w.amount || 0), 0);

  const handleComplete = (withdrawal: Withdrawal) => {
    setSelectedWithdrawal(withdrawal);
    setShowCompleteModal(true);
  };

  const handleReject = (withdrawal: Withdrawal) => {
    setSelectedWithdrawal(withdrawal);
    setShowRejectModal(true);
  };

  const confirmReject = () => {
    if (selectedWithdrawal && rejectReason.trim()) {
      rejectMutation.mutate({ 
        withdrawalId: selectedWithdrawal.id, 
        reason: rejectReason 
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-violet-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-amber-500/10 rounded-full blur-[100px]" />
      </div>

      {/* Header */}
      <div className="relative flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <Link href="/admin">
            <button className="p-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20 hover:border-violet-500/40 transition-all duration-300">
              <ArrowLeft className="w-5 h-5" />
            </button>
          </Link>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white via-violet-200 to-amber-200 bg-clip-text text-transparent">
              إدارة طلبات السحب
            </h1>
            <p className="text-white/40 text-sm mt-1">مراجعة والموافقة على طلبات السحب - يتم إرسال إيميل تلقائي للمستخدم</p>
          </div>
        </div>
        <button 
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20 hover:border-violet-500/40 transition-all duration-300"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Email Notification Info */}
      <div className="relative flex items-center gap-3 p-4 rounded-xl bg-violet-500/10 border border-violet-500/20">
        <Mail className="w-5 h-5 text-violet-400 flex-shrink-0" />
        <p className="text-violet-200 text-sm">
          <span className="font-bold">إشعارات تلقائية:</span> عند الموافقة أو الرفض أو إتمام التحويل، يتم إرسال إيميل تلقائي للمستخدم بحالة طلبه
        </p>
      </div>

      {/* Stats */}
      <div className="relative grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-violet-500/15">
              <ArrowUpCircle className="w-6 h-6 text-violet-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي الطلبات</p>
              <p className="text-2xl font-bold text-white">{withdrawals.length}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-amber-500/20 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-amber-500/15">
              <Clock className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">بانتظار الموافقة</p>
              <p className="text-2xl font-bold text-amber-400">{pendingCount}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-blue-500/20 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-blue-500/15">
              <Send className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">بانتظار التحويل</p>
              <p className="text-2xl font-bold text-blue-400">{approvedCount}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-emerald-500/20 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-emerald-500/15">
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">مكتمل</p>
              <p className="text-2xl font-bold text-emerald-400">
                {withdrawals.filter((w: Withdrawal) => w.status === "completed").length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {pendingCount > 0 && (
        <div className="relative flex items-center gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <p className="text-amber-200">
            يوجد <span className="font-bold">{pendingCount}</span> طلب سحب بانتظار الموافقة بإجمالي <span className="font-bold" dir="ltr">{formatCurrency(totalPending)}</span>
          </p>
        </div>
      )}
      
      {approvedCount > 0 && (
        <div className="relative flex items-center gap-3 p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
          <Send className="w-5 h-5 text-blue-400 flex-shrink-0" />
          <p className="text-blue-200">
            يوجد <span className="font-bold">{approvedCount}</span> طلب سحب تمت الموافقة عليه وبانتظار التحويل اليدوي بإجمالي <span className="font-bold" dir="ltr">{formatCurrency(totalApproved)}</span>
          </p>
        </div>
      )}

      {/* Search and Filter */}
      <div className="relative flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" />
          <input
            type="text"
            placeholder="بحث بالبريد أو عنوان المحفظة..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pr-12 pl-4 py-3 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 text-white placeholder:text-white/30 focus:outline-none focus:border-violet-500/40 transition-colors"
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          {[
            { value: "all", label: "الكل" },
            { value: "pending_approval", label: "بانتظار الموافقة" },
            { value: "approved", label: "بانتظار التحويل" },
            { value: "completed", label: "مكتمل" },
            { value: "rejected", label: "مرفوض" },
          ].map((item) => (
            <button
              key={item.value}
              onClick={() => setFilterStatus(item.value as any)}
              className={cn(
                "px-4 py-3 rounded-xl border transition-all duration-300",
                filterStatus === item.value
                  ? "bg-violet-500/20 border-violet-500/40 text-violet-300"
                  : "bg-[rgba(18,18,28,0.6)] border-violet-500/15 text-white/50 hover:border-violet-500/30"
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Withdrawals Table */}
      <div className="relative rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
        
        {isLoading ? (
          <div className="p-6 space-y-4">
            {[1, 2, 3, 4, 5].map(i => (
              <Skeleton key={i} className="h-16 w-full bg-white/10" />
            ))}
          </div>
        ) : paginatedWithdrawals.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-white/50 border-b border-violet-500/10">
                  <th className="px-6 py-4 font-medium text-sm">المستخدم</th>
                  <th className="px-6 py-4 font-medium text-sm">المبلغ</th>
                  <th className="px-6 py-4 font-medium text-sm">العنوان</th>
                  <th className="px-6 py-4 font-medium text-sm">الشبكة</th>
                  <th className="px-6 py-4 font-medium text-sm">الحالة</th>
                  <th className="px-6 py-4 font-medium text-sm">التاريخ</th>
                  <th className="px-6 py-4 font-medium text-sm">الإجراءات</th>
                </tr>
              </thead>
              <tbody>
                {paginatedWithdrawals.map((withdrawal: Withdrawal) => (
                  <tr
                    key={withdrawal.id}
                    className="border-t border-white/5 hover:bg-violet-500/5 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div>
                        <p className="text-white font-medium">{withdrawal.user_name || "بدون اسم"}</p>
                        <p className="text-white/50 text-sm">{withdrawal.user_email}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-white font-bold text-lg" dir="ltr">
                      {formatCurrency(withdrawal.amount_usd || withdrawal.amount)}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <code className="text-xs text-violet-300 bg-violet-500/10 px-2 py-1 rounded">
                          {(withdrawal.wallet_address || withdrawal.to_address).slice(0, 10)}...{(withdrawal.wallet_address || withdrawal.to_address).slice(-8)}
                        </code>
                        <button
                          onClick={() => copyToClipboard(withdrawal.wallet_address || withdrawal.to_address)}
                          className="p-1 rounded hover:bg-white/10 transition-colors"
                          title="نسخ العنوان"
                        >
                          <Copy className="w-4 h-4 text-white/50" />
                        </button>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded bg-white/5 text-white/70 text-sm">
                        {withdrawal.network}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      {getStatusBadge(withdrawal.status)}
                    </td>
                    <td className="px-6 py-4 text-white/50 text-sm">
                      {format(new Date(withdrawal.created_at || withdrawal.requested_at), "dd/MM/yyyy HH:mm")}
                    </td>
                    <td className="px-6 py-4">
                      {withdrawal.status === "pending_approval" ? (
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => approveMutation.mutate(withdrawal.id)}
                            disabled={approveMutation.isPending}
                            className="flex items-center gap-1 p-2 rounded-lg bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-colors disabled:opacity-50"
                            title="موافقة وإرسال إيميل"
                          >
                            <CheckCircle className="w-4 h-4" />
                            <Mail className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => handleReject(withdrawal)}
                            disabled={rejectMutation.isPending}
                            className="flex items-center gap-1 p-2 rounded-lg bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors disabled:opacity-50"
                            title="رفض مع السبب"
                          >
                            <XCircle className="w-4 h-4" />
                            <Mail className="w-3 h-3" />
                          </button>
                        </div>
                      ) : withdrawal.status === "approved" ? (
                        <button
                          onClick={() => handleComplete(withdrawal)}
                          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 transition-colors"
                          title="تأكيد التحويل وإرسال إيميل"
                        >
                          <Send className="w-4 h-4" />
                          <span className="text-sm">تأكيد التحويل</span>
                        </button>
                      ) : withdrawal.status === "completed" && withdrawal.tx_hash ? (
                        <a
                          href={`https://bscscan.com/tx/${withdrawal.tx_hash}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 text-emerald-400 hover:text-emerald-300 text-sm"
                        >
                          <ExternalLink className="w-4 h-4" />
                          عرض TX
                        </a>
                      ) : (
                        <span className="text-white/30 text-sm">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-16">
            <ArrowUpCircle className="w-16 h-16 text-white/10 mb-4" />
            <p className="text-white/40 text-lg">لا توجد طلبات سحب</p>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-violet-500/10">
            <p className="text-white/50 text-sm">
              عرض {((currentPage - 1) * itemsPerPage) + 1} - {Math.min(currentPage * itemsPerPage, filteredWithdrawals.length)} من {filteredWithdrawals.length}
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="p-2 rounded-lg bg-violet-500/10 text-violet-400 hover:bg-violet-500/20 transition-colors disabled:opacity-50"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
              <span className="text-white/70 px-3">{currentPage} / {totalPages}</span>
              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="p-2 rounded-lg bg-violet-500/10 text-violet-400 hover:bg-violet-500/20 transition-colors disabled:opacity-50"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Reject Withdrawal Modal */}
      {showRejectModal && selectedWithdrawal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[rgba(18,18,28,0.95)] border border-red-500/20 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <XCircle className="w-6 h-6 text-red-400" />
                رفض طلب السحب
              </h2>
              <button
                onClick={() => {
                  setShowRejectModal(false);
                  setSelectedWithdrawal(null);
                  setRejectReason("");
                }}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <XCircle className="w-5 h-5 text-white/50" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                <p className="text-red-200 text-sm mb-2">تفاصيل الطلب:</p>
                <div className="space-y-2 text-white/70 text-sm">
                  <p>المستخدم: <span className="text-white">{selectedWithdrawal.user_email}</span></p>
                  <p>المبلغ: <span className="text-white font-bold" dir="ltr">{formatCurrency(selectedWithdrawal.amount_usd || selectedWithdrawal.amount)}</span></p>
                  <p>الشبكة: <span className="text-white">{selectedWithdrawal.network}</span></p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <div className="flex items-center gap-2 text-amber-200 text-sm">
                  <Mail className="w-4 h-4" />
                  <span>سيتم إرسال إيميل للمستخدم يتضمن سبب الرفض</span>
                </div>
              </div>

              <div>
                <label className="block text-white/70 text-sm mb-2">سبب الرفض <span className="text-red-400">*</span></label>
                <textarea
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  placeholder="اكتب سبب رفض طلب السحب..."
                  className="w-full p-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-white/30 focus:outline-none focus:border-red-500/40 transition-colors resize-none h-32"
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={confirmReject}
                  disabled={!rejectReason.trim() || rejectMutation.isPending}
                  className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {rejectMutation.isPending ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <XCircle className="w-5 h-5" />
                      تأكيد الرفض وإرسال الإيميل
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowRejectModal(false);
                    setSelectedWithdrawal(null);
                    setRejectReason("");
                  }}
                  className="px-6 py-3 rounded-xl bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 transition-colors"
                >
                  إلغاء
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Complete Withdrawal Modal */}
      {showCompleteModal && selectedWithdrawal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl bg-[rgba(18,18,28,0.95)] border border-violet-500/20 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Send className="w-6 h-6 text-blue-400" />
                تأكيد إتمام التحويل
              </h2>
              <button
                onClick={() => {
                  setShowCompleteModal(false);
                  setSelectedWithdrawal(null);
                  setTxHash("");
                }}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <XCircle className="w-5 h-5 text-white/50" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <p className="text-blue-200 text-sm mb-2">تفاصيل السحب:</p>
                <div className="space-y-2 text-white/70 text-sm">
                  <p>المستخدم: <span className="text-white">{selectedWithdrawal.user_email}</span></p>
                  <p>المبلغ: <span className="text-white font-bold" dir="ltr">{formatCurrency(selectedWithdrawal.amount_usd || selectedWithdrawal.amount)}</span></p>
                  <p>الشبكة: <span className="text-white">{selectedWithdrawal.network}</span></p>
                </div>
              </div>

              <div>
                <label className="block text-white/70 text-sm mb-2">عنوان المحفظة (للتحويل إليه)</label>
                <div className="flex items-center gap-2">
                  <code className="flex-1 p-3 rounded-xl bg-white/5 text-violet-300 text-sm font-mono break-all">
                    {selectedWithdrawal.wallet_address || selectedWithdrawal.to_address}
                  </code>
                  <button
                    onClick={() => copyToClipboard(selectedWithdrawal.wallet_address || selectedWithdrawal.to_address)}
                    className="p-3 rounded-xl bg-violet-500/10 text-violet-400 hover:bg-violet-500/20 transition-colors"
                  >
                    <Copy className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-white/70 text-sm mb-2">رابط المعاملة (TX Hash) - اختياري</label>
                <input
                  type="text"
                  value={txHash}
                  onChange={(e) => setTxHash(e.target.value)}
                  placeholder="0x..."
                  className="w-full p-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-white/30 focus:outline-none focus:border-violet-500/40 transition-colors font-mono text-sm"
                  dir="ltr"
                />
              </div>

              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <div className="flex items-center gap-2 text-emerald-200 text-sm">
                  <Mail className="w-4 h-4" />
                  <span>سيتم إرسال إيميل للمستخدم يؤكد إتمام التحويل مع رابط المعاملة</span>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => completeMutation.mutate({ 
                    withdrawalId: selectedWithdrawal.id, 
                    txHash: txHash || undefined 
                  })}
                  disabled={completeMutation.isPending}
                  className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/30 transition-colors disabled:opacity-50"
                >
                  {completeMutation.isPending ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <CheckCircle className="w-5 h-5" />
                      تأكيد الإتمام وإرسال الإيميل
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowCompleteModal(false);
                    setSelectedWithdrawal(null);
                    setTxHash("");
                  }}
                  className="px-6 py-3 rounded-xl bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 transition-colors"
                >
                  إلغاء
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
