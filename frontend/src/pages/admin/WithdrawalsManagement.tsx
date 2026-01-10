import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { adminAPI } from "../../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { Skeleton } from "../../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import { 
  ArrowUpCircle, 
  CheckCircle, 
  XCircle, 
  Clock, 
  RefreshCw,
  AlertTriangle,
  Copy,
  ExternalLink,
  Search,
  Filter,
} from "lucide-react";
import { cn } from "../../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";

interface Withdrawal {
  id: number;
  user_id: number;
  user?: { email: string; full_name?: string };
  amount: number;
  to_address: string;
  network: string;
  coin: string;
  status: string;
  tx_hash?: string;
  admin_note?: string;
  created_at: string;
  processed_at?: string;
}

export default function WithdrawalsManagement() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("pending");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedWithdrawal, setSelectedWithdrawal] = useState<Withdrawal | null>(null);
  const [rejectReason, setRejectReason] = useState("");

  // Fetch pending withdrawals
  const { data: pendingWithdrawals = [], isLoading: loadingPending, refetch: refetchPending } = useQuery({
    queryKey: ["/api/v1/admin/withdrawals/pending"],
    queryFn: () => adminAPI.getPendingWithdrawals().then(res => res.data),
  });

  // Review withdrawal mutation
  const reviewMutation = useMutation({
    mutationFn: ({ id, action, reason }: { id: number; action: "approve" | "reject"; reason?: string }) =>
      adminAPI.reviewWithdrawal(id, action, reason),
    onSuccess: (_, variables) => {
      toast.success(variables.action === "approve" ? "تمت الموافقة على السحب" : "تم رفض السحب");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/withdrawals/pending"] });
      setSelectedWithdrawal(null);
      setRejectReason("");
    },
    onError: () => {
      toast.error("فشل في معالجة الطلب");
    },
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("تم النسخ");
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
            <CheckCircle className="w-3 h-3 ml-1" />
            مكتمل
          </Badge>
        );
      case "pending":
      case "pending_approval":
        return (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
            <Clock className="w-3 h-3 ml-1" />
            قيد الانتظار
          </Badge>
        );
      case "rejected":
        return (
          <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/20">
            <XCircle className="w-3 h-3 ml-1" />
            مرفوض
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  // Filter withdrawals
  const filteredWithdrawals = pendingWithdrawals.filter((w: Withdrawal) => {
    return (
      w.user?.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      w.to_address.toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  // Calculate totals
  const totalPendingAmount = pendingWithdrawals.reduce((sum: number, w: Withdrawal) => sum + w.amount, 0);

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">إدارة السحوبات</h1>
          <p className="text-muted-foreground text-sm">مراجعة وإدارة طلبات السحب</p>
        </div>
        <button
          onClick={() => refetchPending()}
          className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-yellow-500/10">
                <Clock className="w-5 h-5 text-yellow-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">طلبات معلقة</p>
                <p className="text-xl font-bold">{pendingWithdrawals.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg bg-primary/10">
                <ArrowUpCircle className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">إجمالي المبلغ المعلق</p>
                <p className="text-xl font-bold" dir="ltr">{formatCurrency(totalPendingAmount)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className={cn(
          "border-2",
          pendingWithdrawals.length > 0 ? "border-yellow-500/50" : "border-green-500/50"
        )}>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              {pendingWithdrawals.length > 0 ? (
                <>
                  <AlertTriangle className="w-8 h-8 text-yellow-500" />
                  <div>
                    <p className="font-medium text-yellow-500">يوجد طلبات تحتاج مراجعة</p>
                    <p className="text-sm text-muted-foreground">يرجى مراجعة الطلبات المعلقة</p>
                  </div>
                </>
              ) : (
                <>
                  <CheckCircle className="w-8 h-8 text-green-500" />
                  <div>
                    <p className="font-medium text-green-500">لا توجد طلبات معلقة</p>
                    <p className="text-sm text-muted-foreground">تم معالجة جميع الطلبات</p>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-4">
          <div className="relative">
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="بحث بالإيميل أو العنوان..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pr-10 px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
            />
          </div>
        </CardContent>
      </Card>

      {/* Withdrawals List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Clock className="w-5 h-5 text-yellow-500" />
            طلبات السحب المعلقة
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loadingPending ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : filteredWithdrawals.length > 0 ? (
            <div className="space-y-4">
              {filteredWithdrawals.map((withdrawal: Withdrawal) => (
                <div
                  key={withdrawal.id}
                  className="p-4 border border-border rounded-lg hover:border-primary/50 transition-colors"
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-start gap-4">
                      <div className="p-3 rounded-lg bg-yellow-500/10">
                        <ArrowUpCircle className="w-5 h-5 text-yellow-500" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <p className="font-medium">{withdrawal.user?.email || "مستخدم"}</p>
                          {getStatusBadge(withdrawal.status)}
                        </div>
                        <p className="text-2xl font-bold" dir="ltr">{formatCurrency(withdrawal.amount)}</p>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <span>{withdrawal.network}</span>
                          <span>•</span>
                          <span>{withdrawal.coin}</span>
                          <span>•</span>
                          <span>{format(new Date(withdrawal.created_at), "dd/MM/yyyy HH:mm")}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <code className="text-xs bg-muted px-2 py-1 rounded max-w-[200px] truncate" dir="ltr">
                          {withdrawal.to_address}
                        </code>
                        <button
                          onClick={() => copyToClipboard(withdrawal.to_address)}
                          className="p-1 hover:bg-muted rounded"
                        >
                          <Copy className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => reviewMutation.mutate({ id: withdrawal.id, action: "approve" })}
                          disabled={reviewMutation.isPending}
                          className="flex-1 px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-500/90 disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          <CheckCircle className="w-4 h-4" />
                          موافقة
                        </button>
                        <button
                          onClick={() => setSelectedWithdrawal(withdrawal)}
                          disabled={reviewMutation.isPending}
                          className="flex-1 px-4 py-2 bg-destructive text-destructive-foreground rounded-lg font-medium hover:bg-destructive/90 disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          <XCircle className="w-4 h-4" />
                          رفض
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <p className="text-muted-foreground">لا توجد طلبات سحب معلقة</p>
              <p className="text-sm text-muted-foreground mt-1">تم معالجة جميع الطلبات</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Reject Modal */}
      {selectedWithdrawal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2 text-destructive">
                <XCircle className="w-5 h-5" />
                رفض طلب السحب
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground">المستخدم</p>
                <p className="font-medium">{selectedWithdrawal.user?.email}</p>
                <p className="text-sm text-muted-foreground mt-2">المبلغ</p>
                <p className="font-bold text-lg" dir="ltr">{formatCurrency(selectedWithdrawal.amount)}</p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">سبب الرفض</label>
                <textarea
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none resize-none"
                  rows={3}
                  placeholder="أدخل سبب الرفض (اختياري)"
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    reviewMutation.mutate({
                      id: selectedWithdrawal.id,
                      action: "reject",
                      reason: rejectReason,
                    });
                  }}
                  disabled={reviewMutation.isPending}
                  className="flex-1 py-2 bg-destructive text-destructive-foreground rounded-lg font-medium hover:bg-destructive/90 disabled:opacity-50"
                >
                  {reviewMutation.isPending ? "جاري الرفض..." : "تأكيد الرفض"}
                </button>
                <button
                  onClick={() => {
                    setSelectedWithdrawal(null);
                    setRejectReason("");
                  }}
                  className="flex-1 py-2 bg-muted rounded-lg font-medium hover:bg-muted/80"
                >
                  إلغاء
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
