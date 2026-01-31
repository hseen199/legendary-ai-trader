import { useState } from "react";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { adminAPI } from "../../services/api";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { Skeleton } from "../../components/ui/skeleton";
import { 
  Users, 
  Search,
  Filter,
  MoreVertical,
  UserCheck,
  UserX,
  Mail,
  Calendar,
  DollarSign,
  Eye,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Shield,
  Crown,
  ArrowLeft,
  LogIn,
  Plus,
  Minus,
  Wallet,
} from "lucide-react";
import { format } from "date-fns";
import toast from "react-hot-toast";
import { Link, useLocation } from "wouter";
import { cn } from "../../lib/utils";

interface User {
  id: number;
  email: string;
  full_name?: string;
  phone?: string;
  is_active: boolean;
  is_admin: boolean;
  vip_level?: number;
  created_at: string;
  last_login?: string;
  total_deposited?: number;
  current_value?: number;
  // Backend field names (for compatibility)
  current_value_usd?: number;
  units?: number;
  status?: string;
}

export default function UsersManagement() {
  const queryClient = useQueryClient();
  const [, setLocation] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState<"all" | "active" | "suspended">("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [balanceModalUser, setBalanceModalUser] = useState<User | null>(null);
  const [balanceAmount, setBalanceAmount] = useState("");
  const [balanceReason, setBalanceReason] = useState("");
  const [balanceOperation, setBalanceOperation] = useState<"add" | "deduct">("add");
  const itemsPerPage = 20;

  // Impersonate user function
  const handleImpersonate = async (user: User) => {
    try {
      // Store admin token for later restoration
      const adminToken = localStorage.getItem('token');
      if (adminToken) {
        localStorage.setItem('admin_token_backup', adminToken);
      }
      
      // Call API to get impersonation token
      const response = await adminAPI.impersonateUser(user.id);
      const { access_token, user_email, user_name } = response.data;
      
      // Store impersonation info
      localStorage.setItem('token', access_token);
      localStorage.setItem('impersonating', 'true');
      localStorage.setItem('impersonated_user', user_email);
      localStorage.setItem('impersonating_user', JSON.stringify({
        id: user.id,
        email: user_email,
        full_name: user_name
      }));
      
      toast.success(`جاري الدخول كمستخدم: ${user_email}`);
      
      // Redirect to dashboard
      window.location.href = '/dashboard';
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'فشل في الدخول كمستخدم');
    }
  };

  // Fetch users
  const { data: users = [], isLoading, refetch } = useQuery({
    queryKey: ["/api/v1/admin/users"],
    queryFn: () => adminAPI.getUsers(0, 500).then(res => res.data),
  });

  // Suspend user mutation
  const suspendMutation = useMutation({
    mutationFn: (userId: number) => adminAPI.suspendUser(userId),
    onSuccess: () => {
      toast.success("تم إيقاف المستخدم بنجاح");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || 'فشل في إيقاف المستخدم';
      if (errorMessage.includes('already')) {
        toast.error('تم معالجة هذا الطلب مسبقاً. جاري تحديث القائمة...');
        queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
      } else {
        toast.error(errorMessage);
      }
    },
  });

  // Activate user mutation
  const activateMutation = useMutation({
    mutationFn: (userId: number) => adminAPI.activateUser(userId),
    onSuccess: () => {
      toast.success("تم تفعيل المستخدم بنجاح");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.detail || 'فشل في تفعيل المستخدم';
      if (errorMessage.includes('already')) {
        toast.error('تم معالجة هذا الطلب مسبقاً. جاري تحديث القائمة...');
        queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
      } else {
        toast.error(errorMessage);
      }
    },
  });

  // Adjust balance mutation
  const adjustBalanceMutation = useMutation({
    mutationFn: ({ userId, data }: { userId: number; data: { amount_usd: number; reason: string; operation: "add" | "deduct" } }) =>
      adminAPI.adjustUserBalance(userId, data),
    onSuccess: (response) => {
      const data = response.data;
      toast.success(
        `تم ${data.operation === "add" ? "إضافة" : "خصم"} $${data.amount_adjusted_usd.toFixed(2)} ${data.operation === "add" ? "إلى" : "من"} رصيد ${data.user_email}`
      );
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
      setBalanceModalUser(null);
      setBalanceAmount("");
      setBalanceReason("");
      setBalanceOperation("add");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || "فشل في تعديل الرصيد");
    },
  });

  const handleAdjustBalance = () => {
    if (!balanceModalUser) return;
    const amount = parseFloat(balanceAmount);
    if (isNaN(amount) || amount <= 0) {
      toast.error("يرجى إدخال مبلغ صحيح");
      return;
    }
    if (!balanceReason.trim()) {
      toast.error("يرجى إدخال سبب التعديل");
      return;
    }
    adjustBalanceMutation.mutate({
      userId: balanceModalUser.id,
      data: {
        amount_usd: amount,
        reason: balanceReason.trim(),
        operation: balanceOperation,
      },
    });
  };

  // Filter users
  const filteredUsers = users.filter((user: User) => {
    const matchesSearch = 
      user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (user.full_name?.toLowerCase() || "").includes(searchTerm.toLowerCase());
    
    const matchesStatus = 
      filterStatus === "all" ||
      (filterStatus === "active" && user.is_active) ||
      (filterStatus === "suspended" && !user.is_active);
    
    return matchesSearch && matchesStatus;
  });

  // Pagination
  const totalPages = Math.ceil(filteredUsers.length / itemsPerPage);
  const paginatedUsers = filteredUsers.slice(
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

  return (
    <div className="min-h-screen bg-[#08080c] p-4 md:p-6 space-y-6">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-violet-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-blue-500/10 rounded-full blur-[100px]" />
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
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white via-violet-200 to-blue-200 bg-clip-text text-transparent">
              إدارة المستخدمين
            </h1>
            <p className="text-white/40 text-sm mt-1">عرض وإدارة جميع المستخدمين</p>
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

      {/* Stats */}
      <div className="relative grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-blue-500/15">
              <Users className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">إجمالي المستخدمين</p>
              <p className="text-2xl font-bold text-white">{users.length}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-emerald-500/15">
              <UserCheck className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">مستخدمين نشطين</p>
              <p className="text-2xl font-bold text-emerald-400">{users.filter((u: User) => u.is_active).length}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-red-500/15">
              <UserX className="w-6 h-6 text-red-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">موقوفين</p>
              <p className="text-2xl font-bold text-red-400">{users.filter((u: User) => !u.is_active).length}</p>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 p-5">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-amber-500/15">
              <Crown className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <p className="text-sm text-white/50">مشرفين</p>
              <p className="text-2xl font-bold text-amber-400">{users.filter((u: User) => u.is_admin).length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="relative flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" />
          <input
            type="text"
            placeholder="بحث بالاسم أو البريد الإلكتروني..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pr-12 pl-4 py-3 rounded-xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 text-white placeholder:text-white/30 focus:outline-none focus:border-violet-500/40 transition-colors"
          />
        </div>
        <div className="flex gap-2">
          {["all", "active", "suspended"].map((status) => (
            <button
              key={status}
              onClick={() => setFilterStatus(status as any)}
              className={cn(
                "px-4 py-3 rounded-xl border transition-all duration-300",
                filterStatus === status
                  ? "bg-violet-500/20 border-violet-500/40 text-violet-300"
                  : "bg-[rgba(18,18,28,0.6)] border-violet-500/15 text-white/50 hover:border-violet-500/30"
              )}
            >
              {status === "all" ? "الكل" : status === "active" ? "نشط" : "موقوف"}
            </button>
          ))}
        </div>
      </div>

      {/* Users Table */}
      <div className="relative rounded-2xl bg-[rgba(18,18,28,0.6)] backdrop-blur-xl border border-violet-500/15 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
        
        {isLoading ? (
          <div className="p-6 space-y-4">
            {[1, 2, 3, 4, 5].map(i => (
              <Skeleton key={i} className="h-16 w-full bg-white/10" />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-right text-white/50 border-b border-violet-500/10">
                  <th className="px-6 py-4 font-medium text-sm">المستخدم</th>
                  <th className="px-6 py-4 font-medium text-sm">الحالة</th>
                  <th className="px-6 py-4 font-medium text-sm">الإيداعات</th>
                  <th className="px-6 py-4 font-medium text-sm">القيمة الحالية</th>
                  <th className="px-6 py-4 font-medium text-sm">تاريخ التسجيل</th>
                  <th className="px-6 py-4 font-medium text-sm">الإجراءات</th>
                </tr>
              </thead>
              <tbody>
                {paginatedUsers.map((user: User) => (
                  <tr
                    key={user.id}
                    className="border-t border-white/5 hover:bg-violet-500/5 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-semibold">
                          {user.full_name?.[0] || user.email[0].toUpperCase()}
                        </div>
                        <div>
                          <p className="text-white font-medium">{user.full_name || "بدون اسم"}</p>
                          <p className="text-white/50 text-sm">{user.email}</p>
                        </div>
                        {user.is_admin && (
                          <span className="px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 text-xs border border-amber-500/25">
                            أدمن
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={cn(
                        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border",
                        user.is_active
                          ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                          : "bg-red-500/15 text-red-400 border-red-500/25"
                      )}>
                        <span className={cn(
                          "w-1.5 h-1.5 rounded-full",
                          user.is_active ? "bg-emerald-400" : "bg-red-400"
                        )} />
                        {user.is_active ? "نشط" : "موقوف"}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-white font-medium" dir="ltr">
                      {formatCurrency(user.total_deposited || 0)}
                    </td>
                    <td className="px-6 py-4 text-emerald-400 font-medium" dir="ltr">
                      {formatCurrency(user.current_value_usd || user.current_value || 0)}
                    </td>
                    <td className="px-6 py-4 text-white/50 text-sm">
                      {format(new Date(user.created_at), "dd/MM/yyyy")}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setSelectedUser(user)}
                          className="p-2 rounded-lg bg-violet-500/10 text-violet-400 hover:bg-violet-500/20 transition-colors"
                          title="عرض التفاصيل"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setBalanceModalUser(user)}
                          className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                          title="تعديل الرصيد"
                        >
                          <Wallet className="w-4 h-4" />
                        </button>
                        {!user.is_admin && (
                          <button
                            onClick={() => handleImpersonate(user)}
                            className="p-2 rounded-lg bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 transition-colors"
                            title="الدخول كمستخدم"
                          >
                            <LogIn className="w-4 h-4" />
                          </button>
                        )}
                        {user.is_active ? (
                          <button
                            onClick={() => suspendMutation.mutate(user.id)}
                            disabled={suspendMutation.isPending}
                            className="p-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-50"
                          >
                            <UserX className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => activateMutation.mutate(user.id)}
                            disabled={activateMutation.isPending}
                            className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
                          >
                            <UserCheck className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-violet-500/10">
            <p className="text-white/50 text-sm">
              عرض {((currentPage - 1) * itemsPerPage) + 1} - {Math.min(currentPage * itemsPerPage, filteredUsers.length)} من {filteredUsers.length}
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

      {/* Balance Adjustment Modal */}
      {balanceModalUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl bg-[rgba(18,18,28,0.95)] backdrop-blur-xl border border-violet-500/20 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-white">تعديل الرصيد</h3>
              <button
                onClick={() => {
                  setBalanceModalUser(null);
                  setBalanceAmount("");
                  setBalanceReason("");
                  setBalanceOperation("add");
                }}
                className="p-2 rounded-lg bg-white/5 text-white/50 hover:bg-white/10 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="flex items-center gap-4 p-4 rounded-xl bg-white/5">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white text-lg font-bold">
                {balanceModalUser.full_name?.[0] || balanceModalUser.email[0].toUpperCase()}
              </div>
              <div>
                <p className="text-white font-semibold">{balanceModalUser.full_name || "بدون اسم"}</p>
                <p className="text-white/50 text-sm">{balanceModalUser.email}</p>
                <p className="text-emerald-400 text-sm mt-1">
                  الرصيد الحالي: {formatCurrency(balanceModalUser.current_value_usd || balanceModalUser.current_value || 0)}
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex gap-2">
                <button
                  onClick={() => setBalanceOperation("add")}
                  className={cn(
                    "flex-1 py-3 rounded-xl border transition-all duration-300 flex items-center justify-center gap-2 font-medium",
                    balanceOperation === "add"
                      ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-400"
                      : "bg-white/5 border-white/10 text-white/50 hover:border-white/20"
                  )}
                >
                  <Plus className="w-4 h-4" />
                  إضافة رصيد
                </button>
                <button
                  onClick={() => setBalanceOperation("deduct")}
                  className={cn(
                    "flex-1 py-3 rounded-xl border transition-all duration-300 flex items-center justify-center gap-2 font-medium",
                    balanceOperation === "deduct"
                      ? "bg-red-500/20 border-red-500/40 text-red-400"
                      : "bg-white/5 border-white/10 text-white/50 hover:border-white/20"
                  )}
                >
                  <Minus className="w-4 h-4" />
                  خصم رصيد
                </button>
              </div>

              <div className="space-y-2">
                <Label className="text-white/70">المبلغ (دولار)</Label>
                <Input
                  type="number"
                  placeholder="0.00"
                  value={balanceAmount}
                  onChange={(e) => setBalanceAmount(e.target.value)}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white text-lg"
                  dir="ltr"
                />
              </div>

              <div className="space-y-2">
                <Label className="text-white/70">سبب التعديل</Label>
                <Input
                  type="text"
                  placeholder="مثال: مكافأة إحالة / تصحيح خطأ..."
                  value={balanceReason}
                  onChange={(e) => setBalanceReason(e.target.value)}
                  className="bg-[#1a1a2e] border-violet-500/20 text-white"
                />
              </div>
            </div>

            {balanceAmount && parseFloat(balanceAmount) > 0 && (
              <div className={cn(
                "p-4 rounded-xl border",
                balanceOperation === "add"
                  ? "bg-emerald-500/10 border-emerald-500/20"
                  : "bg-red-500/10 border-red-500/20"
              )}>
                <p className={balanceOperation === "add" ? "text-emerald-400" : "text-red-400"}>
                  سيتم {balanceOperation === "add" ? "إضافة" : "خصم"} <strong>${parseFloat(balanceAmount).toFixed(2)}</strong> {balanceOperation === "add" ? "إلى" : "من"} رصيد المستخدم
                </p>
                <p className="text-white/50 text-sm mt-1">
                  الرصيد الجديد المتوقع: {formatCurrency(
                    (balanceModalUser.current_value_usd || balanceModalUser.current_value || 0) + 
                    (balanceOperation === "add" ? parseFloat(balanceAmount) : -parseFloat(balanceAmount))
                  )}
                </p>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleAdjustBalance}
                disabled={adjustBalanceMutation.isPending || !balanceAmount || !balanceReason}
                className={cn(
                  "flex-1 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
                  balanceOperation === "add"
                    ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                    : "bg-red-500 hover:bg-red-600 text-white"
                )}
              >
                {adjustBalanceMutation.isPending ? "جاري التنفيذ..." : "تأكيد التعديل"}
              </button>
              <button
                onClick={() => {
                  setBalanceModalUser(null);
                  setBalanceAmount("");
                  setBalanceReason("");
                  setBalanceOperation("add");
                }}
                className="flex-1 py-3 rounded-xl bg-white/5 text-white/70 hover:bg-white/10 transition-colors font-medium"
              >
                إلغاء
              </button>
            </div>
          </div>
        </div>
      )}

      {/* User Details Modal */}
      {selectedUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl bg-[rgba(18,18,28,0.95)] backdrop-blur-xl border border-violet-500/20 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-white">تفاصيل المستخدم</h3>
              <button
                onClick={() => setSelectedUser(null)}
                className="p-2 rounded-lg bg-white/5 text-white/50 hover:bg-white/10 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold">
                {selectedUser.full_name?.[0] || selectedUser.email[0].toUpperCase()}
              </div>
              <div>
                <p className="text-white text-lg font-semibold">{selectedUser.full_name || "بدون اسم"}</p>
                <p className="text-white/50">{selectedUser.email}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">الحالة</p>
                <span className={cn(
                  "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold",
                  selectedUser.is_active
                    ? "bg-emerald-500/15 text-emerald-400"
                    : "bg-red-500/15 text-red-400"
                )}>
                  {selectedUser.is_active ? "نشط" : "موقوف"}
                </span>
              </div>
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">الصلاحية</p>
                <span className={cn(
                  "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold",
                  selectedUser.is_admin
                    ? "bg-amber-500/15 text-amber-400"
                    : "bg-blue-500/15 text-blue-400"
                )}>
                  {selectedUser.is_admin ? "مشرف" : "مستخدم"}
                </span>
              </div>
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">إجمالي الإيداعات</p>
                <p className="text-white font-semibold" dir="ltr">{formatCurrency(selectedUser.total_deposited || 0)}</p>
              </div>
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">القيمة الحالية</p>
                <p className="text-emerald-400 font-semibold" dir="ltr">{formatCurrency(selectedUser.current_value_usd || selectedUser.current_value || 0)}</p>
              </div>
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">تاريخ التسجيل</p>
                <p className="text-white">{format(new Date(selectedUser.created_at), "dd/MM/yyyy HH:mm")}</p>
              </div>
              <div className="p-4 rounded-xl bg-white/5">
                <p className="text-white/50 text-sm mb-1">آخر دخول</p>
                <p className="text-white">
                  {selectedUser.last_login 
                    ? format(new Date(selectedUser.last_login), "dd/MM/yyyy HH:mm")
                    : "لم يسجل دخول"
                  }
                </p>
              </div>
            </div>

            <div className="flex gap-3 flex-wrap">
              <button
                onClick={() => {
                  setBalanceModalUser(selectedUser);
                  setSelectedUser(null);
                }}
                className="flex-1 py-3 rounded-xl bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 hover:bg-emerald-500/25 transition-colors font-medium flex items-center justify-center gap-2"
              >
                <Wallet className="w-4 h-4" />
                تعديل الرصيد
              </button>
              {!selectedUser.is_admin && (
                <button
                  onClick={() => {
                    handleImpersonate(selectedUser);
                    setSelectedUser(null);
                  }}
                  className="flex-1 py-3 rounded-xl bg-blue-500/15 text-blue-400 border border-blue-500/25 hover:bg-blue-500/25 transition-colors font-medium flex items-center justify-center gap-2"
                >
                  <LogIn className="w-4 h-4" />
                  الدخول كمستخدم
                </button>
              )}
              {selectedUser.is_active ? (
                <button
                  onClick={() => {
                    suspendMutation.mutate(selectedUser.id);
                    setSelectedUser(null);
                  }}
                  className="flex-1 py-3 rounded-xl bg-red-500/15 text-red-400 border border-red-500/25 hover:bg-red-500/25 transition-colors font-medium"
                >
                  إيقاف المستخدم
                </button>
              ) : (
                <button
                  onClick={() => {
                    activateMutation.mutate(selectedUser.id);
                    setSelectedUser(null);
                  }}
                  className="flex-1 py-3 rounded-xl bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 hover:bg-emerald-500/25 transition-colors font-medium"
                >
                  تفعيل المستخدم
                </button>
              )}
              <button
                onClick={() => setSelectedUser(null)}
                className="flex-1 py-3 rounded-xl bg-white/5 text-white/70 hover:bg-white/10 transition-colors font-medium"
              >
                إغلاق
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
