import { useState } from "react";
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
} from "lucide-react";
import { cn } from "../../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";

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
}

export default function UsersManagement() {
  const queryClient = useQueryClient();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState<"all" | "active" | "suspended">("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const itemsPerPage = 20;

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
    onError: () => {
      toast.error("فشل في إيقاف المستخدم");
    },
  });

  // Activate user mutation
  const activateMutation = useMutation({
    mutationFn: (userId: number) => adminAPI.activateUser(userId),
    onSuccess: () => {
      toast.success("تم تفعيل المستخدم بنجاح");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/admin/users"] });
    },
    onError: () => {
      toast.error("فشل في تفعيل المستخدم");
    },
  });

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

  // Stats
  const totalUsers = users.length;
  const activeUsers = users.filter((u: User) => u.is_active).length;
  const suspendedUsers = users.filter((u: User) => !u.is_active).length;
  const vipUsers = users.filter((u: User) => (u.vip_level || 0) > 0).length;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  const getVipBadge = (level: number) => {
    const colors: Record<number, string> = {
      1: "bg-blue-500/10 text-blue-500 border-blue-500/20",
      2: "bg-purple-500/10 text-purple-500 border-purple-500/20",
      3: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
      4: "bg-gradient-to-r from-yellow-500 to-orange-500 text-white border-0",
    };
    const names: Record<number, string> = {
      1: "برونزي",
      2: "فضي",
      3: "ذهبي",
      4: "بلاتيني",
    };
    return (
      <Badge variant="outline" className={colors[level] || ""}>
        <Crown className="w-3 h-3 ml-1" />
        {names[level] || `VIP ${level}`}
      </Badge>
    );
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">إدارة المستخدمين</h1>
          <p className="text-muted-foreground text-sm">عرض وإدارة جميع المستخدمين</p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          تحديث
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">إجمالي المستخدمين</span>
            </div>
            <p className="text-xl font-bold">{totalUsers}</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <UserCheck className="w-4 h-4 text-green-500" />
              <span className="text-sm text-muted-foreground">نشط</span>
            </div>
            <p className="text-xl font-bold text-green-500">{activeUsers}</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <UserX className="w-4 h-4 text-destructive" />
              <span className="text-sm text-muted-foreground">موقوف</span>
            </div>
            <p className="text-xl font-bold text-destructive">{suspendedUsers}</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Crown className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-muted-foreground">VIP</span>
            </div>
            <p className="text-xl font-bold text-yellow-500">{vipUsers}</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="بحث بالإيميل أو الاسم..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pr-10 px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
              />
            </div>

            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as any)}
                className="px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
              >
                <option value="all">جميع الحالات</option>
                <option value="active">نشط</option>
                <option value="suspended">موقوف</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Users Table */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-6 space-y-3">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : paginatedUsers.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-muted">
                    <tr className="text-right text-muted-foreground">
                      <th className="px-4 py-3 font-medium">المستخدم</th>
                      <th className="px-4 py-3 font-medium">الحالة</th>
                      <th className="px-4 py-3 font-medium">VIP</th>
                      <th className="px-4 py-3 font-medium">الإيداعات</th>
                      <th className="px-4 py-3 font-medium">القيمة الحالية</th>
                      <th className="px-4 py-3 font-medium">تاريخ التسجيل</th>
                      <th className="px-4 py-3 font-medium">الإجراءات</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedUsers.map((user: User) => (
                      <tr
                        key={user.id}
                        className="border-t border-border hover:bg-muted/50 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                              <span className="text-primary font-medium">
                                {user.email[0].toUpperCase()}
                              </span>
                            </div>
                            <div>
                              <p className="font-medium">{user.full_name || "بدون اسم"}</p>
                              <p className="text-sm text-muted-foreground" dir="ltr">{user.email}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          {user.is_admin ? (
                            <Badge variant="outline" className="bg-purple-500/10 text-purple-500 border-purple-500/20">
                              <Shield className="w-3 h-3 ml-1" />
                              أدمن
                            </Badge>
                          ) : user.is_active ? (
                            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                              <UserCheck className="w-3 h-3 ml-1" />
                              نشط
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/20">
                              <UserX className="w-3 h-3 ml-1" />
                              موقوف
                            </Badge>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {user.vip_level && user.vip_level > 0 ? (
                            getVipBadge(user.vip_level)
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </td>
                        <td className="px-4 py-3" dir="ltr">
                          {formatCurrency(user.total_deposited || 0)}
                        </td>
                        <td className="px-4 py-3" dir="ltr">
                          {formatCurrency(user.current_value || 0)}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {format(new Date(user.created_at), "dd/MM/yyyy")}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setSelectedUser(user)}
                              className="p-2 hover:bg-muted rounded-lg transition-colors"
                              title="عرض التفاصيل"
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            {!user.is_admin && (
                              user.is_active ? (
                                <button
                                  onClick={() => suspendMutation.mutate(user.id)}
                                  disabled={suspendMutation.isPending}
                                  className="p-2 hover:bg-destructive/10 text-destructive rounded-lg transition-colors"
                                  title="إيقاف"
                                >
                                  <UserX className="w-4 h-4" />
                                </button>
                              ) : (
                                <button
                                  onClick={() => activateMutation.mutate(user.id)}
                                  disabled={activateMutation.isPending}
                                  className="p-2 hover:bg-green-500/10 text-green-500 rounded-lg transition-colors"
                                  title="تفعيل"
                                >
                                  <UserCheck className="w-4 h-4" />
                                </button>
                              )
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-border">
                  <p className="text-sm text-muted-foreground">
                    عرض {(currentPage - 1) * itemsPerPage + 1} إلى{" "}
                    {Math.min(currentPage * itemsPerPage, filteredUsers.length)} من{" "}
                    {filteredUsers.length} مستخدم
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="p-2 rounded-lg hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-muted-foreground">
                      صفحة {currentPage} من {totalPages}
                    </span>
                    <button
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="p-2 rounded-lg hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">لا يوجد مستخدمين</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* User Details Modal */}
      {selectedUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>تفاصيل المستخدم</CardTitle>
                <button
                  onClick={() => setSelectedUser(null)}
                  className="p-2 hover:bg-muted rounded-lg"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <span className="text-2xl text-primary font-medium">
                    {selectedUser.email[0].toUpperCase()}
                  </span>
                </div>
                <div>
                  <p className="text-lg font-medium">{selectedUser.full_name || "بدون اسم"}</p>
                  <p className="text-muted-foreground" dir="ltr">{selectedUser.email}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">الحالة</p>
                  <p className="font-medium">{selectedUser.is_active ? "نشط" : "موقوف"}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">مستوى VIP</p>
                  <p className="font-medium">{selectedUser.vip_level || "عادي"}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">إجمالي الإيداعات</p>
                  <p className="font-medium" dir="ltr">{formatCurrency(selectedUser.total_deposited || 0)}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">القيمة الحالية</p>
                  <p className="font-medium" dir="ltr">{formatCurrency(selectedUser.current_value || 0)}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">تاريخ التسجيل</p>
                  <p className="font-medium">{format(new Date(selectedUser.created_at), "dd/MM/yyyy")}</p>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">آخر دخول</p>
                  <p className="font-medium">
                    {selectedUser.last_login 
                      ? format(new Date(selectedUser.last_login), "dd/MM/yyyy HH:mm")
                      : "-"
                    }
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                {!selectedUser.is_admin && (
                  selectedUser.is_active ? (
                    <button
                      onClick={() => {
                        suspendMutation.mutate(selectedUser.id);
                        setSelectedUser(null);
                      }}
                      className="flex-1 py-2 bg-destructive text-destructive-foreground rounded-lg font-medium hover:bg-destructive/90"
                    >
                      إيقاف المستخدم
                    </button>
                  ) : (
                    <button
                      onClick={() => {
                        activateMutation.mutate(selectedUser.id);
                        setSelectedUser(null);
                      }}
                      className="flex-1 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-500/90"
                    >
                      تفعيل المستخدم
                    </button>
                  )
                )}
                <button
                  onClick={() => setSelectedUser(null)}
                  className="flex-1 py-2 bg-muted rounded-lg font-medium hover:bg-muted/80"
                >
                  إغلاق
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
