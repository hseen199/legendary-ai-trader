// صفحة الإشعارات الكاملة
// يُضاف إلى /opt/asinax/frontend/src/pages/NotificationsPage.tsx

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { format } from "date-fns";
import { ar } from "date-fns/locale";
import {
  Bell,
  Check,
  CheckCheck,
  Trash2,
  ArrowDownCircle,
  ArrowUpCircle,
  Wallet,
  Gift,
  AlertCircle,
  Settings,
  Filter,
  RefreshCw
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import api from "@/services/api";
import { toast } from "@/hooks/use-toast";

interface Notification {
  id: number;
  type: string;
  title: string;
  message: string;
  data?: Record<string, any>;
  is_read: boolean;
  read_at?: string;
  created_at: string;
}

export default function NotificationsPage() {
  const queryClient = useQueryClient();
  const [language, setLanguage] = useState<"ar" | "en">("ar");
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    const savedLang = localStorage.getItem("language") || "ar";
    setLanguage(savedLang as "ar" | "en");
  }, []);

  // Fetch all notifications
  const { data: notifications = [], isLoading, refetch } = useQuery({
    queryKey: ["notifications", filter],
    queryFn: async () => {
      let url = "/notifications?limit=50";
      if (filter === "unread") {
        url += "&unread_only=true";
      }
      const res = await api.get(url);
      return res.data;
    },
    refetchInterval: 30000,
  });

  // Fetch unread count
  const { data: countData } = useQuery({
    queryKey: ["notifications-count"],
    queryFn: async () => {
      const res = await api.get("/notifications/count");
      return res.data;
    },
    refetchInterval: 15000,
  });

  // Mark as read mutation
  const markAsReadMutation = useMutation({
    mutationFn: async (id: number) => {
      await api.post(`/notifications/${id}/read`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      queryClient.invalidateQueries({ queryKey: ["notifications-count"] });
    },
  });

  // Mark all as read mutation
  const markAllAsReadMutation = useMutation({
    mutationFn: async () => {
      await api.post("/notifications/read-all");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      queryClient.invalidateQueries({ queryKey: ["notifications-count"] });
      toast({ 
        title: language === "ar" ? "تم تحديد الكل كمقروء" : "All marked as read" 
      });
    },
  });

  const unreadCount = countData?.unread_count || 0;

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "deposit":
        return <ArrowDownCircle className="w-5 h-5 text-green-500" />;
      case "withdrawal":
        return <ArrowUpCircle className="w-5 h-5 text-orange-500" />;
      case "balance":
        return <Wallet className="w-5 h-5 text-blue-500" />;
      case "referral":
        return <Gift className="w-5 h-5 text-purple-500" />;
      case "system":
        return <Settings className="w-5 h-5 text-gray-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const getTypeBadge = (type: string) => {
    const typeConfig: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
      deposit: { label: language === "ar" ? "إيداع" : "Deposit", variant: "default" },
      withdrawal: { label: language === "ar" ? "سحب" : "Withdrawal", variant: "secondary" },
      balance: { label: language === "ar" ? "رصيد" : "Balance", variant: "outline" },
      referral: { label: language === "ar" ? "إحالة" : "Referral", variant: "default" },
      system: { label: language === "ar" ? "نظام" : "System", variant: "secondary" },
    };
    const config = typeConfig[type] || { label: type, variant: "outline" as const };
    return <Badge variant={config.variant} className="text-xs">{config.label}</Badge>;
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return format(date, "dd/MM/yyyy HH:mm", { locale: language === "ar" ? ar : undefined });
  };

  const formatRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return language === "ar" ? "الآن" : "Just now";
    if (diffMins < 60) return language === "ar" ? `منذ ${diffMins} دقيقة` : `${diffMins}m ago`;
    if (diffHours < 24) return language === "ar" ? `منذ ${diffHours} ساعة` : `${diffHours}h ago`;
    if (diffDays < 7) return language === "ar" ? `منذ ${diffDays} يوم` : `${diffDays}d ago`;
    return formatTime(dateString);
  };

  // Filter notifications
  const filteredNotifications = filter === "all" 
    ? notifications 
    : filter === "unread" 
      ? notifications.filter((n: Notification) => !n.is_read)
      : notifications.filter((n: Notification) => n.type === filter);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Bell className="w-6 h-6" />
            {language === "ar" ? "الإشعارات" : "Notifications"}
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            {unreadCount > 0 
              ? (language === "ar" 
                  ? `لديك ${unreadCount} إشعار غير مقروء` 
                  : `You have ${unreadCount} unread notification(s)`)
              : (language === "ar" 
                  ? "لا توجد إشعارات جديدة" 
                  : "No new notifications")
            }
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Filter */}
          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger className="w-[140px]">
              <Filter className="w-4 h-4 ml-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{language === "ar" ? "الكل" : "All"}</SelectItem>
              <SelectItem value="unread">{language === "ar" ? "غير مقروء" : "Unread"}</SelectItem>
              <SelectItem value="deposit">{language === "ar" ? "إيداع" : "Deposit"}</SelectItem>
              <SelectItem value="withdrawal">{language === "ar" ? "سحب" : "Withdrawal"}</SelectItem>
              <SelectItem value="referral">{language === "ar" ? "إحالة" : "Referral"}</SelectItem>
              <SelectItem value="system">{language === "ar" ? "نظام" : "System"}</SelectItem>
            </SelectContent>
          </Select>

          {/* Refresh */}
          <Button variant="outline" size="icon" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4" />
          </Button>

          {/* Mark all as read */}
          {unreadCount > 0 && (
            <Button 
              variant="outline" 
              onClick={() => markAllAsReadMutation.mutate()}
              disabled={markAllAsReadMutation.isPending}
            >
              <CheckCheck className="w-4 h-4 ml-2" />
              {language === "ar" ? "قراءة الكل" : "Mark all read"}
            </Button>
          )}
        </div>
      </div>

      {/* Notifications List */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-4 space-y-4">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="flex gap-4">
                  <Skeleton className="w-10 h-10 rounded-full" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-1/3" />
                    <Skeleton className="h-3 w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          ) : filteredNotifications.length > 0 ? (
            <div className="divide-y">
              {filteredNotifications.map((notification: Notification) => (
                <div
                  key={notification.id}
                  className={cn(
                    "p-4 hover:bg-muted/50 transition-colors cursor-pointer",
                    !notification.is_read && "bg-primary/5"
                  )}
                  onClick={() => {
                    if (!notification.is_read) {
                      markAsReadMutation.mutate(notification.id);
                    }
                  }}
                >
                  <div className="flex gap-4">
                    {/* Icon */}
                    <div className={cn(
                      "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0",
                      notification.type === "deposit" && "bg-green-500/10",
                      notification.type === "withdrawal" && "bg-orange-500/10",
                      notification.type === "balance" && "bg-blue-500/10",
                      notification.type === "referral" && "bg-purple-500/10",
                      notification.type === "system" && "bg-gray-500/10",
                    )}>
                      {getNotificationIcon(notification.type)}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className={cn(
                            "text-sm",
                            !notification.is_read && "font-semibold"
                          )}>
                            {notification.title}
                          </p>
                          {getTypeBadge(notification.type)}
                          {!notification.is_read && (
                            <div className="w-2 h-2 rounded-full bg-primary" />
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground whitespace-nowrap">
                          {formatRelativeTime(notification.created_at)}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {notification.message}
                      </p>
                      {notification.data && Object.keys(notification.data).length > 0 && (
                        <div className="mt-2 p-2 bg-muted rounded text-xs">
                          {notification.data.amount && (
                            <span className="font-mono">
                              {language === "ar" ? "المبلغ: " : "Amount: "}
                              ${notification.data.amount}
                            </span>
                          )}
                          {notification.data.status && (
                            <span className="mr-4">
                              {language === "ar" ? "الحالة: " : "Status: "}
                              {notification.data.status}
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    {!notification.is_read && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="flex-shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          markAsReadMutation.mutate(notification.id);
                        }}
                      >
                        <Check className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-12 text-center text-muted-foreground">
              <Bell className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">
                {language === "ar" ? "لا توجد إشعارات" : "No notifications"}
              </p>
              <p className="text-sm mt-1">
                {filter !== "all" 
                  ? (language === "ar" 
                      ? "جرب تغيير الفلتر لرؤية المزيد" 
                      : "Try changing the filter to see more")
                  : (language === "ar" 
                      ? "ستظهر الإشعارات هنا عند وصولها" 
                      : "Notifications will appear here when they arrive")
                }
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
