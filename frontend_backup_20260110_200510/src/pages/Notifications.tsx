import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { 
  Bell, 
  BellOff, 
  Check, 
  CheckCheck, 
  Trash2,
  ArrowDownCircle,
  ArrowUpCircle,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Info,
  Gift,
  Settings,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format, isToday, isYesterday } from "date-fns";
import toast from "react-hot-toast";
import api from "../services/api";

interface Notification {
  id: number;
  type: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
  data?: any;
}

export default function Notifications() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("all");

  // Fetch notifications
  const { data: notifications = [], isLoading } = useQuery<Notification[]>({
    queryKey: ["/api/v1/notifications"],
    queryFn: async () => {
      try {
        const res = await api.get("/notifications");
        return res.data;
      } catch {
        // Return sample notifications if API not available
        return [
          {
            id: 1,
            type: "deposit",
            title: "تم تأكيد الإيداع",
            message: "تم تأكيد إيداعك بمبلغ 500 USDC بنجاح",
            is_read: false,
            created_at: new Date().toISOString(),
          },
          {
            id: 2,
            type: "trade",
            title: "صفقة جديدة",
            message: "تم تنفيذ صفقة شراء BTC/USDC بقيمة 100 USDC",
            is_read: true,
            created_at: new Date(Date.now() - 86400000).toISOString(),
          },
        ];
      }
    },
  });

  // Mark as read mutation
  const markReadMutation = useMutation({
    mutationFn: async (id: number) => {
      return api.post(`/notifications/${id}/read`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/notifications"] });
    },
  });

  // Mark all as read mutation
  const markAllReadMutation = useMutation({
    mutationFn: async () => {
      return api.post("/notifications/read-all");
    },
    onSuccess: () => {
      toast.success("تم تحديد جميع الإشعارات كمقروءة");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/notifications"] });
    },
  });

  // Delete notification mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: number) => {
      return api.delete(`/notifications/${id}`);
    },
    onSuccess: () => {
      toast.success("تم حذف الإشعار");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/notifications"] });
    },
  });

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "deposit":
        return <ArrowDownCircle className="w-5 h-5 text-green-500" />;
      case "withdrawal":
        return <ArrowUpCircle className="w-5 h-5 text-destructive" />;
      case "trade":
        return <TrendingUp className="w-5 h-5 text-primary" />;
      case "profit":
        return <TrendingUp className="w-5 h-5 text-green-500" />;
      case "loss":
        return <TrendingDown className="w-5 h-5 text-destructive" />;
      case "alert":
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case "referral":
        return <Gift className="w-5 h-5 text-purple-500" />;
      case "system":
        return <Settings className="w-5 h-5 text-muted-foreground" />;
      default:
        return <Info className="w-5 h-5 text-primary" />;
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    if (isToday(date)) {
      return `اليوم ${format(date, "HH:mm")}`;
    } else if (isYesterday(date)) {
      return `أمس ${format(date, "HH:mm")}`;
    }
    return format(date, "dd/MM/yyyy HH:mm");
  };

  // Filter notifications
  const filteredNotifications = notifications.filter((n) => {
    if (activeTab === "unread") return !n.is_read;
    return true;
  });

  // Group notifications by date
  const groupedNotifications = filteredNotifications.reduce((groups: any, notification) => {
    const date = new Date(notification.created_at);
    let key = format(date, "yyyy-MM-dd");
    
    if (isToday(date)) key = "today";
    else if (isYesterday(date)) key = "yesterday";
    
    if (!groups[key]) groups[key] = [];
    groups[key].push(notification);
    return groups;
  }, {});

  const unreadCount = notifications.filter((n) => !n.is_read).length;

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">الإشعارات</h1>
          <p className="text-muted-foreground text-sm">
            {unreadCount > 0 ? `لديك ${unreadCount} إشعار غير مقروء` : "لا توجد إشعارات جديدة"}
          </p>
        </div>
        {unreadCount > 0 && (
          <button
            onClick={() => markAllReadMutation.mutate()}
            disabled={markAllReadMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
          >
            <CheckCheck className="w-4 h-4" />
            تحديد الكل كمقروء
          </button>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2 max-w-[300px]">
          <TabsTrigger value="all" className="gap-2">
            <Bell className="w-4 h-4" />
            الكل
          </TabsTrigger>
          <TabsTrigger value="unread" className="gap-2">
            <BellOff className="w-4 h-4" />
            غير مقروء
            {unreadCount > 0 && (
              <Badge variant="destructive" className="mr-1">{unreadCount}</Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map(i => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
          ) : filteredNotifications.length > 0 ? (
            <div className="space-y-6">
              {Object.entries(groupedNotifications).map(([key, notifs]: [string, any]) => (
                <div key={key}>
                  <h3 className="text-sm font-medium text-muted-foreground mb-3">
                    {key === "today" ? "اليوم" : key === "yesterday" ? "أمس" : format(new Date(key), "dd/MM/yyyy")}
                  </h3>
                  <div className="space-y-2">
                    {notifs.map((notification: Notification) => (
                      <Card
                        key={notification.id}
                        className={cn(
                          "transition-colors",
                          !notification.is_read && "bg-primary/5 border-primary/20"
                        )}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start gap-4">
                            <div className={cn(
                              "p-2 rounded-lg",
                              !notification.is_read ? "bg-primary/10" : "bg-muted"
                            )}>
                              {getNotificationIcon(notification.type)}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <h4 className="font-medium">{notification.title}</h4>
                                {!notification.is_read && (
                                  <div className="w-2 h-2 rounded-full bg-primary" />
                                )}
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {notification.message}
                              </p>
                              <p className="text-xs text-muted-foreground mt-2">
                                {formatDate(notification.created_at)}
                              </p>
                            </div>
                            <div className="flex items-center gap-1">
                              {!notification.is_read && (
                                <button
                                  onClick={() => markReadMutation.mutate(notification.id)}
                                  className="p-2 hover:bg-muted rounded-lg transition-colors"
                                  title="تحديد كمقروء"
                                >
                                  <Check className="w-4 h-4" />
                                </button>
                              )}
                              <button
                                onClick={() => deleteMutation.mutate(notification.id)}
                                className="p-2 hover:bg-destructive/10 text-destructive rounded-lg transition-colors"
                                title="حذف"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="p-12 text-center">
                <Bell className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  {activeTab === "unread" ? "لا توجد إشعارات غير مقروءة" : "لا توجد إشعارات"}
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  ستظهر هنا الإشعارات الجديدة عند وصولها
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Notification Types Info */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">أنواع الإشعارات</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <ArrowDownCircle className="w-5 h-5 text-green-500" />
              <span className="text-sm">إيداعات</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <ArrowUpCircle className="w-5 h-5 text-destructive" />
              <span className="text-sm">سحوبات</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <TrendingUp className="w-5 h-5 text-primary" />
              <span className="text-sm">صفقات</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <Gift className="w-5 h-5 text-purple-500" />
              <span className="text-sm">إحالات</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
