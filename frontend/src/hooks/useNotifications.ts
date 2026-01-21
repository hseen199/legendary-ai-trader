// Hook للإشعارات في الوقت الفعلي
// /opt/asinax/frontend/src/hooks/useNotifications.ts

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import api from '@/services/api';
import toast from 'react-hot-toast';
import { useLanguage } from '@/lib/i18n';

interface Notification {
  id: number;
  type: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
  data?: any;
}

interface UseNotificationsOptions {
  enabled?: boolean;
  refetchInterval?: number; // بالمللي ثانية
  showToastOnNew?: boolean;
}

export function useNotifications(options: UseNotificationsOptions = {}) {
  const {
    enabled = true,
    refetchInterval = 10000, // تحديث كل 10 ثواني (بدلاً من 30)
    showToastOnNew = true,
  } = options;

  const { language } = useLanguage();
  const queryClient = useQueryClient();
  const [previousCount, setPreviousCount] = useState<number | null>(null);

  // جلب عدد الإشعارات غير المقروءة
  const { data: unreadCount = 0, refetch: refetchCount } = useQuery({
    queryKey: ['notifications-unread-count'],
    queryFn: async () => {
      try {
        const res = await api.get('/notifications/unread-count');
        return res.data.count || 0;
      } catch {
        return 0;
      }
    },
    enabled,
    refetchInterval,
    staleTime: 5000,
  });

  // جلب الإشعارات
  const {
    data: notifications = [],
    isLoading,
    refetch: refetchNotifications,
  } = useQuery<Notification[]>({
    queryKey: ['notifications'],
    queryFn: async () => {
      try {
        const res = await api.get('/notifications');
        return res.data;
      } catch {
        return [];
      }
    },
    enabled,
    refetchInterval,
    staleTime: 5000,
  });

  // إظهار toast عند وصول إشعار جديد
  useEffect(() => {
    if (previousCount !== null && unreadCount > previousCount && showToastOnNew) {
      const newCount = unreadCount - previousCount;
      toast(
        language === 'ar'
          ? `لديك ${newCount} إشعار${newCount > 1 ? 'ات' : ''} جديد${newCount > 1 ? 'ة' : ''}`
          : `You have ${newCount} new notification${newCount > 1 ? 's' : ''}`,
        {
          icon: '🔔',
          duration: 4000,
        }
      );
    }
    setPreviousCount(unreadCount);
  }, [unreadCount, previousCount, showToastOnNew, language]);

  // تحديد إشعار كمقروء
  const markAsRead = useCallback(async (notificationId: number) => {
    try {
      await api.post(`/notifications/${notificationId}/read`);
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      queryClient.invalidateQueries({ queryKey: ['notifications-unread-count'] });
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  }, [queryClient]);

  // تحديد جميع الإشعارات كمقروءة
  const markAllAsRead = useCallback(async () => {
    try {
      await api.post('/notifications/read-all');
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      queryClient.invalidateQueries({ queryKey: ['notifications-unread-count'] });
      toast.success(
        language === 'ar'
          ? 'تم تحديد جميع الإشعارات كمقروءة'
          : 'All notifications marked as read'
      );
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
    }
  }, [queryClient, language]);

  // حذف إشعار
  const deleteNotification = useCallback(async (notificationId: number) => {
    try {
      await api.delete(`/notifications/${notificationId}`);
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      queryClient.invalidateQueries({ queryKey: ['notifications-unread-count'] });
      toast.success(
        language === 'ar' ? 'تم حذف الإشعار' : 'Notification deleted'
      );
    } catch (error) {
      console.error('Error deleting notification:', error);
    }
  }, [queryClient, language]);

  // تحديث يدوي
  const refresh = useCallback(() => {
    refetchCount();
    refetchNotifications();
  }, [refetchCount, refetchNotifications]);

  return {
    notifications,
    unreadCount,
    isLoading,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    refresh,
  };
}

export default useNotifications;
