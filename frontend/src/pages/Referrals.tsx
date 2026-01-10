import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { 
  Users, 
  Link2, 
  Copy, 
  CheckCircle, 
  Gift, 
  TrendingUp,
  Share2,
  DollarSign,
  UserPlus,
  Crown,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import api from "../services/api";

interface Referral {
  id: number;
  email: string;
  full_name?: string;
  created_at: string;
  total_deposited: number;
  status: string;
}

interface ReferralStats {
  total_referrals: number;
  active_referrals: number;
  total_earnings: number;
  pending_earnings: number;
  referral_code: string;
}

export default function Referrals() {
  const { user } = useAuth();
  const [copied, setCopied] = useState(false);

  // Fetch referral stats
  const { data: stats, isLoading: loadingStats } = useQuery<ReferralStats>({
    queryKey: ["/api/v1/referrals/stats"],
    queryFn: async () => {
      try {
        const res = await api.get("/referrals/stats");
        return res.data;
      } catch {
        // Return default stats if API not available
        return {
          total_referrals: 0,
          active_referrals: 0,
          total_earnings: 0,
          pending_earnings: 0,
          referral_code: user?.id?.toString() || "REF001",
        };
      }
    },
  });

  // Fetch referrals list
  const { data: referrals = [], isLoading: loadingReferrals } = useQuery<Referral[]>({
    queryKey: ["/api/v1/referrals/list"],
    queryFn: async () => {
      try {
        const res = await api.get("/referrals/list");
        return res.data;
      } catch {
        return [];
      }
    },
  });

  const referralCode = stats?.referral_code || user?.id?.toString() || "REF001";
  const referralLink = `${window.location.origin}/register?ref=${referralCode}`;

  const copyLink = () => {
    navigator.clipboard.writeText(referralLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success("تم نسخ رابط الإحالة");
  };

  const copyCode = () => {
    navigator.clipboard.writeText(referralCode);
    toast.success("تم نسخ كود الإحالة");
  };

  const shareLink = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: "انضم إلى Legendary AI Trader",
          text: "انضم إلى منصة التداول الذكي واحصل على مكافأة ترحيبية!",
          url: referralLink,
        });
      } catch (err) {
        copyLink();
      }
    } else {
      copyLink();
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">برنامج الإحالات</h1>
        <p className="text-muted-foreground text-sm">ادعُ أصدقاءك واكسب عمولات</p>
      </div>

      {/* Referral Link Card */}
      <Card className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Gift className="w-6 h-6 text-primary" />
                <h2 className="text-xl font-bold">رابط الإحالة الخاص بك</h2>
              </div>
              <p className="text-muted-foreground">
                شارك هذا الرابط مع أصدقائك واحصل على <span className="text-primary font-bold">5%</span> من إيداعاتهم كعمولة!
              </p>
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 p-3 bg-background rounded-lg">
                <code className="flex-1 text-sm break-all" dir="ltr">
                  {referralLink}
                </code>
                <button
                  onClick={copyLink}
                  className="p-2 hover:bg-muted rounded-lg transition-colors"
                >
                  {copied ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <Copy className="w-5 h-5" />
                  )}
                </button>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={copyLink}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
                >
                  <Copy className="w-4 h-4" />
                  نسخ الرابط
                </button>
                <button
                  onClick={shareLink}
                  className="flex items-center justify-center gap-2 px-4 py-2 bg-muted rounded-lg font-medium hover:bg-muted/80"
                >
                  <Share2 className="w-4 h-4" />
                  مشاركة
                </button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">إجمالي الإحالات</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold">{stats?.total_referrals || 0}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <UserPlus className="w-4 h-4 text-green-500" />
              <span className="text-sm text-muted-foreground">إحالات نشطة</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-16" />
            ) : (
              <p className="text-xl font-bold text-green-500">{stats?.active_referrals || 0}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-muted-foreground">إجمالي الأرباح</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-20" />
            ) : (
              <p className="text-xl font-bold" dir="ltr">{formatCurrency(stats?.total_earnings || 0)}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-orange-500" />
              <span className="text-sm text-muted-foreground">أرباح معلقة</span>
            </div>
            {loadingStats ? (
              <Skeleton className="h-7 w-20" />
            ) : (
              <p className="text-xl font-bold text-orange-500" dir="ltr">{formatCurrency(stats?.pending_earnings || 0)}</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Referral Code */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Link2 className="w-5 h-5" />
            كود الإحالة
          </CardTitle>
          <CardDescription>يمكن لأصدقائك استخدام هذا الكود عند التسجيل</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1 p-4 bg-muted rounded-lg text-center">
              <p className="text-3xl font-bold font-mono tracking-widest" dir="ltr">{referralCode}</p>
            </div>
            <button
              onClick={copyCode}
              className="p-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              <Copy className="w-5 h-5" />
            </button>
          </div>
        </CardContent>
      </Card>

      {/* How it works */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">كيف يعمل برنامج الإحالات؟</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-3">
                <Share2 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-medium mb-1">1. شارك الرابط</h3>
              <p className="text-sm text-muted-foreground">
                شارك رابط الإحالة الخاص بك مع أصدقائك
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-3">
                <UserPlus className="w-6 h-6 text-green-500" />
              </div>
              <h3 className="font-medium mb-1">2. يسجلون ويودعون</h3>
              <p className="text-sm text-muted-foreground">
                عندما يسجل صديقك ويودع، تحصل على عمولة
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-yellow-500/10 flex items-center justify-center mx-auto mb-3">
                <Gift className="w-6 h-6 text-yellow-500" />
              </div>
              <h3 className="font-medium mb-1">3. اكسب المكافآت</h3>
              <p className="text-sm text-muted-foreground">
                احصل على 5% من كل إيداع يقوم به المُحال
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Referrals List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">قائمة الإحالات</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingReferrals ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : referrals.length > 0 ? (
            <div className="space-y-3">
              {referrals.map((referral) => (
                <div
                  key={referral.id}
                  className="flex items-center justify-between p-4 bg-muted/50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <span className="text-primary font-medium">
                        {referral.email[0].toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{referral.full_name || referral.email}</p>
                      <p className="text-sm text-muted-foreground">
                        انضم في {format(new Date(referral.created_at), "dd/MM/yyyy")}
                      </p>
                    </div>
                  </div>
                  <div className="text-left">
                    <p className="font-medium" dir="ltr">{formatCurrency(referral.total_deposited)}</p>
                    <Badge variant="outline" className={cn(
                      referral.status === "active" 
                        ? "bg-green-500/10 text-green-500 border-green-500/20"
                        : "bg-muted text-muted-foreground"
                    )}>
                      {referral.status === "active" ? "نشط" : "غير نشط"}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">لا توجد إحالات بعد</p>
              <p className="text-sm text-muted-foreground mt-1">
                شارك رابط الإحالة مع أصدقائك للبدء
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* VIP Levels */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Crown className="w-5 h-5 text-yellow-500" />
            مستويات VIP
          </CardTitle>
          <CardDescription>كلما زادت إحالاتك، زادت مكافآتك</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border-2 border-blue-500/30 bg-blue-500/5">
              <Badge className="bg-blue-500 mb-2">برونزي</Badge>
              <p className="text-sm text-muted-foreground">5+ إحالات</p>
              <p className="font-bold text-blue-500">5% عمولة</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-purple-500/30 bg-purple-500/5">
              <Badge className="bg-purple-500 mb-2">فضي</Badge>
              <p className="text-sm text-muted-foreground">15+ إحالات</p>
              <p className="font-bold text-purple-500">7% عمولة</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-yellow-500/30 bg-yellow-500/5">
              <Badge className="bg-yellow-500 text-black mb-2">ذهبي</Badge>
              <p className="text-sm text-muted-foreground">30+ إحالات</p>
              <p className="font-bold text-yellow-500">10% عمولة</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-gradient-to-r from-yellow-500 to-orange-500 bg-gradient-to-r from-yellow-500/5 to-orange-500/5">
              <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white mb-2">بلاتيني</Badge>
              <p className="text-sm text-muted-foreground">50+ إحالات</p>
              <p className="font-bold bg-gradient-to-r from-yellow-500 to-orange-500 bg-clip-text text-transparent">15% عمولة</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
