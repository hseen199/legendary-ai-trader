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
import { useLanguage } from '@/lib/i18n';

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
  const { t, language } = useLanguage();
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
    toast.success(t.referrals.linkCopied);
  };

  const copyCode = () => {
    navigator.clipboard.writeText(referralCode);
    toast.success(t.referrals.codeCopied);
  };

  const shareLink = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: t.referrals.joinAsinax,
          text: t.referrals.inviteFriends,
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
        <h1 className="text-2xl font-bold">{t.referrals.title}</h1>
        <p className="text-muted-foreground text-sm">{t.referrals.subtitle}</p>
      </div>

      {/* Referral Link Card */}
      <Card className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Gift className="w-6 h-6 text-primary" />
                <h2 className="text-xl font-bold">{t.referrals.yourLink}</h2>
              </div>
              <p className="text-muted-foreground">
                {t.referrals.shareDesc}
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
                  {t.referrals.copyLink}
                </button>
                <button
                  onClick={shareLink}
                  className="flex items-center justify-center gap-2 px-4 py-2 bg-muted rounded-lg font-medium hover:bg-muted/80"
                >
                  <Share2 className="w-4 h-4" />
                  {t.referrals.share}
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
              <span className="text-sm text-muted-foreground">{t.referrals.totalReferrals}</span>
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
              <span className="text-sm text-muted-foreground">{t.referrals.activeReferrals}</span>
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
              <span className="text-sm text-muted-foreground">{t.referrals.totalEarnings}</span>
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
              <span className="text-sm text-muted-foreground">{t.referrals.pendingEarnings}</span>
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
            {t.referrals.yourCode}
          </CardTitle>
          <CardDescription>
            {language === 'ar' ? 'يمكن لأصدقائك استخدام هذا الكود عند التسجيل' : 'Your friends can use this code when registering'}
          </CardDescription>
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
          <CardTitle className="text-lg">{t.referrals.howItWorks}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-3">
                <Share2 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-medium mb-1">{language === 'ar' ? '1. شارك الرابط' : '1. Share Link'}</h3>
              <p className="text-sm text-muted-foreground">
                {t.referrals.step1Desc}
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-3">
                <UserPlus className="w-6 h-6 text-green-500" />
              </div>
              <h3 className="font-medium mb-1">{language === 'ar' ? '2. يسجلون ويودعون' : '2. They Register & Deposit'}</h3>
              <p className="text-sm text-muted-foreground">
                {t.referrals.step2Desc}
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 rounded-full bg-yellow-500/10 flex items-center justify-center mx-auto mb-3">
                <Gift className="w-6 h-6 text-yellow-500" />
              </div>
              <h3 className="font-medium mb-1">{language === 'ar' ? '3. اكسب المكافآت' : '3. Earn Rewards'}</h3>
              <p className="text-sm text-muted-foreground">
                {t.referrals.step3Desc}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Referrals List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{t.referrals.referralsList}</CardTitle>
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
                        {language === 'ar' ? 'انضم في' : 'Joined'} {format(new Date(referral.created_at), "dd/MM/yyyy")}
                      </p>
                    </div>
                  </div>
                  <div className={language === 'ar' ? 'text-left' : 'text-right'}>
                    <p className="font-medium" dir="ltr">{formatCurrency(referral.total_deposited)}</p>
                    <Badge variant="outline" className={cn(
                      referral.status === "active" 
                        ? "bg-green-500/10 text-green-500 border-green-500/20"
                        : "bg-muted text-muted-foreground"
                    )}>
                      {referral.status === "active" 
                        ? (language === 'ar' ? 'نشط' : 'Active') 
                        : (language === 'ar' ? 'غير نشط' : 'Inactive')}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">{t.referrals.noReferrals}</p>
              <p className="text-sm text-muted-foreground mt-1">
                {t.referrals.noReferralsDesc}
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
            {t.referrals.vipLevels}
          </CardTitle>
          <CardDescription>{t.referrals.vipDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border-2 border-blue-500/30 bg-blue-500/5">
              <Badge className="bg-blue-500 mb-2">{t.referrals.bronze}</Badge>
              <p className="text-sm text-muted-foreground">5+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-blue-500">$10 {t.referrals.reward}</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-purple-500/30 bg-purple-500/5">
              <Badge className="bg-purple-500 mb-2">{t.referrals.silver}</Badge>
              <p className="text-sm text-muted-foreground">15+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-purple-500">$12 {t.referrals.reward}</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-yellow-500/30 bg-yellow-500/5">
              <Badge className="bg-yellow-500 text-black mb-2">{t.referrals.gold}</Badge>
              <p className="text-sm text-muted-foreground">30+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-yellow-500">$15 {t.referrals.reward}</p>
            </div>
            <div className="p-4 rounded-lg border-2 border-gradient-to-r from-yellow-500 to-orange-500 bg-gradient-to-r from-yellow-500/5 to-orange-500/5">
              <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white mb-2">{t.referrals.platinum}</Badge>
              <p className="text-sm text-muted-foreground">50+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold bg-gradient-to-r from-yellow-500 to-orange-500 bg-clip-text text-transparent">$15+ {t.referrals.reward}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
