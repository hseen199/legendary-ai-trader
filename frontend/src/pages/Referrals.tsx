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
  MessageCircle,
  Send,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import api from "../services/api";
import { useLanguage } from '@/lib/i18n';

// Social Media Icons
const WhatsAppIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
    <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/>
  </svg>
);

const TelegramIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
    <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>
  </svg>
);

const TwitterIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
  </svg>
);

const FacebookIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor">
    <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
  </svg>
);

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
    queryKey: ["/api/v1/marketing/referral/stats"],
    queryFn: async () => {
      try {
        const res = await api.get("/marketing/referral/stats");
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
    queryKey: ["/api/v1/marketing/referral/my"],
    queryFn: async () => {
      try {
        const res = await api.get("/marketing/referral/my");
        return res.data;
      } catch {
        return [];
      }
    },
  });

  const referralCode = stats?.referral_code || user?.id?.toString() || "REF001";
  const referralLink = `${window.location.origin}/register?ref=${referralCode}`;
  
  const shareMessage = language === 'ar' 
    ? `انضم إلى ASINAX واستثمر بذكاء مع الوكيل الذكي! استخدم رابط الإحالة الخاص بي: ${referralLink}`
    : `Join ASINAX and invest smartly with AI Agent! Use my referral link: ${referralLink}`;

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

  // Social sharing functions
  const shareToWhatsApp = () => {
    const url = `https://wa.me/?text=${encodeURIComponent(shareMessage)}`;
    window.open(url, '_blank');
  };

  const shareToTelegram = () => {
    const url = `https://t.me/share/url?url=${encodeURIComponent(referralLink)}&text=${encodeURIComponent(shareMessage)}`;
    window.open(url, '_blank');
  };

  const shareToTwitter = () => {
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareMessage)}`;
    window.open(url, '_blank');
  };

  const shareToFacebook = () => {
    const url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(referralLink)}`;
    window.open(url, '_blank');
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value || 0);
  };

  // Calculate current level
  const totalReferrals = stats?.total_referrals || 0;
  const currentLevel = totalReferrals >= 50 ? 'platinum' : totalReferrals >= 30 ? 'gold' : totalReferrals >= 15 ? 'silver' : totalReferrals >= 5 ? 'bronze' : 'starter';
  const nextLevelTarget = currentLevel === 'starter' ? 5 : currentLevel === 'bronze' ? 15 : currentLevel === 'silver' ? 30 : currentLevel === 'gold' ? 50 : 100;
  const progressToNextLevel = Math.min((totalReferrals / nextLevelTarget) * 100, 100);

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{t.referrals.title}</h1>
        <p className="text-muted-foreground text-sm">{t.referrals.subtitle}</p>
      </div>

      {/* Current Level Progress */}
      <Card className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border-yellow-500/20">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-yellow-500/20">
                <Crown className="w-6 h-6 text-yellow-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{language === 'ar' ? 'مستواك الحالي' : 'Your Level'}</p>
                <p className="text-xl font-bold capitalize">{currentLevel === 'starter' ? (language === 'ar' ? 'مبتدئ' : 'Starter') : t.referrals[currentLevel as keyof typeof t.referrals]}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">{language === 'ar' ? 'التقدم للمستوى التالي' : 'Progress to Next'}</p>
              <p className="text-lg font-bold">{totalReferrals}/{nextLevelTarget}</p>
            </div>
          </div>
          <div className="w-full bg-muted rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-yellow-500 to-orange-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progressToNextLevel}%` }}
            />
          </div>
        </CardContent>
      </Card>

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

      {/* Social Share Buttons */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Share2 className="w-5 h-5" />
            {language === 'ar' ? 'شارك عبر وسائل التواصل' : 'Share on Social Media'}
          </CardTitle>
          <CardDescription>
            {language === 'ar' ? 'شارك رابط الإحالة مع أصدقائك بنقرة واحدة' : 'Share your referral link with one click'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button
              onClick={shareToWhatsApp}
              className="flex items-center justify-center gap-2 p-4 rounded-lg bg-green-500/10 hover:bg-green-500/20 text-green-500 transition-colors"
            >
              <WhatsAppIcon />
              <span className="font-medium">WhatsApp</span>
            </button>
            <button
              onClick={shareToTelegram}
              className="flex items-center justify-center gap-2 p-4 rounded-lg bg-blue-500/10 hover:bg-blue-500/20 text-blue-500 transition-colors"
            >
              <TelegramIcon />
              <span className="font-medium">Telegram</span>
            </button>
            <button
              onClick={shareToTwitter}
              className="flex items-center justify-center gap-2 p-4 rounded-lg bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 transition-colors"
            >
              <TwitterIcon />
              <span className="font-medium">X (Twitter)</span>
            </button>
            <button
              onClick={shareToFacebook}
              className="flex items-center justify-center gap-2 p-4 rounded-lg bg-blue-600/10 hover:bg-blue-600/20 text-blue-600 transition-colors"
            >
              <FacebookIcon />
              <span className="font-medium">Facebook</span>
            </button>
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
            <div className={cn(
              "p-4 rounded-lg border-2",
              currentLevel === 'bronze' ? "border-blue-500 bg-blue-500/10" : "border-blue-500/30 bg-blue-500/5"
            )}>
              <Badge className="bg-blue-500 mb-2">{t.referrals.bronze}</Badge>
              <p className="text-sm text-muted-foreground">5+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-blue-500">$10 {t.referrals.reward}</p>
            </div>
            <div className={cn(
              "p-4 rounded-lg border-2",
              currentLevel === 'silver' ? "border-purple-500 bg-purple-500/10" : "border-purple-500/30 bg-purple-500/5"
            )}>
              <Badge className="bg-purple-500 mb-2">{t.referrals.silver}</Badge>
              <p className="text-sm text-muted-foreground">15+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-purple-500">$12 {t.referrals.reward}</p>
            </div>
            <div className={cn(
              "p-4 rounded-lg border-2",
              currentLevel === 'gold' ? "border-yellow-500 bg-yellow-500/10" : "border-yellow-500/30 bg-yellow-500/5"
            )}>
              <Badge className="bg-yellow-500 text-black mb-2">{t.referrals.gold}</Badge>
              <p className="text-sm text-muted-foreground">30+ {language === 'ar' ? 'إحالات' : 'referrals'}</p>
              <p className="font-bold text-yellow-500">$15 {t.referrals.reward}</p>
            </div>
            <div className={cn(
              "p-4 rounded-lg border-2",
              currentLevel === 'platinum' ? "border-orange-500 bg-gradient-to-r from-yellow-500/10 to-orange-500/10" : "border-gradient-to-r from-yellow-500 to-orange-500 bg-gradient-to-r from-yellow-500/5 to-orange-500/5"
            )}>
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
