/**
 * AdvancedReferrals Component
 * لوحة إحالات متقدمة مع مستويات ومكافآت
 */
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Users, 
  Gift, 
  Trophy,
  Share2,
  Copy,
  CheckCircle,
  Star,
  TrendingUp,
  Wallet,
  Crown,
  Medal,
  Award,
  ExternalLink,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { cn } from '../lib/utils';
import toast from 'react-hot-toast';
import api from '../services/api';

interface AdvancedReferralsProps {
  language?: 'ar' | 'en';
  referralCode?: string;
}

// مستويات الإحالات
const referralTiers = [
  {
    id: 'bronze',
    nameAr: 'برونزي',
    nameEn: 'Bronze',
    icon: <Medal className="h-6 w-6" />,
    color: 'text-orange-600 bg-orange-500/10 border-orange-500/30',
    minReferrals: 0,
    maxReferrals: 4,
    commissionRate: 5,
    bonusAr: 'عمولة 5% من أرباح المُحالين',
    bonusEn: '5% commission on referrals\' profits',
  },
  {
    id: 'silver',
    nameAr: 'فضي',
    nameEn: 'Silver',
    icon: <Award className="h-6 w-6" />,
    color: 'text-gray-400 bg-gray-500/10 border-gray-500/30',
    minReferrals: 5,
    maxReferrals: 14,
    commissionRate: 7,
    bonusAr: 'عمولة 7% + مكافأة $25 لكل 5 إحالات',
    bonusEn: '7% commission + $25 bonus per 5 referrals',
  },
  {
    id: 'gold',
    nameAr: 'ذهبي',
    nameEn: 'Gold',
    icon: <Trophy className="h-6 w-6" />,
    color: 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30',
    minReferrals: 15,
    maxReferrals: 29,
    commissionRate: 10,
    bonusAr: 'عمولة 10% + مكافأة $50 لكل 5 إحالات',
    bonusEn: '10% commission + $50 bonus per 5 referrals',
  },
  {
    id: 'platinum',
    nameAr: 'بلاتيني',
    nameEn: 'Platinum',
    icon: <Crown className="h-6 w-6" />,
    color: 'text-purple-500 bg-purple-500/10 border-purple-500/30',
    minReferrals: 30,
    maxReferrals: Infinity,
    commissionRate: 15,
    bonusAr: 'عمولة 15% + مكافآت VIP حصرية',
    bonusEn: '15% commission + exclusive VIP rewards',
  },
];

// بيانات تجريبية
const mockReferralData = {
  totalReferrals: 8,
  activeReferrals: 6,
  pendingReferrals: 2,
  totalEarnings: 156.50,
  pendingEarnings: 23.00,
  currentTier: 'silver',
  referralCode: 'ASINAX-ABC123',
  referralLink: 'https://asinax.cloud/ref/ABC123',
  recentReferrals: [
    { id: 1, name: 'أحمد م.', date: '2025-01-20', status: 'active', earnings: 45.00 },
    { id: 2, name: 'سارة ك.', date: '2025-01-18', status: 'active', earnings: 32.50 },
    { id: 3, name: 'محمد ع.', date: '2025-01-15', status: 'pending', earnings: 0 },
    { id: 4, name: 'فاطمة ح.', date: '2025-01-12', status: 'active', earnings: 28.00 },
    { id: 5, name: 'خالد س.', date: '2025-01-10', status: 'active', earnings: 51.00 },
  ],
};

export function AdvancedReferrals({ 
  language = 'ar',
  referralCode: propReferralCode,
}: AdvancedReferralsProps) {
  const isRTL = language === 'ar';
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // جلب بيانات الإحالات
  const { data: referralData = mockReferralData } = useQuery({
    queryKey: ['/api/v1/user/referrals'],
    queryFn: async () => {
      try {
        const res = await api.get('/user/referrals');
        return res.data;
      } catch {
        return mockReferralData;
      }
    },
  });

  const currentTier = referralTiers.find(t => t.id === referralData.currentTier) || referralTiers[0];
  const nextTier = referralTiers.find(t => t.minReferrals > referralData.totalReferrals);
  
  const progressToNextTier = nextTier 
    ? ((referralData.totalReferrals - currentTier.minReferrals) / (nextTier.minReferrals - currentTier.minReferrals)) * 100
    : 100;

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success(isRTL ? 'تم النسخ!' : 'Copied!');
    setTimeout(() => setCopied(false), 2000);
  };

  const shareOnSocial = (platform: string) => {
    const text = isRTL 
      ? `انضم إلى ASINAX واحصل على مكافأة ترحيبية! استخدم رمز الإحالة: ${referralData.referralCode}`
      : `Join ASINAX and get a welcome bonus! Use referral code: ${referralData.referralCode}`;
    
    const url = referralData.referralLink;
    
    let shareUrl = '';
    switch (platform) {
      case 'twitter':
        shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
        break;
      case 'facebook':
        shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
        break;
      case 'whatsapp':
        shareUrl = `https://wa.me/?text=${encodeURIComponent(text + ' ' + url)}`;
        break;
      case 'telegram':
        shareUrl = `https://t.me/share/url?url=${encodeURIComponent(url)}&text=${encodeURIComponent(text)}`;
        break;
    }
    
    window.open(shareUrl, '_blank', 'width=600,height=400');
  };

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5 text-primary" />
          {isRTL ? 'برنامج الإحالات' : 'Referral Program'}
        </CardTitle>
        <CardDescription>
          {isRTL 
            ? 'ادعُ أصدقاءك واكسب مكافآت على كل إحالة ناجحة'
            : 'Invite your friends and earn rewards for every successful referral'
          }
        </CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 mb-6">
            <TabsTrigger value="overview">
              {isRTL ? 'نظرة عامة' : 'Overview'}
            </TabsTrigger>
            <TabsTrigger value="tiers">
              {isRTL ? 'المستويات' : 'Tiers'}
            </TabsTrigger>
            <TabsTrigger value="history">
              {isRTL ? 'السجل' : 'History'}
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Current Tier Card */}
            <div className={cn(
              "p-4 rounded-lg border-2",
              currentTier.color
            )}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={cn("p-2 rounded-full", currentTier.color)}>
                    {currentTier.icon}
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      {isRTL ? 'مستواك الحالي' : 'Your Current Tier'}
                    </p>
                    <p className="text-xl font-bold">
                      {isRTL ? currentTier.nameAr : currentTier.nameEn}
                    </p>
                  </div>
                </div>
                <div className="text-left">
                  <p className="text-3xl font-bold">{currentTier.commissionRate}%</p>
                  <p className="text-sm text-muted-foreground">
                    {isRTL ? 'عمولة' : 'Commission'}
                  </p>
                </div>
              </div>
              
              {nextTier && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span>{referralData.totalReferrals} {isRTL ? 'إحالة' : 'referrals'}</span>
                    <span>{nextTier.minReferrals} {isRTL ? 'للمستوى التالي' : 'for next tier'}</span>
                  </div>
                  <Progress value={progressToNextTier} className="h-2" />
                </div>
              )}
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <Users className="h-6 w-6 mx-auto mb-2 text-primary" />
                <p className="text-2xl font-bold">{referralData.totalReferrals}</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'إجمالي الإحالات' : 'Total Referrals'}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <CheckCircle className="h-6 w-6 mx-auto mb-2 text-green-500" />
                <p className="text-2xl font-bold">{referralData.activeReferrals}</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'إحالات نشطة' : 'Active Referrals'}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <Wallet className="h-6 w-6 mx-auto mb-2 text-blue-500" />
                <p className="text-2xl font-bold">${referralData.totalEarnings.toFixed(2)}</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'إجمالي الأرباح' : 'Total Earnings'}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 text-center">
                <Gift className="h-6 w-6 mx-auto mb-2 text-purple-500" />
                <p className="text-2xl font-bold">${referralData.pendingEarnings.toFixed(2)}</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'أرباح معلقة' : 'Pending Earnings'}
                </p>
              </div>
            </div>

            {/* Referral Link */}
            <div className="space-y-3">
              <h4 className="font-semibold">
                {isRTL ? 'رابط الإحالة الخاص بك' : 'Your Referral Link'}
              </h4>
              <div className="flex gap-2">
                <div className="flex-1 p-3 rounded-lg bg-muted font-mono text-sm truncate">
                  {referralData.referralLink}
                </div>
                <Button
                  variant="outline"
                  onClick={() => copyToClipboard(referralData.referralLink)}
                >
                  {copied ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              
              <div className="flex gap-2">
                <div className="flex-1 p-3 rounded-lg bg-muted text-center">
                  <p className="text-xs text-muted-foreground mb-1">
                    {isRTL ? 'رمز الإحالة' : 'Referral Code'}
                  </p>
                  <p className="font-bold">{referralData.referralCode}</p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => copyToClipboard(referralData.referralCode)}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Share Buttons */}
            <div className="space-y-3">
              <h4 className="font-semibold flex items-center gap-2">
                <Share2 className="h-4 w-4" />
                {isRTL ? 'شارك عبر' : 'Share via'}
              </h4>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  className="bg-[#25D366]/10 hover:bg-[#25D366]/20 border-[#25D366]/30"
                  onClick={() => shareOnSocial('whatsapp')}
                >
                  <span className="ml-2">WhatsApp</span>
                </Button>
                <Button
                  variant="outline"
                  className="bg-[#0088cc]/10 hover:bg-[#0088cc]/20 border-[#0088cc]/30"
                  onClick={() => shareOnSocial('telegram')}
                >
                  <span className="ml-2">Telegram</span>
                </Button>
                <Button
                  variant="outline"
                  className="bg-[#1DA1F2]/10 hover:bg-[#1DA1F2]/20 border-[#1DA1F2]/30"
                  onClick={() => shareOnSocial('twitter')}
                >
                  <span className="ml-2">Twitter</span>
                </Button>
                <Button
                  variant="outline"
                  className="bg-[#4267B2]/10 hover:bg-[#4267B2]/20 border-[#4267B2]/30"
                  onClick={() => shareOnSocial('facebook')}
                >
                  <span className="ml-2">Facebook</span>
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* Tiers Tab */}
          <TabsContent value="tiers" className="space-y-4">
            {referralTiers.map((tier, index) => {
              const isCurrentTier = tier.id === currentTier.id;
              const isUnlocked = referralData.totalReferrals >= tier.minReferrals;
              
              return (
                <div
                  key={tier.id}
                  className={cn(
                    "p-4 rounded-lg border-2 transition-all",
                    isCurrentTier ? tier.color : "border-muted",
                    !isUnlocked && "opacity-50"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "p-2 rounded-full",
                        isUnlocked ? tier.color : "bg-muted text-muted-foreground"
                      )}>
                        {tier.icon}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <p className="font-bold">
                            {isRTL ? tier.nameAr : tier.nameEn}
                          </p>
                          {isCurrentTier && (
                            <Badge>{isRTL ? 'الحالي' : 'Current'}</Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {tier.minReferrals}+ {isRTL ? 'إحالة' : 'referrals'}
                        </p>
                      </div>
                    </div>
                    <div className="text-left">
                      <p className="text-2xl font-bold">{tier.commissionRate}%</p>
                      <p className="text-xs text-muted-foreground">
                        {isRTL ? tier.bonusAr : tier.bonusEn}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="space-y-4">
            {referralData.recentReferrals.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>{isRTL ? 'لا توجد إحالات بعد' : 'No referrals yet'}</p>
              </div>
            ) : (
              <div className="space-y-2">
                {referralData.recentReferrals.map((referral: any) => (
                  <div
                    key={referral.id}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <span className="text-primary font-bold">
                          {referral.name.charAt(0)}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium">{referral.name}</p>
                        <p className="text-sm text-muted-foreground">{referral.date}</p>
                      </div>
                    </div>
                    <div className="text-left">
                      <Badge variant={referral.status === 'active' ? 'default' : 'secondary'}>
                        {referral.status === 'active' 
                          ? (isRTL ? 'نشط' : 'Active')
                          : (isRTL ? 'معلق' : 'Pending')
                        }
                      </Badge>
                      <p className="text-sm font-medium mt-1">
                        ${referral.earnings.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default AdvancedReferrals;
