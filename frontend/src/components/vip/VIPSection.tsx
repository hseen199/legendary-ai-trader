/**
 * VIPSection.tsx - قسم VIP المحسّن
 * يُضاف إلى /opt/asinax/frontend/src/components/vip/
 */

import React, { useState, useEffect } from 'react';
import {
  Award,
  Star,
  Crown,
  Diamond,
  Gem,
  Shield,
  Zap,
  Gift,
  TrendingUp,
  Check,
  X,
  ChevronRight
} from 'lucide-react';

// Types
interface VIPLevel {
  key: string;
  nameAr: string;
  nameEn: string;
  icon: string;
  color: string;
  minDeposit: number;
  maxDeposit: number;
  performanceFee: number;
  referralBonus: number;
  benefits: {
    prioritySupport: boolean;
    weeklyReports: boolean;
    dailyReports: boolean;
    dedicatedManager: boolean;
    earlyAccess: boolean;
  };
}

interface UserVIPStatus {
  currentLevel: string;
  totalDeposited: number;
  nextLevel: VIPLevel | null;
  amountToNextLevel: number;
  progress: number;
}

// VIP Levels Data
const VIP_LEVELS: VIPLevel[] = [
  {
    key: 'bronze',
    nameAr: 'برونزي',
    nameEn: 'Bronze',
    icon: '🥉',
    color: '#CD7F32',
    minDeposit: 0,
    maxDeposit: 999,
    performanceFee: 20,
    referralBonus: 5,
    benefits: {
      prioritySupport: false,
      weeklyReports: false,
      dailyReports: false,
      dedicatedManager: false,
      earlyAccess: false
    }
  },
  {
    key: 'silver',
    nameAr: 'فضي',
    nameEn: 'Silver',
    icon: '🥈',
    color: '#C0C0C0',
    minDeposit: 1000,
    maxDeposit: 4999,
    performanceFee: 18,
    referralBonus: 7,
    benefits: {
      prioritySupport: true,
      weeklyReports: true,
      dailyReports: false,
      dedicatedManager: false,
      earlyAccess: false
    }
  },
  {
    key: 'gold',
    nameAr: 'ذهبي',
    nameEn: 'Gold',
    icon: '🥇',
    color: '#FFD700',
    minDeposit: 5000,
    maxDeposit: 24999,
    performanceFee: 15,
    referralBonus: 10,
    benefits: {
      prioritySupport: true,
      weeklyReports: true,
      dailyReports: true,
      dedicatedManager: false,
      earlyAccess: true
    }
  },
  {
    key: 'platinum',
    nameAr: 'بلاتيني',
    nameEn: 'Platinum',
    icon: '💎',
    color: '#E5E4E2',
    minDeposit: 25000,
    maxDeposit: 99999,
    performanceFee: 12,
    referralBonus: 12,
    benefits: {
      prioritySupport: true,
      weeklyReports: true,
      dailyReports: true,
      dedicatedManager: true,
      earlyAccess: true
    }
  },
  {
    key: 'diamond',
    nameAr: 'ماسي',
    nameEn: 'Diamond',
    icon: '💠',
    color: '#B9F2FF',
    minDeposit: 100000,
    maxDeposit: Infinity,
    performanceFee: 10,
    referralBonus: 15,
    benefits: {
      prioritySupport: true,
      weeklyReports: true,
      dailyReports: true,
      dedicatedManager: true,
      earlyAccess: true
    }
  }
];

// ============ المكونات الفرعية ============

// بطاقة مستوى VIP
const VIPLevelCard: React.FC<{
  level: VIPLevel;
  isCurrentLevel: boolean;
  isLocked: boolean;
  onSelect: () => void;
}> = ({ level, isCurrentLevel, isLocked, onSelect }) => (
  <div
    onClick={!isLocked ? onSelect : undefined}
    className={`
      relative rounded-2xl p-6 transition-all duration-300 cursor-pointer
      ${isCurrentLevel 
        ? 'ring-2 ring-offset-2 ring-offset-gray-900' 
        : 'hover:scale-105'
      }
      ${isLocked ? 'opacity-50 cursor-not-allowed' : ''}
    `}
    style={{
      background: `linear-gradient(135deg, ${level.color}30 0%, ${level.color}10 100%)`,
      borderColor: level.color,
      borderWidth: isCurrentLevel ? '2px' : '1px',
      borderStyle: 'solid',
      ringColor: level.color
    }}
  >
    {isCurrentLevel && (
      <div 
        className="absolute -top-3 right-4 px-3 py-1 rounded-full text-xs font-bold"
        style={{ backgroundColor: level.color, color: '#000' }}
      >
        مستواك الحالي
      </div>
    )}
    
    <div className="text-center mb-4">
      <span className="text-5xl">{level.icon}</span>
      <h3 className="text-xl font-bold text-white mt-2">{level.nameAr}</h3>
      <p className="text-gray-400 text-sm">{level.nameEn}</p>
    </div>
    
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-gray-400">الحد الأدنى</span>
        <span className="text-white font-semibold">
          ${level.minDeposit.toLocaleString()}
        </span>
      </div>
      
      <div className="flex justify-between items-center">
        <span className="text-gray-400">رسوم الأداء</span>
        <span className="text-emerald-400 font-semibold">
          {level.performanceFee}%
        </span>
      </div>
      
      <div className="flex justify-between items-center">
        <span className="text-gray-400">مكافأة الإحالة</span>
        <span className="text-blue-400 font-semibold">
          {level.referralBonus}%
        </span>
      </div>
    </div>
    
    <div className="mt-4 pt-4 border-t border-gray-700">
      <div className="grid grid-cols-5 gap-1">
        {Object.entries(level.benefits).map(([key, enabled], index) => (
          <div 
            key={key}
            className={`p-2 rounded text-center ${enabled ? 'text-emerald-400' : 'text-gray-600'}`}
            title={getBenefitName(key)}
          >
            {enabled ? <Check size={16} /> : <X size={16} />}
          </div>
        ))}
      </div>
    </div>
  </div>
);

// تفاصيل المزايا
const BenefitsDetail: React.FC<{ level: VIPLevel }> = ({ level }) => {
  const benefits = [
    {
      key: 'prioritySupport',
      name: 'دعم أولوي',
      description: 'أولوية في الرد على استفساراتك',
      icon: <Shield className="text-blue-400" size={24} />
    },
    {
      key: 'weeklyReports',
      name: 'تقارير أسبوعية',
      description: 'تقرير أداء مفصل كل أسبوع',
      icon: <TrendingUp className="text-green-400" size={24} />
    },
    {
      key: 'dailyReports',
      name: 'تقارير يومية',
      description: 'ملخص يومي لأداء محفظتك',
      icon: <Zap className="text-yellow-400" size={24} />
    },
    {
      key: 'dedicatedManager',
      name: 'مدير حساب مخصص',
      description: 'مدير حساب شخصي لخدمتك',
      icon: <Crown className="text-purple-400" size={24} />
    },
    {
      key: 'earlyAccess',
      name: 'وصول مبكر',
      description: 'الوصول المبكر للميزات الجديدة',
      icon: <Star className="text-orange-400" size={24} />
    }
  ];
  
  return (
    <div className="bg-gray-800/50 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Gift className="text-emerald-400" />
        مزايا مستوى {level.nameAr}
      </h3>
      
      <div className="space-y-4">
        {benefits.map((benefit) => {
          const isEnabled = level.benefits[benefit.key as keyof typeof level.benefits];
          
          return (
            <div 
              key={benefit.key}
              className={`flex items-center gap-4 p-3 rounded-lg ${
                isEnabled ? 'bg-emerald-500/10' : 'bg-gray-700/30'
              }`}
            >
              <div className={isEnabled ? '' : 'opacity-30'}>
                {benefit.icon}
              </div>
              <div className="flex-1">
                <div className={`font-medium ${isEnabled ? 'text-white' : 'text-gray-500'}`}>
                  {benefit.name}
                </div>
                <div className={`text-sm ${isEnabled ? 'text-gray-400' : 'text-gray-600'}`}>
                  {benefit.description}
                </div>
              </div>
              <div>
                {isEnabled ? (
                  <Check className="text-emerald-400" size={20} />
                ) : (
                  <X className="text-gray-600" size={20} />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// شريط التقدم للترقية
const UpgradeProgress: React.FC<{ status: UserVIPStatus }> = ({ status }) => {
  const currentLevel = VIP_LEVELS.find(l => l.key === status.currentLevel);
  const nextLevel = status.nextLevel;
  
  if (!nextLevel) {
    return (
      <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-xl p-6 text-center">
        <Diamond className="mx-auto text-purple-400 mb-3" size={48} />
        <h3 className="text-xl font-bold text-white">أنت في أعلى مستوى!</h3>
        <p className="text-gray-400 mt-2">تهانينا، أنت تستمتع بجميع المزايا الحصرية</p>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-800/50 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{currentLevel?.icon}</span>
          <div>
            <div className="text-white font-semibold">{currentLevel?.nameAr}</div>
            <div className="text-gray-400 text-sm">مستواك الحالي</div>
          </div>
        </div>
        
        <ChevronRight className="text-gray-500" size={24} />
        
        <div className="flex items-center gap-3">
          <span className="text-3xl">{nextLevel.icon}</span>
          <div>
            <div className="text-white font-semibold">{nextLevel.nameAr}</div>
            <div className="text-gray-400 text-sm">المستوى التالي</div>
          </div>
        </div>
      </div>
      
      <div className="relative">
        <div className="w-full h-4 bg-gray-700 rounded-full overflow-hidden">
          <div 
            className="h-full rounded-full transition-all duration-500"
            style={{ 
              width: `${status.progress}%`,
              background: `linear-gradient(90deg, ${currentLevel?.color} 0%, ${nextLevel.color} 100%)`
            }}
          />
        </div>
        <div className="flex justify-between mt-2 text-sm">
          <span className="text-gray-400">
            ${status.totalDeposited.toLocaleString()}
          </span>
          <span className="text-gray-400">
            ${nextLevel.minDeposit.toLocaleString()}
          </span>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
        <div className="flex items-center justify-between">
          <span className="text-gray-300">المبلغ المطلوب للترقية</span>
          <span className="text-emerald-400 font-bold text-lg">
            ${status.amountToNextLevel.toLocaleString()}
          </span>
        </div>
        <button className="w-full mt-3 py-2 bg-emerald-500 text-white rounded-lg font-semibold hover:bg-emerald-600 transition">
          أودع الآن للترقية
        </button>
      </div>
    </div>
  );
};

// ============ المكون الرئيسي ============

const VIPSection: React.FC = () => {
  const [selectedLevel, setSelectedLevel] = useState<VIPLevel | null>(null);
  const [userStatus, setUserStatus] = useState<UserVIPStatus | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchVIPStatus();
  }, []);
  
  const fetchVIPStatus = async () => {
    try {
      // const response = await api.get('/vip/status');
      // setUserStatus(response.data);
      
      // بيانات تجريبية
      setUserStatus({
        currentLevel: 'gold',
        totalDeposited: 15000,
        nextLevel: VIP_LEVELS.find(l => l.key === 'platinum') || null,
        amountToNextLevel: 10000,
        progress: 60
      });
      
      setSelectedLevel(VIP_LEVELS.find(l => l.key === 'gold') || null);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching VIP status:', error);
      setLoading(false);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-500" />
      </div>
    );
  }
  
  return (
    <div className="space-y-8 p-6">
      {/* العنوان */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white mb-2">
          <span className="text-emerald-400">VIP</span> مستويات
        </h1>
        <p className="text-gray-400">
          استمتع بمزايا حصرية كلما زاد استثمارك
        </p>
      </div>
      
      {/* شريط التقدم */}
      {userStatus && <UpgradeProgress status={userStatus} />}
      
      {/* بطاقات المستويات */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {VIP_LEVELS.map((level) => {
          const isCurrentLevel = userStatus?.currentLevel === level.key;
          const currentIndex = VIP_LEVELS.findIndex(l => l.key === userStatus?.currentLevel);
          const levelIndex = VIP_LEVELS.findIndex(l => l.key === level.key);
          const isLocked = levelIndex > currentIndex + 1;
          
          return (
            <VIPLevelCard
              key={level.key}
              level={level}
              isCurrentLevel={isCurrentLevel}
              isLocked={isLocked}
              onSelect={() => setSelectedLevel(level)}
            />
          );
        })}
      </div>
      
      {/* تفاصيل المزايا */}
      {selectedLevel && <BenefitsDetail level={selectedLevel} />}
      
      {/* جدول المقارنة */}
      <div className="bg-gray-800/50 rounded-xl p-6 overflow-x-auto">
        <h3 className="text-lg font-semibold text-white mb-4">مقارنة المستويات</h3>
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-right py-3 text-gray-400">الميزة</th>
              {VIP_LEVELS.map(level => (
                <th 
                  key={level.key} 
                  className="text-center py-3"
                  style={{ color: level.color }}
                >
                  {level.icon} {level.nameAr}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-700/50">
              <td className="py-3 text-gray-300">رسوم الأداء</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3 text-emerald-400">
                  {level.performanceFee}%
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-3 text-gray-300">مكافأة الإحالة</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3 text-blue-400">
                  {level.referralBonus}%
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-3 text-gray-300">دعم أولوي</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3">
                  {level.benefits.prioritySupport ? 
                    <Check className="mx-auto text-emerald-400" size={18} /> : 
                    <X className="mx-auto text-gray-600" size={18} />
                  }
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-3 text-gray-300">تقارير أسبوعية</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3">
                  {level.benefits.weeklyReports ? 
                    <Check className="mx-auto text-emerald-400" size={18} /> : 
                    <X className="mx-auto text-gray-600" size={18} />
                  }
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-3 text-gray-300">تقارير يومية</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3">
                  {level.benefits.dailyReports ? 
                    <Check className="mx-auto text-emerald-400" size={18} /> : 
                    <X className="mx-auto text-gray-600" size={18} />
                  }
                </td>
              ))}
            </tr>
            <tr>
              <td className="py-3 text-gray-300">مدير حساب مخصص</td>
              {VIP_LEVELS.map(level => (
                <td key={level.key} className="text-center py-3">
                  {level.benefits.dedicatedManager ? 
                    <Check className="mx-auto text-emerald-400" size={18} /> : 
                    <X className="mx-auto text-gray-600" size={18} />
                  }
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Helper function
function getBenefitName(key: string): string {
  const names: Record<string, string> = {
    prioritySupport: 'دعم أولوي',
    weeklyReports: 'تقارير أسبوعية',
    dailyReports: 'تقارير يومية',
    dedicatedManager: 'مدير حساب',
    earlyAccess: 'وصول مبكر'
  };
  return names[key] || key;
}

export default VIPSection;
