/**
 * Tooltip Definitions
 * تعريفات التلميحات لجميع عناصر المنصة
 */

export interface TooltipDefinition {
  id: string;
  textAr: string;
  textEn: string;
}

// تلميحات لوحة التحكم
export const dashboardTooltips: Record<string, TooltipDefinition> = {
  portfolioValue: {
    id: 'portfolio-value',
    textAr: 'القيمة الإجمالية لاستثمارك بالدولار الأمريكي، تتحدث تلقائياً بناءً على أداء السوق',
    textEn: 'Total value of your investment in USD, automatically updated based on market performance',
  },
  totalProfit: {
    id: 'total-profit',
    textAr: 'إجمالي الأرباح أو الخسائر منذ بداية استثمارك',
    textEn: 'Total profits or losses since the start of your investment',
  },
  profitPercentage: {
    id: 'profit-percentage',
    textAr: 'نسبة الربح أو الخسارة مقارنة بالمبلغ المودع',
    textEn: 'Profit or loss percentage compared to deposited amount',
  },
  navValue: {
    id: 'nav-value',
    textAr: 'صافي قيمة الأصول (NAV) - سعر الوحدة الاستثمارية الحالي. يتغير يومياً بناءً على أداء المحفظة',
    textEn: 'Net Asset Value (NAV) - current investment unit price. Changes daily based on portfolio performance',
  },
  unitsOwned: {
    id: 'units-owned',
    textAr: 'عدد الوحدات الاستثمارية التي تمتلكها. قيمة محفظتك = الوحدات × سعر NAV',
    textEn: 'Number of investment units you own. Your portfolio value = Units × NAV price',
  },
  agentStatus: {
    id: 'agent-status',
    textAr: 'حالة الوكيل الذكي - نشط يعني أنه يعمل ويتداول، متوقف يعني أنه في وضع الانتظار',
    textEn: 'AI agent status - Active means it is working and trading, Stopped means it is on standby',
  },
  lastTrade: {
    id: 'last-trade',
    textAr: 'تاريخ ووقت آخر صفقة نفذها الوكيل الذكي',
    textEn: 'Date and time of the last trade executed by the AI agent',
  },
  navChart: {
    id: 'nav-chart',
    textAr: 'رسم بياني يوضح تطور سعر NAV خلال الفترة المحددة. الخط الصاعد يعني أرباح',
    textEn: 'Chart showing NAV price evolution over the selected period. Rising line means profits',
  },
  recentTrades: {
    id: 'recent-trades',
    textAr: 'آخر الصفقات التي نفذها الوكيل الذكي مع تفاصيل كل صفقة',
    textEn: 'Latest trades executed by the AI agent with details of each trade',
  },
  refreshButton: {
    id: 'refresh-button',
    textAr: 'اضغط لتحديث البيانات والحصول على آخر المعلومات',
    textEn: 'Click to refresh data and get the latest information',
  },
};

// تلميحات المحفظة
export const walletTooltips: Record<string, TooltipDefinition> = {
  availableBalance: {
    id: 'available-balance',
    textAr: 'الرصيد المتاح للسحب بعد انتهاء فترة التسوية (7 أيام من آخر إيداع)',
    textEn: 'Balance available for withdrawal after settlement period (7 days from last deposit)',
  },
  lockedBalance: {
    id: 'locked-balance',
    textAr: 'الرصيد المقفل خلال فترة التسوية. سيصبح متاحاً للسحب بعد 7 أيام',
    textEn: 'Locked balance during settlement period. Will become available for withdrawal after 7 days',
  },
  depositButton: {
    id: 'deposit-button',
    textAr: 'اضغط لإيداع أموال جديدة في حسابك. الحد الأدنى 100 USDC',
    textEn: 'Click to deposit new funds to your account. Minimum 100 USDC',
  },
  withdrawButton: {
    id: 'withdraw-button',
    textAr: 'اضغط لسحب أموال من حسابك. متاح فقط بعد انتهاء فترة التسوية',
    textEn: 'Click to withdraw funds from your account. Only available after settlement period',
  },
  networkSelect: {
    id: 'network-select',
    textAr: 'اختر الشبكة التي ستستخدمها للإيداع. BNB أرخص، Solana أسرع',
    textEn: 'Choose the network you will use for deposit. BNB is cheaper, Solana is faster',
  },
  depositAddress: {
    id: 'deposit-address',
    textAr: 'عنوان المحفظة لإرسال USDC إليه. تأكد من استخدام الشبكة الصحيحة!',
    textEn: 'Wallet address to send USDC to. Make sure to use the correct network!',
  },
  copyAddress: {
    id: 'copy-address',
    textAr: 'اضغط لنسخ العنوان إلى الحافظة',
    textEn: 'Click to copy address to clipboard',
  },
  depositFee: {
    id: 'deposit-fee',
    textAr: 'رسوم الإيداع 1% من المبلغ المودع',
    textEn: 'Deposit fee is 1% of the deposited amount',
  },
  withdrawFee: {
    id: 'withdraw-fee',
    textAr: 'رسوم السحب 1% + رسوم الشبكة',
    textEn: 'Withdrawal fee is 1% + network fees',
  },
  settlementPeriod: {
    id: 'settlement-period',
    textAr: 'فترة التسوية 7 أيام من تاريخ الإيداع لحماية الصندوق',
    textEn: 'Settlement period is 7 days from deposit date to protect the fund',
  },
};

// تلميحات الصفقات
export const tradestooltips: Record<string, TooltipDefinition> = {
  tradeType: {
    id: 'trade-type',
    textAr: 'نوع الصفقة: شراء (BUY) أو بيع (SELL)',
    textEn: 'Trade type: BUY or SELL',
  },
  tradePair: {
    id: 'trade-pair',
    textAr: 'زوج التداول - العملة المتداولة مقابل USDC',
    textEn: 'Trading pair - the traded currency against USDC',
  },
  entryPrice: {
    id: 'entry-price',
    textAr: 'سعر الدخول - السعر الذي تم الشراء أو البيع عنده',
    textEn: 'Entry price - the price at which the buy or sell was executed',
  },
  currentPrice: {
    id: 'current-price',
    textAr: 'السعر الحالي للعملة في السوق',
    textEn: 'Current market price of the currency',
  },
  profitLoss: {
    id: 'profit-loss',
    textAr: 'الربح أو الخسارة من هذه الصفقة',
    textEn: 'Profit or loss from this trade',
  },
  tradeStatus: {
    id: 'trade-status',
    textAr: 'حالة الصفقة: مفتوحة (نشطة) أو مغلقة (منتهية)',
    textEn: 'Trade status: Open (active) or Closed (completed)',
  },
  tradeAmount: {
    id: 'trade-amount',
    textAr: 'كمية العملة المتداولة في هذه الصفقة',
    textEn: 'Amount of currency traded in this transaction',
  },
};

// تلميحات الإعدادات
export const settingsTooltips: Record<string, TooltipDefinition> = {
  language: {
    id: 'language',
    textAr: 'اختر لغة واجهة المنصة',
    textEn: 'Choose the platform interface language',
  },
  theme: {
    id: 'theme',
    textAr: 'اختر المظهر: داكن أو فاتح',
    textEn: 'Choose theme: Dark or Light',
  },
  notifications: {
    id: 'notifications',
    textAr: 'إعدادات الإشعارات - اختر ما تريد تلقي إشعارات عنه',
    textEn: 'Notification settings - choose what you want to receive notifications about',
  },
  twoFactor: {
    id: 'two-factor',
    textAr: 'المصادقة الثنائية - طبقة حماية إضافية لحسابك',
    textEn: 'Two-factor authentication - additional security layer for your account',
  },
  changePassword: {
    id: 'change-password',
    textAr: 'تغيير كلمة المرور الحالية',
    textEn: 'Change your current password',
  },
  startTour: {
    id: 'start-tour',
    textAr: 'ابدأ جولة تعريفية للتعرف على المنصة',
    textEn: 'Start an introductory tour to learn about the platform',
  },
};

// تلميحات الإحالات
export const referralsTooltips: Record<string, TooltipDefinition> = {
  referralCode: {
    id: 'referral-code',
    textAr: 'رمز الإحالة الفريد الخاص بك. شاركه مع أصدقائك',
    textEn: 'Your unique referral code. Share it with your friends',
  },
  totalReferrals: {
    id: 'total-referrals',
    textAr: 'عدد الأشخاص الذين سجلوا باستخدام رمزك',
    textEn: 'Number of people who registered using your code',
  },
  totalEarnings: {
    id: 'total-earnings',
    textAr: 'إجمالي المكافآت التي حصلت عليها من الإحالات',
    textEn: 'Total rewards you earned from referrals',
  },
  shareLink: {
    id: 'share-link',
    textAr: 'رابط مباشر للتسجيل يتضمن رمز الإحالة الخاص بك',
    textEn: 'Direct registration link that includes your referral code',
  },
};

// دالة للحصول على نص التلميح حسب اللغة
export function getTooltipText(tooltip: TooltipDefinition, language: 'ar' | 'en'): string {
  return language === 'ar' ? tooltip.textAr : tooltip.textEn;
}

export default {
  dashboard: dashboardTooltips,
  wallet: walletTooltips,
  trades: tradestooltips,
  settings: settingsTooltips,
  referrals: referralsTooltips,
};
