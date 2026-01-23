/**
 * FAQSection Component
 * قسم الأسئلة الشائعة المحسن
 * يتضمن بحث، تصنيفات، وتصميم احترافي
 */
import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  ChevronDown,
  HelpCircle,
  Wallet,
  TrendingUp,
  Shield,
  User,
  CreditCard,
  Bot,
  ArrowUpDown,
} from 'lucide-react';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';

// تعريف الأسئلة الشائعة
interface FAQItem {
  id: number;
  question: string;
  questionEn: string;
  answer: string;
  answerEn: string;
  category: string;
  categoryEn: string;
}

const faqData: FAQItem[] = [
  // قسم الحساب
  {
    id: 1,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'كيف أسجل حساب جديد؟',
    questionEn: 'How do I create a new account?',
    answer: 'يمكنك التسجيل بسهولة عبر الضغط على زر "إنشاء حساب" في الصفحة الرئيسية. أدخل بريدك الإلكتروني واختر كلمة مرور قوية. يمكنك أيضاً التسجيل باستخدام حساب Google الخاص بك.',
    answerEn: 'You can easily register by clicking the "Create Account" button on the homepage. Enter your email and choose a strong password. You can also sign up using your Google account.',
  },
  {
    id: 2,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'كيف أغير كلمة المرور؟',
    questionEn: 'How do I change my password?',
    answer: 'اذهب إلى صفحة الإعدادات > الأمان > تغيير كلمة المرور. أدخل كلمة المرور الحالية ثم الجديدة مرتين للتأكيد.',
    answerEn: 'Go to Settings > Security > Change Password. Enter your current password, then the new one twice to confirm.',
  },
  {
    id: 3,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'هل بياناتي آمنة؟',
    questionEn: 'Is my data secure?',
    answer: 'نعم، نستخدم تشفير AES-256 لحماية بياناتك. كما نوفر مصادقة ثنائية (2FA) لحماية إضافية لحسابك.',
    answerEn: 'Yes, we use AES-256 encryption to protect your data. We also offer two-factor authentication (2FA) for additional account protection.',
  },
  {
    id: 4,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'نسيت كلمة المرور، ماذا أفعل؟',
    questionEn: 'I forgot my password, what should I do?',
    answer: 'اضغط على "نسيت كلمة المرور" في صفحة تسجيل الدخول. سنرسل لك رابط إعادة تعيين كلمة المرور على بريدك الإلكتروني.',
    answerEn: 'Click "Forgot Password" on the login page. We will send you a password reset link to your email.',
  },
  
  // قسم الإيداع
  {
    id: 5,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هو الحد الأدنى للإيداع؟',
    questionEn: 'What is the minimum deposit?',
    answer: 'الحد الأدنى للإيداع هو 100 USDC. هذا المبلغ يسمح لك بالمشاركة في صندوق التداول والاستفادة من أداء الوكيل الذكي.',
    answerEn: 'The minimum deposit is 100 USDC. This amount allows you to participate in the trading fund and benefit from the AI agent performance.',
  },
  {
    id: 6,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هي العملات المدعومة للإيداع؟',
    questionEn: 'What currencies are supported for deposit?',
    answer: 'نحن ندعم USDC على شبكتين: BNB Smart Chain (BEP20) و Solana. اختر الشبكة التي تناسبك حسب رسوم التحويل.',
    answerEn: 'We support USDC on two networks: BNB Smart Chain (BEP20) and Solana. Choose the network that suits you based on transfer fees.',
  },
  {
    id: 7,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'كم يستغرق تأكيد الإيداع؟',
    questionEn: 'How long does deposit confirmation take?',
    answer: 'عادة يتم تأكيد الإيداع خلال 5-15 دقيقة حسب ازدحام الشبكة. ستتلقى إشعاراً فور تأكيد الإيداع.',
    answerEn: 'Usually, deposits are confirmed within 5-15 minutes depending on network congestion. You will receive a notification once the deposit is confirmed.',
  },
  {
    id: 8,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هي رسوم الإيداع؟',
    questionEn: 'What are the deposit fees?',
    answer: 'رسوم الإيداع هي 1% من المبلغ المودع. هذه الرسوم تغطي تكاليف الشبكة والمعالجة.',
    answerEn: 'Deposit fees are 1% of the deposited amount. These fees cover network and processing costs.',
  },
  
  // قسم السحب
  {
    id: 9,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'كيف يمكنني سحب أرباحي؟',
    questionEn: 'How can I withdraw my profits?',
    answer: 'اذهب إلى صفحة المحفظة > سحب. أدخل المبلغ وعنوان محفظتك. سيتم معالجة السحب خلال 24-48 ساعة.',
    answerEn: 'Go to Wallet page > Withdraw. Enter the amount and your wallet address. Withdrawal will be processed within 24-48 hours.',
  },
  {
    id: 10,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'ما هي رسوم السحب؟',
    questionEn: 'What are the withdrawal fees?',
    answer: 'رسوم السحب هي 1% من المبلغ المسحوب بالإضافة إلى رسوم الشبكة. الحد الأدنى للسحب هو 50 USDC.',
    answerEn: 'Withdrawal fees are 1% of the withdrawn amount plus network fees. The minimum withdrawal is 50 USDC.',
  },
  {
    id: 11,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'هل هناك فترة انتظار للسحب؟',
    questionEn: 'Is there a waiting period for withdrawal?',
    answer: 'نعم، هناك فترة تسوية مدتها 7 أيام من تاريخ آخر إيداع. بعد ذلك يمكنك السحب في أي وقت.',
    answerEn: 'Yes, there is a 7-day settlement period from the date of your last deposit. After that, you can withdraw at any time.',
  },
  
  // قسم التداول
  {
    id: 12,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'كيف يعمل الوكيل الذكي؟',
    questionEn: 'How does the AI agent work?',
    answer: 'الوكيل الذكي يستخدم خوارزميات متقدمة لتحليل السوق واتخاذ قرارات التداول. يعتمد على مؤشرات فنية مثل RSI، MACD، والمتوسطات المتحركة لتحديد أفضل فرص الشراء والبيع.',
    answerEn: 'The AI agent uses advanced algorithms to analyze the market and make trading decisions. It relies on technical indicators like RSI, MACD, and moving averages to identify the best buy and sell opportunities.',
  },
  {
    id: 13,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'ما هو NAV؟',
    questionEn: 'What is NAV?',
    answer: 'NAV (صافي قيمة الأصول) هو سعر الوحدة الاستثمارية. يتغير يومياً بناءً على أداء المحفظة. عند الإيداع، تحصل على وحدات بسعر NAV الحالي.',
    answerEn: 'NAV (Net Asset Value) is the price of an investment unit. It changes daily based on portfolio performance. When you deposit, you get units at the current NAV price.',
  },
  {
    id: 14,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'هل يمكنني خسارة أموالي؟',
    questionEn: 'Can I lose my money?',
    answer: 'التداول في العملات الرقمية يحمل مخاطر. قد تزيد أو تنقص قيمة استثمارك. الوكيل الذكي يستخدم استراتيجيات إدارة المخاطر لتقليل الخسائر المحتملة.',
    answerEn: 'Trading in cryptocurrencies carries risks. Your investment value may increase or decrease. The AI agent uses risk management strategies to minimize potential losses.',
  },
  {
    id: 15,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'أين يمكنني رؤية الصفقات؟',
    questionEn: 'Where can I see the trades?',
    answer: 'يمكنك رؤية جميع الصفقات في صفحة "الصفقات". تعرض الصفحة الصفقات النشطة وسجل الصفقات السابقة مع تفاصيل كاملة.',
    answerEn: 'You can see all trades on the "Trades" page. The page shows active trades and history of previous trades with full details.',
  },
  
  // قسم الرسوم
  {
    id: 20,
    category: 'الرسوم',
    categoryEn: 'Fees',
    question: 'ما هي رسوم الأداء وكيف تُحسب؟',
    questionEn: 'What are performance fees and how are they calculated?',
    answer: 'رسوم الأداء هي 15% من أرباحك الشهرية فقط. هذا يعني أننا نأخذ رسوماً فقط عندما تربح. مثال: إذا ربحت 100$ هذا الشهر، سيتم خصم 15$ كرسوم أداء وستحصل على 85$ صافي ربح. إذا لم تربح أو خسرت، لا تُخصم أي رسوم أداء.',
    answerEn: 'Performance fees are 15% of your monthly profits only. This means we only charge fees when you profit. Example: If you profit $100 this month, $15 will be deducted as performance fee and you will receive $85 net profit. If you did not profit or lost, no performance fees are charged.',
  },
  {
    id: 21,
    category: 'الرسوم',
    categoryEn: 'Fees',
    question: 'ما هي علامة الذروة المائية (High-Water Mark)؟',
    questionEn: 'What is the High-Water Mark?',
    answer: 'علامة الذروة المائية هي آلية لحمايتك. ببساطة: لا نأخذ رسوم أداء إلا عندما يتجاوز رصيدك أعلى قيمة وصل إليها سابقاً. مثال: إذا وصل رصيدك إلى 1000$ ثم انخفض إلى 900$ ثم ارتفع إلى 950$، لن نأخذ رسوم أداء لأن الرصيد لم يتجاوز 1000$ بعد. فقط عندما يتجاوز 1000$ نأخذ 15% من الزيادة.',
    answerEn: 'The High-Water Mark is a mechanism to protect you. Simply: we only charge performance fees when your balance exceeds the highest value it has ever reached. Example: If your balance reached $1000 then dropped to $900 then rose to $950, we will not charge performance fees because the balance has not exceeded $1000 yet. Only when it exceeds $1000, we take 15% of the increase.',
  },
  {
    id: 22,
    category: 'الرسوم',
    categoryEn: 'Fees',
    question: 'ملخص جميع الرسوم',
    questionEn: 'Summary of all fees',
    answer: 'هناك ثلاثة أنواع من الرسوم:\n\n1️⃣ رسوم الإيداع: 1% من المبلغ المودع\n2️⃣ رسوم السحب: 1% من المبلغ المسحوب\n3️⃣ رسوم الأداء: 15% من الأرباح الشهرية فقط (مع حماية علامة الذروة المائية)\n\nلا توجد رسوم مخفية أخرى.',
    answerEn: 'There are three types of fees:\n\n1️⃣ Deposit fees: 1% of the deposited amount\n2️⃣ Withdrawal fees: 1% of the withdrawn amount\n3️⃣ Performance fees: 15% of monthly profits only (with High-Water Mark protection)\n\nThere are no other hidden fees.',
  },
  
  // قسم الإحالات
  {
    id: 16,
    category: 'الإحالات',
    categoryEn: 'Referrals',
    question: 'كيف يعمل برنامج الإحالات؟',
    questionEn: 'How does the referral program work?',
    answer: 'شارك رمز الإحالة الخاص بك مع أصدقائك. عندما يسجلون ويقومون بأول إيداع (100 USDC على الأقل)، تحصل على مكافأة 10 دولار.',
    answerEn: 'Share your referral code with friends. When they register and make their first deposit (at least 100 USDC), you get a $10 reward.',
  },
  {
    id: 17,
    category: 'الإحالات',
    categoryEn: 'Referrals',
    question: 'أين أجد رمز الإحالة الخاص بي؟',
    questionEn: 'Where can I find my referral code?',
    answer: 'يمكنك العثور على رمز الإحالة في صفحة "الإحالات" من القائمة الجانبية. يمكنك نسخ الرمز أو مشاركة الرابط مباشرة.',
    answerEn: 'You can find your referral code on the "Referrals" page from the sidebar menu. You can copy the code or share the link directly.',
  },
  
  // قسم الأمان
  {
    id: 18,
    category: 'الأمان',
    categoryEn: 'Security',
    question: 'كيف أفعّل المصادقة الثنائية؟',
    questionEn: 'How do I enable two-factor authentication?',
    answer: 'اذهب إلى الإعدادات > الأمان > المصادقة الثنائية. اتبع التعليمات لربط تطبيق المصادقة (مثل Google Authenticator) بحسابك.',
    answerEn: 'Go to Settings > Security > Two-Factor Authentication. Follow the instructions to link an authenticator app (like Google Authenticator) to your account.',
  },
  {
    id: 19,
    category: 'الأمان',
    categoryEn: 'Security',
    question: 'ماذا أفعل إذا لاحظت نشاطاً مشبوهاً؟',
    questionEn: 'What should I do if I notice suspicious activity?',
    answer: 'غيّر كلمة المرور فوراً وتواصل مع فريق الدعم. راجع سجل تسجيل الدخول في الإعدادات للتحقق من أي نشاط غير معتاد.',
    answerEn: 'Change your password immediately and contact the support team. Review the login history in settings to check for any unusual activity.',
  },
];

// تصنيفات الأسئلة مع الأيقونات
const categories = [
  { id: 'all', name: 'الكل', nameEn: 'All', icon: HelpCircle },
  { id: 'الحساب', name: 'الحساب', nameEn: 'Account', icon: User },
  { id: 'الإيداع', name: 'الإيداع', nameEn: 'Deposit', icon: CreditCard },
  { id: 'السحب', name: 'السحب', nameEn: 'Withdrawal', icon: ArrowUpDown },
  { id: 'الرسوم', name: 'الرسوم', nameEn: 'Fees', icon: CreditCard },
  { id: 'التداول', name: 'التداول', nameEn: 'Trading', icon: TrendingUp },
  { id: 'الإحالات', name: 'الإحالات', nameEn: 'Referrals', icon: Wallet },
  { id: 'الأمان', name: 'الأمان', nameEn: 'Security', icon: Shield },
];

interface FAQSectionProps {
  language?: 'ar' | 'en';
  className?: string;
}

export function FAQSection({ language = 'ar', className }: FAQSectionProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const isRTL = language === 'ar';

  // تصفية الأسئلة
  const filteredFaqs = useMemo(() => {
    return faqData.filter((faq) => {
      const matchesSearch =
        searchTerm === '' ||
        faq.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
        faq.questionEn.toLowerCase().includes(searchTerm.toLowerCase()) ||
        faq.answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
        faq.answerEn.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesCategory =
        selectedCategory === 'all' || faq.category === selectedCategory;

      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, selectedCategory]);

  const toggleExpand = (id: number) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className={cn("space-y-6", className)} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* شريط البحث */}
      <div className="relative">
        <Search className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        <Input
          type="text"
          placeholder={isRTL ? 'ابحث في الأسئلة الشائعة...' : 'Search FAQ...'}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pr-10 h-12 text-base"
        />
      </div>

      {/* التصنيفات */}
      <div className="flex flex-wrap gap-2">
        {categories.map((cat) => {
          const Icon = cat.icon;
          const isSelected = selectedCategory === cat.id;
          return (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all",
                isSelected
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted hover:bg-muted/80 text-muted-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {isRTL ? cat.name : cat.nameEn}
            </button>
          );
        })}
      </div>

      {/* عدد النتائج */}
      <p className="text-sm text-muted-foreground">
        {isRTL 
          ? `عرض ${filteredFaqs.length} من ${faqData.length} سؤال`
          : `Showing ${filteredFaqs.length} of ${faqData.length} questions`
        }
      </p>

      {/* قائمة الأسئلة */}
      <div className="space-y-3">
        <AnimatePresence>
          {filteredFaqs.map((faq) => (
            <motion.div
              key={faq.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="border rounded-lg overflow-hidden bg-card"
            >
              {/* السؤال */}
              <button
                onClick={() => toggleExpand(faq.id)}
                className="w-full flex items-center justify-between p-4 text-right hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Badge variant="secondary" className="text-xs">
                    {isRTL ? faq.category : faq.categoryEn}
                  </Badge>
                  <span className="font-medium">
                    {isRTL ? faq.question : faq.questionEn}
                  </span>
                </div>
                <ChevronDown
                  className={cn(
                    "h-5 w-5 text-muted-foreground transition-transform",
                    expandedId === faq.id && "rotate-180"
                  )}
                />
              </button>

              {/* الإجابة */}
              <AnimatePresence>
                {expandedId === faq.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="p-4 pt-0 text-muted-foreground leading-relaxed border-t bg-muted/30">
                      {isRTL ? faq.answer : faq.answerEn}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* رسالة عدم وجود نتائج */}
        {filteredFaqs.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            <HelpCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">
              {isRTL ? 'لم يتم العثور على نتائج' : 'No results found'}
            </p>
            <p className="text-sm mt-2">
              {isRTL 
                ? 'جرب البحث بكلمات مختلفة أو اختر تصنيفاً آخر'
                : 'Try searching with different words or select another category'
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FAQSection;
