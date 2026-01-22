import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Input } from "../components/ui/input";
import { 
  MessageSquare, 
  Plus, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Send,
  HelpCircle,
  FileText,
  Search,
  User,
  CreditCard,
  ArrowUpDown,
  TrendingUp,
  Wallet,
  Shield,
  Bot,
  BookOpen,
  Lightbulb,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import api from "../services/api";
import { useLanguage } from '@/lib/i18n';
import { motion, AnimatePresence } from "framer-motion";

interface Ticket {
  id: number;
  subject: string;
  message: string;
  status: string;
  priority: string;
  created_at: string;
  updated_at?: string;
  replies?: TicketReply[];
}

interface TicketReply {
  id: number;
  message: string;
  is_admin: boolean;
  created_at: string;
}

// تعريف الأسئلة الشائعة الشاملة
interface FAQItem {
  id: number;
  question: string;
  questionEn: string;
  answer: string;
  answerEn: string;
  category: string;
  categoryEn: string;
}

const comprehensiveFAQs: FAQItem[] = [
  // قسم الحساب
  {
    id: 1,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'كيف أسجل حساب جديد؟',
    questionEn: 'How do I create a new account?',
    answer: 'يمكنك التسجيل بسهولة عبر الضغط على زر "إنشاء حساب" في الصفحة الرئيسية. أدخل بريدك الإلكتروني واختر كلمة مرور قوية. يمكنك أيضاً التسجيل باستخدام حساب Google الخاص بك للتسجيل السريع.',
    answerEn: 'You can easily register by clicking the "Create Account" button on the homepage. Enter your email and choose a strong password. You can also sign up using your Google account for quick registration.',
  },
  {
    id: 2,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'كيف أغير كلمة المرور؟',
    questionEn: 'How do I change my password?',
    answer: 'اذهب إلى صفحة الإعدادات من القائمة الجانبية، ثم اختر قسم "الأمان"، ومن هناك يمكنك تغيير كلمة المرور. أدخل كلمة المرور الحالية ثم الجديدة مرتين للتأكيد.',
    answerEn: 'Go to the Settings page from the sidebar menu, then select the "Security" section, and from there you can change your password. Enter your current password, then the new one twice to confirm.',
  },
  {
    id: 3,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'هل بياناتي آمنة؟',
    questionEn: 'Is my data secure?',
    answer: 'نعم، نحن نأخذ أمان بياناتك على محمل الجد. نستخدم تشفير AES-256 لحماية بياناتك، ونوفر مصادقة ثنائية (2FA) لحماية إضافية لحسابك. جميع الاتصالات مشفرة باستخدام HTTPS.',
    answerEn: 'Yes, we take your data security very seriously. We use AES-256 encryption to protect your data, and we offer two-factor authentication (2FA) for additional account protection. All communications are encrypted using HTTPS.',
  },
  {
    id: 4,
    category: 'الحساب',
    categoryEn: 'Account',
    question: 'نسيت كلمة المرور، ماذا أفعل؟',
    questionEn: 'I forgot my password, what should I do?',
    answer: 'اضغط على "نسيت كلمة المرور" في صفحة تسجيل الدخول. أدخل بريدك الإلكتروني وسنرسل لك رابط إعادة تعيين كلمة المرور. تحقق من صندوق الوارد أو مجلد الرسائل غير المرغوب فيها.',
    answerEn: 'Click "Forgot Password" on the login page. Enter your email and we will send you a password reset link. Check your inbox or spam folder.',
  },
  
  // قسم الإيداع
  {
    id: 5,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هو الحد الأدنى للإيداع؟',
    questionEn: 'What is the minimum deposit?',
    answer: 'الحد الأدنى للإيداع هو 100 USDC. هذا المبلغ يسمح لك بالمشاركة في صندوق التداول والاستفادة من أداء الوكيل الذكي. يمكنك إيداع أي مبلغ أكبر من ذلك.',
    answerEn: 'The minimum deposit is 100 USDC. This amount allows you to participate in the trading fund and benefit from the AI agent performance. You can deposit any amount greater than that.',
  },
  {
    id: 6,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هي العملات المدعومة للإيداع؟',
    questionEn: 'What currencies are supported for deposit?',
    answer: 'نحن ندعم USDC (عملة مستقرة مرتبطة بالدولار) على شبكتين: BNB Smart Chain (BEP20) برسوم منخفضة، و Solana برسوم منخفضة جداً. اختر الشبكة التي تناسبك حسب المحفظة التي تستخدمها.',
    answerEn: 'We support USDC (a stablecoin pegged to the dollar) on two networks: BNB Smart Chain (BEP20) with low fees, and Solana with very low fees. Choose the network that suits you based on the wallet you use.',
  },
  {
    id: 7,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'كم يستغرق تأكيد الإيداع؟',
    questionEn: 'How long does deposit confirmation take?',
    answer: 'عادة يتم تأكيد الإيداع خلال 5-15 دقيقة حسب ازدحام الشبكة. ستتلقى إشعاراً فور تأكيد الإيداع وإضافته لرصيدك. يمكنك متابعة حالة الإيداع من صفحة المحفظة.',
    answerEn: 'Usually, deposits are confirmed within 5-15 minutes depending on network congestion. You will receive a notification once the deposit is confirmed and added to your balance. You can track the deposit status from the Wallet page.',
  },
  {
    id: 8,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'ما هي رسوم الإيداع؟',
    questionEn: 'What are the deposit fees?',
    answer: 'رسوم الإيداع هي 1% من المبلغ المودع. هذه الرسوم تغطي تكاليف الشبكة والمعالجة. مثال: إذا أودعت 100 USDC، سيتم إضافة 99 USDC لرصيدك.',
    answerEn: 'Deposit fees are 1% of the deposited amount. These fees cover network and processing costs. Example: If you deposit 100 USDC, 99 USDC will be added to your balance.',
  },
  {
    id: 9,
    category: 'الإيداع',
    categoryEn: 'Deposit',
    question: 'كيف أقوم بالإيداع؟',
    questionEn: 'How do I make a deposit?',
    answer: 'اذهب إلى صفحة المحفظة > اضغط على "إيداع" > اختر المبلغ والشبكة > ستحصل على عنوان محفظة لإرسال USDC إليه. تأكد من إرسال USDC فقط على الشبكة الصحيحة.',
    answerEn: 'Go to the Wallet page > Click "Deposit" > Choose the amount and network > You will get a wallet address to send USDC to. Make sure to send USDC only on the correct network.',
  },
  
  // قسم السحب
  {
    id: 10,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'كيف يمكنني سحب أرباحي؟',
    questionEn: 'How can I withdraw my profits?',
    answer: 'اذهب إلى صفحة المحفظة > اضغط على "سحب" > أدخل المبلغ وعنوان محفظتك الخارجية. سيتم معالجة السحب خلال 24-48 ساعة عمل. ستتلقى إشعاراً عند إتمام السحب.',
    answerEn: 'Go to the Wallet page > Click "Withdraw" > Enter the amount and your external wallet address. Withdrawal will be processed within 24-48 business hours. You will receive a notification when the withdrawal is complete.',
  },
  {
    id: 11,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'ما هي رسوم السحب؟',
    questionEn: 'What are the withdrawal fees?',
    answer: 'رسوم السحب هي 1% من المبلغ المسحوب بالإضافة إلى رسوم الشبكة (تختلف حسب الشبكة). الحد الأدنى للسحب هو 50 USDC.',
    answerEn: 'Withdrawal fees are 1% of the withdrawn amount plus network fees (varies by network). The minimum withdrawal is 50 USDC.',
  },
  {
    id: 12,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'هل هناك فترة انتظار للسحب؟',
    questionEn: 'Is there a waiting period for withdrawal?',
    answer: 'نعم، هناك فترة تسوية مدتها 7 أيام من تاريخ آخر إيداع. هذا لحماية الصندوق وضمان استقرار التداول. بعد انتهاء الفترة، يمكنك السحب في أي وقت.',
    answerEn: 'Yes, there is a 7-day settlement period from the date of your last deposit. This is to protect the fund and ensure trading stability. After the period ends, you can withdraw at any time.',
  },
  {
    id: 13,
    category: 'السحب',
    categoryEn: 'Withdrawal',
    question: 'لماذا السحب مقفل؟',
    questionEn: 'Why is withdrawal locked?',
    answer: 'السحب قد يكون مقفلاً لأحد الأسباب التالية: 1) لم تمر 7 أيام على آخر إيداع، 2) رصيدك أقل من الحد الأدنى للسحب (50 USDC)، 3) هناك عملية سحب معلقة بالفعل.',
    answerEn: 'Withdrawal may be locked for one of the following reasons: 1) 7 days have not passed since your last deposit, 2) Your balance is less than the minimum withdrawal (50 USDC), 3) There is already a pending withdrawal.',
  },
  
  // قسم التداول
  {
    id: 14,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'كيف يعمل الوكيل الذكي؟',
    questionEn: 'How does the AI agent work?',
    answer: 'الوكيل الذكي يستخدم خوارزميات متقدمة للذكاء الاصطناعي لتحليل السوق واتخاذ قرارات التداول. يعتمد على مؤشرات فنية مثل RSI، MACD، والمتوسطات المتحركة، بالإضافة إلى تحليل حجم التداول والزخم لتحديد أفضل فرص الشراء والبيع.',
    answerEn: 'The AI agent uses advanced artificial intelligence algorithms to analyze the market and make trading decisions. It relies on technical indicators like RSI, MACD, and moving averages, as well as volume and momentum analysis to identify the best buy and sell opportunities.',
  },
  {
    id: 15,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'ما هو NAV؟',
    questionEn: 'What is NAV?',
    answer: 'NAV (صافي قيمة الأصول) هو سعر الوحدة الاستثمارية في الصندوق. يتغير يومياً بناءً على أداء المحفظة. عند الإيداع، تحصل على وحدات بسعر NAV الحالي. عند السحب، يتم حساب قيمة وحداتك بسعر NAV وقت السحب.',
    answerEn: 'NAV (Net Asset Value) is the price of an investment unit in the fund. It changes daily based on portfolio performance. When you deposit, you get units at the current NAV price. When you withdraw, your units value is calculated at the NAV price at withdrawal time.',
  },
  {
    id: 16,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'هل يمكنني خسارة أموالي؟',
    questionEn: 'Can I lose my money?',
    answer: 'التداول في العملات الرقمية يحمل مخاطر. قد تزيد أو تنقص قيمة استثمارك حسب ظروف السوق. الوكيل الذكي يستخدم استراتيجيات إدارة المخاطر مثل وقف الخسارة وتنويع المحفظة لتقليل الخسائر المحتملة، لكن لا يمكن ضمان الأرباح.',
    answerEn: 'Trading in cryptocurrencies carries risks. Your investment value may increase or decrease depending on market conditions. The AI agent uses risk management strategies like stop-loss and portfolio diversification to minimize potential losses, but profits cannot be guaranteed.',
  },
  {
    id: 17,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'أين يمكنني رؤية الصفقات؟',
    questionEn: 'Where can I see the trades?',
    answer: 'يمكنك رؤية جميع الصفقات في صفحة "الصفقات" من القائمة الجانبية. تعرض الصفحة الصفقات النشطة حالياً وسجل الصفقات السابقة مع تفاصيل كاملة تشمل: العملة، نوع الصفقة (شراء/بيع)، السعر، الكمية، والربح/الخسارة.',
    answerEn: 'You can see all trades on the "Trades" page from the sidebar menu. The page shows currently active trades and history of previous trades with full details including: currency, trade type (buy/sell), price, quantity, and profit/loss.',
  },
  {
    id: 18,
    category: 'التداول',
    categoryEn: 'Trading',
    question: 'كم مرة يتداول الوكيل؟',
    questionEn: 'How often does the agent trade?',
    answer: 'يعمل الوكيل الذكي على مدار الساعة 24/7. عدد الصفقات يعتمد على ظروف السوق وفرص التداول المتاحة. قد يكون هناك عدة صفقات في اليوم أو لا توجد صفقات إذا لم تتوفر فرص جيدة.',
    answerEn: 'The AI agent works around the clock 24/7. The number of trades depends on market conditions and available trading opportunities. There may be several trades per day or no trades if good opportunities are not available.',
  },
  
  // قسم الإحالات
  {
    id: 19,
    category: 'الإحالات',
    categoryEn: 'Referrals',
    question: 'كيف يعمل برنامج الإحالات؟',
    questionEn: 'How does the referral program work?',
    answer: 'شارك رمز الإحالة الخاص بك مع أصدقائك. عندما يسجلون باستخدام رمزك ويقومون بأول إيداع (100 USDC على الأقل)، تحصل على مكافأة 10 دولار تُضاف لرصيدك تلقائياً.',
    answerEn: 'Share your referral code with friends. When they register using your code and make their first deposit (at least 100 USDC), you get a $10 reward automatically added to your balance.',
  },
  {
    id: 20,
    category: 'الإحالات',
    categoryEn: 'Referrals',
    question: 'أين أجد رمز الإحالة الخاص بي؟',
    questionEn: 'Where can I find my referral code?',
    answer: 'يمكنك العثور على رمز الإحالة في صفحة "الإحالات" من القائمة الجانبية. يمكنك نسخ الرمز أو مشاركة الرابط المباشر عبر وسائل التواصل الاجتماعي.',
    answerEn: 'You can find your referral code on the "Referrals" page from the sidebar menu. You can copy the code or share the direct link via social media.',
  },
  
  // قسم الأمان
  {
    id: 21,
    category: 'الأمان',
    categoryEn: 'Security',
    question: 'كيف أفعّل المصادقة الثنائية؟',
    questionEn: 'How do I enable two-factor authentication?',
    answer: 'اذهب إلى الإعدادات > الأمان > المصادقة الثنائية. اتبع التعليمات لربط تطبيق المصادقة (مثل Google Authenticator أو Authy) بحسابك. ستحتاج لإدخال رمز من التطبيق عند كل تسجيل دخول.',
    answerEn: 'Go to Settings > Security > Two-Factor Authentication. Follow the instructions to link an authenticator app (like Google Authenticator or Authy) to your account. You will need to enter a code from the app at each login.',
  },
  {
    id: 22,
    category: 'الأمان',
    categoryEn: 'Security',
    question: 'ماذا أفعل إذا لاحظت نشاطاً مشبوهاً؟',
    questionEn: 'What should I do if I notice suspicious activity?',
    answer: 'غيّر كلمة المرور فوراً وتواصل مع فريق الدعم عبر فتح تذكرة جديدة. راجع سجل تسجيل الدخول في الإعدادات للتحقق من أي نشاط غير معتاد. إذا كان لديك 2FA مفعّل، تأكد من أن جهازك آمن.',
    answerEn: 'Change your password immediately and contact the support team by opening a new ticket. Review the login history in settings to check for any unusual activity. If you have 2FA enabled, make sure your device is secure.',
  },
  {
    id: 23,
    category: 'الأمان',
    categoryEn: 'Security',
    question: 'هل أموالي مؤمّنة؟',
    questionEn: 'Are my funds insured?',
    answer: 'نحن نتخذ إجراءات أمنية صارمة لحماية الأموال، لكن التداول في العملات الرقمية يحمل مخاطر. ننصح بعدم استثمار أكثر مما يمكنك تحمل خسارته.',
    answerEn: 'We take strict security measures to protect funds, but trading in cryptocurrencies carries risks. We advise not to invest more than you can afford to lose.',
  },
];

// تصنيفات الأسئلة مع الأيقونات
const faqCategories = [
  { id: 'all', name: 'الكل', nameEn: 'All', icon: HelpCircle, color: 'text-primary' },
  { id: 'الحساب', name: 'الحساب', nameEn: 'Account', icon: User, color: 'text-blue-500' },
  { id: 'الإيداع', name: 'الإيداع', nameEn: 'Deposit', icon: CreditCard, color: 'text-green-500' },
  { id: 'السحب', name: 'السحب', nameEn: 'Withdrawal', icon: ArrowUpDown, color: 'text-orange-500' },
  { id: 'التداول', name: 'التداول', nameEn: 'Trading', icon: TrendingUp, color: 'text-purple-500' },
  { id: 'الإحالات', name: 'الإحالات', nameEn: 'Referrals', icon: Wallet, color: 'text-pink-500' },
  { id: 'الأمان', name: 'الأمان', nameEn: 'Security', icon: Shield, color: 'text-red-500' },
];

export default function Support() {
  const { t, language } = useLanguage();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("faq");
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [showNewTicket, setShowNewTicket] = useState(false);
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  
  // New ticket form
  const [newSubject, setNewSubject] = useState("");
  const [newMessage, setNewMessage] = useState("");
  const [newPriority, setNewPriority] = useState("medium");
  
  // Reply form
  const [replyMessage, setReplyMessage] = useState("");

  const isRTL = language === 'ar';

  // Filter FAQs based on search and category
  const filteredFaqs = comprehensiveFAQs.filter((faq) => {
    const searchLower = searchTerm.toLowerCase();
    const matchesSearch =
      searchTerm === '' ||
      faq.question.toLowerCase().includes(searchLower) ||
      faq.questionEn.toLowerCase().includes(searchLower) ||
      faq.answer.toLowerCase().includes(searchLower) ||
      faq.answerEn.toLowerCase().includes(searchLower);

    const matchesCategory =
      selectedCategory === 'all' || faq.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  // Fetch tickets
  const { data: tickets = [], isLoading: loadingTickets } = useQuery<Ticket[]>({
    queryKey: ["/api/v1/support/tickets/my"],
    queryFn: async () => {
      try {
        const res = await api.get("/support/tickets/my");
        return res.data;
      } catch {
        return [];
      }
    },
  });

  // Create ticket mutation
  const createTicketMutation = useMutation({
    mutationFn: async (data: { subject: string; message: string; priority: string }) => {
      const res = await api.post("/support/tickets", data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets/my"] });
      setShowNewTicket(false);
      setNewSubject("");
      setNewMessage("");
      setNewPriority("medium");
      toast.success(language === 'ar' ? 'تم إنشاء التذكرة بنجاح' : 'Ticket created successfully');
    },
    onError: () => {
      toast.error(language === 'ar' ? 'فشل في إنشاء التذكرة' : 'Failed to create ticket');
    },
  });

  // Reply mutation
  const replyMutation = useMutation({
    mutationFn: async (data: { ticketId: number; message: string }) => {
      const res = await api.post(`/support/tickets/${data.ticketId}/reply`, {
        message: data.message,
      });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets/my"] });
      setReplyMessage("");
      toast.success(language === 'ar' ? 'تم إرسال الرد' : 'Reply sent');
    },
    onError: () => {
      toast.error(language === 'ar' ? 'فشل في إرسال الرد' : 'Failed to send reply');
    },
  });

  const handleCreateTicket = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubject.trim() || !newMessage.trim()) return;
    createTicketMutation.mutate({
      subject: newSubject,
      message: newMessage,
      priority: newPriority,
    });
  };

  const handleReply = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedTicket || !replyMessage.trim()) return;
    replyMutation.mutate({
      ticketId: selectedTicket.id,
      message: replyMessage,
    });
  };

  const getStatusBadge = (status: string) => {
    const statusConfig: Record<string, { color: string; icon: any; label: string; labelEn: string }> = {
      open: { color: "bg-blue-500/20 text-blue-400", icon: Clock, label: "مفتوح", labelEn: "Open" },
      in_progress: { color: "bg-yellow-500/20 text-yellow-400", icon: AlertCircle, label: "قيد المعالجة", labelEn: "In Progress" },
      resolved: { color: "bg-green-500/20 text-green-400", icon: CheckCircle, label: "تم الحل", labelEn: "Resolved" },
      closed: { color: "bg-gray-500/20 text-gray-400", icon: CheckCircle, label: "مغلق", labelEn: "Closed" },
    };
    const config = statusConfig[status] || statusConfig.open;
    const Icon = config.icon;
    return (
      <Badge className={cn("flex items-center gap-1", config.color)}>
        <Icon className="w-3 h-3" />
        {language === 'ar' ? config.label : config.labelEn}
      </Badge>
    );
  };

  return (
    <div className="container mx-auto p-4 md:p-6 max-w-5xl" dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="p-4 rounded-full bg-primary/10">
            <HelpCircle className="h-10 w-10 text-primary" />
          </div>
        </div>
        <h1 className="text-3xl font-bold mb-2">{t.support.title}</h1>
        <p className="text-muted-foreground text-lg">
          {language === 'ar' 
            ? 'نحن هنا لمساعدتك! ابحث في الأسئلة الشائعة أو تواصل معنا'
            : 'We are here to help! Search FAQ or contact us'
          }
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 h-12">
          <TabsTrigger value="faq" className="flex items-center gap-2 text-base">
            <BookOpen className="w-4 h-4" />
            {t.support.faq}
          </TabsTrigger>
          <TabsTrigger value="tickets" className="flex items-center gap-2 text-base">
            <MessageSquare className="w-4 h-4" />
            {t.support.myTickets}
          </TabsTrigger>
        </TabsList>

        {/* FAQ Tab */}
        <TabsContent value="faq" className="space-y-6">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input
              type="text"
              placeholder={language === 'ar' ? 'ابحث في الأسئلة الشائعة...' : 'Search FAQ...'}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pr-12 h-14 text-lg bg-card border-2"
            />
          </div>

          {/* Category Filters */}
          <div className="flex flex-wrap gap-2">
            {faqCategories.map((cat) => {
              const Icon = cat.icon;
              const isSelected = selectedCategory === cat.id;
              return (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-all",
                    isSelected
                      ? "bg-primary text-primary-foreground shadow-lg"
                      : "bg-card hover:bg-muted border"
                  )}
                >
                  <Icon className={cn("h-4 w-4", !isSelected && cat.color)} />
                  {isRTL ? cat.name : cat.nameEn}
                </button>
              );
            })}
          </div>

          {/* Results Count */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {isRTL 
                ? `عرض ${filteredFaqs.length} من ${comprehensiveFAQs.length} سؤال`
                : `Showing ${filteredFaqs.length} of ${comprehensiveFAQs.length} questions`
              }
            </p>
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="text-sm text-primary hover:underline"
              >
                {isRTL ? 'مسح البحث' : 'Clear search'}
              </button>
            )}
          </div>

          {/* FAQ List */}
          <div className="space-y-3">
            <AnimatePresence>
              {filteredFaqs.map((faq) => {
                const categoryConfig = faqCategories.find(c => c.id === faq.category);
                const CategoryIcon = categoryConfig?.icon || HelpCircle;
                
                return (
                  <motion.div
                    key={faq.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Card 
                      className={cn(
                        "overflow-hidden transition-all cursor-pointer hover:shadow-md",
                        expandedFaq === faq.id && "ring-2 ring-primary/50"
                      )}
                      onClick={() => setExpandedFaq(expandedFaq === faq.id ? null : faq.id)}
                    >
                      <CardContent className="p-0">
                        {/* Question */}
                        <div className="flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
                          <div className="flex items-center gap-3 flex-1">
                            <div className={cn("p-2 rounded-lg bg-muted", categoryConfig?.color)}>
                              <CategoryIcon className="h-4 w-4" />
                            </div>
                            <div className="flex-1">
                              <Badge variant="secondary" className="text-xs mb-1">
                                {isRTL ? faq.category : faq.categoryEn}
                              </Badge>
                              <p className="font-medium text-base">
                                {isRTL ? faq.question : faq.questionEn}
                              </p>
                            </div>
                          </div>
                          <ChevronDown
                            className={cn(
                              "h-5 w-5 text-muted-foreground transition-transform flex-shrink-0 mr-2",
                              expandedFaq === faq.id && "rotate-180"
                            )}
                          />
                        </div>

                        {/* Answer */}
                        <AnimatePresence>
                          {expandedFaq === faq.id && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.2 }}
                              className="overflow-hidden"
                            >
                              <div className="p-4 pt-0 border-t bg-muted/30">
                                <div className="flex items-start gap-3 pt-4">
                                  <Lightbulb className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                                  <p className="text-muted-foreground leading-relaxed">
                                    {isRTL ? faq.answer : faq.answerEn}
                                  </p>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {/* No Results */}
            {filteredFaqs.length === 0 && (
              <Card className="p-12 text-center">
                <HelpCircle className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
                <h3 className="text-xl font-medium mb-2">
                  {isRTL ? 'لم يتم العثور على نتائج' : 'No results found'}
                </h3>
                <p className="text-muted-foreground mb-4">
                  {isRTL 
                    ? 'جرب البحث بكلمات مختلفة أو اختر تصنيفاً آخر'
                    : 'Try searching with different words or select another category'
                  }
                </p>
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setSelectedCategory('all');
                  }}
                  className="text-primary hover:underline"
                >
                  {isRTL ? 'عرض جميع الأسئلة' : 'Show all questions'}
                </button>
              </Card>
            )}
          </div>

          {/* Contact Support Card */}
          <Card className="bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-full bg-primary/20">
                  <MessageSquare className="h-6 w-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-lg mb-1">
                    {isRTL ? 'لم تجد إجابة لسؤالك؟' : "Didn't find an answer?"}
                  </h3>
                  <p className="text-muted-foreground">
                    {isRTL 
                      ? 'تواصل مع فريق الدعم وسنرد عليك في أقرب وقت'
                      : 'Contact our support team and we will respond as soon as possible'
                    }
                  </p>
                </div>
                <button
                  onClick={() => {
                    setActiveTab('tickets');
                    setShowNewTicket(true);
                  }}
                  className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
                >
                  {isRTL ? 'تواصل معنا' : 'Contact Us'}
                </button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tickets Tab */}
        <TabsContent value="tickets" className="space-y-6">
          {/* New Ticket Button */}
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold">{t.support.myTickets}</h2>
            <button
              onClick={() => setShowNewTicket(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              {t.support.newTicket}
            </button>
          </div>

          {/* Tickets List */}
          {loadingTickets ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : tickets.length === 0 ? (
            <Card className="p-12 text-center">
              <MessageSquare className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-xl font-medium mb-2">
                {language === 'ar' ? 'لا توجد تذاكر' : 'No tickets'}
              </h3>
              <p className="text-muted-foreground mb-4">
                {language === 'ar' 
                  ? 'لم تقم بإنشاء أي تذاكر دعم بعد'
                  : "You haven't created any support tickets yet"
                }
              </p>
              <button
                onClick={() => setShowNewTicket(true)}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg"
              >
                {t.support.newTicket}
              </button>
            </Card>
          ) : (
            <div className="space-y-4">
              {tickets.map((ticket) => (
                <Card
                  key={ticket.id}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-md",
                    selectedTicket?.id === ticket.id && "ring-2 ring-primary"
                  )}
                  onClick={() => setSelectedTicket(selectedTicket?.id === ticket.id ? null : ticket)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          {getStatusBadge(ticket.status)}
                          <Badge variant="outline" className="text-xs">
                            #{ticket.id}
                          </Badge>
                        </div>
                        <h3 className="font-medium text-lg">{ticket.subject}</h3>
                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                          {ticket.message}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          {format(new Date(ticket.created_at), "yyyy/MM/dd HH:mm")}
                        </p>
                      </div>
                      <ChevronDown
                        className={cn(
                          "h-5 w-5 text-muted-foreground transition-transform",
                          selectedTicket?.id === ticket.id && "rotate-180"
                        )}
                      />
                    </div>

                    {/* Expanded Content */}
                    {selectedTicket?.id === ticket.id && (
                      <div className="mt-4 pt-4 border-t space-y-4">
                        {/* Replies */}
                        {ticket.replies && ticket.replies.length > 0 && (
                          <div className="space-y-3">
                            {ticket.replies.map((reply) => (
                              <div
                                key={reply.id}
                                className={cn(
                                  "p-3 rounded-lg",
                                  reply.is_admin
                                    ? "bg-primary/10 mr-8"
                                    : "bg-muted ml-8"
                                )}
                              >
                                <div className="flex items-center gap-2 mb-1">
                                  <Badge variant={reply.is_admin ? "default" : "secondary"} className="text-xs">
                                    {reply.is_admin 
                                      ? (language === 'ar' ? 'الدعم' : 'Support')
                                      : (language === 'ar' ? 'أنت' : 'You')
                                    }
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {format(new Date(reply.created_at), "yyyy/MM/dd HH:mm")}
                                  </span>
                                </div>
                                <p className="text-sm">{reply.message}</p>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Reply Form */}
                        {ticket.status !== "closed" && (
                          <form onSubmit={handleReply} className="flex gap-2">
                            <input
                              type="text"
                              value={replyMessage}
                              onChange={(e) => setReplyMessage(e.target.value)}
                              placeholder={language === 'ar' ? 'اكتب ردك...' : 'Write your reply...'}
                              className="flex-1 px-4 py-2 bg-muted rounded-lg border focus:border-primary focus:outline-none"
                              onClick={(e) => e.stopPropagation()}
                            />
                            <button
                              type="submit"
                              disabled={!replyMessage.trim() || replyMutation.isPending}
                              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <Send className="w-4 h-4" />
                            </button>
                          </form>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* New Ticket Modal */}
      {showNewTicket && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                {t.support.newTicket}
              </CardTitle>
              <CardDescription>
                {language === 'ar' 
                  ? 'أخبرنا بمشكلتك وسنرد عليك في أقرب وقت'
                  : 'Tell us your issue and we will respond as soon as possible'
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateTicket} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.subject}</label>
                  <input
                    type="text"
                    value={newSubject}
                    onChange={(e) => setNewSubject(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border focus:border-primary focus:outline-none"
                    placeholder={language === 'ar' ? 'موضوع التذكرة' : 'Ticket subject'}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.message}</label>
                  <textarea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-3 bg-muted rounded-lg border focus:border-primary focus:outline-none resize-none"
                    placeholder={language === 'ar' ? 'اكتب رسالتك هنا...' : 'Write your message here...'}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.priority}</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border focus:border-primary focus:outline-none"
                  >
                    <option value="low">{t.support.low}</option>
                    <option value="medium">{t.support.medium}</option>
                    <option value="high">{t.support.high}</option>
                  </select>
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowNewTicket(false)}
                    className="flex-1 px-4 py-3 border rounded-lg hover:bg-muted transition-colors"
                  >
                    {language === 'ar' ? 'إلغاء' : 'Cancel'}
                  </button>
                  <button
                    type="submit"
                    disabled={createTicketMutation.isPending || !newSubject.trim() || !newMessage.trim()}
                    className="flex-1 px-4 py-3 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 transition-colors"
                  >
                    {createTicketMutation.isPending 
                      ? (language === 'ar' ? 'جاري الإرسال...' : 'Sending...') 
                      : t.support.submit
                    }
                  </button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
