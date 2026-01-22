import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useLanguage } from "@/lib/i18n";
import { useEffect, useState, useRef } from "react";
import {
  Bot,
  Shield,
  Wallet,
  Users,
  Brain,
  Zap,
  ArrowLeft,
  ArrowRight,
  TrendingUp,
  Clock,
  DollarSign,
  Activity,
  Moon,
  Sun,
  Globe,
  CheckCircle2,
  Star,
  ChevronLeft,
  ChevronRight,
  Play,
  Lock,
  BarChart3,
  PieChart,
  LineChart,
  Sparkles,
  Award,
  Target,
  Rocket,
  Mail,
  Phone,
  MapPin,
  Facebook,
  Twitter,
  Instagram,
  Linkedin,
  Youtube,
  Send,
  ArrowUpRight,
  Quote,
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

// مكون العداد المتحرك
const AnimatedCounter = ({ end, duration = 2000, prefix = "", suffix = "" }: { end: number; duration?: number; prefix?: string; suffix?: string }) => {
  const [count, setCount] = useState(0);
  const countRef = useRef<HTMLSpanElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (countRef.current) {
      observer.observe(countRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible) return;

    let startTime: number;
    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      setCount(Math.floor(progress * end));
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }, [isVisible, end, duration]);

  return <span ref={countRef}>{prefix}{count.toLocaleString()}{suffix}</span>;
};

// مكون شعارات العملات المتحركة
const CryptoLogosMarquee = () => {
  const cryptos = [
    { name: "Bitcoin", symbol: "BTC", logo: "https://cryptologos.cc/logos/bitcoin-btc-logo.png" },
    { name: "Ethereum", symbol: "ETH", logo: "https://cryptologos.cc/logos/ethereum-eth-logo.png" },
    { name: "Solana", symbol: "SOL", logo: "https://cryptologos.cc/logos/solana-sol-logo.png" },
    { name: "Cardano", symbol: "ADA", logo: "https://cryptologos.cc/logos/cardano-ada-logo.png" },
    { name: "Polygon", symbol: "MATIC", logo: "https://cryptologos.cc/logos/polygon-matic-logo.png" },
    { name: "Avalanche", symbol: "AVAX", logo: "https://cryptologos.cc/logos/avalanche-avax-logo.png" },
    { name: "Chainlink", symbol: "LINK", logo: "https://cryptologos.cc/logos/chainlink-link-logo.png" },
    { name: "Polkadot", symbol: "DOT", logo: "https://cryptologos.cc/logos/polkadot-new-dot-logo.png" },
    { name: "Uniswap", symbol: "UNI", logo: "https://cryptologos.cc/logos/uniswap-uni-logo.png" },
    { name: "Litecoin", symbol: "LTC", logo: "https://cryptologos.cc/logos/litecoin-ltc-logo.png" },
  ];

  return (
    <div className="relative overflow-hidden py-8">
      <div className="flex animate-marquee gap-12">
        {[...cryptos, ...cryptos].map((crypto, index) => (
          <div
            key={index}
            className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-white/5 border border-white/10 hover:border-violet-500/50 hover:bg-violet-500/10 transition-all duration-300 min-w-fit"
          >
            <img src={crypto.logo} alt={crypto.name} className="w-8 h-8" />
            <div>
              <p className="font-semibold text-white">{crypto.symbol}</p>
              <p className="text-xs text-white/50">{crypto.name}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// مكون الرسم البياني المتحرك
const AnimatedChart = () => {
  const [data, setData] = useState<number[]>([]);
  
  useEffect(() => {
    // بيانات وهمية للرسم البياني
    const initialData = [20, 25, 30, 28, 35, 40, 38, 45, 50, 48, 55, 60, 58, 65, 70, 68, 75, 80, 78, 85];
    setData(initialData);
  }, []);

  const maxValue = Math.max(...data);
  const minValue = Math.min(...data);

  return (
    <div className="relative h-64 w-full">
      {/* Grid lines */}
      <div className="absolute inset-0 flex flex-col justify-between">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="border-t border-white/5 w-full" />
        ))}
      </div>
      
      {/* Chart */}
      <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(139, 92, 246, 0.5)" />
            <stop offset="100%" stopColor="rgba(139, 92, 246, 0)" />
          </linearGradient>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#8B5CF6" />
            <stop offset="50%" stopColor="#A855F7" />
            <stop offset="100%" stopColor="#D946EF" />
          </linearGradient>
        </defs>
        
        {/* Area fill */}
        <path
          d={`
            M 0 ${256 - ((data[0] - minValue) / (maxValue - minValue)) * 200}
            ${data.map((value, index) => {
              const x = (index / (data.length - 1)) * 100;
              const y = 256 - ((value - minValue) / (maxValue - minValue)) * 200;
              return `L ${x}% ${y}`;
            }).join(' ')}
            L 100% 256
            L 0 256
            Z
          `}
          fill="url(#chartGradient)"
          className="animate-fade-in"
        />
        
        {/* Line */}
        <path
          d={`
            M 0 ${256 - ((data[0] - minValue) / (maxValue - minValue)) * 200}
            ${data.map((value, index) => {
              const x = (index / (data.length - 1)) * 100;
              const y = 256 - ((value - minValue) / (maxValue - minValue)) * 200;
              return `L ${x}% ${y}`;
            }).join(' ')}
          `}
          fill="none"
          stroke="url(#lineGradient)"
          strokeWidth="3"
          className="animate-draw-line"
          style={{ strokeDasharray: 1000, strokeDashoffset: 1000, animation: 'drawLine 2s ease-out forwards' }}
        />
        
        {/* Dots */}
        {data.map((value, index) => {
          const x = (index / (data.length - 1)) * 100;
          const y = 256 - ((value - minValue) / (maxValue - minValue)) * 200;
          return (
            <circle
              key={index}
              cx={`${x}%`}
              cy={y}
              r="4"
              fill="#8B5CF6"
              className="animate-pulse"
              style={{ animationDelay: `${index * 0.1}s` }}
            />
          );
        })}
      </svg>
      
      {/* Current value indicator */}
      <div className="absolute top-4 right-4 bg-violet-500/20 backdrop-blur-sm border border-violet-500/30 rounded-xl px-4 py-2">
        <p className="text-xs text-violet-300">القيمة الحالية</p>
        <p className="text-2xl font-bold text-white">+18.5%</p>
      </div>
    </div>
  );
};

const Landing = () => {
  const { language, setLanguage, t } = useLanguage();
  const { theme, setTheme } = useTheme();
  const [currentTestimonial, setCurrentTestimonial] = useState(0);
  
  const ArrowIcon = language === 'ar' ? ArrowLeft : ArrowRight;

  // الإحصائيات
  const stats = [
    { value: 125000, label: t.landing.totalAssets, icon: DollarSign, prefix: "$", suffix: "+" },
    { value: 18, label: t.landing.avgReturn, icon: TrendingUp, prefix: "+", suffix: "%" },
    { value: 50, label: t.landing.activeParticipants, icon: Users, prefix: "", suffix: "+" },
    { value: 99, label: language === 'ar' ? "نسبة الأمان" : "Security Rate", icon: Shield, prefix: "", suffix: "%" },
  ];

  // الإحصائيات الحية
  const liveStats = [
    { label: language === 'ar' ? "إجمالي الصفقات" : "Total Trades", value: 15847, icon: Activity },
    { label: language === 'ar' ? "المستخدمين النشطين" : "Active Users", value: 234, icon: Users },
    { label: language === 'ar' ? "نسبة النجاح" : "Success Rate", value: 87, suffix: "%", icon: Target },
    { label: language === 'ar' ? "وقت التشغيل" : "Uptime", value: 99.9, suffix: "%", icon: Clock },
  ];

  // المميزات
  const features = [
    {
      icon: <Bot className="w-7 h-7" />,
      title: t.landing.feature1Title,
      description: t.landing.feature1Desc,
      gradient: "from-violet-500 to-purple-600",
      highlights: [
        language === 'ar' ? "تحليل RSI و MACD" : "RSI & MACD Analysis",
        language === 'ar' ? "تحليل المشاعر" : "Sentiment Analysis",
        language === 'ar' ? "تعلم آلي متقدم" : "Advanced ML",
      ],
    },
    {
      icon: <Shield className="w-7 h-7" />,
      title: t.landing.feature2Title,
      description: t.landing.feature2Desc,
      gradient: "from-emerald-500 to-teal-600",
      highlights: [
        language === 'ar' ? "تشفير AES-256" : "AES-256 Encryption",
        language === 'ar' ? "مصادقة ثنائية" : "2FA Authentication",
        language === 'ar' ? "مراقبة 24/7" : "24/7 Monitoring",
      ],
    },
    {
      icon: <Wallet className="w-7 h-7" />,
      title: t.landing.feature3Title,
      description: t.landing.feature3Desc,
      gradient: "from-amber-500 to-orange-600",
      highlights: [
        language === 'ar' ? "سحب خلال 24-48 ساعة" : "24-48h Withdrawal",
        language === 'ar' ? "بدون رسوم مخفية" : "No Hidden Fees",
        language === 'ar' ? "دعم USDC" : "USDC Support",
      ],
    },
    {
      icon: <Users className="w-7 h-7" />,
      title: t.landing.feature4Title,
      description: t.landing.feature4Desc,
      gradient: "from-pink-500 to-rose-600",
      highlights: [
        language === 'ar' ? "نظام NAV شفاف" : "Transparent NAV",
        language === 'ar' ? "توزيع عادل" : "Fair Distribution",
        language === 'ar' ? "تقارير فورية" : "Real-time Reports",
      ],
    },
    {
      icon: <Brain className="w-7 h-7" />,
      title: t.landing.feature5Title,
      description: t.landing.feature5Desc,
      gradient: "from-cyan-500 to-blue-600",
      highlights: [
        language === 'ar' ? "لوحة تحكم شاملة" : "Comprehensive Dashboard",
        language === 'ar' ? "تحليلات متقدمة" : "Advanced Analytics",
        language === 'ar' ? "سجل كامل" : "Full History",
      ],
    },
    {
      icon: <Zap className="w-7 h-7" />,
      title: t.landing.feature6Title,
      description: t.landing.feature6Desc,
      gradient: "from-indigo-500 to-violet-600",
      highlights: [
        language === 'ar' ? "BSC & Solana" : "BSC & Solana",
        language === 'ar' ? "حد أدنى $100" : "Min $100",
        language === 'ar' ? "تأكيد سريع" : "Fast Confirmation",
      ],
    },
  ];

  // الخطوات
  const steps = [
    { 
      number: "1", 
      title: t.landing.step1Title, 
      description: t.landing.step1Desc,
      icon: <Users className="w-6 h-6" />,
    },
    { 
      number: "2", 
      title: t.landing.step2Title, 
      description: t.landing.step2Desc,
      icon: <Wallet className="w-6 h-6" />,
    },
    { 
      number: "3", 
      title: t.landing.step3Title, 
      description: t.landing.step3Desc,
      icon: <Bot className="w-6 h-6" />,
    },
    { 
      number: "4", 
      title: t.landing.step4Title, 
      description: t.landing.step4Desc,
      icon: <TrendingUp className="w-6 h-6" />,
    },
  ];

  // الشهادات
  const testimonials = [
    {
      name: language === 'ar' ? "أحمد محمد" : "Ahmed Mohammed",
      role: language === 'ar' ? "مستثمر" : "Investor",
      avatar: "https://randomuser.me/api/portraits/men/1.jpg",
      content: language === 'ar' 
        ? "منصة رائعة! حققت أرباحاً ممتازة خلال الأشهر الماضية. الوكيل الذكي يعمل بشكل مذهل والدعم الفني سريع جداً."
        : "Amazing platform! I've made excellent profits over the past months. The AI agent works amazingly and technical support is very fast.",
      rating: 5,
    },
    {
      name: language === 'ar' ? "سارة أحمد" : "Sara Ahmed",
      role: language === 'ar' ? "متداولة" : "Trader",
      avatar: "https://randomuser.me/api/portraits/women/2.jpg",
      content: language === 'ar'
        ? "أفضل منصة تداول جماعي استخدمتها. الشفافية في نظام NAV تجعلني أثق تماماً في المنصة. أنصح بها بشدة!"
        : "Best collective trading platform I've used. The transparency in the NAV system makes me fully trust the platform. Highly recommend!",
      rating: 5,
    },
    {
      name: language === 'ar' ? "خالد العمري" : "Khaled Al-Omari",
      role: language === 'ar' ? "رجل أعمال" : "Businessman",
      avatar: "https://randomuser.me/api/portraits/men/3.jpg",
      content: language === 'ar'
        ? "استثمرت مبلغاً كبيراً وأنا راضٍ تماماً عن النتائج. الأمان ممتاز والسحب سريع. شكراً لفريق ASINAX!"
        : "I invested a large amount and I'm completely satisfied with the results. Security is excellent and withdrawal is fast. Thanks to the ASINAX team!",
      rating: 5,
    },
    {
      name: language === 'ar' ? "نورة السعيد" : "Noura Al-Saeed",
      role: language === 'ar' ? "مستثمرة" : "Investor",
      avatar: "https://randomuser.me/api/portraits/women/4.jpg",
      content: language === 'ar'
        ? "كنت متخوفة في البداية لكن بعد تجربة المنصة لعدة أشهر، أصبحت من أكبر المعجبين. العوائد ممتازة والتجربة سلسة."
        : "I was hesitant at first but after trying the platform for several months, I became a big fan. Returns are excellent and the experience is smooth.",
      rating: 5,
    },
  ];

  // التنقل بين الشهادات
  const nextTestimonial = () => {
    setCurrentTestimonial((prev) => (prev + 1) % testimonials.length);
  };

  const prevTestimonial = () => {
    setCurrentTestimonial((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  // Auto-rotate testimonials
  useEffect(() => {
    const interval = setInterval(nextTestimonial, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#08080c] text-white overflow-hidden relative">
      {/* CSS Animations */}
      <style>{`
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
          50% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.8); }
        }
        .animate-glow {
          animation: glow 2s ease-in-out infinite;
        }
        @keyframes drawLine {
          to { stroke-dashoffset: 0; }
        }
        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient-shift {
          animation: gradient-shift 3s ease infinite;
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }
        .animate-pulse-ring {
          animation: pulse-ring 2s ease-out infinite;
        }
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-up {
          animation: slide-up 0.6s ease-out forwards;
        }
        @keyframes rotate-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-rotate-slow {
          animation: rotate-slow 20s linear infinite;
        }
      `}</style>

      {/* Animated Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        {/* Gradient Orbs */}
        <div className="absolute w-[600px] h-[600px] rounded-full bg-violet-500/20 blur-[120px] top-[5%] left-[5%] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute w-[500px] h-[500px] rounded-full bg-pink-500/15 blur-[100px] bottom-[10%] right-[5%] animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
        <div className="absolute w-[400px] h-[400px] rounded-full bg-purple-600/15 blur-[80px] top-[40%] left-[40%] animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />
        <div className="absolute w-[300px] h-[300px] rounded-full bg-cyan-500/10 blur-[60px] top-[60%] right-[20%] animate-pulse" style={{ animationDuration: '9s', animationDelay: '1s' }} />
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:50px_50px]" />
        
        {/* Floating Particles */}
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-violet-400/30 rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${5 + Math.random() * 5}s`,
            }}
          />
        ))}
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#08080c]/80 backdrop-blur-xl border-b border-violet-500/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="absolute inset-0 bg-violet-500/50 rounded-xl blur-lg group-hover:blur-xl transition-all duration-300" />
              <div className="relative w-10 h-10 rounded-xl overflow-hidden shadow-[0_0_20px_rgba(139,92,246,0.4)] group-hover:shadow-[0_0_30px_rgba(139,92,246,0.6)] transition-all duration-300">
                <img 
                  src="/logo-header.png?v=1768667205" 
                  alt="ASINAX Logo" 
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
            <span className="font-bold text-xl bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">ASINAX</span>
          </Link>
          
          <div className="hidden md:flex items-center gap-6">
            <a href="#features" className="text-white/70 hover:text-white transition-colors">{language === 'ar' ? 'المميزات' : 'Features'}</a>
            <a href="#how-it-works" className="text-white/70 hover:text-white transition-colors">{language === 'ar' ? 'كيف يعمل' : 'How it Works'}</a>
            <a href="#testimonials" className="text-white/70 hover:text-white transition-colors">{language === 'ar' ? 'آراء العملاء' : 'Testimonials'}</a>
            <Link to="/about" className="text-white/70 hover:text-white transition-colors">{language === 'ar' ? 'من نحن' : 'About Us'}</Link>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
              className="p-2.5 rounded-xl bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 hover:border-violet-500/40 transition-all duration-300"
            >
              <Globe className="w-5 h-5 text-violet-400" />
            </button>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2.5 rounded-xl bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 hover:border-violet-500/40 transition-all duration-300"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5 text-violet-400" /> : <Moon className="w-5 h-5 text-violet-400" />}
            </button>
            <Link to="/login">
              <Button variant="ghost" className="text-white/80 hover:text-white hover:bg-violet-500/10 transition-all duration-300">
                {t.landing.login}
              </Button>
            </Link>
            <Link to="/register">
              <Button className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-[0_4px_20px_rgba(139,92,246,0.4)] hover:shadow-[0_6px_30px_rgba(139,92,246,0.5)] transition-all duration-300">
                {t.landing.register}
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className="text-center lg:text-start">
              {/* Status Badge */}
              <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-violet-500/10 border border-violet-500/30 mb-8 animate-slide-up backdrop-blur-sm">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                <span className="text-violet-200 font-medium">{t.landing.tradingActive}</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
                <span className="text-white">{t.landing.heroTitle1}</span>
                <br />
                <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient-shift bg-[length:200%_auto]">
                  ASINAX CRYPTO AI
                </span>
              </h1>

              {/* Description */}
              <p className="text-lg md:text-xl text-white/60 mb-8 animate-slide-up leading-relaxed" style={{ animationDelay: '0.2s' }}>
                {t.landing.heroDescription}
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 mb-8 animate-slide-up" style={{ animationDelay: '0.3s' }}>
                <Link to="/register">
                  <Button size="lg" className="text-lg px-8 py-6 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 group animate-glow">
                    {t.landing.startTrading}
                    <ArrowIcon className="w-5 h-5 mx-2 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </Link>
                <a href="#features">
                  <Button variant="outline" size="lg" className="text-lg px-8 py-6 border-violet-500/40 text-white hover:bg-violet-500/10 hover:border-violet-500/60 transition-all duration-300">
                    <Play className="w-5 h-5 mx-2" />
                    {t.landing.learnMore}
                  </Button>
                </a>
              </div>

              {/* Trust Badges */}
              <div className="flex items-center justify-center lg:justify-start gap-6 animate-slide-up" style={{ animationDelay: '0.4s' }}>
                <div className="flex items-center gap-2 text-white/50">
                  <Shield className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm">{language === 'ar' ? 'آمن 100%' : '100% Secure'}</span>
                </div>
                <div className="flex items-center gap-2 text-white/50">
                  <Lock className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm">{language === 'ar' ? 'تشفير متقدم' : 'Advanced Encryption'}</span>
                </div>
                <div className="flex items-center gap-2 text-white/50">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm">{language === 'ar' ? 'موثوق' : 'Verified'}</span>
                </div>
              </div>
            </div>

            {/* Right Content - Chart & Stats */}
            <div className="relative animate-slide-up" style={{ animationDelay: '0.5s' }}>
              {/* Decorative Ring */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-[400px] h-[400px] rounded-full border border-violet-500/20 animate-rotate-slow" />
                <div className="absolute w-[350px] h-[350px] rounded-full border border-purple-500/15 animate-rotate-slow" style={{ animationDirection: 'reverse', animationDuration: '25s' }} />
              </div>
              
              {/* Chart Card */}
              <div className="relative bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 rounded-3xl p-6 shadow-[0_20px_60px_rgba(139,92,246,0.2)]">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className="text-white/50 text-sm">{language === 'ar' ? 'أداء المحفظة' : 'Portfolio Performance'}</p>
                    <p className="text-2xl font-bold text-white">$125,847.32</p>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400">
                    <TrendingUp className="w-4 h-4" />
                    <span className="font-semibold">+18.5%</span>
                  </div>
                </div>
                
                <AnimatedChart />
                
                {/* Mini Stats */}
                <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-white/10">
                  <div className="text-center">
                    <p className="text-white/50 text-xs mb-1">{language === 'ar' ? 'اليوم' : 'Today'}</p>
                    <p className="text-emerald-400 font-semibold">+2.4%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-white/50 text-xs mb-1">{language === 'ar' ? 'الأسبوع' : 'Week'}</p>
                    <p className="text-emerald-400 font-semibold">+8.7%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-white/50 text-xs mb-1">{language === 'ar' ? 'الشهر' : 'Month'}</p>
                    <p className="text-emerald-400 font-semibold">+18.5%</p>
                  </div>
                </div>
              </div>
              
              {/* Floating Cards */}
              <div className="absolute -top-4 -right-4 bg-[rgba(15,15,25,0.9)] backdrop-blur-xl border border-violet-500/30 rounded-2xl p-4 shadow-xl animate-float">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-xs text-white/50">{language === 'ar' ? 'صفقة جديدة' : 'New Trade'}</p>
                    <p className="font-semibold text-white">BTC/USDC</p>
                  </div>
                </div>
              </div>
              
              <div className="absolute -bottom-4 -left-4 bg-[rgba(15,15,25,0.9)] backdrop-blur-xl border border-violet-500/30 rounded-2xl p-4 shadow-xl animate-float" style={{ animationDelay: '1s' }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-violet-500/20 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-violet-400" />
                  </div>
                  <div>
                    <p className="text-xs text-white/50">{language === 'ar' ? 'الوكيل الذكي' : 'AI Agent'}</p>
                    <p className="font-semibold text-emerald-400">{language === 'ar' ? 'نشط' : 'Active'}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-20 animate-slide-up" style={{ animationDelay: '0.6s' }}>
            {stats.map((stat, index) => (
              <div
                key={index}
                className="group p-6 rounded-2xl bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/40 hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)] transition-all duration-400 hover:-translate-y-2"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-purple-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <stat.icon className="w-6 h-6 text-violet-400" />
                </div>
                <div className="text-3xl font-bold text-white mb-1 group-hover:text-violet-300 transition-colors">
                  <AnimatedCounter end={stat.value} prefix={stat.prefix} suffix={stat.suffix} />
                </div>
                <div className="text-sm text-white/50">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Crypto Logos Marquee */}
      <section className="py-12 relative z-10 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-4">
          <p className="text-center text-white/40 text-sm mb-6">{language === 'ar' ? 'العملات المدعومة للتداول' : 'Supported Trading Currencies'}</p>
          <CryptoLogosMarquee />
        </div>
      </section>

      {/* Live Stats Section */}
      <section className="py-16 relative z-10">
        <div className="max-w-7xl mx-auto px-4">
          <div className="bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-pink-500/10 backdrop-blur-xl border border-violet-500/20 rounded-3xl p-8">
            <div className="flex items-center justify-center gap-2 mb-8">
              <div className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </div>
              <span className="text-white font-medium">{language === 'ar' ? 'إحصائيات حية' : 'Live Statistics'}</span>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {liveStats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="w-14 h-14 mx-auto rounded-2xl bg-violet-500/20 flex items-center justify-center mb-4">
                    <stat.icon className="w-7 h-7 text-violet-400" />
                  </div>
                  <p className="text-3xl font-bold text-white mb-1">
                    <AnimatedCounter end={stat.value} suffix={stat.suffix || ""} />
                  </p>
                  <p className="text-white/50 text-sm">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>


      {/* Features Section */}
      <section id="features" className="py-24 px-4 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/30 mb-6">
              <Sparkles className="w-4 h-4 text-violet-400" />
              <span className="text-violet-300 text-sm font-medium">{language === 'ar' ? 'مميزات حصرية' : 'Exclusive Features'}</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {t.landing.whyChooseUs}
            </h2>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              {t.landing.whyChooseUsDesc}
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="group p-6 rounded-2xl bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/40 hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)] transition-all duration-400 hover:-translate-y-2"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center text-white mb-5 group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(139,92,246,0.4)] transition-all duration-300`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-violet-200 transition-colors">{feature.title}</h3>
                <p className="text-white/50 leading-relaxed mb-4">{feature.description}</p>
                
                {/* Feature Highlights */}
                <div className="space-y-2">
                  {feature.highlights.map((highlight, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-white/60">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>{highlight}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 px-4 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/30 mb-6">
              <Rocket className="w-4 h-4 text-violet-400" />
              <span className="text-violet-300 text-sm font-medium">{language === 'ar' ? 'ابدأ الآن' : 'Get Started'}</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {t.landing.howItWorks}
            </h2>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              {t.landing.howItWorksDesc}
            </p>
          </div>
          
          <div className="relative">
            {/* Connection Line */}
            <div className="hidden md:block absolute top-24 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-violet-500/50 to-transparent" />
            
            <div className="grid md:grid-cols-4 gap-8">
              {steps.map((step, index) => (
                <div key={index} className="relative text-center group">
                  {/* Step Number */}
                  <div className="relative mx-auto mb-6">
                    <div className="absolute inset-0 bg-violet-500/30 rounded-full blur-xl group-hover:blur-2xl transition-all" />
                    <div className="relative w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white shadow-[0_0_30px_rgba(139,92,246,0.4)] group-hover:shadow-[0_0_50px_rgba(139,92,246,0.6)] transition-all duration-300 group-hover:scale-110">
                      <span className="text-2xl font-bold">{step.number}</span>
                    </div>
                    {/* Icon Badge */}
                    <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-full bg-[#08080c] border-2 border-violet-500 flex items-center justify-center">
                      {step.icon}
                    </div>
                  </div>
                  
                  <h3 className="text-lg font-semibold mb-2 text-white group-hover:text-violet-200 transition-colors">{step.title}</h3>
                  <p className="text-white/50 text-sm">{step.description}</p>
                </div>
              ))}
            </div>
          </div>
          
          {/* CTA after steps */}
          <div className="text-center mt-16">
            <Link to="/register">
              <Button size="lg" className="text-lg px-10 py-6 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 group">
                {language === 'ar' ? 'ابدأ رحلتك الآن' : 'Start Your Journey Now'}
                <ArrowIcon className="w-5 h-5 mx-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-24 px-4 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/30 mb-6">
              <Star className="w-4 h-4 text-violet-400" />
              <span className="text-violet-300 text-sm font-medium">{language === 'ar' ? 'آراء عملائنا' : 'Customer Reviews'}</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {language === 'ar' ? 'ماذا يقول عملاؤنا' : 'What Our Clients Say'}
            </h2>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              {language === 'ar' ? 'آلاف المستثمرين يثقون بمنصتنا لتحقيق أهدافهم المالية' : 'Thousands of investors trust our platform to achieve their financial goals'}
            </p>
          </div>
          
          {/* Testimonials Carousel */}
          <div className="relative max-w-4xl mx-auto">
            {/* Navigation Buttons */}
            <button
              onClick={prevTestimonial}
              className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 md:-translate-x-12 w-12 h-12 rounded-full bg-violet-500/20 border border-violet-500/30 flex items-center justify-center text-white hover:bg-violet-500/30 transition-all z-10"
            >
              <ChevronLeft className="w-6 h-6" />
            </button>
            <button
              onClick={nextTestimonial}
              className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 md:translate-x-12 w-12 h-12 rounded-full bg-violet-500/20 border border-violet-500/30 flex items-center justify-center text-white hover:bg-violet-500/30 transition-all z-10"
            >
              <ChevronRight className="w-6 h-6" />
            </button>
            
            {/* Testimonial Card */}
            <div className="bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 rounded-3xl p-8 md:p-12 relative overflow-hidden">
              {/* Quote Icon */}
              <div className="absolute top-6 right-6 opacity-10">
                <Quote className="w-24 h-24 text-violet-400" />
              </div>
              
              <div className="relative">
                {/* Stars */}
                <div className="flex items-center gap-1 mb-6">
                  {[...Array(testimonials[currentTestimonial].rating)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                  ))}
                </div>
                
                {/* Content */}
                <p className="text-xl md:text-2xl text-white/90 leading-relaxed mb-8">
                  "{testimonials[currentTestimonial].content}"
                </p>
                
                {/* Author */}
                <div className="flex items-center gap-4">
                  <img
                    src={testimonials[currentTestimonial].avatar}
                    alt={testimonials[currentTestimonial].name}
                    className="w-14 h-14 rounded-full border-2 border-violet-500/50"
                  />
                  <div>
                    <p className="font-semibold text-white">{testimonials[currentTestimonial].name}</p>
                    <p className="text-white/50 text-sm">{testimonials[currentTestimonial].role}</p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Dots Indicator */}
            <div className="flex items-center justify-center gap-2 mt-6">
              {testimonials.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentTestimonial(index)}
                  className={`w-2.5 h-2.5 rounded-full transition-all ${
                    index === currentTestimonial
                      ? 'bg-violet-500 w-8'
                      : 'bg-white/20 hover:bg-white/40'
                  }`}
                />
              ))}
            </div>
          </div>
          
          {/* Trust Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-16">
            <div className="text-center p-6 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-3xl font-bold text-white mb-1">4.9/5</p>
              <p className="text-white/50 text-sm">{language === 'ar' ? 'تقييم المستخدمين' : 'User Rating'}</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-3xl font-bold text-white mb-1">500+</p>
              <p className="text-white/50 text-sm">{language === 'ar' ? 'مستثمر نشط' : 'Active Investors'}</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-3xl font-bold text-white mb-1">$2M+</p>
              <p className="text-white/50 text-sm">{language === 'ar' ? 'أرباح موزعة' : 'Profits Distributed'}</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-3xl font-bold text-white mb-1">24/7</p>
              <p className="text-white/50 text-sm">{language === 'ar' ? 'دعم متواصل' : 'Continuous Support'}</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 relative z-10">
        <div className="max-w-5xl mx-auto">
          <div className="relative bg-gradient-to-br from-violet-600/20 via-purple-600/20 to-pink-600/20 backdrop-blur-xl border border-violet-500/30 rounded-3xl p-12 md:p-16 overflow-hidden">
            {/* Background Decoration */}
            <div className="absolute inset-0 overflow-hidden">
              <div className="absolute -top-20 -right-20 w-60 h-60 bg-violet-500/30 rounded-full blur-3xl" />
              <div className="absolute -bottom-20 -left-20 w-60 h-60 bg-purple-500/30 rounded-full blur-3xl" />
            </div>
            
            <div className="relative text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 mb-6">
                <Rocket className="w-4 h-4 text-white" />
                <span className="text-white text-sm font-medium">{language === 'ar' ? 'ابدأ مجاناً' : 'Start Free'}</span>
              </div>
              
              <h2 className="text-3xl md:text-5xl font-bold mb-6 text-white">
                {t.landing.ctaTitle}
              </h2>
              <p className="text-xl text-white/70 mb-10 max-w-2xl mx-auto">
                {t.landing.ctaDescription}
              </p>
              
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link to="/register">
                  <Button size="lg" className="text-lg px-10 py-6 bg-white text-violet-600 hover:bg-white/90 shadow-[0_8px_30px_rgba(255,255,255,0.3)] transition-all duration-300 group">
                    {t.landing.startTrading}
                    <ArrowIcon className="w-5 h-5 mx-2 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </Link>
                <Link to="/contact">
                  <Button variant="outline" size="lg" className="text-lg px-10 py-6 border-white/40 text-white hover:bg-white/10 transition-all duration-300">
                    {language === 'ar' ? 'تواصل معنا' : 'Contact Us'}
                  </Button>
                </Link>
              </div>
              
              {/* Features List */}
              <div className="flex flex-wrap items-center justify-center gap-6 mt-10">
                <div className="flex items-center gap-2 text-white/70">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span>{language === 'ar' ? 'بدون رسوم تسجيل' : 'No Registration Fees'}</span>
                </div>
                <div className="flex items-center gap-2 text-white/70">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span>{language === 'ar' ? 'سحب في أي وقت' : 'Withdraw Anytime'}</span>
                </div>
                <div className="flex items-center gap-2 text-white/70">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span>{language === 'ar' ? 'دعم على مدار الساعة' : '24/7 Support'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-16">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
            {/* Brand */}
            <div>
              <Link to="/" className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl overflow-hidden">
                  <img 
                    src="/logo-header.png?v=1768667205" 
                    alt="ASINAX Logo" 
                    className="w-full h-full object-contain"
                  />
                </div>
                <span className="font-bold text-xl text-white">ASINAX</span>
              </Link>
              <p className="text-white/50 mb-6 leading-relaxed">
                {language === 'ar' 
                  ? 'منصة التداول الجماعي الذكية الأولى من نوعها. نستخدم أحدث تقنيات الذكاء الاصطناعي لتحقيق أفضل النتائج.'
                  : 'The first smart collective trading platform of its kind. We use the latest AI technologies to achieve the best results.'}
              </p>
              {/* Social Links */}
              <div className="flex items-center gap-3">
                <a href="#" className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/50 hover:text-white hover:bg-violet-500/20 hover:border-violet-500/50 transition-all">
                  <Twitter className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/50 hover:text-white hover:bg-violet-500/20 hover:border-violet-500/50 transition-all">
                  <Facebook className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/50 hover:text-white hover:bg-violet-500/20 hover:border-violet-500/50 transition-all">
                  <Instagram className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/50 hover:text-white hover:bg-violet-500/20 hover:border-violet-500/50 transition-all">
                  <Linkedin className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/50 hover:text-white hover:bg-violet-500/20 hover:border-violet-500/50 transition-all">
                  <Youtube className="w-5 h-5" />
                </a>
              </div>
            </div>
            
            {/* Quick Links */}
            <div>
              <h4 className="font-semibold text-white mb-6">{language === 'ar' ? 'روابط سريعة' : 'Quick Links'}</h4>
              <ul className="space-y-3">
                <li><Link to="/about" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'من نحن' : 'About Us'}</Link></li>
                <li><a href="#features" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'المميزات' : 'Features'}</a></li>
                <li><a href="#how-it-works" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'كيف يعمل' : 'How it Works'}</a></li>
                <li><Link to="/transparency" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'الشفافية' : 'Transparency'}</Link></li>
                <li><Link to="/blog" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'المدونة' : 'Blog'}</Link></li>
              </ul>
            </div>
            
            {/* Support */}
            <div>
              <h4 className="font-semibold text-white mb-6">{language === 'ar' ? 'الدعم' : 'Support'}</h4>
              <ul className="space-y-3">
                <li><Link to="/support" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'مركز المساعدة' : 'Help Center'}</Link></li>
                <li><Link to="/contact" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'تواصل معنا' : 'Contact Us'}</Link></li>
                <li><Link to="/privacy" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'سياسة الخصوصية' : 'Privacy Policy'}</Link></li>
                <li><Link to="/terms" className="text-white/50 hover:text-white transition-colors">{language === 'ar' ? 'الشروط والأحكام' : 'Terms & Conditions'}</Link></li>
              </ul>
            </div>
            
            {/* Contact */}
            <div>
              <h4 className="font-semibold text-white mb-6">{language === 'ar' ? 'تواصل معنا' : 'Contact Us'}</h4>
              <ul className="space-y-4">
                <li className="flex items-center gap-3 text-white/50">
                  <Mail className="w-5 h-5 text-violet-400" />
                  <span>support@asinax.cloud</span>
                </li>
                <li className="flex items-center gap-3 text-white/50">
                  <Clock className="w-5 h-5 text-violet-400" />
                  <span>{language === 'ar' ? 'دعم 24/7' : '24/7 Support'}</span>
                </li>
              </ul>
              
              {/* Newsletter */}
              <div className="mt-6">
                <p className="text-white/70 text-sm mb-3">{language === 'ar' ? 'اشترك في نشرتنا البريدية' : 'Subscribe to our newsletter'}</p>
                <div className="flex gap-2">
                  <input
                    type="email"
                    placeholder={language === 'ar' ? 'بريدك الإلكتروني' : 'Your email'}
                    className="flex-1 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-white/30 focus:outline-none focus:border-violet-500/50"
                  />
                  <button className="px-4 py-2 rounded-xl bg-violet-500 hover:bg-violet-600 text-white transition-colors">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Bottom Bar */}
          <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-white/40 text-sm">
              © 2024 ASINAX. {language === 'ar' ? 'جميع الحقوق محفوظة.' : 'All rights reserved.'}
            </p>
            <div className="flex items-center gap-6">
              <Link to="/privacy" className="text-white/40 text-sm hover:text-white transition-colors">
                {language === 'ar' ? 'الخصوصية' : 'Privacy'}
              </Link>
              <Link to="/terms" className="text-white/40 text-sm hover:text-white transition-colors">
                {language === 'ar' ? 'الشروط' : 'Terms'}
              </Link>
              <Link to="/contact" className="text-white/40 text-sm hover:text-white transition-colors">
                {language === 'ar' ? 'تواصل' : 'Contact'}
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
