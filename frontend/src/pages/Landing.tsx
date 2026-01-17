import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useLanguage } from "@/lib/i18n";
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
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

const Landing = () => {
  const { language, setLanguage, t } = useLanguage();
  const { theme, setTheme } = useTheme();
  
  const ArrowIcon = language === 'ar' ? ArrowLeft : ArrowRight;

  const stats = [
    { value: "+$125,000", label: t.landing.totalAssets, icon: DollarSign },
    { value: "+18.5%", label: t.landing.avgReturn, icon: TrendingUp },
    { value: "24/7", label: t.landing.continuousTrading, icon: Clock },
    { value: "+50", label: t.landing.activeParticipants, icon: Users },
  ];

  const features = [
    {
      icon: <Bot className="w-7 h-7" />,
      title: t.landing.feature1Title,
      description: t.landing.feature1Desc,
      gradient: "from-violet-500 to-purple-600",
    },
    {
      icon: <Shield className="w-7 h-7" />,
      title: t.landing.feature2Title,
      description: t.landing.feature2Desc,
      gradient: "from-emerald-500 to-teal-600",
    },
    {
      icon: <Wallet className="w-7 h-7" />,
      title: t.landing.feature3Title,
      description: t.landing.feature3Desc,
      gradient: "from-amber-500 to-orange-600",
    },
    {
      icon: <Users className="w-7 h-7" />,
      title: t.landing.feature4Title,
      description: t.landing.feature4Desc,
      gradient: "from-pink-500 to-rose-600",
    },
    {
      icon: <Brain className="w-7 h-7" />,
      title: t.landing.feature5Title,
      description: t.landing.feature5Desc,
      gradient: "from-cyan-500 to-blue-600",
    },
    {
      icon: <Zap className="w-7 h-7" />,
      title: t.landing.feature6Title,
      description: t.landing.feature6Desc,
      gradient: "from-indigo-500 to-violet-600",
    },
  ];

  const steps = [
    { number: "1", title: t.landing.step1Title, description: t.landing.step1Desc },
    { number: "2", title: t.landing.step2Title, description: t.landing.step2Desc },
    { number: "3", title: t.landing.step3Title, description: t.landing.step3Desc },
    { number: "4", title: t.landing.step4Title, description: t.landing.step4Desc },
  ];

  return (
    <div className="min-h-screen bg-[#08080c] text-white overflow-hidden relative">
      {/* Animated Background Orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute w-[500px] h-[500px] rounded-full bg-violet-500/20 blur-[100px] top-[10%] left-[10%] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute w-[400px] h-[400px] rounded-full bg-pink-500/15 blur-[100px] bottom-[20%] right-[10%] animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
        <div className="absolute w-[300px] h-[300px] rounded-full bg-purple-600/15 blur-[80px] top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#08080c]/80 backdrop-blur-xl border-b border-violet-500/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl overflow-hidden shadow-[0_0_20px_rgba(139,92,246,0.4)] group-hover:shadow-[0_0_30px_rgba(139,92,246,0.6)] transition-all duration-300">
              <img 
                src="/logo-header.png" 
                alt="ASINAX Logo" 
                className="w-full h-full object-contain"
              />
            </div>
            <span className="font-bold text-xl bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">ASINAX</span>
          </Link>
          
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
        <div className="max-w-6xl mx-auto text-center">
          {/* Status Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-violet-500/10 border border-violet-500/30 mb-8 animate-fade-in-up backdrop-blur-sm">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
            <span className="text-violet-200 font-medium">{t.landing.tradingActive}</span>
          </div>

          {/* Main Heading */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            <span className="text-white">{t.landing.heroTitle1}</span>
            <br />
            <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient-shift bg-[length:200%_auto]">
              ASINAX CRYPTO AI
            </span>
          </h1>

          {/* Description */}
          <p className="text-lg md:text-xl text-white/60 max-w-3xl mx-auto mb-10 animate-fade-in-up leading-relaxed" style={{ animationDelay: '0.2s' }}>
            {t.landing.heroDescription}
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            <Link to="/register">
              <Button size="lg" className="text-lg px-8 py-6 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 group">
                {t.landing.startTrading}
                <ArrowIcon className="w-5 h-5 mx-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <a href="#features">
              <Button variant="outline" size="lg" className="text-lg px-8 py-6 border-violet-500/40 text-white hover:bg-violet-500/10 hover:border-violet-500/60 transition-all duration-300">
                {t.landing.learnMore}
              </Button>
            </a>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
            {stats.map((stat, index) => (
              <div
                key={index}
                className="p-6 rounded-2xl bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/40 hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)] transition-all duration-400 hover:-translate-y-1 group"
                style={{ animationDelay: `${0.5 + index * 0.1}s` }}
              >
                <div className="text-2xl md:text-3xl font-bold text-white mb-1 group-hover:text-violet-300 transition-colors">{stat.value}</div>
                <div className="text-sm text-white/50">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
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
                className="group p-6 rounded-2xl bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 hover:border-violet-500/40 hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)] transition-all duration-400 hover:-translate-y-2 animate-fade-in-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center text-white mb-5 group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(139,92,246,0.4)] transition-all duration-300`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-violet-200 transition-colors">{feature.title}</h3>
                <p className="text-white/50 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {t.landing.howItWorks}
            </h2>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              {t.landing.howItWorksDesc}
            </p>
          </div>
          
          <div className="grid md:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="relative text-center group animate-fade-in-up" style={{ animationDelay: `${index * 0.15}s` }}>
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold shadow-[0_0_30px_rgba(139,92,246,0.4)] group-hover:shadow-[0_0_50px_rgba(139,92,246,0.6)] transition-all duration-300 group-hover:scale-110 animate-pulse-glow">
                  {step.number}
                </div>
                {index < steps.length - 1 && (
                  <div 
                    className="hidden md:block absolute top-10 left-1/2 w-full h-0.5 bg-gradient-to-r from-violet-500/50 to-transparent" 
                    style={{ transform: language === 'ar' ? 'scaleX(-1)' : 'none' }} 
                  />
                )}
                <h3 className="text-lg font-semibold mb-2 text-white group-hover:text-violet-200 transition-colors">{step.title}</h3>
                <p className="text-white/50 text-sm">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 relative z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-violet-500/10 via-transparent to-purple-500/10" />
        <div className="max-w-4xl mx-auto text-center relative">
          <h2 className="text-3xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent animate-fade-in-up">
            {t.landing.readyToStart}
          </h2>
          <p className="text-white/50 text-lg mb-10 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            {t.landing.readyToStartDesc}
          </p>
          <Link to="/register" className="animate-fade-in-up inline-block" style={{ animationDelay: '0.2s' }}>
            <Button size="lg" className="text-lg px-12 py-7 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 group animate-pulse-glow">
              {t.landing.createAccount}
              <ArrowIcon className="w-5 h-5 mx-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-violet-500/10 relative z-10 bg-[#08080c]/80 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(139,92,246,0.3)]">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-lg text-white">ASINAX</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-white/50">
              <Link to="/privacy" className="hover:text-violet-400 transition-colors">{t.landing.privacy}</Link>
              <Link to="/terms" className="hover:text-violet-400 transition-colors">{t.landing.terms}</Link>
              <Link to="/contact" className="hover:text-violet-400 transition-colors">{t.landing.contact}</Link>
            </div>
            <p className="text-sm text-white/40">
              © 2025 ASINAX. {t.landing.allRightsReserved}
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
