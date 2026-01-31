import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Wallet,
  TrendingUp,
  Settings,
  Users,
  Bell,
  LogOut,
  Menu,
  X,
  ChevronDown,
  Bot,
  PieChart,
  Gift,
  HeadphonesIcon,
  Shield,
  ArrowDownCircle,
  Globe,
  Sun,
  Moon,
  ArrowLeft,
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { useTheme } from '@/components/theme-provider';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useNotifications } from '@/hooks/useNotifications';

const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { language, setLanguage, t, dir } = useLanguage();
  const { theme, setTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  // استخدام hook الإشعارات الجديد
  const { unreadCount } = useNotifications({
    enabled: isAuthenticated,
    refetchInterval: 10000,
    showToastOnNew: true,
  });

  // Check if impersonating a user
  const isImpersonating = localStorage.getItem("impersonating") === "true";
  const impersonatedUser = localStorage.getItem("impersonated_user");

  const handleReturnToAdmin = () => {
    const adminToken = localStorage.getItem("admin_token_backup");
    if (adminToken) {
      localStorage.setItem("token", adminToken);
      localStorage.removeItem("admin_token_backup");
      localStorage.removeItem("impersonating");
      localStorage.removeItem("impersonated_user");
      window.location.href = "/admin";
    }
  };

  const userLinks = [
    { path: '/dashboard', icon: LayoutDashboard, label: t.navbar.dashboard },
    { path: '/portfolio', icon: PieChart, label: t.navbar.portfolio },
    { path: '/wallet', icon: Wallet, label: t.navbar.wallet },
    { path: '/trades', icon: TrendingUp, label: t.navbar.trades },
    { path: '/referrals', icon: Gift, label: t.navbar.referrals },
    { path: '/support', icon: HeadphonesIcon, label: t.navbar.support },
  ];

  const adminLinks = [
    { path: '/admin', icon: Shield, label: t.navbar.admin },
    { path: '/admin/users', icon: Users, label: t.navbar.users },
    { path: '/admin/withdrawals', icon: Wallet, label: t.navbar.withdrawals },
    { path: '/admin/deposits', icon: ArrowDownCircle, label: 'الإيداعات' },
    { path: '/admin/agent', icon: Bot, label: 'التحكم بالوكيل' },
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  // تحديد الروابط المعروضة بناءً على نوع المستخدم
  const displayedLinks = user?.is_admin ? adminLinks : userLinks;

  return (
    <>
      {/* شريط التنبيه عند الدخول كمستخدم */}
      {isImpersonating && (
        <div className="fixed top-0 left-0 right-0 z-[100] bg-amber-500 text-black py-2 px-4 flex items-center justify-between">
          <span className="font-medium">
            ⚠️ أنت تتصفح كـ: {impersonatedUser}
          </span>
          <button
            onClick={handleReturnToAdmin}
            className="flex items-center gap-2 bg-black text-white px-4 py-1 rounded-lg hover:bg-gray-800 transition-colors font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            العودة للأدمن
          </button>
        </div>
      )}

      <nav className={cn(
        "sticky z-50 bg-background/80 backdrop-blur-md border-b border-border",
        isImpersonating ? "top-10" : "top-0"
      )} dir={dir}>
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to={isAuthenticated ? (user?.is_admin ? '/admin' : '/dashboard') : '/'} className="flex items-center gap-2">
              <div className="w-9 h-9 bg-gradient-to-br from-primary to-primary/60 rounded-xl flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                ASINAX
              </span>
            </Link>

            {/* Desktop Navigation */}
            {isAuthenticated && (
              <div className="hidden lg:flex items-center gap-1">
                {displayedLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                      isActive(link.path)
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    )}
                  >
                    <link.icon className="w-4 h-4" />
                    {link.label}
                  </Link>
                ))}
              </div>
            )}

            {/* Right Side */}
            <div className="flex items-center gap-2">
              {/* Language Toggle */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
                className="relative"
                title={language === 'ar' ? 'Switch to English' : 'التبديل للعربية'}
              >
                <Globe className="h-5 w-5" />
                <span className="absolute -bottom-1 -right-1 text-[10px] font-bold bg-primary text-primary-foreground rounded px-1">
                  {language === 'ar' ? 'EN' : 'ع'}
                </span>
              </Button>

              {/* Theme Toggle */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                title={theme === 'dark' ? t.settings.lightMode : t.settings.darkMode}
              >
                {theme === 'dark' ? (
                  <Sun className="h-5 w-5" />
                ) : (
                  <Moon className="h-5 w-5" />
                )}
              </Button>

              {isAuthenticated ? (
                <>
                  {/* Notifications */}
                  <Link to="/notifications">
                    <Button variant="ghost" size="icon" className="relative">
                      <Bell className={cn(
                        "h-5 w-5 transition-all",
                        unreadCount > 0 && "animate-pulse"
                      )} />
                      {unreadCount > 0 && (
                        <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] bg-destructive text-destructive-foreground text-xs font-bold rounded-full flex items-center justify-center px-1 animate-bounce">
                          {unreadCount > 99 ? '99+' : unreadCount}
                        </span>
                      )}
                    </Button>
                  </Link>

                  {/* User Menu */}
                  <div className="relative">
                    <button
                      onClick={() => setUserMenuOpen(!userMenuOpen)}
                      className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-muted transition-colors"
                    >
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center text-white text-sm font-medium">
                        {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                      </div>
                      <span className="hidden md:block text-sm font-medium">
                        {user?.full_name || user?.email?.split('@')[0]}
                      </span>
                      {user?.is_admin && (
                        <span className="hidden md:inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary/20 text-primary">
                          أدمن
                        </span>
                      )}
                      <ChevronDown className="w-4 h-4 text-muted-foreground hidden md:block" />
                    </button>

                    {/* Dropdown */}
                    {userMenuOpen && (
                      <>
                        <div 
                          className="fixed inset-0 z-40"
                          onClick={() => setUserMenuOpen(false)}
                        />
                        <div className="absolute left-0 mt-2 w-48 bg-card rounded-lg shadow-lg border border-border z-50">
                          <div className="p-3 border-b border-border">
                            <p className="font-medium text-sm">{user?.full_name || 'مستخدم'}</p>
                            <p className="text-xs text-muted-foreground" dir="ltr">{user?.email}</p>
                            {user?.is_admin && (
                              <span className="inline-flex items-center mt-1 px-2 py-0.5 rounded text-xs font-medium bg-primary/20 text-primary">
                                مدير النظام
                              </span>
                            )}
                          </div>
                          <Link
                            to="/settings"
                            onClick={() => setUserMenuOpen(false)}
                            className="flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted transition-colors"
                          >
                            <Settings className="w-4 h-4" />
                            {t.navbar.settings}
                          </Link>
                          <button
                            onClick={() => {
                              setUserMenuOpen(false);
                              handleLogout();
                            }}
                            className="flex items-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-muted transition-colors w-full"
                          >
                            <LogOut className="w-4 h-4" />
                            {t.navbar.logout}
                          </button>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Mobile menu button */}
                  <button
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    className="lg:hidden p-2 rounded-lg hover:bg-muted transition-colors"
                  >
                    {mobileMenuOpen ? (
                      <X className="w-5 h-5" />
                    ) : (
                      <Menu className="w-5 h-5" />
                    )}
                  </button>
                </>
              ) : (
                <div className="flex items-center gap-2">
                  <Link to="/login">
                    <Button variant="ghost" size="sm">
                      {t.auth.loginButton}
                    </Button>
                  </Link>
                  <Link to="/register">
                    <Button size="sm" className="bg-gradient-to-r from-primary to-primary/80">
                      {t.auth.registerButton}
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {isAuthenticated && mobileMenuOpen && (
          <div className="lg:hidden border-t border-border bg-card">
            <div className="px-4 py-3 space-y-1">
              {displayedLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    isActive(link.path)
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  <link.icon className="w-5 h-5" />
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        )}
      </nav>
    </>
  );
};

export default Navbar;
