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
  Globe,
  Sun,
  Moon,
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { useTheme } from '@/components/theme-provider';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { language, setLanguage, t, dir } = useLanguage();
  const { theme, setTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

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
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border" dir={dir}>
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to={isAuthenticated ? '/dashboard' : '/'} className="flex items-center gap-2">
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
              {userLinks.map((link) => (
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
              
              {user?.is_admin && (
                <div className="h-6 w-px bg-border mx-2" />
              )}
              
              {user?.is_admin && adminLinks.map((link) => (
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
                    <Bell className="h-5 w-5" />
                    <span className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full" />
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
            {userLinks.map((link) => (
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
            
            {user?.is_admin && (
              <>
                <div className="border-t border-border my-2" />
                <p className="px-3 py-1 text-xs text-muted-foreground font-medium">{t.navbar.admin}</p>
                {adminLinks.map((link) => (
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
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
