import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';
import { 
  Sun, 
  Moon, 
  LogOut, 
  User, 
  Settings, 
  Bell,
  Menu,
  X,
  LayoutDashboard,
  Wallet,
  TrendingUp,
  PieChart,
  Users,
  Gift,
  HelpCircle,
  Shield,
  ChevronDown,
} from 'lucide-react';
import { cn } from '../../lib/utils';

const Navbar: React.FC = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path: string) => location.pathname === path;

  const userLinks = [
    { path: '/dashboard', label: 'لوحة التحكم', icon: LayoutDashboard },
    { path: '/portfolio', label: 'المحفظة', icon: PieChart },
    { path: '/wallet', label: 'المحفظة المالية', icon: Wallet },
    { path: '/trades', label: 'الصفقات', icon: TrendingUp },
    { path: '/referrals', label: 'الإحالات', icon: Gift },
    { path: '/support', label: 'الدعم', icon: HelpCircle },
  ];

  const adminLinks = [
    { path: '/admin', label: 'لوحة الأدمن', icon: Shield },
    { path: '/admin/users', label: 'المستخدمين', icon: Users },
    { path: '/admin/withdrawals', label: 'السحوبات', icon: Wallet },
  ];

  return (
    <nav className="bg-card border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold">L</span>
              </div>
              <span className="font-bold text-xl hidden sm:block">
                Legendary AI
              </span>
            </Link>
          </div>

          {/* Desktop Navigation Links */}
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
              
              {/* Admin Dropdown */}
              {user?.is_admin && (
                <div className="relative group">
                  <button className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    location.pathname.startsWith('/admin')
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}>
                    <Shield className="w-4 h-4" />
                    الإدارة
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  <div className="absolute right-0 mt-1 w-48 bg-card rounded-lg shadow-lg border border-border opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                    {adminLinks.map((link) => (
                      <Link
                        key={link.path}
                        to={link.path}
                        className={cn(
                          "flex items-center gap-2 px-4 py-2 text-sm transition-colors first:rounded-t-lg last:rounded-b-lg",
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
                </div>
              )}
            </div>
          )}

          {/* Right side */}
          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-muted-foreground" />
              ) : (
                <Moon className="w-5 h-5 text-muted-foreground" />
              )}
            </button>

            {isAuthenticated ? (
              <>
                {/* Notifications */}
                <Link 
                  to="/notifications"
                  className="p-2 rounded-lg hover:bg-muted transition-colors relative"
                >
                  <Bell className="w-5 h-5 text-muted-foreground" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full"></span>
                </Link>

                {/* User menu */}
                <div className="relative">
                  <button 
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-2 p-2 rounded-lg hover:bg-muted transition-colors"
                  >
                    <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-primary" />
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
                          الإعدادات
                        </Link>
                        <button
                          onClick={() => {
                            setUserMenuOpen(false);
                            handleLogout();
                          }}
                          className="flex items-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-muted transition-colors w-full"
                        >
                          <LogOut className="w-4 h-4" />
                          تسجيل الخروج
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
                <Link 
                  to="/login" 
                  className="px-4 py-2 text-sm font-medium hover:bg-muted rounded-lg transition-colors"
                >
                  تسجيل الدخول
                </Link>
                <Link 
                  to="/register" 
                  className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                >
                  إنشاء حساب
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
                <p className="px-3 py-1 text-xs text-muted-foreground font-medium">الإدارة</p>
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
