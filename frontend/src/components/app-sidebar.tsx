import { Link, useLocation } from "wouter";
import {
  LayoutDashboard,
  Wallet,
  History,
  Settings,
  LogOut,
  TrendingUp,
  Bot,
  PieChart,
  Bell,
  BookOpen,
  Users,
  HelpCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/lib/i18n";
import { AIThinkingPulse } from "@/components/ai-thinking-pulse";

export function AppSidebar() {
  const [location] = useLocation();
  const { user } = useAuth();
  const { t, dir } = useLanguage();

  const menuItems = [
    {
      title: t.nav.dashboard,
      url: "/",
      icon: LayoutDashboard,
    },
    {
      title: t.nav.wallet,
      url: "/wallet",
      icon: Wallet,
    },
    {
      title: t.nav.trades,
      url: "/trades",
      icon: History,
    },
    {
      title: t.nav.market,
      url: "/market",
      icon: TrendingUp,
    },
    {
      title: t.nav.aiAnalysis,
      url: "/ai-analysis",
      icon: Bot,
    },
    {
      title: t.nav.stats,
      url: "/stats",
      icon: PieChart,
    },
    {
      title: t.nav.notifications,
      url: "/notifications",
      icon: Bell,
    },
    {
      title: t.nav.ledger || "سجل المعاملات",
      url: "/ledger",
      icon: BookOpen,
    },
  ];

  const getUserInitials = () => {
    if (user?.firstName && user?.lastName) {
      return `${user.firstName[0]}${user.lastName[0]}`;
    }
    if (user?.email) {
      return user.email[0].toUpperCase();
    }
    return "م";
  };

  return (
    <Sidebar 
      side={dir === "rtl" ? "right" : "left"} 
      collapsible="icon"
      className="bg-[rgba(12,12,18,0.9)] backdrop-blur-xl border-violet-500/10"
    >
      <SidebarHeader className="p-4 border-b border-violet-500/10 relative overflow-visible">
        {/* Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-violet-500/5 to-transparent pointer-events-none" />
        
        <div className="flex items-center gap-3 relative">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl overflow-hidden shadow-[0_0_20px_rgba(139,92,246,0.4)]">
              <img 
                src="/logo-sidebar.png?v=1768667205" 
                alt="ASINAX Logo" 
                className="w-full h-full object-contain"
              />
            </div>
            <div className="absolute -inset-1 bg-violet-500/20 rounded-xl blur-md -z-10 animate-pulse" />
          </div>
          <div className="flex flex-col group-data-[collapsible=icon]:hidden">
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg bg-gradient-to-r from-white to-violet-300 bg-clip-text text-transparent">
                ASINAX
              </span>
              <AIThinkingPulse size="sm" isActive={true} />
            </div>
            <span className="text-xs text-white/40">CRYPTO AI</span>
          </div>
        </div>
      </SidebarHeader>
      
      <SidebarContent className="py-4">
        <SidebarGroup>
          <SidebarGroupLabel className="text-violet-400/70 text-xs font-semibold uppercase tracking-wider px-4 mb-2">
            {t.common.mainMenu}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-1 px-2">
              {menuItems.map((item) => {
                const isActive = location === item.url;
                return (
                  <SidebarMenuItem key={item.title} className="relative">
                    {/* Active Indicator */}
                    {isActive && (
                      <div 
                        className="absolute inset-y-1 end-0 w-1 bg-gradient-to-b from-violet-500 via-purple-500 to-pink-500 rounded-full shadow-[0_0_10px_rgba(139,92,246,0.5)]" 
                      />
                    )}
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      tooltip={item.title}
                      className={`
                        relative rounded-xl px-3 py-2.5 transition-all duration-300
                        ${isActive 
                          ? 'bg-violet-500/15 text-violet-300 shadow-[0_0_20px_rgba(139,92,246,0.1)]' 
                          : 'text-white/50 hover:text-violet-300 hover:bg-violet-500/10'
                        }
                      `}
                    >
                      <Link href={item.url} data-testid={`link-${item.url.slice(1) || 'dashboard'}`}>
                        <item.icon 
                          className={`w-5 h-5 transition-all duration-300 ${
                            isActive 
                              ? 'text-violet-400 drop-shadow-[0_0_8px_rgba(139,92,246,0.5)]' 
                              : ''
                          }`} 
                        />
                        <span className="font-medium">{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Divider */}
        <div className="my-4 mx-4 h-px bg-gradient-to-r from-transparent via-violet-500/20 to-transparent" />

        {/* Secondary Menu */}
        <SidebarGroup>
          <SidebarGroupLabel className="text-violet-400/70 text-xs font-semibold uppercase tracking-wider px-4 mb-2">
            {t.navbar.settings}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-1 px-2">
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={location === '/referrals'}
                  tooltip={t.navbar.referrals}
                  className={`
                    relative rounded-xl px-3 py-2.5 transition-all duration-300
                    ${location === '/referrals'
                      ? 'bg-violet-500/15 text-violet-300' 
                      : 'text-white/50 hover:text-violet-300 hover:bg-violet-500/10'
                    }
                  `}
                >
                  <Link href="/referrals">
                    <Users className="w-5 h-5" />
                    <span className="font-medium">{t.navbar.referrals}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={location === '/support'}
                  tooltip={t.navbar.support}
                  className={`
                    relative rounded-xl px-3 py-2.5 transition-all duration-300
                    ${location === '/support'
                      ? 'bg-violet-500/15 text-violet-300' 
                      : 'text-white/50 hover:text-violet-300 hover:bg-violet-500/10'
                    }
                  `}
                >
                  <Link href="/support">
                    <HelpCircle className="w-5 h-5" />
                    <span className="font-medium">{t.navbar.support}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={location === '/settings'}
                  tooltip={t.navbar.settings}
                  className={`
                    relative rounded-xl px-3 py-2.5 transition-all duration-300
                    ${location === '/settings'
                      ? 'bg-violet-500/15 text-violet-300' 
                      : 'text-white/50 hover:text-violet-300 hover:bg-violet-500/10'
                    }
                  `}
                >
                  <Link href="/settings">
                    <Settings className="w-5 h-5" />
                    <span className="font-medium">{t.navbar.settings}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-violet-500/10">
        <div className="flex items-center gap-3 group-data-[collapsible=icon]:justify-center">
          <Avatar className="w-10 h-10 ring-2 ring-violet-500/30 ring-offset-2 ring-offset-[#0c0c12]">
            <AvatarImage src={user?.profileImageUrl || undefined} className="object-cover" />
            <AvatarFallback className="bg-gradient-to-br from-violet-500 to-purple-600 text-white text-sm font-semibold">
              {getUserInitials()}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 group-data-[collapsible=icon]:hidden min-w-0">
            <p className="text-sm font-medium text-white truncate">
              {user?.firstName && user?.lastName 
                ? `${user.firstName} ${user.lastName}`
                : user?.email || t.common.user}
            </p>
            <p className="text-xs text-white/40 truncate">{user?.email}</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="group-data-[collapsible=icon]:hidden text-white/50 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all duration-300"
            onClick={async () => {
              try {
                await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
                window.location.href = "/login";
              } catch (e) {
                window.location.href = "/login";
              }
            }}
            data-testid="button-logout"
          >
            <LogOut className="w-4 h-4" />
          </Button>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
