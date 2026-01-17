import { Switch, Route } from "wouter";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { LanguageProvider, useLanguage } from "@/lib/i18n";
import { ThemeToggle } from "@/components/theme-toggle";
import { LanguageToggle } from "@/components/language-toggle";
import { AppSidebar } from "@/components/app-sidebar";
import { useAuth } from "@/hooks/useAuth";
import NotFound from "@/pages/not-found";
import Landing from "@/pages/landing";
import Dashboard from "@/pages/dashboard";
import Trades from "@/pages/trades";
import Market from "@/pages/market";
import AiAnalysis from "@/pages/ai-analysis";
import Stats from "@/pages/stats";
import Wallet from "@/pages/wallet";
import Notifications from "@/pages/notifications";
import Ledger from "@/pages/ledger";
import Register from "@/pages/register";
import Login from "@/pages/login";
import ForgotPassword from "@/pages/forgot-password";
import Privacy from "@/pages/privacy";
import Terms from "@/pages/terms";
import About from "@/pages/about";
import Contact from "@/pages/contact";
import Portfolio from "@/pages/portfolio";
import Settings from "@/pages/settings";
import { NotificationBell } from "@/components/notification-bell";
import { ParticlesBackground } from "@/components/particles-background";
import { FloatingActionButton } from "@/components/floating-action-button";
import { LiveIndicator } from "@/components/live-indicator";

function AuthenticatedRouter() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/wallet" component={Wallet} />
      <Route path="/trades" component={Trades} />
      <Route path="/market" component={Market} />
      <Route path="/ai-analysis" component={AiAnalysis} />
      <Route path="/stats" component={Stats} />
      <Route path="/notifications" component={Notifications} />
      <Route path="/ledger" component={Ledger} />
      <Route path="/portfolio" component={Portfolio} />
      <Route path="/settings" component={Settings} />
      <Route component={NotFound} />
    </Switch>
  );
}

function AuthenticatedLayout() {
  const { dir } = useLanguage();
  const style = {
    "--sidebar-width": "18rem",
    "--sidebar-width-icon": "4rem",
  };

  return (
    <SidebarProvider style={style as React.CSSProperties}>
      <ParticlesBackground count={8} />
      <div className="flex h-screen w-full relative z-10" dir={dir}>
        <AppSidebar />
        <div className="flex flex-col flex-1 min-w-0">
          <header className="flex items-center justify-between gap-4 p-3 border-b border-border sticky top-0 z-50 bg-background/95 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <SidebarTrigger />
              <LiveIndicator isConnected={true} />
            </div>
            <div className="flex items-center gap-2">
              <NotificationBell />
              <LanguageToggle />
              <ThemeToggle />
            </div>
          </header>
          <main className="flex-1 overflow-auto">
            <AuthenticatedRouter />
          </main>
        </div>
      </div>
      <FloatingActionButton />
    </SidebarProvider>
  );
}

function AppContent() {
  const { isAuthenticated, isLoading } = useAuth();
  const { dir, t } = useLanguage();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background" dir={dir}>
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">{t.common.loading}</p>
        </div>
      </div>
    );
  }

  return (
    <Switch>
      <Route path="/register" component={Register} />
      <Route path="/login" component={Login} />
      <Route path="/forgot-password" component={ForgotPassword} />
      <Route path="/privacy" component={Privacy} />
      <Route path="/terms" component={Terms} />
      <Route path="/about" component={About} />
      <Route path="/contact" component={Contact} />
      <Route>
        {isAuthenticated ? <AuthenticatedLayout /> : <Landing />}
      </Route>
    </Switch>
  );
}

function App() {
  return (
    <LanguageProvider>
      <TooltipProvider>
        <AppContent />
      </TooltipProvider>
    </LanguageProvider>
  );
}

export default App;
