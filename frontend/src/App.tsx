import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { LanguageProvider } from './lib/i18n';

// Components
import Navbar from './components/common/Navbar';
import ImpersonationBanner from './components/ImpersonationBanner';

// Pages - Public
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import AuthCallback from './pages/AuthCallback';
import Privacy from './pages/Privacy';
import Terms from './pages/Terms';
import Contact from './pages/Contact';

// Pages - User
import DashboardNew from './pages/DashboardNew';
import PortfolioNew from './pages/PortfolioNew';
import WalletNew from './pages/WalletNew';
import TradesNew from './pages/TradesNew';
import SettingsNew from './pages/SettingsNew';
import Referrals from './pages/Referrals';
import Support from './pages/Support';
import Notifications from './pages/Notifications';

// Pages - Admin
import AdminDashboardNew from './pages/admin/AdminDashboardNew';
import UsersManagement from './pages/admin/UsersManagement';
import WithdrawalsManagement from './pages/admin/WithdrawalsManagement';

// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode; adminOnly?: boolean }> = ({
  children,
  adminOnly = false,
}) => {
  const { isAuthenticated, isLoading, user } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (adminOnly && !user?.is_admin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Public Route Component (redirect if authenticated)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Layout with Navbar
const Layout: React.FC<{ children: React.ReactNode; showNavbar?: boolean }> = ({
  children,
  showNavbar = true,
}) => {
  return (
    <div className="min-h-screen bg-background">
      <ImpersonationBanner />
      {showNavbar && <Navbar />}
      <main>{children}</main>
    </div>
  );
};

function AppRoutes() {
  return (
    <Routes>
      {/* Public Routes */}
      <Route
        path="/"
        element={
          <Layout showNavbar={false}>
            <Landing />
          </Layout>
        }
      />
      <Route
        path="/login"
        element={
          <PublicRoute>
            <Layout showNavbar={false}>
              <Login />
            </Layout>
          </PublicRoute>
        }
      />
      <Route
        path="/register"
        element={
          <PublicRoute>
            <Layout showNavbar={false}>
              <Register />
            </Layout>
          </PublicRoute>
        }
      />
      <Route
        path="/auth/callback"
        element={
          <Layout showNavbar={false}>
            <AuthCallback />
          </Layout>
        }
      />
      <Route
        path="/privacy"
        element={
          <Layout showNavbar={false}>
            <Privacy />
          </Layout>
        }
      />
      <Route
        path="/terms"
        element={
          <Layout showNavbar={false}>
            <Terms />
          </Layout>
        }
      />
      <Route
        path="/contact"
        element={
          <Layout showNavbar={false}>
            <Contact />
          </Layout>
        }
      />

      {/* Protected User Routes */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Layout>
              <DashboardNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/portfolio"
        element={
          <ProtectedRoute>
            <Layout>
              <PortfolioNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/wallet"
        element={
          <ProtectedRoute>
            <Layout>
              <WalletNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/deposit"
        element={
          <ProtectedRoute>
            <Layout>
              <WalletNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/withdraw"
        element={
          <ProtectedRoute>
            <Layout>
              <WalletNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/trades"
        element={
          <ProtectedRoute>
            <Layout>
              <TradesNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <Layout>
              <SettingsNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/referrals"
        element={
          <ProtectedRoute>
            <Layout>
              <Referrals />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/support"
        element={
          <ProtectedRoute>
            <Layout>
              <Support />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/notifications"
        element={
          <ProtectedRoute>
            <Layout>
              <Notifications />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* Admin Routes */}
      <Route
        path="/admin"
        element={
          <ProtectedRoute adminOnly>
            <Layout>
              <AdminDashboardNew />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/users"
        element={
          <ProtectedRoute adminOnly>
            <Layout>
              <UsersManagement />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/withdrawals"
        element={
          <ProtectedRoute adminOnly>
            <Layout>
              <WithdrawalsManagement />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* Catch all - redirect to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ThemeProvider>
          <LanguageProvider>
            <AuthProvider>
              <AppRoutes />
              <Toaster
                position="top-center"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: 'hsl(var(--card))',
                    color: 'hsl(var(--card-foreground))',
                    border: '1px solid hsl(var(--border))',
                  },
                }}
              />
            </AuthProvider>
          </LanguageProvider>
        </ThemeProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
