import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { Mail, Lock, Eye, EyeOff, Bot, ArrowRight, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

// Google Icon Component
const GoogleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24">
    <path
      fill="#4285F4"
      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
    />
    <path
      fill="#34A853"
      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
    />
    <path
      fill="#FBBC05"
      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
    />
    <path
      fill="#EA4335"
      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
    />
  </svg>
);

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    // Check for Google OAuth error
    const googleError = searchParams.get('error');
    if (googleError) {
      if (googleError === 'google_auth_failed') {
        toast.error(t.auth.googleLoginFailed);
      } else if (googleError === 'token_failed') {
        toast.error(t.auth.googleLoginFailed);
      } else if (googleError === 'userinfo_failed') {
        toast.error(t.auth.googleLoginFailed);
      } else if (googleError === 'no_email') {
        toast.error(t.auth.googleLoginFailed);
      }
    }
  }, [searchParams, t]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      await login(email, password);
      toast.success(t.common.success);
      navigate('/dashboard');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || t.common.error;
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    // Redirect to Google OAuth endpoint
    window.location.href = 'https://asinax.cloud/api/v1/auth/google/login';
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#08080c] py-12 px-4 relative overflow-hidden">
      {/* Animated Background Orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute w-[500px] h-[500px] rounded-full bg-violet-500/20 blur-[100px] top-[10%] left-[10%] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute w-[400px] h-[400px] rounded-full bg-pink-500/15 blur-[100px] bottom-[20%] right-[10%] animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
        <div className="absolute w-[300px] h-[300px] rounded-full bg-purple-600/15 blur-[80px] top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />
      </div>

      <div className="max-w-md w-full relative z-10">
        {/* Card */}
        <div className="rounded-3xl bg-[rgba(15,15,25,0.8)] backdrop-blur-xl border border-violet-500/20 p-8 shadow-[0_20px_60px_rgba(0,0,0,0.5)] animate-fade-in-up">
          {/* Header */}
          <div className="text-center mb-8">
            <Link to="/" className="inline-block">
              <div className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-[0_0_30px_rgba(139,92,246,0.4)] hover:shadow-[0_0_40px_rgba(139,92,246,0.5)] transition-all duration-300 hover:scale-105 overflow-hidden">
                <img 
                  src="/logo-login.png" 
                  alt="ASINAX Logo" 
                  className="w-full h-full object-contain"
                />
              </div>
            </Link>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {t.auth.loginTitle}
            </h1>
            <p className="text-white/50 mt-2">
              {t.auth.welcomeBack}
            </p>
          </div>

          {/* Google Login Button */}
          <button
            onClick={handleGoogleLogin}
            className="w-full py-3.5 rounded-xl bg-white hover:bg-gray-100 text-gray-800 font-medium shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-3 mb-6 animate-fade-in-up"
          >
            <GoogleIcon />
            <span>{t.auth.loginWithGoogle}</span>
          </button>

          {/* Divider */}
          <div className="flex items-center gap-4 mb-6 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
            <span className="text-white/30 text-sm">{t.auth.orWithEmail}</span>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-400 animate-shake">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.email}
              </label>
              <div className="relative group">
                <Mail className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="example@email.com"
                  required
                  dir="ltr"
                />
              </div>
            </div>

            {/* Password */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.password}
              </label>
              <div className="relative group">
                <Lock className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-12 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="••••••••"
                  required
                  dir="ltr"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-violet-400/60 hover:text-violet-400 transition-colors"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Forgot Password */}
            <div className="flex justify-end animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
              <Link to="/forgot-password" className="text-sm text-violet-400 hover:text-violet-300 transition-colors">
                {t.auth.forgotPassword}
              </Link>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white font-semibold shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group animate-fade-in-up"
              style={{ animationDelay: '0.3s' }}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>{t.common.loading}</span>
                </>
              ) : (
                <>
                  <span>{t.auth.loginButton}</span>
                  <ArrowRight className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </form>

          {/* Footer */}
          <p className="text-center text-white/50 mt-6 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
            {t.auth.noAccount}{' '}
            <Link to="/register" className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
              {t.auth.createAccount}
            </Link>
          </p>
        </div>

        {/* Back to Home */}
        <div className="text-center mt-6 animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
          <Link to="/" className="text-white/40 hover:text-violet-400 transition-colors text-sm">
            ← {t.auth.backToHome}
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
