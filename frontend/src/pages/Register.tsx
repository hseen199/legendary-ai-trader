import React, { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { Mail, Lock, Eye, EyeOff, User, Phone, Bot, ArrowRight, Gift } from 'lucide-react';
import toast from 'react-hot-toast';

const Register: React.FC = () => {
  const [searchParams] = useSearchParams();
  const referralCode = searchParams.get('ref') || '';
  const { t } = useLanguage();
  
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    password: '',
    confirmPassword: '',
    referral_code: referralCode,
  });
  const [showPassword, setShowPassword] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (formData.password !== formData.confirmPassword) {
      toast.error(t.auth.passwordsNotMatch);
      return;
    }
    
    if (!agreedToTerms) {
      toast.error(t.auth.mustAgreeTerms);
      return;
    }

    setIsLoading(true);
    try {
      await register(formData);
      toast.success(t.auth.accountCreated);
      navigate('/dashboard');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || t.common.error);
    } finally {
      setIsLoading(false);
    }
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
              <div className="w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-[0_0_30px_rgba(139,92,246,0.4)] hover:shadow-[0_0_40px_rgba(139,92,246,0.5)] transition-all duration-300 hover:scale-105">
                <Bot className="w-8 h-8 text-white" />
              </div>
            </Link>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {t.auth.registerTitle}
            </h1>
            <p className="text-white/50 mt-2">
              {t.auth.joinPlatform}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Full Name */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.fullName}
              </label>
              <div className="relative group">
                <User className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type="text"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleChange}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="Ahmed Mohamed"
                />
              </div>
            </div>

            {/* Email */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.email} *
              </label>
              <div className="relative group">
                <Mail className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="example@email.com"
                  required
                  dir="ltr"
                />
              </div>
            </div>

            {/* Phone */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.phone}
              </label>
              <div className="relative group">
                <Phone className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type="tel"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="+971 50 123 4567"
                  dir="ltr"
                />
              </div>
            </div>

            {/* Password */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.password} *
              </label>
              <div className="relative group">
                <Lock className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-12 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="••••••••"
                  required
                  minLength={8}
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

            {/* Confirm Password */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
              <label className="block text-sm font-medium text-white/70 mb-2">
                {t.auth.confirmPassword} *
              </label>
              <div className="relative group">
                <Lock className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  className="w-full bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                  placeholder="••••••••"
                  required
                  dir="ltr"
                />
              </div>
            </div>

            {/* Referral Code */}
            {referralCode && (
              <div className="animate-fade-in-up" style={{ animationDelay: '0.35s' }}>
                <label className="block text-sm font-medium text-white/70 mb-2">
                  {t.referrals.yourCode}
                </label>
                <div className="relative group">
                  <Gift className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-emerald-400" />
                  <input
                    type="text"
                    name="referral_code"
                    value={formData.referral_code}
                    onChange={handleChange}
                    className="w-full bg-emerald-500/10 border border-emerald-500/30 rounded-xl py-3.5 pr-12 pl-4 text-emerald-400 focus:outline-none"
                    readOnly
                  />
                </div>
              </div>
            )}

            {/* Terms */}
            <div className="flex items-start gap-3 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
              <input
                type="checkbox"
                id="terms"
                checked={agreedToTerms}
                onChange={(e) => setAgreedToTerms(e.target.checked)}
                className="mt-1 w-4 h-4 rounded border-violet-500/30 bg-violet-500/10 text-violet-500 focus:ring-violet-500/50"
              />
              <label htmlFor="terms" className="text-sm text-white/50">
                {t.auth.agreeToTerms}{' '}
                <Link to="/terms" className="text-violet-400 hover:text-violet-300 transition-colors">
                  {t.landing.terms}
                </Link>{' '}
                &{' '}
                <Link to="/privacy" className="text-violet-400 hover:text-violet-300 transition-colors">
                  {t.landing.privacy}
                </Link>
              </label>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white font-semibold shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group animate-fade-in-up"
              style={{ animationDelay: '0.45s' }}
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
                  <span>{t.auth.registerButton}</span>
                  <ArrowRight className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center gap-4 my-6 animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
            <span className="text-white/30 text-sm">{t.auth.orContinueWith}</span>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-violet-500/30 to-transparent" />
          </div>

          {/* Footer */}
          <p className="text-center text-white/50 animate-fade-in-up" style={{ animationDelay: '0.55s' }}>
            {t.auth.hasAccount}{' '}
            <Link to="/login" className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
              {t.auth.loginHere}
            </Link>
          </p>
        </div>

        {/* Back to Home */}
        <div className="text-center mt-6 animate-fade-in-up" style={{ animationDelay: '0.6s' }}>
          <Link to="/" className="text-white/40 hover:text-violet-400 transition-colors text-sm">
            ← {t.auth.backToHome}
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Register;
