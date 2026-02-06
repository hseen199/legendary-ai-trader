import { BotIcon } from "../components/BotIcon";
import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { Mail, Lock, Eye, EyeOff, User, ArrowRight, AlertCircle, KeyRound, RefreshCw, Gift } from 'lucide-react';
import toast from 'react-hot-toast';
import api from '@/services/api';

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

const Register: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [referralCode, setReferralCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [acceptTerms, setAcceptTerms] = useState(false);
  const { refreshUser } = useAuth();
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  
  // OTP State
  const [step, setStep] = useState<'register' | 'otp'>('register');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [resendTimer, setResendTimer] = useState(0);
  const otpInputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Get referral code from URL
  useEffect(() => {
    const ref = searchParams.get('ref');
    if (ref) {
      setReferralCode(ref);
    }
  }, [searchParams]);

  // Resend timer countdown
  useEffect(() => {
    if (resendTimer > 0) {
      const timer = setTimeout(() => setResendTimer(resendTimer - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [resendTimer]);

  // Handle OTP input change
  const handleOtpChange = (index: number, value: string) => {
    if (value.length > 1) {
      const pastedValue = value.slice(0, 6).split('');
      const newOtp = [...otp];
      pastedValue.forEach((char, i) => {
        if (index + i < 6) {
          newOtp[index + i] = char;
        }
      });
      setOtp(newOtp);
      const nextIndex = Math.min(index + pastedValue.length, 5);
      otpInputRefs.current[nextIndex]?.focus();
    } else {
      const newOtp = [...otp];
      newOtp[index] = value;
      setOtp(newOtp);
      
      if (value && index < 5) {
        otpInputRefs.current[index + 1]?.focus();
      }
    }
  };

  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
    }
  };

  // Step 1: Register
  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('كلمات المرور غير متطابقة');
      return;
    }

    if (password.length < 8) {
      setError('كلمة المرور يجب أن تكون 8 أحرف على الأقل');
      return;
    }

    if (!acceptTerms) {
      setError('يجب الموافقة على الشروط والأحكام');
      return;
    }

    setIsLoading(true);
    
    try {
      await api.post('/auth/register', {
        email: email.toLowerCase(),
        password,
        full_name: fullName,
        referral_code: referralCode || undefined
      });
      
      setStep('otp');
      setResendTimer(60);
      toast.success('تم إرسال رمز التحقق إلى بريدك الإلكتروني', { id: 'otp-sent-register' });
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || t.common.error;
      setError(errorMessage);
      toast.error(errorMessage, { id: 'register-error' });
    } finally {
      setIsLoading(false);
    }
  };

  // Step 2: Verify OTP
  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const otpCode = otp.join('');
    
    if (otpCode.length !== 6) {
      toast.error('يرجى إدخال رمز التحقق كاملاً', { id: 'otp-incomplete' });
      return;
    }
    
    setError('');
    setIsLoading(true);
    
    try {
      const response = await api.post('/auth/register/verify', {
        email: email.toLowerCase(),
        otp: otpCode
      });
      
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        await refreshUser();
        // استخدام ID ثابت لمنع التكرار
        toast.success('تم إنشاء حسابك بنجاح! مرحباً بك في ASINAX', { id: 'register-success' });
        navigate('/dashboard');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'رمز التحقق غير صحيح';
      setError(errorMessage);
      toast.error(errorMessage, { id: 'verify-error' });
      setOtp(['', '', '', '', '', '']);
      otpInputRefs.current[0]?.focus();
    } finally {
      setIsLoading(false);
    }
  };

  // Resend OTP
  const handleResendOtp = async () => {
    if (resendTimer > 0) return;
    
    setIsLoading(true);
    try {
      await api.post('/auth/otp/resend', { email: email.toLowerCase() });
      setResendTimer(60);
      toast.success('تم إرسال رمز التحقق مرة أخرى', { id: 'otp-resent' });
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'فشل في إعادة إرسال الرمز';
      toast.error(errorMessage, { id: 'resend-error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToRegister = () => {
    setStep('register');
    setOtp(['', '', '', '', '', '']);
    setError('');
  };

  const handleGoogleLogin = () => {
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
                  src="/logo-login.png?v=1768667205" 
                  alt="ASINAX Logo" 
                  className="w-full h-full object-contain"
                />
              </div>
            </Link>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {step === 'register' ? t.auth.registerTitle : 'تأكيد البريد الإلكتروني'}
            </h1>
            <p className="text-white/50 mt-2">
              {step === 'register' ? t.auth.joinUs : 'أدخل رمز التحقق المرسل إلى بريدك'}
            </p>
          </div>

          {step === 'register' ? (
            <>
              {/* Google Login Button */}
              <button
                onClick={handleGoogleLogin}
                className="w-full py-3.5 rounded-xl bg-white hover:bg-gray-100 text-gray-800 font-medium shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-3 mb-6 animate-fade-in-up"
              >
                <GoogleIcon />
                <span>{t.auth.registerWithGoogle}</span>
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

              {/* Register Form */}
              <form onSubmit={handleRegisterSubmit} className="space-y-5">
                {/* Full Name */}
                <div className="animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                  <label className="block text-sm font-medium text-white/70 mb-2">
                    {t.auth.fullName}
                  </label>
                  <div className="relative group">
                    <User className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                      placeholder={t.auth.fullNamePlaceholder}
                      required
                    />
                  </div>
                </div>

                {/* Email */}
                <div className="animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
                  <label className="block text-sm font-medium text-white/70 mb-2">
                    {t.auth.email}
                  </label>
                  <div className="relative group">
                    <Mail className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                      placeholder={t.auth.emailPlaceholder}
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
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pr-12 pl-12 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                      placeholder={t.auth.passwordPlaceholder}
                      required
                      dir="ltr"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute left-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                {/* Confirm Password */}
                <div className="animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
                  <label className="block text-sm font-medium text-white/70 mb-2">
                    {t.auth.confirmPassword}
                  </label>
                  <div className="relative group">
                    <Lock className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                    <input
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pr-12 pl-12 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                      placeholder={t.auth.confirmPasswordPlaceholder}
                      required
                      dir="ltr"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute left-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70 transition-colors"
                    >
                      {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                {/* Referral Code */}
                <div className="animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
                  <label className="block text-sm font-medium text-white/70 mb-2">
                    {t.auth.referralCode} <span className="text-white/30">({t.common.optional})</span>
                  </label>
                  <div className="relative group">
                    <Gift className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-violet-400/60 group-focus-within:text-violet-400 transition-colors" />
                    <input
                      type="text"
                      value={referralCode}
                      onChange={(e) => setReferralCode(e.target.value.toUpperCase())}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pr-12 pl-4 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                      placeholder={t.auth.referralCodePlaceholder}
                      dir="ltr"
                    />
                  </div>
                </div>

                {/* Terms Checkbox */}
                <div className="flex items-start gap-3 animate-fade-in-up" style={{ animationDelay: '0.35s' }}>
                  <input
                    type="checkbox"
                    id="terms"
                    checked={acceptTerms}
                    onChange={(e) => setAcceptTerms(e.target.checked)}
                    className="mt-1 w-4 h-4 rounded border-white/20 bg-white/5 text-violet-500 focus:ring-violet-500/20"
                  />
                  <label htmlFor="terms" className="text-sm text-white/50">
                    {t.auth.agreeToTerms}{' '}
                    <Link to="/terms" className="text-violet-400 hover:text-violet-300">
                      {t.auth.termsOfService}
                    </Link>
                    {' '}{t.common.and}{' '}
                    <Link to="/privacy" className="text-violet-400 hover:text-violet-300">
                      {t.auth.privacyPolicy}
                    </Link>
                  </label>
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isLoading || !acceptTerms}
                  className="w-full py-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-semibold shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed animate-fade-in-up"
                  style={{ animationDelay: '0.4s' }}
                >
                  {isLoading ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <span>{t.auth.register}</span>
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </form>
            </>
          ) : (
            <>
              {/* OTP Form */}
              <form onSubmit={handleOtpSubmit} className="space-y-6">
                {/* Error Message */}
                {error && (
                  <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-400 animate-shake">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm">{error}</span>
                  </div>
                )}

                {/* OTP Inputs */}
                <div className="flex justify-center gap-2" dir="ltr" style={{ direction: "ltr" }}>
                  {otp.map((digit, index) => (
                    <input
                      key={index}
                      ref={(el) => (otpInputRefs.current[index] = el)}
                      type="text"
                      inputMode="numeric"
                      maxLength={6}
                      value={digit}
                      onChange={(e) => handleOtpChange(index, e.target.value.replace(/\D/g, ''))}
                      onKeyDown={(e) => handleOtpKeyDown(index, e)}
                      className="w-12 h-14 text-center text-xl font-bold bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                    />
                  ))}
                </div>

                {/* Resend OTP */}
                <div className="text-center">
                  {resendTimer > 0 ? (
                    <p className="text-white/50 text-sm">
                      إعادة الإرسال بعد {resendTimer} ثانية
                    </p>
                  ) : (
                    <button
                      type="button"
                      onClick={handleResendOtp}
                      disabled={isLoading}
                      className="text-violet-400 hover:text-violet-300 text-sm transition-colors"
                    >
                      إعادة إرسال الرمز
                    </button>
                  )}
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-semibold shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <KeyRound className="w-5 h-5" />
                      <span>تأكيد التسجيل</span>
                    </>
                  )}
                </button>

                {/* Back Button */}
                <button
                  type="button"
                  onClick={handleBackToRegister}
                  className="w-full py-3 text-white/50 hover:text-white transition-colors text-sm"
                >
                  العودة للتسجيل
                </button>
              </form>
            </>
          )}

          {/* Login Link */}
          <p className="mt-8 text-center text-white/50 animate-fade-in-up" style={{ animationDelay: '0.45s' }}>
            {t.auth.haveAccount}{' '}
            <Link to="/login" className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
              {t.auth.login}
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Register;
