import { BotIcon } from "../components/BotIcon";
import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '@/lib/i18n';
import { Mail, Lock, Eye, EyeOff, User, Phone, Bot, ArrowRight, Gift, KeyRound, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import api from '@/services/api';

const Register: React.FC = () => {
  const [searchParams] = useSearchParams();
  const referralCode = searchParams.get('ref') || '';
  const { t } = useLanguage();
  const { refreshUser } = useAuth();
  const navigate = useNavigate();
  
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
  const [error, setError] = useState('');
  
  // OTP State
  const [step, setStep] = useState<'register' | 'otp'>('register');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [resendTimer, setResendTimer] = useState(0);
  const otpInputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Resend timer countdown
  useEffect(() => {
    if (resendTimer > 0) {
      const timer = setTimeout(() => setResendTimer(resendTimer - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [resendTimer]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle OTP input change
  const handleOtpChange = (index: number, value: string) => {
    if (value.length > 1) {
      // Handle paste
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
      
      // Move to next input
      if (value && index < 5) {
        otpInputRefs.current[index + 1]?.focus();
      }
    }
  };

  // Handle OTP key down
  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
    }
  };

  // Step 1: Submit registration form
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
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
      const response = await api.post('/auth/register', {
        email: formData.email.toLowerCase(),
        password: formData.password,
        full_name: formData.full_name,
        phone: formData.phone,
        referral_code: formData.referral_code
      });
      
      if (response.data.requires_otp) {
        setStep('otp');
        setResendTimer(60);
        toast.success('تم إرسال رمز التحقق إلى بريدك الإلكتروني');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || t.common.error;
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Step 2: Verify OTP
  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const otpCode = otp.join('');
    
    if (otpCode.length !== 6) {
      toast.error('يرجى إدخال رمز التحقق كاملاً');
      return;
    }
    
    setError('');
    setIsLoading(true);
    
    try {
      const response = await api.post('/auth/register/verify', {
        email: formData.email.toLowerCase(),
        otp: otpCode
      });
      
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        await refreshUser();
        toast.success('تم إنشاء حسابك بنجاح! مرحباً بك في ASINAX');
        navigate('/dashboard');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'رمز التحقق غير صحيح';
      setError(errorMessage);
      toast.error(errorMessage);
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
      await api.post('/auth/resend-otp', { email: formData.email.toLowerCase() });
      setResendTimer(60);
      toast.success('تم إرسال رمز التحقق مرة أخرى');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'فشل في إعادة إرسال الرمز';
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Go back to registration form
  const handleBackToRegister = () => {
    setStep('register');
    setOtp(['', '', '', '', '', '']);
    setError('');
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
                <BotIcon className="w-8 h-8 text-white" />
              </div>
            </Link>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              {step === 'register' ? t.auth.registerTitle : 'تأكيد البريد الإلكتروني'}
            </h1>
            <p className="text-white/50 mt-2">
              {step === 'register' ? t.auth.joinPlatform : 'أدخل رمز التحقق المرسل إلى بريدك'}
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-400 animate-shake">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {step === 'register' ? (
            <>
              {/* Registration Form */}
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
            </>
          ) : (
            <>
              {/* OTP Verification Form */}
              {/* Email Display */}
              <div className="mb-6 p-4 rounded-xl bg-violet-500/10 border border-violet-500/20">
                <p className="text-white/60 text-sm text-center">
                  تم إرسال رمز التحقق إلى
                </p>
                <p className="text-violet-400 text-center font-medium mt-1" dir="ltr">
                  {formData.email}
                </p>
              </div>

              <form onSubmit={handleOtpSubmit} className="space-y-6">
                {/* OTP Inputs */}
                <div className="animate-fade-in-up">
                  <label className="block text-sm font-medium text-white/70 mb-4 text-center">
                    أدخل رمز التحقق المكون من 6 أرقام
                  </label>
                  <div className="flex justify-center gap-3" dir="ltr">
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
                        className="w-12 h-14 text-center text-2xl font-bold bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl text-white focus:outline-none focus:border-violet-500/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] transition-all duration-300"
                        autoFocus={index === 0}
                      />
                    ))}
                  </div>
                </div>

                {/* Resend OTP */}
                <div className="text-center animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                  {resendTimer > 0 ? (
                    <p className="text-white/50 text-sm">
                      إعادة الإرسال بعد <span className="text-violet-400 font-medium">{resendTimer}</span> ثانية
                    </p>
                  ) : (
                    <button
                      type="button"
                      onClick={handleResendOtp}
                      disabled={isLoading}
                      className="text-violet-400 hover:text-violet-300 text-sm font-medium transition-colors flex items-center gap-2 mx-auto"
                    >
                      <RefreshCw className="w-4 h-4" />
                      إعادة إرسال الرمز
                    </button>
                  )}
                </div>

                {/* Submit OTP */}
                <button
                  type="submit"
                  disabled={isLoading || otp.join('').length !== 6}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white font-semibold shadow-[0_8px_30px_rgba(139,92,246,0.4)] hover:shadow-[0_12px_40px_rgba(139,92,246,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group animate-fade-in-up"
                  style={{ animationDelay: '0.2s' }}
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>جاري التحقق...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-5 h-5" />
                      <span>تأكيد وإنشاء الحساب</span>
                    </>
                  )}
                </button>

                {/* Back Button */}
                <button
                  type="button"
                  onClick={handleBackToRegister}
                  className="w-full py-3 rounded-xl border border-violet-500/20 text-white/70 hover:text-white hover:border-violet-500/40 transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: '0.3s' }}
                >
                  العودة للتسجيل
                </button>
              </form>
            </>
          )}
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
