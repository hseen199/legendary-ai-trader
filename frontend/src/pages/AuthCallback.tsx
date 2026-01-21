import { BotIcon } from "../components/BotIcon";
import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {  CheckCircle, XCircle, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { useLanguage } from '@/lib/i18n';

const AuthCallback: React.FC = () => {
  const { t, language } = useLanguage();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { refreshUser } = useAuth();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('جاري تسجيل الدخول...');

  useEffect(() => {
    const handleCallback = async () => {
      const token = searchParams.get('token');
      const error = searchParams.get('error');

      if (error) {
        setStatus('error');
        setMessage('فشل تسجيل الدخول. يرجى المحاولة مرة أخرى.');
        toast.error('فشل تسجيل الدخول');
        setTimeout(() => navigate('/login'), 2000);
        return;
      }

      if (token) {
        try {
          // Store token and refresh user
          localStorage.setItem('token', token);
          await refreshUser();
          setStatus('success');
          setMessage('تم تسجيل الدخول بنجاح!');
          toast.success('مرحباً بك في ASINAX!');
          setTimeout(() => navigate('/dashboard'), 1500);
        } catch (err) {
          setStatus('error');
          setMessage('حدث خطأ أثناء تسجيل الدخول.');
          toast.error('خطأ في المصادقة');
          setTimeout(() => navigate('/login'), 2000);
        }
      } else {
        setStatus('error');
        setMessage('لم يتم العثور على رمز الوصول.');
        toast.error('خطأ في المصادقة');
        setTimeout(() => navigate('/login'), 2000);
      }
    };

    handleCallback();
  }, [searchParams, navigate, refreshUser]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#08080c] relative overflow-hidden">
      {/* Animated Background Orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute w-[500px] h-[500px] rounded-full bg-violet-500/20 blur-[100px] top-[10%] left-[10%] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute w-[400px] h-[400px] rounded-full bg-pink-500/15 blur-[100px] bottom-[20%] right-[10%] animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
      </div>

      <div className="relative z-10 text-center">
        {/* Logo */}
        <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-[0_0_40px_rgba(139,92,246,0.5)]">
          <BotIcon className="w-10 h-10 text-white" />
        </div>

        {/* Status Icon */}
        <div className="mb-4">
          {status === 'loading' && (
            <Loader2 className="w-12 h-12 text-violet-400 animate-spin mx-auto" />
          )}
          {status === 'success' && (
            <CheckCircle className="w-12 h-12 text-green-400 mx-auto animate-bounce" />
          )}
          {status === 'error' && (
            <XCircle className="w-12 h-12 text-red-400 mx-auto" />
          )}
        </div>

        {/* Message */}
        <h2 className="text-2xl font-bold text-white mb-2">{message}</h2>
        
        {status === 'loading' && (
          <p className="text-white/50">يرجى الانتظار...</p>
        )}
        
        {status === 'success' && (
          <p className="text-white/50">جاري تحويلك إلى لوحة التحكم...</p>
        )}
        
        {status === 'error' && (
          <p className="text-white/50">جاري تحويلك إلى صفحة تسجيل الدخول...</p>
        )}
      </div>
    </div>
  );
};

export default AuthCallback;
