import React, { useEffect, useState, useCallback } from 'react';
import { MessageCircle, X, Send, Headphones } from 'lucide-react';

declare global {
  interface Window {
    chatwootSDK: {
      run: (config: { websiteToken: string; baseUrl: string }) => void;
    };
    $chatwoot: {
      toggle: (state?: 'open' | 'close') => void;
      setUser: (identifier: string, user: object) => void;
      isOpen: boolean;
    };
    chatwootSettings: {
      hideMessageBubble: boolean;
      position: string;
      locale: string;
      type: string;
    };
  }
}

const ChatWidget: React.FC = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);

  useEffect(() => {
    // إعدادات Chatwoot قبل التحميل
    window.chatwootSettings = {
      hideMessageBubble: true, // إخفاء الزر الافتراضي
      position: 'left',
      locale: 'ar',
      type: 'standard'
    };

    // تحميل Chatwoot SDK
    const loadChatwoot = () => {
      const BASE_URL = 'https://support.asinax.cloud';
      
      // التحقق من أن SDK لم يتم تحميله مسبقاً
      if (window.$chatwoot) {
        setIsLoaded(true);
        return;
      }

      const script = document.createElement('script');
      script.src = `${BASE_URL}/packs/js/sdk.js`;
      script.async = true;
      script.defer = true;

      script.onload = () => {
        if (window.chatwootSDK) {
          window.chatwootSDK.run({
            websiteToken: 'cyxLY7zioAGCh4c5hPyxwLX3',
            baseUrl: BASE_URL
          });
        }

        // انتظار تهيئة Chatwoot
        const checkReady = setInterval(() => {
          if (window.$chatwoot) {
            setIsLoaded(true);
            clearInterval(checkReady);
          }
        }, 100);
        
        // إيقاف الانتظار بعد 5 ثواني
        setTimeout(() => clearInterval(checkReady), 5000);
      };

      document.body.appendChild(script);
    };

    loadChatwoot();

    // إخفاء التلميح بعد 10 ثواني
    const tooltipTimer = setTimeout(() => setShowTooltip(false), 10000);

    return () => {
      clearTimeout(tooltipTimer);
    };
  }, []);

  const toggleChat = useCallback(() => {
    if (window.$chatwoot) {
      if (isOpen) {
        window.$chatwoot.toggle('close');
      } else {
        window.$chatwoot.toggle('open');
      }
      setIsOpen(!isOpen);
      setShowTooltip(false);
    } else {
      // فتح صفحة الدعم إذا لم يتم تحميل Chatwoot
      window.open('https://support.asinax.cloud/widget?website_token=cyxLY7zioAGCh4c5hPyxwLX3', '_blank', 'width=400,height=600');
    }
  }, [isOpen]);

  // إخفاء زر Chatwoot الافتراضي فقط (وليس نافذة الدردشة)
  useEffect(() => {
    const style = document.createElement('style');
    style.id = 'chatwoot-custom-style';
    style.textContent = `
      .woot-widget-bubble,
      .woot--bubble-holder {
        display: none !important;
      }
      .woot-widget-holder {
        z-index: 9998 !important;
      }
    `;
    document.head.appendChild(style);

    return () => {
      const existingStyle = document.getElementById('chatwoot-custom-style');
      if (existingStyle) {
        document.head.removeChild(existingStyle);
      }
    };
  }, []);

  return (
    <>
      {/* التلميح */}
      {showTooltip && !isOpen && (
        <div 
          className="fixed bottom-24 left-6 z-[9998] bg-white dark:bg-gray-800 text-gray-800 dark:text-white px-4 py-3 rounded-xl shadow-2xl text-sm font-medium transition-all duration-500 animate-bounce"
          style={{
            boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          }}
        >
          <div className="flex items-center gap-2">
            <Headphones className="w-5 h-5 text-purple-500" />
            <span className="font-bold">تحتاج مساعدة؟</span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">فريق الدعم متواجد الآن</p>
          <div className="absolute -bottom-2 left-6 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-white dark:border-t-gray-800" />
        </div>
      )}

      {/* زر الدردشة المخصص */}
      <button
        onClick={toggleChat}
        className="fixed bottom-6 left-6 z-[9999] flex items-center justify-center w-16 h-16 rounded-full text-white shadow-2xl transition-all duration-300 hover:scale-110 focus:outline-none focus:ring-4 focus:ring-purple-300"
        style={{
          background: 'linear-gradient(135deg, #8B5CF6 0%, #3B82F6 50%, #06B6D4 100%)',
          boxShadow: '0 8px 32px rgba(139, 92, 246, 0.5), 0 0 0 4px rgba(139, 92, 246, 0.1)',
        }}
        aria-label={isOpen ? "إغلاق الدردشة" : "فتح الدردشة"}
      >
        {isOpen ? (
          <X className="w-7 h-7" />
        ) : (
          <>
            <MessageCircle className="w-7 h-7" />
            {/* نقطة الإشعار الخضراء */}
            <span 
              className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-2 border-white flex items-center justify-center"
              style={{
                boxShadow: '0 0 10px rgba(34, 197, 94, 0.5)',
              }}
            >
              <span className="w-2 h-2 bg-white rounded-full animate-ping" />
            </span>
          </>
        )}
      </button>

      {/* نص تحت الزر */}
      {!isOpen && (
        <div className="fixed bottom-1 left-6 z-[9998] text-center w-16">
          <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">دردشة</span>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
