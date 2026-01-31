import { useState, useEffect } from "react";
import { AlertTriangle, ArrowLeft } from "lucide-react";

export default function ImpersonationBanner() {
  const [isImpersonating, setIsImpersonating] = useState(false);
  const [impersonatedEmail, setImpersonatedEmail] = useState<string>("");

  useEffect(() => {
    const checkImpersonation = () => {
      const impersonating = localStorage.getItem('impersonating') === 'true';
      const email = localStorage.getItem('impersonated_user') || '';
      setIsImpersonating(impersonating);
      setImpersonatedEmail(email);
    };

    checkImpersonation();
    
    // Listen for storage changes
    window.addEventListener('storage', checkImpersonation);
    
    // Check periodically
    const interval = setInterval(checkImpersonation, 1000);
    
    return () => {
      window.removeEventListener('storage', checkImpersonation);
      clearInterval(interval);
    };
  }, []);

  const handleExitImpersonation = () => {
    // Restore admin token
    const adminToken = localStorage.getItem('admin_token_backup');
    if (adminToken) {
      localStorage.setItem('token', adminToken);
      localStorage.removeItem('admin_token_backup');
    }
    localStorage.removeItem('impersonating');
    localStorage.removeItem('impersonated_user');
    setIsImpersonating(false);
    
    // Force reload to refresh auth state
    window.location.href = '/admin';
  };

  if (!isImpersonating) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-[100] bg-gradient-to-r from-amber-500 to-orange-500 text-white py-2 px-4 shadow-lg">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 animate-pulse" />
          <span className="font-medium">
            ⚠️ أنت تتصفح المنصة كـ: 
            <span className="font-bold mx-2 bg-white/20 px-2 py-0.5 rounded">{impersonatedEmail}</span>
          </span>
        </div>
        <button
          onClick={handleExitImpersonation}
          className="flex items-center gap-2 px-4 py-1.5 bg-black text-white hover:bg-gray-800 rounded-lg transition-colors font-medium"
        >
          <ArrowLeft className="w-4 h-4" />
          العودة للوحة الأدمن
        </button>
      </div>
    </div>
  );
}
