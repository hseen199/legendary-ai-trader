import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { AlertTriangle, LogOut, User } from "lucide-react";

interface ImpersonatedUser {
  id: number;
  email: string;
  full_name?: string;
}

export default function ImpersonationBanner() {
  const [, setLocation] = useLocation();
  const [impersonatedUser, setImpersonatedUser] = useState<ImpersonatedUser | null>(null);

  useEffect(() => {
    const checkImpersonation = () => {
      const userData = localStorage.getItem('impersonating_user');
      if (userData) {
        setImpersonatedUser(JSON.parse(userData));
      } else {
        setImpersonatedUser(null);
      }
    };

    checkImpersonation();
    // Listen for storage changes
    window.addEventListener('storage', checkImpersonation);
    return () => window.removeEventListener('storage', checkImpersonation);
  }, []);

  const handleExitImpersonation = () => {
    // Restore admin token
    const adminToken = localStorage.getItem('admin_token_backup');
    if (adminToken) {
      localStorage.setItem('token', adminToken);
      localStorage.removeItem('admin_token_backup');
    }
    localStorage.removeItem('impersonating_user');
    setImpersonatedUser(null);
    setLocation('/admin/users');
  };

  if (!impersonatedUser) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-[100] bg-gradient-to-r from-amber-500 to-orange-500 text-white py-2 px-4 shadow-lg">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 animate-pulse" />
          <span className="font-medium">
            أنت تشاهد المنصة كمستخدم: 
            <span className="font-bold mx-2">{impersonatedUser.full_name || impersonatedUser.email}</span>
          </span>
        </div>
        <button
          onClick={handleExitImpersonation}
          className="flex items-center gap-2 px-4 py-1.5 bg-white/20 hover:bg-white/30 rounded-lg transition-colors font-medium"
        >
          <LogOut className="w-4 h-4" />
          العودة للوحة الأدمن
        </button>
      </div>
    </div>
  );
}
