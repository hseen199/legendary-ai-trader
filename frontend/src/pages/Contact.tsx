import { Link } from "react-router-dom";
import { Bot, ArrowRight, Mail, MessageSquare, Clock, Globe, Send } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import { useLanguage } from '@/lib/i18n';

const Contact = () => {
  const { t, language } = useLanguage();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    toast.success("تم إرسال رسالتك بنجاح! سنتواصل معك قريباً.");
    setFormData({ name: "", email: "", subject: "", message: "" });
    setIsSubmitting(false);
  };

  return (
    <div className="min-h-screen bg-[#08080c] text-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#08080c]/80 backdrop-blur-xl border-b border-violet-500/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(139,92,246,0.4)]">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <span className="font-bold text-xl bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">ASINAX</span>
          </Link>
          <Link to="/" className="flex items-center gap-2 text-violet-400 hover:text-violet-300 transition-colors">
            <ArrowRight className="w-4 h-4" />
            العودة للرئيسية
          </Link>
        </div>
      </nav>

      {/* Content */}
      <div className="pt-24 pb-16 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl mb-6 shadow-[0_0_30px_rgba(139,92,246,0.4)]">
              <Mail className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              تواصل معنا
            </h1>
            <p className="text-white/60">نحن هنا لمساعدتك. تواصل معنا في أي وقت!</p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Contact Info */}
            <div className="space-y-6">
              <div className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
                <h2 className="text-xl font-bold mb-6">معلومات التواصل</h2>
                
                <div className="space-y-4">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-violet-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Mail className="w-5 h-5 text-violet-400" />
                    </div>
                    <div>
                      <h3 className="font-medium text-white">{t.settings.email}</h3>
                      <p className="text-white/60 text-sm">support@asinax.com</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
                      <MessageSquare className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                      <h3 className="font-medium text-white">{t.support.title}</h3>
                      <p className="text-white/60 text-sm">متاح من خلال لوحة التحكم</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-amber-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Clock className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                      <h3 className="font-medium text-white">ساعات العمل</h3>
                      <p className="text-white/60 text-sm">24/7 - على مدار الساعة</p>
                    </div>
                  </div>

                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-pink-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Globe className="w-5 h-5 text-pink-400" />
                    </div>
                    <div>
                      <h3 className="font-medium text-white">الموقع</h3>
                      <p className="text-white/60 text-sm">منصة عالمية - خدمة رقمية</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* FAQ Link */}
              <div className="bg-gradient-to-br from-violet-500/20 to-purple-500/20 backdrop-blur-xl border border-violet-500/30 rounded-2xl p-6">
                <h3 className="font-bold mb-2">هل لديك سؤال شائع؟</h3>
                <p className="text-white/60 text-sm mb-4">
                  قد تجد إجابتك في قسم الأسئلة الشائعة
                </p>
                <Link 
                  to="/support" 
                  className="inline-flex items-center gap-2 text-violet-400 hover:text-violet-300 transition-colors font-medium"
                >
                  زيارة صفحة الدعم
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            </div>

            {/* Contact Form */}
            <div className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-6">أرسل رسالة</h2>
              
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-white/70 mb-2">الاسم</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                    className="w-full px-4 py-3 bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl text-white placeholder-white/30 focus:border-violet-500/50 focus:outline-none transition-colors"
                    placeholder="أدخل اسمك"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/70 mb-2">{t.settings.email}</label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    required
                    className="w-full px-4 py-3 bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl text-white placeholder-white/30 focus:border-violet-500/50 focus:outline-none transition-colors"
                    placeholder="أدخل بريدك الإلكتروني"
                    dir="ltr"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/70 mb-2">الموضوع</label>
                  <input
                    type="text"
                    value={formData.subject}
                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                    required
                    className="w-full px-4 py-3 bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl text-white placeholder-white/30 focus:border-violet-500/50 focus:outline-none transition-colors"
                    placeholder="موضوع الرسالة"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-white/70 mb-2">الرسالة</label>
                  <textarea
                    value={formData.message}
                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                    required
                    rows={5}
                    className="w-full px-4 py-3 bg-[rgba(15,15,25,0.8)] border border-violet-500/20 rounded-xl text-white placeholder-white/30 focus:border-violet-500/50 focus:outline-none transition-colors resize-none"
                    placeholder="اكتب رسالتك هنا..."
                  />
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white font-medium rounded-xl shadow-[0_4px_20px_rgba(139,92,246,0.4)] hover:shadow-[0_6px_30px_rgba(139,92,246,0.5)] transition-all duration-300 disabled:opacity-50"
                >
                  {isSubmitting ? (
                    <>جاري الإرسال...</>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      إرسال الرسالة
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-violet-500/10 bg-[#08080c]/80">
        <div className="max-w-4xl mx-auto text-center text-white/40 text-sm">
          © 2025 ASINAX. جميع الحقوق محفوظة.
        </div>
      </footer>
    </div>
  );
};

export default Contact;
