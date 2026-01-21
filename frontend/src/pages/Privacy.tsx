import { Link } from "react-router-dom";
import {  ArrowRight, Shield, Lock, Eye, Database, UserCheck, Mail } from "lucide-react";
import { useLanguage } from "@/lib/i18n";

const Privacy = () => {
  const { t, language } = useLanguage();

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
              <Shield className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              سياسة الخصوصية
            </h1>
            <p className="text-white/60">آخر تحديث: يناير 2025</p>
          </div>

          {/* Content Sections */}
          <div className="space-y-8">
            {/* Introduction */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Lock className="w-5 h-5 text-violet-400" />
                مقدمة
              </h2>
              <p className="text-white/70 leading-relaxed">
                نحن في ASINAX نلتزم بحماية خصوصيتك وبياناتك الشخصية. توضح هذه السياسة كيفية جمع واستخدام وحماية معلوماتك عند استخدام منصتنا للتداول الذكي.
              </p>
            </section>

            {/* Data Collection */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Database className="w-5 h-5 text-violet-400" />
                البيانات التي نجمعها
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">معلومات الحساب:</strong> الاسم، البريد الإلكتروني، وكلمة المرور المشفرة</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">بيانات المعاملات:</strong> سجل الإيداعات والسحوبات والصفقات</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">بيانات الاستخدام:</strong> معلومات الجهاز، عنوان IP، وسجل النشاط</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">عناوين المحافظ:</strong> عناوين العملات الرقمية للإيداع والسحب</span>
                </li>
              </ul>
            </section>

            {/* Data Usage */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Eye className="w-5 h-5 text-violet-400" />
                كيف نستخدم بياناتك
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>تقديم خدمات التداول الذكي وإدارة حسابك</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>معالجة المعاملات المالية والتحقق منها</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>تحسين تجربة المستخدم وتطوير خدماتنا</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>التواصل معك بخصوص حسابك والتحديثات المهمة</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>الامتثال للمتطلبات القانونية والتنظيمية</span>
                </li>
              </ul>
            </section>

            {/* Data Protection */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Shield className="w-5 h-5 text-violet-400" />
                حماية البيانات
              </h2>
              <p className="text-white/70 leading-relaxed mb-4">
                نستخدم أحدث تقنيات الأمان لحماية بياناتك:
              </p>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>تشفير SSL/TLS لجميع الاتصالات</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>تشفير كلمات المرور باستخدام خوارزميات متقدمة</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>مراقبة أمنية على مدار الساعة</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>نسخ احتياطية منتظمة للبيانات</span>
                </li>
              </ul>
            </section>

            {/* User Rights */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <UserCheck className="w-5 h-5 text-violet-400" />
                حقوقك
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>الوصول إلى بياناتك الشخصية وتحديثها</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>طلب حذف حسابك وبياناتك</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>الاعتراض على معالجة بياناتك</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>تصدير بياناتك بتنسيق قابل للقراءة</span>
                </li>
              </ul>
            </section>

            {/* Contact */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Mail className="w-5 h-5 text-violet-400" />
                تواصل معنا
              </h2>
              <p className="text-white/70 leading-relaxed">
                إذا كان لديك أي أسئلة حول سياسة الخصوصية أو كيفية تعاملنا مع بياناتك، يرجى التواصل معنا عبر صفحة الدعم الفني أو البريد الإلكتروني: support@asinax.com
              </p>
            </section>
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

export default Privacy;
