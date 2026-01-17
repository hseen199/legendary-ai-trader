import { Link } from "react-router-dom";
import { Bot, ArrowRight, FileText, AlertTriangle, CheckCircle, XCircle, Scale, Clock } from "lucide-react";
import { useLanguage } from "@/lib/i18n";

const Terms = () => {
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
              <FileText className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
              شروط الخدمة
            </h1>
            <p className="text-white/60">آخر تحديث: يناير 2025</p>
          </div>

          {/* Content Sections */}
          <div className="space-y-8">
            {/* Introduction */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Scale className="w-5 h-5 text-violet-400" />
                مقدمة
              </h2>
              <p className="text-white/70 leading-relaxed">
                مرحباً بك في منصة ASINAX للتداول الذكي. باستخدامك لهذه المنصة، فإنك توافق على الالتزام بهذه الشروط والأحكام. يرجى قراءتها بعناية قبل استخدام خدماتنا.
              </p>
            </section>

            {/* Eligibility */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                شروط الأهلية
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>يجب أن يكون عمرك 18 عاماً على الأقل</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>يجب أن تكون مؤهلاً قانونياً لاستخدام خدمات التداول في بلدك</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>يجب تقديم معلومات صحيحة ودقيقة عند التسجيل</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>يجب الحفاظ على سرية بيانات حسابك</span>
                </li>
              </ul>
            </section>

            {/* Services */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Bot className="w-5 h-5 text-violet-400" />
                الخدمات المقدمة
              </h2>
              <p className="text-white/70 leading-relaxed mb-4">
                توفر ASINAX منصة للتداول الجماعي الذكي تشمل:
              </p>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>وكيل تداول ذكي يعمل على مدار الساعة</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>نظام إدارة المحافظ الجماعية (NAV)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>خدمات الإيداع والسحب بالعملات الرقمية</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>برنامج الإحالات والمكافآت</span>
                </li>
              </ul>
            </section>

            {/* Risk Warning */}
            <section className="bg-[rgba(239,68,68,0.1)] backdrop-blur-xl border border-red-500/30 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                تحذير المخاطر
              </h2>
              <p className="text-white/70 leading-relaxed mb-4">
                <strong className="text-red-400">تنبيه هام:</strong> تداول العملات الرقمية ينطوي على مخاطر عالية وقد يؤدي إلى خسارة رأس المال بالكامل.
              </p>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>الأداء السابق لا يضمن النتائج المستقبلية</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>لا تستثمر أكثر مما يمكنك تحمل خسارته</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>أسعار العملات الرقمية شديدة التقلب</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>نوصي بالتشاور مع مستشار مالي مؤهل</span>
                </li>
              </ul>
            </section>

            {/* Deposits & Withdrawals */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Clock className="w-5 h-5 text-violet-400" />
                الإيداع والسحب
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">الحد الأدنى للإيداع:</strong> 100 USDC</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">فترة القفل:</strong> 7 أيام من آخر إيداع قبل السحب</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">معالجة السحب:</strong> خلال 24-48 ساعة عمل</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-violet-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span><strong className="text-white">الشبكات المدعومة:</strong> BSC (BEP20) و Solana</span>
                </li>
              </ul>
            </section>

            {/* Prohibited Activities */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <XCircle className="w-5 h-5 text-red-400" />
                الأنشطة المحظورة
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>استخدام المنصة لغسيل الأموال أو أي نشاط غير قانوني</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>إنشاء حسابات متعددة لنفس الشخص</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>محاولة اختراق أو التلاعب بالنظام</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>مشاركة بيانات الحساب مع أطراف ثالثة</span>
                </li>
              </ul>
            </section>

            {/* Limitation of Liability */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Scale className="w-5 h-5 text-violet-400" />
                حدود المسؤولية
              </h2>
              <p className="text-white/70 leading-relaxed">
                ASINAX غير مسؤولة عن أي خسائر ناتجة عن تقلبات السوق، أو أخطاء المستخدم، أو ظروف خارجة عن السيطرة. نحن نبذل قصارى جهدنا لتوفير خدمة موثوقة، لكن لا نضمن تحقيق أرباح أو نتائج محددة.
              </p>
            </section>

            {/* Changes to Terms */}
            <section className="bg-[rgba(15,15,25,0.6)] backdrop-blur-xl border border-violet-500/15 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
                <FileText className="w-5 h-5 text-violet-400" />
                تعديل الشروط
              </h2>
              <p className="text-white/70 leading-relaxed">
                نحتفظ بالحق في تعديل هذه الشروط في أي وقت. سيتم إخطارك بأي تغييرات جوهرية عبر البريد الإلكتروني أو من خلال إشعار على المنصة. استمرارك في استخدام الخدمة بعد التعديلات يعني موافقتك على الشروط الجديدة.
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

export default Terms;
