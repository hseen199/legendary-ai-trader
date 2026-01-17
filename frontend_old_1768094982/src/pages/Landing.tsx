import React from 'react';
import { Link } from 'react-router-dom';
import {
  TrendingUp,
  Shield,
  Clock,
  Users,
  BarChart3,
  Zap,
  CheckCircle,
  ArrowLeft,
} from 'lucide-react';

const Landing: React.FC = () => {
  const features = [
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: 'تداول آلي بالذكاء الاصطناعي',
      description: 'بوت تداول متطور يعمل على مدار الساعة لتحقيق أفضل العوائد',
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: 'أمان عالي',
      description: 'حماية متقدمة لأموالك مع تشفير من الدرجة الأولى',
    },
    {
      icon: <Clock className="w-6 h-6" />,
      title: 'سحب سهل',
      description: 'اسحب أموالك في أي وقت بعد فترة القفل القصيرة',
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: 'استثمار جماعي',
      description: 'استفد من قوة الاستثمار الجماعي مع مستثمرين آخرين',
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: 'شفافية كاملة',
      description: 'تابع جميع الصفقات والأداء في الوقت الفعلي',
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'بدء سريع',
      description: 'سجل وأودع وابدأ الاستثمار في دقائق معدودة',
    },
  ];

  const steps = [
    {
      number: '1',
      title: 'إنشاء حساب',
      description: 'سجل حساباً جديداً بالإيميل فقط',
    },
    {
      number: '2',
      title: 'إيداع الأموال',
      description: 'أودع USDT في محفظتك الخاصة',
    },
    {
      number: '3',
      title: 'البوت يعمل',
      description: 'البوت يتداول تلقائياً نيابة عنك',
    },
    {
      number: '4',
      title: 'اجمع الأرباح',
      description: 'تابع أرباحك واسحبها متى شئت',
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            استثمر بذكاء مع الذكاء الاصطناعي
          </h1>
          <p className="text-xl md:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto">
            منصة استثمار جماعي في العملات الرقمية مدعومة ببوت تداول ذكي يعمل على مدار الساعة
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/register"
              className="bg-white text-primary-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-primary-50 transition-colors"
            >
              ابدأ الاستثمار الآن
            </Link>
            <Link
              to="/how-it-works"
              className="border-2 border-white text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-white/10 transition-colors flex items-center justify-center gap-2"
            >
              كيف يعمل؟
              <ArrowLeft className="w-5 h-5" />
            </Link>
          </div>
        </div>

        {/* Stats */}
        <div className="max-w-4xl mx-auto mt-16 grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="text-center">
            <p className="text-4xl font-bold">+500</p>
            <p className="text-primary-200">مستثمر نشط</p>
          </div>
          <div className="text-center">
            <p className="text-4xl font-bold">$2M+</p>
            <p className="text-primary-200">إجمالي الأصول</p>
          </div>
          <div className="text-center">
            <p className="text-4xl font-bold">+25%</p>
            <p className="text-primary-200">متوسط العائد السنوي</p>
          </div>
          <div className="text-center">
            <p className="text-4xl font-bold">24/7</p>
            <p className="text-primary-200">تداول مستمر</p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-gray-50 dark:bg-dark-200">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white mb-4">
            لماذا تختارنا؟
          </h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-12 max-w-2xl mx-auto">
            نقدم لك أفضل تجربة استثمار في العملات الرقمية مع ميزات متقدمة وأمان عالي
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="card p-6 hover:shadow-lg transition-shadow"
              >
                <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900 rounded-lg flex items-center justify-center text-primary-600 dark:text-primary-400 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white mb-4">
            كيف يعمل؟
          </h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-12">
            أربع خطوات بسيطة لبدء رحلة الاستثمار
          </p>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="text-center relative">
                <div className="w-16 h-16 bg-primary-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                  {step.number}
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {step.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {step.description}
                </p>
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-8 left-0 w-full h-0.5 bg-primary-200 dark:bg-primary-800 -z-10" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-primary-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            جاهز لبدء الاستثمار؟
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            انضم إلى مئات المستثمرين الذين يحققون أرباحاً مع منصتنا
          </p>
          <Link
            to="/register"
            className="inline-block bg-white text-primary-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-primary-50 transition-colors"
          >
            إنشاء حساب مجاني
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold">C</span>
                </div>
                <span className="font-bold text-xl text-white">CryptoInvest</span>
              </div>
              <p className="text-sm">
                منصة استثمار جماعي في العملات الرقمية
              </p>
            </div>

            <div>
              <h4 className="font-bold text-white mb-4">روابط سريعة</h4>
              <ul className="space-y-2 text-sm">
                <li><Link to="/how-it-works" className="hover:text-white">كيف يعمل</Link></li>
                <li><Link to="/faq" className="hover:text-white">الأسئلة الشائعة</Link></li>
                <li><Link to="/about" className="hover:text-white">من نحن</Link></li>
              </ul>
            </div>

            <div>
              <h4 className="font-bold text-white mb-4">قانوني</h4>
              <ul className="space-y-2 text-sm">
                <li><Link to="/terms" className="hover:text-white">شروط الاستخدام</Link></li>
                <li><Link to="/privacy" className="hover:text-white">سياسة الخصوصية</Link></li>
                <li><Link to="/disclaimer" className="hover:text-white">إخلاء المسؤولية</Link></li>
              </ul>
            </div>

            <div>
              <h4 className="font-bold text-white mb-4">تواصل معنا</h4>
              <p className="text-sm">support@cryptoinvest.com</p>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm">
            <p>© 2024 CryptoInvest. جميع الحقوق محفوظة.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
