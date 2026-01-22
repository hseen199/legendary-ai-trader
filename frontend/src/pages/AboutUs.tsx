/**
 * AboutUs Page
 * صفحة من نحن
 */
import React from 'react';
import { 
  Users, 
  Target,
  Shield,
  Zap,
  Globe,
  Award,
  TrendingUp,
  Heart,
  Rocket,
  CheckCircle,
} from 'lucide-react';
import { Card, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { useNavigate } from 'react-router-dom';

interface AboutUsProps {
  language?: 'ar' | 'en';
}

const teamMembers = [
  {
    nameAr: 'فريق التطوير',
    nameEn: 'Development Team',
    roleAr: 'هندسة البرمجيات والذكاء الاصطناعي',
    roleEn: 'Software Engineering & AI',
    icon: <Zap className="h-8 w-8" />,
  },
  {
    nameAr: 'فريق التداول',
    nameEn: 'Trading Team',
    roleAr: 'استراتيجيات التداول والتحليل',
    roleEn: 'Trading Strategies & Analysis',
    icon: <TrendingUp className="h-8 w-8" />,
  },
  {
    nameAr: 'فريق الأمان',
    nameEn: 'Security Team',
    roleAr: 'حماية البيانات والأصول',
    roleEn: 'Data & Asset Protection',
    icon: <Shield className="h-8 w-8" />,
  },
  {
    nameAr: 'فريق الدعم',
    nameEn: 'Support Team',
    roleAr: 'خدمة العملاء على مدار الساعة',
    roleEn: '24/7 Customer Service',
    icon: <Heart className="h-8 w-8" />,
  },
];

const values = [
  {
    icon: <Shield className="h-6 w-6" />,
    titleAr: 'الأمان أولاً',
    titleEn: 'Security First',
    descAr: 'نضع أمان أموالك وبياناتك في المقام الأول باستخدام أحدث تقنيات التشفير والحماية.',
    descEn: 'We prioritize the security of your funds and data using the latest encryption and protection technologies.',
  },
  {
    icon: <Target className="h-6 w-6" />,
    titleAr: 'الشفافية',
    titleEn: 'Transparency',
    descAr: 'نؤمن بالشفافية الكاملة. يمكنك متابعة جميع الصفقات والأداء في الوقت الفعلي.',
    descEn: 'We believe in complete transparency. You can track all trades and performance in real-time.',
  },
  {
    icon: <Zap className="h-6 w-6" />,
    titleAr: 'الابتكار',
    titleEn: 'Innovation',
    descAr: 'نستخدم أحدث تقنيات الذكاء الاصطناعي لتطوير استراتيجيات تداول متقدمة.',
    descEn: 'We use the latest AI technologies to develop advanced trading strategies.',
  },
  {
    icon: <Users className="h-6 w-6" />,
    titleAr: 'المجتمع',
    titleEn: 'Community',
    descAr: 'نبني مجتمعاً من المستثمرين الأذكياء الذين يثقون بالتكنولوجيا.',
    descEn: 'We build a community of smart investors who trust technology.',
  },
];

const milestones = [
  { year: '2024', eventAr: 'تأسيس المنصة', eventEn: 'Platform Founded' },
  { year: '2024', eventAr: 'إطلاق الوكيل الذكي v1', eventEn: 'AI Agent v1 Launch' },
  { year: '2025', eventAr: 'تحديث الوكيل الذكي v2', eventEn: 'AI Agent v2 Update' },
  { year: '2025', eventAr: 'توسيع الخدمات', eventEn: 'Service Expansion' },
];

export function AboutUs({ language = 'ar' }: AboutUsProps) {
  const isRTL = language === 'ar';
  const navigate = useNavigate();

  return (
    <div dir={isRTL ? 'rtl' : 'ltr'} className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative py-20 px-4 bg-gradient-to-b from-primary/10 to-transparent">
        <div className="max-w-4xl mx-auto text-center">
          <Badge className="mb-4" variant="outline">
            {isRTL ? 'من نحن' : 'About Us'}
          </Badge>
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            {isRTL 
              ? 'نحن ASINAX' 
              : 'We are ASINAX'
            }
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            {isRTL 
              ? 'منصة تداول ذكية تستخدم أحدث تقنيات الذكاء الاصطناعي لمساعدتك في تحقيق أهدافك الاستثمارية'
              : 'A smart trading platform using the latest AI technologies to help you achieve your investment goals'
            }
          </p>
          <div className="flex justify-center gap-4">
            <Button size="lg" onClick={() => navigate('/register')}>
              {isRTL ? 'ابدأ الآن' : 'Get Started'}
              <Rocket className="h-5 w-5 mr-2" />
            </Button>
            <Button size="lg" variant="outline" onClick={() => navigate('/support')}>
              {isRTL ? 'تواصل معنا' : 'Contact Us'}
            </Button>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <Badge className="mb-4" variant="secondary">
                {isRTL ? 'مهمتنا' : 'Our Mission'}
              </Badge>
              <h2 className="text-3xl font-bold mb-4">
                {isRTL 
                  ? 'جعل التداول الذكي متاحاً للجميع'
                  : 'Making Smart Trading Accessible to Everyone'
                }
              </h2>
              <p className="text-muted-foreground mb-6">
                {isRTL 
                  ? 'نؤمن بأن الجميع يستحق فرصة للاستثمار الذكي. لذلك طورنا وكيلاً ذكياً يعمل على مدار الساعة لتحقيق أفضل النتائج لمستثمرينا.'
                  : 'We believe everyone deserves a chance at smart investing. That\'s why we developed an AI agent that works around the clock to achieve the best results for our investors.'
                }
              </p>
              <ul className="space-y-3">
                {[
                  isRTL ? 'تداول آلي بدون تدخل' : 'Automated trading without intervention',
                  isRTL ? 'إدارة مخاطر متقدمة' : 'Advanced risk management',
                  isRTL ? 'شفافية كاملة في الأداء' : 'Complete performance transparency',
                  isRTL ? 'دعم فني متواصل' : 'Continuous technical support',
                ].map((item, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-primary" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Card className="p-6 text-center bg-primary/5">
                <p className="text-4xl font-bold text-primary">24/7</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'تداول مستمر' : 'Continuous Trading'}
                </p>
              </Card>
              <Card className="p-6 text-center bg-primary/5">
                <p className="text-4xl font-bold text-primary">AI</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'ذكاء اصطناعي' : 'Artificial Intelligence'}
                </p>
              </Card>
              <Card className="p-6 text-center bg-primary/5">
                <p className="text-4xl font-bold text-primary">100%</p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'شفافية' : 'Transparency'}
                </p>
              </Card>
              <Card className="p-6 text-center bg-primary/5">
                <p className="text-4xl font-bold text-primary">
                  <Globe className="h-10 w-10 mx-auto" />
                </p>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? 'عالمي' : 'Global'}
                </p>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <Badge className="mb-4" variant="secondary">
              {isRTL ? 'قيمنا' : 'Our Values'}
            </Badge>
            <h2 className="text-3xl font-bold">
              {isRTL ? 'ما نؤمن به' : 'What We Believe In'}
            </h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => (
              <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4 text-primary">
                  {value.icon}
                </div>
                <h3 className="font-semibold mb-2">
                  {isRTL ? value.titleAr : value.titleEn}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? value.descAr : value.descEn}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <Badge className="mb-4" variant="secondary">
              {isRTL ? 'فريقنا' : 'Our Team'}
            </Badge>
            <h2 className="text-3xl font-bold">
              {isRTL ? 'خبراء يعملون من أجلك' : 'Experts Working for You'}
            </h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {teamMembers.map((member, index) => (
              <Card key={index} className="p-6 text-center">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center mx-auto mb-4 text-primary">
                  {member.icon}
                </div>
                <h3 className="font-semibold mb-1">
                  {isRTL ? member.nameAr : member.nameEn}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {isRTL ? member.roleAr : member.roleEn}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <Badge className="mb-4" variant="secondary">
              {isRTL ? 'رحلتنا' : 'Our Journey'}
            </Badge>
            <h2 className="text-3xl font-bold">
              {isRTL ? 'محطات مهمة' : 'Key Milestones'}
            </h2>
          </div>
          <div className="relative">
            <div className="absolute right-1/2 transform translate-x-1/2 h-full w-0.5 bg-primary/20" />
            <div className="space-y-8">
              {milestones.map((milestone, index) => (
                <div key={index} className={`flex items-center gap-4 ${index % 2 === 0 ? 'flex-row' : 'flex-row-reverse'}`}>
                  <div className={`flex-1 ${index % 2 === 0 ? 'text-left' : 'text-right'}`}>
                    <Card className="p-4 inline-block">
                      <p className="font-bold text-primary">{milestone.year}</p>
                      <p className="text-sm">
                        {isRTL ? milestone.eventAr : milestone.eventEn}
                      </p>
                    </Card>
                  </div>
                  <div className="w-4 h-4 rounded-full bg-primary z-10" />
                  <div className="flex-1" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            {isRTL ? 'هل أنت مستعد للبدء؟' : 'Ready to Get Started?'}
          </h2>
          <p className="text-muted-foreground mb-8">
            {isRTL 
              ? 'انضم إلى آلاف المستثمرين الذين يثقون بـ ASINAX'
              : 'Join thousands of investors who trust ASINAX'
            }
          </p>
          <Button size="lg" onClick={() => navigate('/register')}>
            {isRTL ? 'إنشاء حساب مجاني' : 'Create Free Account'}
            <Rocket className="h-5 w-5 mr-2" />
          </Button>
        </div>
      </section>
    </div>
  );
}

export default AboutUs;
