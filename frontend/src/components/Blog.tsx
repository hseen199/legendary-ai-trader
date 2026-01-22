/**
 * Blog Page
 * صفحة المدونة والأخبار
 */
import React, { useState } from 'react';
import { 
  Newspaper, 
  Calendar,
  Clock,
  Tag,
  Search,
  ChevronRight,
  BookOpen,
  TrendingUp,
  Shield,
  Lightbulb,
  ArrowRight,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { cn } from '../lib/utils';

interface BlogProps {
  language?: 'ar' | 'en';
}

interface BlogPost {
  id: number;
  category: 'news' | 'education' | 'analysis' | 'tips';
  titleAr: string;
  titleEn: string;
  excerptAr: string;
  excerptEn: string;
  date: string;
  readTime: number;
  featured?: boolean;
}

const blogPosts: BlogPost[] = [
  {
    id: 1,
    category: 'news',
    titleAr: 'تحديث الوكيل الذكي v2.0 - ميزات جديدة',
    titleEn: 'AI Agent v2.0 Update - New Features',
    excerptAr: 'نعلن عن إطلاق الإصدار الثاني من الوكيل الذكي مع تحسينات كبيرة في الأداء واستراتيجيات تداول جديدة.',
    excerptEn: 'We announce the launch of the second version of the AI agent with significant performance improvements and new trading strategies.',
    date: '2025-01-20',
    readTime: 5,
    featured: true,
  },
  {
    id: 2,
    category: 'education',
    titleAr: 'ما هو NAV وكيف يؤثر على استثمارك؟',
    titleEn: 'What is NAV and How Does it Affect Your Investment?',
    excerptAr: 'شرح مفصل لمفهوم صافي قيمة الأصول (NAV) وكيفية استخدامه لتقييم أداء محفظتك الاستثمارية.',
    excerptEn: 'A detailed explanation of Net Asset Value (NAV) and how to use it to evaluate your investment portfolio performance.',
    date: '2025-01-18',
    readTime: 8,
  },
  {
    id: 3,
    category: 'analysis',
    titleAr: 'تحليل أداء السوق - يناير 2025',
    titleEn: 'Market Performance Analysis - January 2025',
    excerptAr: 'نظرة شاملة على أداء سوق العملات الرقمية في يناير 2025 وتوقعات الفترة القادمة.',
    excerptEn: 'A comprehensive look at cryptocurrency market performance in January 2025 and upcoming expectations.',
    date: '2025-01-15',
    readTime: 10,
  },
  {
    id: 4,
    category: 'tips',
    titleAr: '5 نصائح لتحقيق أقصى استفادة من المنصة',
    titleEn: '5 Tips to Get the Most Out of the Platform',
    excerptAr: 'اكتشف أفضل الممارسات والنصائح لتحسين تجربتك الاستثمارية على منصة ASINAX.',
    excerptEn: 'Discover best practices and tips to improve your investment experience on the ASINAX platform.',
    date: '2025-01-12',
    readTime: 6,
  },
  {
    id: 5,
    category: 'education',
    titleAr: 'فهم إدارة المخاطر في التداول الآلي',
    titleEn: 'Understanding Risk Management in Automated Trading',
    excerptAr: 'تعرف على كيفية إدارة الوكيل الذكي للمخاطر وحماية استثماراتك من التقلبات الحادة.',
    excerptEn: 'Learn how the AI agent manages risks and protects your investments from sharp volatility.',
    date: '2025-01-10',
    readTime: 7,
  },
  {
    id: 6,
    category: 'news',
    titleAr: 'إضافة دعم عملات جديدة للتداول',
    titleEn: 'Adding Support for New Trading Currencies',
    excerptAr: 'أضفنا دعم 10 عملات رقمية جديدة لتوسيع فرص التداول وتنويع المحفظة.',
    excerptEn: 'We added support for 10 new cryptocurrencies to expand trading opportunities and diversify portfolios.',
    date: '2025-01-08',
    readTime: 4,
  },
];

const categoryConfig: Record<string, { 
  icon: React.ReactNode; 
  colorClass: string;
  labelAr: string;
  labelEn: string;
}> = {
  news: {
    icon: <Newspaper className="h-4 w-4" />,
    colorClass: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
    labelAr: 'أخبار',
    labelEn: 'News',
  },
  education: {
    icon: <BookOpen className="h-4 w-4" />,
    colorClass: 'bg-green-500/10 text-green-500 border-green-500/30',
    labelAr: 'تعليم',
    labelEn: 'Education',
  },
  analysis: {
    icon: <TrendingUp className="h-4 w-4" />,
    colorClass: 'bg-purple-500/10 text-purple-500 border-purple-500/30',
    labelAr: 'تحليل',
    labelEn: 'Analysis',
  },
  tips: {
    icon: <Lightbulb className="h-4 w-4" />,
    colorClass: 'bg-orange-500/10 text-orange-500 border-orange-500/30',
    labelAr: 'نصائح',
    labelEn: 'Tips',
  },
};

export function Blog({ language = 'ar' }: BlogProps) {
  const isRTL = language === 'ar';
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');

  const filteredPosts = blogPosts.filter(post => {
    const matchesSearch = searchQuery === '' || 
      (isRTL ? post.titleAr : post.titleEn).toLowerCase().includes(searchQuery.toLowerCase()) ||
      (isRTL ? post.excerptAr : post.excerptEn).toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = activeCategory === 'all' || post.category === activeCategory;
    
    return matchesSearch && matchesCategory;
  });

  const featuredPost = blogPosts.find(post => post.featured);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(isRTL ? 'ar-SA' : 'en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  return (
    <div dir={isRTL ? 'rtl' : 'ltr'} className="min-h-screen bg-background p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center mb-12">
          <Badge className="mb-4" variant="outline">
            {isRTL ? 'المدونة' : 'Blog'}
          </Badge>
          <h1 className="text-4xl font-bold mb-4">
            {isRTL ? 'الأخبار والمقالات' : 'News & Articles'}
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            {isRTL 
              ? 'تابع آخر الأخبار والتحديثات والمقالات التعليمية حول التداول والاستثمار'
              : 'Follow the latest news, updates, and educational articles about trading and investing'
            }
          </p>
        </div>

        {/* Search */}
        <div className="relative max-w-md mx-auto">
          <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={isRTL ? 'ابحث في المقالات...' : 'Search articles...'}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pr-10"
          />
        </div>

        {/* Featured Post */}
        {featuredPost && activeCategory === 'all' && searchQuery === '' && (
          <Card className="overflow-hidden bg-gradient-to-r from-primary/10 to-transparent">
            <CardContent className="p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-4">
                    <Badge>{isRTL ? 'مميز' : 'Featured'}</Badge>
                    <Badge variant="outline" className={categoryConfig[featuredPost.category].colorClass}>
                      {categoryConfig[featuredPost.category].icon}
                      <span className="mr-1">
                        {isRTL 
                          ? categoryConfig[featuredPost.category].labelAr 
                          : categoryConfig[featuredPost.category].labelEn
                        }
                      </span>
                    </Badge>
                  </div>
                  <h2 className="text-2xl md:text-3xl font-bold mb-4">
                    {isRTL ? featuredPost.titleAr : featuredPost.titleEn}
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    {isRTL ? featuredPost.excerptAr : featuredPost.excerptEn}
                  </p>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
                    <span className="flex items-center gap-1">
                      <Calendar className="h-4 w-4" />
                      {formatDate(featuredPost.date)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {featuredPost.readTime} {isRTL ? 'دقائق' : 'min read'}
                    </span>
                  </div>
                  <Button>
                    {isRTL ? 'اقرأ المزيد' : 'Read More'}
                    <ArrowRight className="h-4 w-4 mr-2" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Category Tabs */}
        <Tabs value={activeCategory} onValueChange={setActiveCategory}>
          <TabsList className="flex flex-wrap justify-center">
            <TabsTrigger value="all">
              {isRTL ? 'الكل' : 'All'}
            </TabsTrigger>
            {Object.entries(categoryConfig).map(([key, config]) => (
              <TabsTrigger key={key} value={key} className="flex items-center gap-1">
                {config.icon}
                {isRTL ? config.labelAr : config.labelEn}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value={activeCategory} className="mt-6">
            {filteredPosts.length === 0 ? (
              <div className="text-center py-12">
                <Newspaper className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                <p className="text-muted-foreground">
                  {isRTL ? 'لا توجد مقالات مطابقة' : 'No matching articles found'}
                </p>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredPosts.filter(p => !p.featured || activeCategory !== 'all' || searchQuery !== '').map((post) => (
                  <Card key={post.id} className="hover:shadow-lg transition-shadow cursor-pointer group">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge variant="outline" className={categoryConfig[post.category].colorClass}>
                          {categoryConfig[post.category].icon}
                          <span className="mr-1">
                            {isRTL 
                              ? categoryConfig[post.category].labelAr 
                              : categoryConfig[post.category].labelEn
                            }
                          </span>
                        </Badge>
                      </div>
                      <h3 className="font-semibold mb-2 group-hover:text-primary transition-colors">
                        {isRTL ? post.titleAr : post.titleEn}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                        {isRTL ? post.excerptAr : post.excerptEn}
                      </p>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {formatDate(post.date)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {post.readTime} {isRTL ? 'د' : 'min'}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Newsletter Signup */}
        <Card className="bg-gradient-to-r from-primary/10 to-primary/5">
          <CardContent className="p-8 text-center">
            <Newspaper className="h-12 w-12 mx-auto mb-4 text-primary" />
            <h3 className="text-2xl font-bold mb-2">
              {isRTL ? 'اشترك في النشرة الإخبارية' : 'Subscribe to Newsletter'}
            </h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              {isRTL 
                ? 'احصل على آخر الأخبار والتحديثات والنصائح مباشرة في بريدك الإلكتروني'
                : 'Get the latest news, updates, and tips directly in your email'
              }
            </p>
            <div className="flex flex-col sm:flex-row gap-2 max-w-md mx-auto">
              <Input 
                placeholder={isRTL ? 'بريدك الإلكتروني' : 'Your email'}
                className="flex-1"
              />
              <Button>
                {isRTL ? 'اشترك' : 'Subscribe'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default Blog;
