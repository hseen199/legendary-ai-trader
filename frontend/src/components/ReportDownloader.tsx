/**
 * ReportDownloader Component
 * مكون تحميل التقارير الشهرية والأسبوعية
 */
import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { 
  FileText, 
  Download, 
  Calendar,
  Loader2,
  CheckCircle,
  FileDown,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { cn } from '../lib/utils';
import toast from 'react-hot-toast';
import api from '../services/api';

interface ReportDownloaderProps {
  language?: 'ar' | 'en';
}

interface ReportOption {
  id: string;
  type: 'weekly' | 'monthly';
  titleAr: string;
  titleEn: string;
  descAr: string;
  descEn: string;
  icon: React.ReactNode;
}

const reportOptions: ReportOption[] = [
  {
    id: 'weekly',
    type: 'weekly',
    titleAr: 'التقرير الأسبوعي',
    titleEn: 'Weekly Report',
    descAr: 'ملخص أداء محفظتك خلال الأسبوع الماضي',
    descEn: 'Summary of your portfolio performance over the past week',
    icon: <Clock className="h-5 w-5" />,
  },
  {
    id: 'monthly',
    type: 'monthly',
    titleAr: 'التقرير الشهري',
    titleEn: 'Monthly Report',
    descAr: 'تقرير مفصل عن أداء محفظتك خلال الشهر',
    descEn: 'Detailed report on your portfolio performance during the month',
    icon: <Calendar className="h-5 w-5" />,
  },
];

// الأشهر المتاحة للتقارير
const getAvailableMonths = () => {
  const months = [];
  const now = new Date();
  
  for (let i = 0; i < 12; i++) {
    const date = new Date(now.getFullYear(), now.getMonth() - i, 1);
    months.push({
      value: `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`,
      labelAr: date.toLocaleDateString('ar-SA', { month: 'long', year: 'numeric' }),
      labelEn: date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
    });
  }
  
  return months;
};

export function ReportDownloader({ language = 'ar' }: ReportDownloaderProps) {
  const isRTL = language === 'ar';
  const [selectedMonth, setSelectedMonth] = useState<string>('');
  const [downloadingReport, setDownloadingReport] = useState<string | null>(null);
  
  const availableMonths = getAvailableMonths();

  // تحميل التقرير
  const downloadMutation = useMutation({
    mutationFn: async ({ type, month }: { type: string; month?: string }) => {
      const params: any = { type };
      if (month) {
        const [year, monthNum] = month.split('-');
        params.year = year;
        params.month = monthNum;
      }
      
      const response = await api.get('/user/reports/download', {
        params,
        responseType: 'blob',
      });
      
      return response.data;
    },
    onSuccess: (data, variables) => {
      // إنشاء رابط التحميل
      const blob = new Blob([data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      const filename = variables.type === 'weekly' 
        ? `ASINAX_Weekly_Report_${new Date().toISOString().split('T')[0]}.pdf`
        : `ASINAX_Monthly_Report_${variables.month}.pdf`;
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success(isRTL ? 'تم تحميل التقرير بنجاح' : 'Report downloaded successfully');
      setDownloadingReport(null);
    },
    onError: () => {
      toast.error(isRTL ? 'فشل في تحميل التقرير' : 'Failed to download report');
      setDownloadingReport(null);
    },
  });

  const handleDownload = (type: string, month?: string) => {
    if (type === 'monthly' && !month) {
      toast.error(isRTL ? 'يرجى اختيار الشهر' : 'Please select a month');
      return;
    }
    
    setDownloadingReport(type);
    downloadMutation.mutate({ type, month });
  };

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          {isRTL ? 'تحميل التقارير' : 'Download Reports'}
        </CardTitle>
        <CardDescription>
          {isRTL 
            ? 'قم بتحميل تقارير أداء محفظتك بصيغة PDF'
            : 'Download your portfolio performance reports in PDF format'
          }
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Report Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reportOptions.map((report) => (
            <div
              key={report.id}
              className="p-4 rounded-lg border hover:border-primary/50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-full bg-primary/10 text-primary">
                  {report.icon}
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold">
                    {isRTL ? report.titleAr : report.titleEn}
                  </h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {isRTL ? report.descAr : report.descEn}
                  </p>
                  
                  {report.type === 'monthly' && (
                    <div className="mt-3">
                      <Select value={selectedMonth} onValueChange={setSelectedMonth}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder={isRTL ? 'اختر الشهر' : 'Select month'} />
                        </SelectTrigger>
                        <SelectContent>
                          {availableMonths.map((month) => (
                            <SelectItem key={month.value} value={month.value}>
                              {isRTL ? month.labelAr : month.labelEn}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  
                  <Button
                    className="mt-3 w-full"
                    onClick={() => handleDownload(report.type, report.type === 'monthly' ? selectedMonth : undefined)}
                    disabled={downloadingReport === report.type || (report.type === 'monthly' && !selectedMonth)}
                  >
                    {downloadingReport === report.type ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin ml-2" />
                        {isRTL ? 'جاري التحميل...' : 'Downloading...'}
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 ml-2" />
                        {isRTL ? 'تحميل PDF' : 'Download PDF'}
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Recent Reports */}
        <div className="space-y-3">
          <h4 className="font-semibold flex items-center gap-2">
            <FileDown className="h-4 w-4 text-primary" />
            {isRTL ? 'التقارير الأخيرة' : 'Recent Reports'}
          </h4>
          
          <div className="space-y-2">
            {availableMonths.slice(0, 3).map((month, index) => (
              <div
                key={month.value}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
              >
                <div className="flex items-center gap-3">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">
                      {isRTL ? 'التقرير الشهري' : 'Monthly Report'} - {isRTL ? month.labelAr : month.labelEn}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      PDF • {isRTL ? 'متاح للتحميل' : 'Available for download'}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setSelectedMonth(month.value);
                    handleDownload('monthly', month.value);
                  }}
                  disabled={downloadingReport !== null}
                >
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* Info */}
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <div className="flex items-start gap-2">
            <CheckCircle className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-blue-600">
                {isRTL ? 'معلومة' : 'Info'}
              </p>
              <p className="text-muted-foreground">
                {isRTL 
                  ? 'التقارير تتضمن ملخص الأداء، سجل الصفقات، وتحليل المحفظة. يمكنك أيضاً تفعيل استلام التقارير تلقائياً عبر البريد الإلكتروني من إعدادات الإشعارات.'
                  : 'Reports include performance summary, trade history, and portfolio analysis. You can also enable automatic report delivery via email from notification settings.'
                }
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ReportDownloader;
