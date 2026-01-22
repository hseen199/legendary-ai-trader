/**
 * ProfitCalculator Component
 * حاسبة الأرباح المتوقعة
 */
import React, { useState, useMemo } from 'react';
import { 
  Calculator, 
  TrendingUp, 
  DollarSign,
  Calendar,
  Info,
  Sparkles,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';

interface ProfitCalculatorProps {
  language?: 'ar' | 'en';
  currentNav?: number;
  historicalReturn?: number; // العائد التاريخي السنوي
}

export function ProfitCalculator({ 
  language = 'ar',
  currentNav = 1.05,
  historicalReturn = 15, // 15% سنوياً كمثال
}: ProfitCalculatorProps) {
  const isRTL = language === 'ar';
  
  const [investmentAmount, setInvestmentAmount] = useState<number>(1000);
  const [investmentPeriod, setInvestmentPeriod] = useState<number>(12); // بالأشهر
  const [expectedReturn, setExpectedReturn] = useState<number>(historicalReturn);

  // حساب الأرباح المتوقعة
  const calculations = useMemo(() => {
    const monthlyRate = expectedReturn / 100 / 12;
    const months = investmentPeriod;
    
    // الفائدة المركبة
    const finalAmount = investmentAmount * Math.pow(1 + monthlyRate, months);
    const totalProfit = finalAmount - investmentAmount;
    const profitPercentage = (totalProfit / investmentAmount) * 100;
    
    // حساب الوحدات
    const units = investmentAmount / currentNav;
    const finalNavEstimate = currentNav * Math.pow(1 + monthlyRate, months);
    
    // بيانات الرسم البياني
    const chartData = Array.from({ length: months + 1 }, (_, i) => {
      const monthAmount = investmentAmount * Math.pow(1 + monthlyRate, i);
      return {
        month: i,
        amount: monthAmount,
        profit: monthAmount - investmentAmount,
      };
    });

    // سيناريوهات مختلفة
    const scenarios = {
      conservative: {
        rate: expectedReturn * 0.5,
        final: investmentAmount * Math.pow(1 + (expectedReturn * 0.5 / 100 / 12), months),
      },
      moderate: {
        rate: expectedReturn,
        final: finalAmount,
      },
      optimistic: {
        rate: expectedReturn * 1.5,
        final: investmentAmount * Math.pow(1 + (expectedReturn * 1.5 / 100 / 12), months),
      },
    };

    return {
      finalAmount,
      totalProfit,
      profitPercentage,
      units,
      finalNavEstimate,
      chartData,
      scenarios,
      monthlyProfit: totalProfit / months,
    };
  }, [investmentAmount, investmentPeriod, expectedReturn, currentNav]);

  // تنسيق الأرقام - استخدام الأرقام الإنجليزية دائماً
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatNumber = (value: number, decimals: number = 2) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);
  };

  return (
    <Card dir={isRTL ? 'rtl' : 'ltr'}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="h-5 w-5 text-primary" />
          {isRTL ? 'حاسبة الأرباح' : 'Profit Calculator'}
        </CardTitle>
        <CardDescription>
          {isRTL 
            ? 'احسب أرباحك المتوقعة بناءً على مبلغ الاستثمار والفترة الزمنية'
            : 'Calculate your expected profits based on investment amount and time period'
          }
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Input Fields */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* مبلغ الاستثمار */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              {isRTL ? 'مبلغ الاستثمار (USDC)' : 'Investment Amount (USDC)'}
            </Label>
            <Input
              type="number"
              value={investmentAmount}
              onChange={(e) => setInvestmentAmount(Number(e.target.value) || 0)}
              min={100}
              max={1000000}
              className="text-lg"
            />
            <div className="flex gap-1">
              {[500, 1000, 5000, 10000].map((amount) => (
                <button
                  key={amount}
                  onClick={() => setInvestmentAmount(amount)}
                  className={cn(
                    "px-2 py-1 text-xs rounded border transition-colors",
                    investmentAmount === amount 
                      ? "bg-primary text-primary-foreground" 
                      : "hover:bg-muted"
                  )}
                >
                  ${amount.toLocaleString()}
                </button>
              ))}
            </div>
          </div>

          {/* فترة الاستثمار */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              {isRTL ? 'فترة الاستثمار' : 'Investment Period'}
            </Label>
            <div className="pt-2">
              <Slider
                value={[investmentPeriod]}
                onValueChange={([value]) => setInvestmentPeriod(value)}
                min={1}
                max={60}
                step={1}
                className="w-full"
              />
            </div>
            <p className="text-center text-sm text-muted-foreground">
              {investmentPeriod} {isRTL ? 'شهر' : 'months'} 
              ({(investmentPeriod / 12).toFixed(1)} {isRTL ? 'سنة' : 'years'})
            </p>
          </div>

          {/* العائد المتوقع */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              {isRTL ? 'العائد السنوي المتوقع' : 'Expected Annual Return'}
            </Label>
            <div className="pt-2">
              <Slider
                value={[expectedReturn]}
                onValueChange={([value]) => setExpectedReturn(value)}
                min={5}
                max={50}
                step={1}
                className="w-full"
              />
            </div>
            <p className="text-center text-sm text-muted-foreground">
              {expectedReturn}% {isRTL ? 'سنوياً' : 'annually'}
            </p>
          </div>
        </div>

        {/* Results */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
            <p className="text-sm text-muted-foreground">
              {isRTL ? 'المبلغ النهائي' : 'Final Amount'}
            </p>
            <p className="text-2xl font-bold text-green-500">
              {formatCurrency(calculations.finalAmount)}
            </p>
          </div>
          
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <p className="text-sm text-muted-foreground">
              {isRTL ? 'إجمالي الربح' : 'Total Profit'}
            </p>
            <p className="text-2xl font-bold text-blue-500">
              {formatCurrency(calculations.totalProfit)}
            </p>
          </div>
          
          <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
            <p className="text-sm text-muted-foreground">
              {isRTL ? 'نسبة الربح' : 'Profit Percentage'}
            </p>
            <p className="text-2xl font-bold text-purple-500">
              +{formatNumber(calculations.profitPercentage)}%
            </p>
          </div>
          
          <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/20">
            <p className="text-sm text-muted-foreground">
              {isRTL ? 'الربح الشهري' : 'Monthly Profit'}
            </p>
            <p className="text-2xl font-bold text-orange-500">
              {formatCurrency(calculations.monthlyProfit)}
            </p>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={calculations.chartData}>
              <defs>
                <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="month" 
                stroke="#666"
                fontSize={12}
                tickFormatter={(v) => `${v}${isRTL ? 'ش' : 'm'}`}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`}
              />
              <Tooltip 
                formatter={(value: number) => [formatCurrency(value), isRTL ? 'المبلغ' : 'Amount']}
                labelFormatter={(label) => `${isRTL ? 'الشهر' : 'Month'} ${label}`}
              />
              <Area
                type="monotone"
                dataKey="amount"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#profitGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Scenarios */}
        <div className="space-y-3">
          <h4 className="font-semibold flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            {isRTL ? 'سيناريوهات مختلفة' : 'Different Scenarios'}
          </h4>
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg border bg-muted/30">
              <Badge variant="secondary" className="mb-2">
                {isRTL ? 'متحفظ' : 'Conservative'}
              </Badge>
              <p className="text-xs text-muted-foreground">
                {formatNumber(calculations.scenarios.conservative.rate)}% {isRTL ? 'سنوياً' : 'annually'}
              </p>
              <p className="text-lg font-bold">
                {formatCurrency(calculations.scenarios.conservative.final)}
              </p>
            </div>
            <div className="p-3 rounded-lg border bg-primary/10 border-primary/30">
              <Badge className="mb-2">
                {isRTL ? 'معتدل' : 'Moderate'}
              </Badge>
              <p className="text-xs text-muted-foreground">
                {formatNumber(calculations.scenarios.moderate.rate)}% {isRTL ? 'سنوياً' : 'annually'}
              </p>
              <p className="text-lg font-bold">
                {formatCurrency(calculations.scenarios.moderate.final)}
              </p>
            </div>
            <div className="p-3 rounded-lg border bg-muted/30">
              <Badge variant="secondary" className="mb-2">
                {isRTL ? 'متفائل' : 'Optimistic'}
              </Badge>
              <p className="text-xs text-muted-foreground">
                {formatNumber(calculations.scenarios.optimistic.rate)}% {isRTL ? 'سنوياً' : 'annually'}
              </p>
              <p className="text-lg font-bold">
                {formatCurrency(calculations.scenarios.optimistic.final)}
              </p>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <div className="flex items-start gap-2">
            <Info className="h-5 w-5 text-yellow-500 shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-yellow-600">
                {isRTL ? 'تنبيه مهم' : 'Important Notice'}
              </p>
              <p className="text-muted-foreground">
                {isRTL 
                  ? 'هذه الحسابات تقديرية بناءً على الأداء التاريخي ولا تضمن نتائج مستقبلية. التداول ينطوي على مخاطر وقد تخسر جزءاً أو كل استثمارك.'
                  : 'These calculations are estimates based on historical performance and do not guarantee future results. Trading involves risks and you may lose part or all of your investment.'
                }
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ProfitCalculator;
