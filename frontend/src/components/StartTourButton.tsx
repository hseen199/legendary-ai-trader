/**
 * StartTourButton Component
 * زر لبدء جولة التعريف من صفحة الإعدادات
 */
import React from 'react';
import { HelpCircle, PlayCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { useOnboarding } from './OnboardingProvider';

interface StartTourButtonProps {
  language?: 'ar' | 'en';
  variant?: 'button' | 'card';
}

export function StartTourButton({ language = 'ar', variant = 'button' }: StartTourButtonProps) {
  const { startTour, hasCompletedTour } = useOnboarding();
  const isRTL = language === 'ar';

  if (variant === 'card') {
    return (
      <Card className="border-dashed border-2 border-primary/30 bg-primary/5">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-primary/10">
              <HelpCircle className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">
                {isRTL ? 'جولة تعريفية' : 'Platform Tour'}
              </CardTitle>
              <CardDescription>
                {isRTL 
                  ? 'تعرف على جميع ميزات المنصة'
                  : 'Learn about all platform features'
                }
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            {isRTL
              ? hasCompletedTour
                ? 'لقد أكملت الجولة التعريفية سابقاً. يمكنك إعادتها في أي وقت.'
                : 'ابدأ جولة تعريفية للتعرف على جميع ميزات المنصة وكيفية استخدامها.'
              : hasCompletedTour
                ? 'You have completed the tour before. You can restart it anytime.'
                : 'Start a tour to learn about all platform features and how to use them.'
            }
          </p>
          <Button
            onClick={startTour}
            className="w-full flex items-center justify-center gap-2"
          >
            <PlayCircle className="h-4 w-4" />
            {isRTL 
              ? hasCompletedTour ? 'إعادة الجولة' : 'ابدأ الجولة'
              : hasCompletedTour ? 'Restart Tour' : 'Start Tour'
            }
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Button
      onClick={startTour}
      variant="outline"
      className="flex items-center gap-2"
    >
      <PlayCircle className="h-4 w-4" />
      {isRTL 
        ? hasCompletedTour ? 'إعادة الجولة التعريفية' : 'ابدأ الجولة التعريفية'
        : hasCompletedTour ? 'Restart Tour' : 'Start Tour'
      }
    </Button>
  );
}

export default StartTourButton;
