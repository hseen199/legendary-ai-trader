import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { Eye, EyeOff, Loader2, Brain, Mail, Lock, User, Phone } from "lucide-react";
import { useLanguage } from "@/lib/i18n";
import { LanguageToggle } from "@/components/language-toggle";
import { ThemeToggle } from "@/components/theme-toggle";
import { NeuralNetworkBg } from "@/components/neural-network-bg";
import { AIThinkingPulse } from "@/components/ai-thinking-pulse";

export default function Register() {
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const { register: registerUser, login } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const { dir, t, language } = useLanguage();

  const registerSchema = z.object({
    full_name: z.string().optional(),
    email: z.string().email(t.auth.invalidEmail),
    phone: z.string().optional(),
    password: z.string().min(8, t.auth.passwordMinLength),
    confirmPassword: z.string(),
    terms: z.boolean().refine((val) => val === true, {
      message: t.auth.termsRequired,
    }),
    referral_code: z.string().optional(),
  }).refine((data) => data.password === data.confirmPassword, {
    message: t.auth.passwordsDoNotMatch,
    path: ["confirmPassword"],
  });

  type RegisterFormData = z.infer<typeof registerSchema>;

  const registerForm = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      full_name: "",
      email: "",
      phone: "",
      password: "",
      confirmPassword: "",
      terms: false,
      referral_code: "",
    },
  });

  const registerMutation = useMutation({
    mutationFn: async (data: RegisterFormData) => {
      await registerUser({
        email: data.email,
        password: data.password,
        full_name: data.full_name || undefined,
        phone: data.phone || undefined,
        referral_code: data.referral_code || undefined,
      });
      // Auto login after registration
      return await login(data.email, data.password);
    },
    onSuccess: () => {
      toast({
        title: t.auth.registerSuccess,
        description: t.auth.accountCreated,
      });
      navigate("/");
    },
    onError: (error: any) => {
      toast({
        title: t.auth.registerError,
        description: error.message || t.auth.registrationFailed,
        variant: "destructive",
      });
    },
  });

  const onRegisterSubmit = (data: RegisterFormData) => {
    registerMutation.mutate(data);
  };

  return (
    <div className="min-h-screen bg-background" dir={dir}>
      <div className="relative overflow-hidden min-h-screen">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/10 via-purple-500/5 to-background" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-purple-500/10 to-transparent" />
        
        <NeuralNetworkBg nodeCount={10} className="opacity-30" />
        
        <header className="relative z-10 flex items-center justify-between p-4 md:p-6 border-b border-border/50 backdrop-blur-sm">
          <Link href="/">
            <div className="flex items-center gap-3 cursor-pointer">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
              </div>
              <div>
                <span className="font-bold text-xl gradient-text-animate">ASINAX</span>
                <span className="text-xs text-muted-foreground block">CRYPTO AI</span>
              </div>
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <LanguageToggle />
            <ThemeToggle />
            <Link href="/login">
              <Button variant="outline">
                {t.auth.login}
              </Button>
            </Link>
          </div>
        </header>

        <div className="relative z-10 flex items-center justify-center py-10 px-4">
          <Card className="w-full max-w-md login-card-glow border-primary/20 backdrop-blur-sm bg-card/95">
            <CardHeader className="text-center relative">
              <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent rounded-t-lg pointer-events-none" />
              <div className="relative mx-auto mb-4 w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-500/20 border border-primary/30 flex items-center justify-center">
                <Brain className="w-8 h-8 text-primary" />
                <div className="absolute -top-1 -right-1">
                  <AIThinkingPulse size="sm" isActive={true} />
                </div>
              </div>
              <CardTitle className="text-2xl gradient-text-animate">
                {t.auth.createAccount}
              </CardTitle>
              <p className="text-muted-foreground text-sm mt-2">
                {t.auth.joinPlatform}
              </p>
            </CardHeader>
            <CardContent>
              <Form {...registerForm}>
                <form onSubmit={registerForm.handleSubmit(onRegisterSubmit)} className="space-y-4">
                  <FormField
                    control={registerForm.control}
                    name="full_name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.fullName} ({t.common.optional})</FormLabel>
                        <FormControl>
                          <div className="relative">
                            <User className={`absolute ${language === "ar" ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground`} />
                            <Input 
                              placeholder={t.auth.fullNamePlaceholder}
                              className={language === "ar" ? "pr-10" : "pl-10"}
                              {...field}
                            />
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.email} *</FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Mail className={`absolute ${language === "ar" ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground`} />
                            <Input 
                              type="email" 
                              placeholder="example@email.com" 
                              className={language === "ar" ? "pr-10" : "pl-10"}
                              {...field}
                            />
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="phone"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.phone} ({t.common.optional})</FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Phone className={`absolute ${language === "ar" ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground`} />
                            <Input 
                              type="tel"
                              placeholder="+971 50 123 4567"
                              className={language === "ar" ? "pr-10" : "pl-10"}
                              {...field}
                            />
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="password"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.password} *</FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Lock className={`absolute ${language === "ar" ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground`} />
                            <Input 
                              type={showPassword ? "text" : "password"} 
                              placeholder="********" 
                              className={language === "ar" ? "pr-10" : "pl-10"}
                              {...field}
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className={`absolute ${language === "ar" ? "left-1" : "right-1"} top-1/2 -translate-y-1/2`}
                              onClick={() => setShowPassword(!showPassword)}
                            >
                              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </Button>
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="confirmPassword"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.confirmPassword} *</FormLabel>
                        <FormControl>
                          <div className="relative">
                            <Lock className={`absolute ${language === "ar" ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground`} />
                            <Input 
                              type={showConfirmPassword ? "text" : "password"} 
                              placeholder="********" 
                              className={language === "ar" ? "pr-10" : "pl-10"}
                              {...field}
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className={`absolute ${language === "ar" ? "left-1" : "right-1"} top-1/2 -translate-y-1/2`}
                              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                            >
                              {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </Button>
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="referral_code"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.referralCode} ({t.common.optional})</FormLabel>
                        <FormControl>
                          <Input 
                            placeholder={t.auth.referralCodePlaceholder}
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={registerForm.control}
                    name="terms"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-start space-x-3 space-y-0 rtl:space-x-reverse">
                        <FormControl>
                          <Checkbox
                            checked={field.value}
                            onCheckedChange={field.onChange}
                          />
                        </FormControl>
                        <div className="space-y-1 leading-none">
                          <FormLabel className="text-sm font-normal">
                            {t.auth.agreeToTerms}{" "}
                            <Link href="/terms" className="text-primary hover:underline">
                              {t.auth.termsOfService}
                            </Link>{" "}
                            {t.common.and}{" "}
                            <Link href="/privacy" className="text-primary hover:underline">
                              {t.auth.privacyPolicy}
                            </Link>
                          </FormLabel>
                          <FormMessage />
                        </div>
                      </FormItem>
                    )}
                  />

                  <Button 
                    type="submit" 
                    className="w-full bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90" 
                    disabled={registerMutation.isPending}
                  >
                    {registerMutation.isPending ? (
                      <>
                        <Loader2 className={`w-4 h-4 ${language === "ar" ? "ml-2" : "mr-2"} animate-spin`} />
                        {t.auth.creatingAccount}
                      </>
                    ) : (
                      t.auth.createAccount
                    )}
                  </Button>
                </form>
              </Form>

              <div className="mt-6 text-center">
                <p className="text-muted-foreground text-sm">
                  {t.auth.haveAccount}{" "}
                  <Link href="/login" className="text-primary hover:underline font-medium">
                    {t.auth.login}
                  </Link>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
