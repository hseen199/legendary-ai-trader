import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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
import { Eye, EyeOff, Loader2, Brain, Mail, Lock } from "lucide-react";
import { useLanguage } from "@/lib/i18n";
import { LanguageToggle } from "@/components/language-toggle";
import { ThemeToggle } from "@/components/theme-toggle";
import { NeuralNetworkBg } from "@/components/neural-network-bg";
import { AIThinkingPulse } from "@/components/ai-thinking-pulse";

export default function Login() {
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const { login } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const { dir, t, language } = useLanguage();

  const loginSchema = z.object({
    email: z.string().email(t.auth.invalidEmail),
    password: z.string().min(1, t.auth.passwordRequired),
  });

  type LoginFormData = z.infer<typeof loginSchema>;

  const loginForm = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const loginMutation = useMutation({
    mutationFn: async (data: LoginFormData) => {
      return await login(data.email, data.password);
    },
    onSuccess: () => {
      toast({
        title: t.auth.loginSuccess,
        description: t.auth.welcomeBackSubtitle,
      });
      navigate("/");
    },
    onError: (error: any) => {
      toast({
        title: t.auth.loginError,
        description: error.message || t.auth.invalidEmailOrPassword,
        variant: "destructive",
      });
    },
  });

  const onLoginSubmit = (data: LoginFormData) => {
    loginMutation.mutate(data);
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
            <Link href="/register">
              <Button variant="outline">
                {t.auth.register}
              </Button>
            </Link>
          </div>
        </header>

        <div className="relative z-10 flex items-center justify-center py-20 px-4">
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
                {t.auth.login}
              </CardTitle>
              <p className="text-muted-foreground text-sm mt-2">
                {t.auth.welcomeBackSubtitle}
              </p>
            </CardHeader>
            <CardContent>
              <Form {...loginForm}>
                <form onSubmit={loginForm.handleSubmit(onLoginSubmit)} className="space-y-4">
                  <FormField
                    control={loginForm.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t.auth.email}</FormLabel>
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
                    control={loginForm.control}
                    name="password"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <FormLabel>{t.auth.password}</FormLabel>
                          <Link href="/forgot-password" className="text-sm text-primary underline">
                            {t.auth.forgotPassword}
                          </Link>
                        </div>
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

                  <Button 
                    type="submit" 
                    className="w-full bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90" 
                    disabled={loginMutation.isPending}
                  >
                    {loginMutation.isPending ? (
                      <>
                        <Loader2 className={`w-4 h-4 ${language === "ar" ? "ml-2" : "mr-2"} animate-spin`} />
                        {t.auth.loggingIn}
                      </>
                    ) : (
                      t.auth.login
                    )}
                  </Button>
                </form>
              </Form>

              <div className="mt-6 text-center">
                <p className="text-muted-foreground text-sm">
                  {t.auth.noAccount}{" "}
                  <Link href="/register" className="text-primary hover:underline font-medium">
                    {t.auth.createAccount}
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
