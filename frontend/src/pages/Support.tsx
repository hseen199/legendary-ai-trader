import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Skeleton } from "../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { 
  MessageSquare, 
  Plus, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Send,
  HelpCircle,
  FileText,
  Search,
} from "lucide-react";
import { cn } from "../lib/utils";
import { format } from "date-fns";
import toast from "react-hot-toast";
import api from "../services/api";
import { useLanguage } from '@/lib/i18n';

interface Ticket {
  id: number;
  subject: string;
  message: string;
  status: string;
  priority: string;
  created_at: string;
  updated_at?: string;
  replies?: TicketReply[];
}

interface TicketReply {
  id: number;
  message: string;
  is_admin: boolean;
  created_at: string;
}

interface FAQ {
  id: number;
  questionKey: string;
  answerKey: string;
  categoryKey: string;
}

export default function Support() {
  const { t, language } = useLanguage();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("faq");
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showNewTicket, setShowNewTicket] = useState(false);
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  
  // New ticket form
  const [newSubject, setNewSubject] = useState("");
  const [newMessage, setNewMessage] = useState("");
  const [newPriority, setNewPriority] = useState("medium");
  
  // Reply form
  const [replyMessage, setReplyMessage] = useState("");

  // FAQ data with translation keys
  const FAQS: FAQ[] = [
    { id: 1, questionKey: 'faqQ1', answerKey: 'faqA1', categoryKey: 'categoryDeposit' },
    { id: 2, questionKey: 'faqQ2', answerKey: 'faqA2', categoryKey: 'categoryDeposit' },
    { id: 3, questionKey: 'faqQ3', answerKey: 'faqA3', categoryKey: 'categoryDeposit' },
    { id: 4, questionKey: 'faqQ4', answerKey: 'faqA4', categoryKey: 'categoryTrading' },
    { id: 5, questionKey: 'faqQ5', answerKey: 'faqA5', categoryKey: 'categoryTrading' },
    { id: 6, questionKey: 'faqQ6', answerKey: 'faqA6', categoryKey: 'categoryDeposit' },
    { id: 7, questionKey: 'faqQ7', answerKey: 'faqA7', categoryKey: 'categoryAccount' },
  ];

  // Get translated FAQ content
  const getTranslatedFaqs = () => {
    return FAQS.map(faq => ({
      id: faq.id,
      question: (t.support as any)[faq.questionKey] || faq.questionKey,
      answer: (t.support as any)[faq.answerKey] || faq.answerKey,
      category: (t.support as any)[faq.categoryKey] || faq.categoryKey,
    }));
  };

  // Fetch tickets
  const { data: tickets = [], isLoading: loadingTickets } = useQuery<Ticket[]>({
    queryKey: ["/api/v1/support/tickets"],
    queryFn: async () => {
      try {
        const res = await api.get("/support/tickets");
        return res.data;
      } catch {
        return [];
      }
    },
  });

  // Create ticket mutation
  const createTicketMutation = useMutation({
    mutationFn: async (data: { subject: string; message: string; priority: string }) => {
      return api.post("/support/tickets", data);
    },
    onSuccess: () => {
      toast.success(t.support.ticketCreated);
      setShowNewTicket(false);
      setNewSubject("");
      setNewMessage("");
      setNewPriority("medium");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets"] });
    },
    onError: () => {
      toast.error(t.support.ticketFailed);
    },
  });

  // Reply mutation
  const replyMutation = useMutation({
    mutationFn: async ({ ticketId, message }: { ticketId: number; message: string }) => {
      return api.post(`/support/tickets/${ticketId}/reply`, { message });
    },
    onSuccess: () => {
      toast.success(language === 'ar' ? 'تم إرسال الرد' : 'Reply sent');
      setReplyMessage("");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets"] });
    },
    onError: () => {
      toast.error(language === 'ar' ? 'فشل في إرسال الرد' : 'Failed to send reply');
    },
  });

  const handleCreateTicket = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubject.trim() || !newMessage.trim()) {
      toast.error(language === 'ar' ? 'يرجى ملء جميع الحقول' : 'Please fill all fields');
      return;
    }
    createTicketMutation.mutate({
      subject: newSubject,
      message: newMessage,
      priority: newPriority,
    });
  };

  const handleReply = (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyMessage.trim() || !selectedTicket) return;
    replyMutation.mutate({
      ticketId: selectedTicket.id,
      message: replyMessage,
    });
  };

  const translatedFaqs = getTranslatedFaqs();
  const filteredFaqs = translatedFaqs.filter(
    (faq) =>
      faq.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "open":
        return (
          <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20">
            <AlertCircle className="w-3 h-3 ml-1" />
            {t.support.open}
          </Badge>
        );
      case "in_progress":
        return (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
            <Clock className="w-3 h-3 ml-1" />
            {t.support.inProgress}
          </Badge>
        );
      case "resolved":
        return (
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
            <CheckCircle className="w-3 h-3 ml-1" />
            {t.support.resolved}
          </Badge>
        );
      case "closed":
        return (
          <Badge variant="outline" className="bg-muted text-muted-foreground">
            {t.support.closed}
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "high":
        return <Badge variant="destructive">{t.support.high}</Badge>;
      case "medium":
        return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500">{t.support.medium}</Badge>;
      case "low":
        return <Badge variant="outline">{t.support.low}</Badge>;
      default:
        return <Badge variant="outline">{priority}</Badge>;
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">{t.support.title}</h1>
          <p className="text-muted-foreground text-sm">{t.support.subtitle}</p>
        </div>
        <button
          onClick={() => setShowNewTicket(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
          {t.support.newTicket}
        </button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="faq" className="gap-2">
            <HelpCircle className="w-4 h-4" />
            {t.support.faq}
          </TabsTrigger>
          <TabsTrigger value="tickets" className="gap-2">
            <MessageSquare className="w-4 h-4" />
            {t.support.myTickets}
            {tickets.filter(t => t.status !== "closed").length > 0 && (
              <Badge variant="destructive" className="mr-2">
                {tickets.filter(t => t.status !== "closed").length}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        {/* FAQ Tab */}
        <TabsContent value="faq" className="space-y-4 mt-6">
          {/* Search */}
          <div className="relative">
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder={t.support.searchFaq}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pr-10 px-4 py-3 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none placeholder:text-gray-400"
            />
          </div>

          {/* FAQ List */}
          <div className="space-y-3">
            {filteredFaqs.map((faq) => (
              <Card key={faq.id} className="overflow-hidden">
                <button
                  onClick={() => setExpandedFaq(expandedFaq === faq.id ? null : faq.id)}
                  className="w-full p-4 flex items-center justify-between text-right hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <HelpCircle className="w-5 h-5 text-primary flex-shrink-0" />
                    <span className="font-medium">{faq.question}</span>
                  </div>
                  {expandedFaq === faq.id ? (
                    <ChevronUp className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-muted-foreground" />
                  )}
                </button>
                {expandedFaq === faq.id && (
                  <div className="px-4 pb-4 pr-12">
                    <p className="text-muted-foreground">{faq.answer}</p>
                    <Badge variant="outline" className="mt-2">{faq.category}</Badge>
                  </div>
                )}
              </Card>
            ))}
          </div>

          {/* Contact Section */}
          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="p-6 text-center">
              <MessageSquare className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">{t.support.noAnswer}</h3>
              <p className="text-muted-foreground mb-4">{t.support.contactUs}</p>
              <button
                onClick={() => setShowNewTicket(true)}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
              >
                {t.support.createTicket}
              </button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tickets Tab */}
        <TabsContent value="tickets" className="space-y-4 mt-6">
          {loadingTickets ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : tickets.length === 0 ? (
            <Card>
              <CardContent className="p-12 text-center">
                <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">{t.support.noTickets}</h3>
                <p className="text-muted-foreground mb-4">{t.support.noTicketsDesc}</p>
                <button
                  onClick={() => setShowNewTicket(true)}
                  className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
                >
                  {t.support.createTicket}
                </button>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {tickets.map((ticket) => (
                <Card
                  key={ticket.id}
                  className={cn(
                    "cursor-pointer hover:border-primary/50 transition-colors",
                    selectedTicket?.id === ticket.id && "border-primary"
                  )}
                  onClick={() => setSelectedTicket(selectedTicket?.id === ticket.id ? null : ticket)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold">{ticket.subject}</h3>
                          {getStatusBadge(ticket.status)}
                          {getPriorityBadge(ticket.priority)}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2">{ticket.message}</p>
                        <p className="text-xs text-muted-foreground mt-2">
                          {format(new Date(ticket.created_at), "yyyy/MM/dd HH:mm")}
                        </p>
                      </div>
                    </div>

                    {/* Ticket Details */}
                    {selectedTicket?.id === ticket.id && (
                      <div className="mt-4 pt-4 border-t space-y-4">
                        {/* Replies */}
                        {ticket.replies && ticket.replies.length > 0 && (
                          <div className="space-y-3">
                            {ticket.replies.map((reply) => (
                              <div
                                key={reply.id}
                                className={cn(
                                  "p-3 rounded-lg",
                                  reply.is_admin
                                    ? "bg-primary/10 mr-8"
                                    : "bg-muted ml-8"
                                )}
                              >
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="text-xs font-medium">
                                    {reply.is_admin ? (language === 'ar' ? 'الدعم' : 'Support') : (language === 'ar' ? 'أنت' : 'You')}
                                  </span>
                                  <span className="text-xs text-muted-foreground">
                                    {format(new Date(reply.created_at), "HH:mm")}
                                  </span>
                                </div>
                                <p className="text-sm">{reply.message}</p>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Reply Form */}
                        {ticket.status !== "closed" && (
                          <form onSubmit={handleReply} className="flex gap-2">
                            <input
                              type="text"
                              value={replyMessage}
                              onChange={(e) => setReplyMessage(e.target.value)}
                              placeholder={language === 'ar' ? 'اكتب ردك...' : 'Write your reply...'}
                              className="flex-1 px-4 py-2 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none placeholder:text-gray-400"
                            />
                            <button
                              type="submit"
                              disabled={!replyMessage.trim() || replyMutation.isPending}
                              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
                            >
                              <Send className="w-4 h-4" />
                            </button>
                          </form>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* New Ticket Modal */}
      {showNewTicket && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <CardTitle>{t.support.newTicket}</CardTitle>
              <CardDescription>{t.support.contactUs}</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateTicket} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.subject}</label>
                  <input
                    type="text"
                    value={newSubject}
                    onChange={(e) => setNewSubject(e.target.value)}
                    className="w-full px-4 py-2 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none placeholder:text-gray-400"
                    placeholder={language === 'ar' ? 'موضوع التذكرة' : 'Ticket subject'}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.message}</label>
                  <textarea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-2 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none resize-none placeholder:text-gray-400"
                    placeholder={language === 'ar' ? 'اكتب رسالتك هنا...' : 'Write your message here...'}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">{t.support.priority}</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value)}
                    className="w-full px-4 py-2 bg-[#1a1a2e] text-white rounded-lg border border-violet-500/30 focus:border-primary focus:outline-none"
                  >
                    <option value="low">{t.support.low}</option>
                    <option value="medium">{t.support.medium}</option>
                    <option value="high">{t.support.high}</option>
                  </select>
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowNewTicket(false)}
                    className="flex-1 px-4 py-2 border border-border rounded-lg hover:bg-muted"
                  >
                    {language === 'ar' ? 'إلغاء' : 'Cancel'}
                  </button>
                  <button
                    type="submit"
                    disabled={createTicketMutation.isPending}
                    className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
                  >
                    {createTicketMutation.isPending ? (language === 'ar' ? 'جاري الإرسال...' : 'Sending...') : t.support.submit}
                  </button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
