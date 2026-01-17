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
  question: string;
  answer: string;
  category: string;
}

const FAQS: FAQ[] = [
  {
    id: 1,
    question: "كيف يمكنني إيداع الأموال؟",
    answer: "يمكنك إيداع USDC عبر شبكة TRC20. انتقل إلى صفحة المحفظة، انسخ عنوان الإيداع، وأرسل USDC من محفظتك الخارجية. سيتم تأكيد الإيداع خلال 10-30 دقيقة.",
    category: "الإيداع والسحب",
  },
  {
    id: 2,
    question: "ما هو الحد الأدنى للإيداع؟",
    answer: "الحد الأدنى للإيداع هو 100 USDC.",
    category: "الإيداع والسحب",
  },
  {
    id: 3,
    question: "كم يستغرق السحب؟",
    answer: "طلبات السحب تحتاج موافقة الإدارة وعادة تتم معالجتها خلال 24-48 ساعة عمل.",
    category: "الإيداع والسحب",
  },
  {
    id: 4,
    question: "كيف يعمل البوت؟",
    answer: "البوت يستخدم خوارزميات تداول متقدمة تعتمد على مؤشرات RSI وMACD والمتوسطات المتحركة، بالإضافة إلى تحليل الذكاء الاصطناعي للسوق.",
    category: "التداول",
  },
  {
    id: 5,
    question: "ما هو نظام الحصص (NAV)؟",
    answer: "نظام NAV (صافي قيمة الأصول) يحدد قيمة حصتك في الصندوق. عند الإيداع تحصل على وحدات بناءً على سعر NAV الحالي، وعند السحب تحصل على قيمة وحداتك بسعر NAV الحالي.",
    category: "التداول",
  },
  {
    id: 6,
    question: "هل يمكنني سحب أموالي في أي وقت؟",
    answer: "نعم، لكن يجب انتظار 7 أيام من آخر إيداع قبل السحب (فترة القفل).",
    category: "الإيداع والسحب",
  },
  {
    id: 7,
    question: "كيف يعمل برنامج الإحالات؟",
    answer: "شارك رابط الإحالة الخاص بك مع أصدقائك. عندما يسجلون ويودعون، تحصل على 5% من إيداعاتهم كعمولة.",
    category: "الإحالات",
  },
];

export default function Support() {
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
      toast.success("تم إرسال التذكرة بنجاح");
      setShowNewTicket(false);
      setNewSubject("");
      setNewMessage("");
      setNewPriority("medium");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets"] });
    },
    onError: () => {
      toast.error("فشل في إرسال التذكرة");
    },
  });

  // Reply mutation
  const replyMutation = useMutation({
    mutationFn: async ({ ticketId, message }: { ticketId: number; message: string }) => {
      return api.post(`/support/tickets/${ticketId}/reply`, { message });
    },
    onSuccess: () => {
      toast.success("تم إرسال الرد");
      setReplyMessage("");
      queryClient.invalidateQueries({ queryKey: ["/api/v1/support/tickets"] });
    },
    onError: () => {
      toast.error("فشل في إرسال الرد");
    },
  });

  const handleCreateTicket = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubject.trim() || !newMessage.trim()) {
      toast.error("يرجى ملء جميع الحقول");
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

  const filteredFaqs = FAQS.filter(
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
            مفتوحة
          </Badge>
        );
      case "in_progress":
        return (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
            <Clock className="w-3 h-3 ml-1" />
            قيد المعالجة
          </Badge>
        );
      case "resolved":
        return (
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
            <CheckCircle className="w-3 h-3 ml-1" />
            تم الحل
          </Badge>
        );
      case "closed":
        return (
          <Badge variant="outline" className="bg-muted text-muted-foreground">
            مغلقة
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "high":
        return <Badge variant="destructive">عاجل</Badge>;
      case "medium":
        return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500">متوسط</Badge>;
      case "low":
        return <Badge variant="outline">عادي</Badge>;
      default:
        return <Badge variant="outline">{priority}</Badge>;
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">الدعم الفني</h1>
          <p className="text-muted-foreground text-sm">نحن هنا لمساعدتك</p>
        </div>
        <button
          onClick={() => setShowNewTicket(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
          تذكرة جديدة
        </button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="faq" className="gap-2">
            <HelpCircle className="w-4 h-4" />
            الأسئلة الشائعة
          </TabsTrigger>
          <TabsTrigger value="tickets" className="gap-2">
            <MessageSquare className="w-4 h-4" />
            تذاكري
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
              placeholder="ابحث في الأسئلة الشائعة..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pr-10 px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
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

          {/* Contact Info */}
          <Card>
            <CardContent className="p-6">
              <div className="text-center">
                <MessageSquare className="w-12 h-12 text-primary mx-auto mb-4" />
                <h3 className="font-bold text-lg mb-2">لم تجد إجابة لسؤالك؟</h3>
                <p className="text-muted-foreground mb-4">
                  تواصل معنا عبر إنشاء تذكرة دعم وسنرد عليك في أقرب وقت
                </p>
                <button
                  onClick={() => {
                    setActiveTab("tickets");
                    setShowNewTicket(true);
                  }}
                  className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
                >
                  إنشاء تذكرة
                </button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tickets Tab */}
        <TabsContent value="tickets" className="space-y-4 mt-6">
          {loadingTickets ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : tickets.length > 0 ? (
            <div className="space-y-3">
              {tickets.map((ticket) => (
                <Card
                  key={ticket.id}
                  className={cn(
                    "cursor-pointer hover:border-primary/50 transition-colors",
                    selectedTicket?.id === ticket.id && "border-primary"
                  )}
                  onClick={() => setSelectedTicket(ticket)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-medium">{ticket.subject}</h3>
                          {getStatusBadge(ticket.status)}
                          {getPriorityBadge(ticket.priority)}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {ticket.message}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          {format(new Date(ticket.created_at), "dd/MM/yyyy HH:mm")}
                        </p>
                      </div>
                      <FileText className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="p-12 text-center">
                <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">لا توجد تذاكر</p>
                <p className="text-sm text-muted-foreground mt-1">
                  أنشئ تذكرة جديدة للتواصل مع فريق الدعم
                </p>
                <button
                  onClick={() => setShowNewTicket(true)}
                  className="mt-4 px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
                >
                  إنشاء تذكرة
                </button>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* New Ticket Modal */}
      {showNewTicket && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>تذكرة جديدة</CardTitle>
                <button
                  onClick={() => setShowNewTicket(false)}
                  className="p-2 hover:bg-muted rounded-lg"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateTicket} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">الموضوع</label>
                  <input
                    type="text"
                    value={newSubject}
                    onChange={(e) => setNewSubject(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                    placeholder="موضوع التذكرة"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">الأولوية</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                  >
                    <option value="low">عادي</option>
                    <option value="medium">متوسط</option>
                    <option value="high">عاجل</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">الرسالة</label>
                  <textarea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    className="w-full px-4 py-3 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none resize-none"
                    rows={5}
                    placeholder="اكتب رسالتك هنا..."
                  />
                </div>

                <div className="flex gap-3">
                  <button
                    type="submit"
                    disabled={createTicketMutation.isPending}
                    className="flex-1 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    <Send className="w-4 h-4" />
                    {createTicketMutation.isPending ? "جاري الإرسال..." : "إرسال التذكرة"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowNewTicket(false)}
                    className="flex-1 py-3 bg-muted rounded-lg font-medium hover:bg-muted/80"
                  >
                    إلغاء
                  </button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Ticket Details Modal */}
      {selectedTicket && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
            <CardHeader className="flex-shrink-0">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>{selectedTicket.subject}</CardTitle>
                  <div className="flex items-center gap-2 mt-2">
                    {getStatusBadge(selectedTicket.status)}
                    {getPriorityBadge(selectedTicket.priority)}
                  </div>
                </div>
                <button
                  onClick={() => setSelectedTicket(null)}
                  className="p-2 hover:bg-muted rounded-lg"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto space-y-4">
              {/* Original Message */}
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground mb-1">
                  {format(new Date(selectedTicket.created_at), "dd/MM/yyyy HH:mm")}
                </p>
                <p>{selectedTicket.message}</p>
              </div>

              {/* Replies */}
              {selectedTicket.replies?.map((reply) => (
                <div
                  key={reply.id}
                  className={cn(
                    "p-4 rounded-lg",
                    reply.is_admin ? "bg-primary/10 mr-8" : "bg-muted/50 ml-8"
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium">
                      {reply.is_admin ? "فريق الدعم" : "أنت"}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {format(new Date(reply.created_at), "dd/MM/yyyy HH:mm")}
                    </span>
                  </div>
                  <p>{reply.message}</p>
                </div>
              ))}

              {/* Reply Form */}
              {selectedTicket.status !== "closed" && (
                <form onSubmit={handleReply} className="flex gap-2">
                  <input
                    type="text"
                    value={replyMessage}
                    onChange={(e) => setReplyMessage(e.target.value)}
                    className="flex-1 px-4 py-2 bg-muted rounded-lg border border-border focus:border-primary focus:outline-none"
                    placeholder="اكتب ردك..."
                  />
                  <button
                    type="submit"
                    disabled={replyMutation.isPending || !replyMessage.trim()}
                    className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
