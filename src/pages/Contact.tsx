import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mail, Phone, MapPin, MessageCircle, Send, Lock, CheckCircle2,
  AlertTriangle, X, Download, Trash2
} from "lucide-react";

import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";
import * as Ably from "ably";

interface ChatMessage {
  id: string;
  type: "sent" | "received" | "system";
  text: string;
  sender?: string;
  time: string;
}

const CHAT_STORAGE_KEY = "surya_chat_history";
const ABLY_KEY = "3rDong.v8h2mg:Ao2k0ULIPJBuJtcV3Wk_bStlfAM2dMo3Khp7XuPeLeE";
const FORMSPREE_URL = "https://formspree.io/f/xwpgqryb";

const Contact = () => {
  const { isAuthenticated, user, showAuthModal } = useAuth();
  const [formData, setFormData] = useState({
    firstName: "", lastName: "", email: "", phone: "", message: "",
  });
  const [formStatus, setFormStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const ablyRef = useRef<Ably.Realtime | null>(null);
  const channelRef = useRef<Ably.RealtimeChannel | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load chat history from localStorage, purging any legacy prefixed messages
  useEffect(() => {
    const PREFIXED_TEXTS = [
      "Live support connected. Welcome to Surya Systems.",
      "Hello! How can I help you with solar analysis today?",
      "Chat history cleared."
    ];
    const saved = localStorage.getItem(CHAT_STORAGE_KEY);
    if (saved) {
      try {
        const parsed: ChatMessage[] = JSON.parse(saved);
        const cleaned = parsed.filter(
          (msg) => !PREFIXED_TEXTS.some((t) => msg.text.trim().startsWith(t.trim()))
        );
        // If we cleaned any out, persist the cleaned version immediately
        if (cleaned.length !== parsed.length) {
          if (cleaned.length > 0) {
            localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(cleaned));
          } else {
            localStorage.removeItem(CHAT_STORAGE_KEY);
          }
        }
        setChatMessages(cleaned);
      } catch (e) {
        console.error("Failed to parse chat history", e);
      }
    } else {
      setChatMessages([]);
    }
  }, []);

  // Save chat history to localStorage
  useEffect(() => {
    if (chatMessages.length > 0) {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatMessages));
    }
  }, [chatMessages]);

  // Auto scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatOpen]);

  // Handle Ably connection
  useEffect(() => {
    if (isAuthenticated && chatOpen && !ablyRef.current) {
      const ably = new Ably.Realtime({ key: ABLY_KEY, clientId: user?.id || "anonymous" });
      const channel = ably.channels.get("public-support-chat");

      channel.subscribe("message", (msg) => {
        // Only add if it's not from us (to avoid duplicates if Ably echoes)
        if (msg.clientId !== (user?.id || "anonymous")) {
          const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
          setChatMessages((prev) => [
            ...prev,
            {
              id: msg.id || Date.now().toString(),
              type: "received",
              text: msg.data.text,
              sender: msg.data.sender || "Support",
              time: now
            },
          ]);
        }
      });

      ablyRef.current = ably;
      channelRef.current = channel;
    }

    return () => {
      if (ablyRef.current) {
        ablyRef.current.close();
        ablyRef.current = null;
        channelRef.current = null;
      }
    };
  }, [isAuthenticated, chatOpen, user?.id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormStatus("loading");

    try {
      const response = await fetch(FORMSPREE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        setFormStatus("success");
      } else {
        setFormStatus("error");
      }
    } catch (error) {
      console.error("Formspree error:", error);
      setFormStatus("error");
    }
  };

  const sendChat = () => {
    if (!chatInput.trim()) return;
    if (!isAuthenticated) {
      showAuthModal("Please login to send messages in live chat.");
      return;
    }

    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      type: "sent",
      text: chatInput,
      sender: user?.name || "You",
      time: now
    };

    setChatMessages((prev) => [...prev, newMessage]);

    if (channelRef.current) {
      channelRef.current.publish("message", {
        text: chatInput,
        sender: user?.name || "User"
      });
    }

    setChatInput("");

    // Simple auto-reply if Ably is just for show / testing
    if (!channelRef.current) {
      setTimeout(() => {
        const replyTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        setChatMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(), type: "received", sender: "Support Agent",
            text: "This message was sent via Surya Sys Realtime! Our team is processing your request.",
            time: replyTime,
          },
        ]);
      }, 1000);
    }
  };

  const exportChat = () => {
    const text = chatMessages
      .map(m => `[${m.time}] ${m.sender || 'SYSTEM'}: ${m.text}`)
      .join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `surya-systems-chat-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    if (confirm("Clear all chat history?")) {
      const reset = [
        {
          id: "reset-1", type: "system", text: "Chat history cleared.",
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        }
      ];
      setChatMessages(reset as any);
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(reset));
    }
  };

  return (
    <div className="min-h-screen py-16">
      <div className="container max-w-5xl mx-auto px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-3">
            Get In Touch
          </h1>
          <p className="text-muted-foreground max-w-lg mx-auto">
            Questions about solar analysis, subsidies, or our platform? We're here to help.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="glass-card rounded-2xl p-8">
              <h3 className="font-display font-bold text-foreground mb-6">Send a Message</h3>

              {formStatus === "success" ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8"
                >
                  <CheckCircle2 className="w-12 h-12 text-success mx-auto mb-4" />
                  <h4 className="font-display font-bold text-foreground mb-2">Message Sent!</h4>
                  <p className="text-sm text-muted-foreground mb-6">
                    Our team will respond via Formspree within 24 hours.
                  </p>
                  <button
                    onClick={() => setFormStatus("idle")}
                    className="text-primary text-sm hover:underline"
                  >
                    Send another message
                  </button>
                </motion.div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                        First Name
                      </label>
                      <input
                        type="text"
                        name="firstName"
                        required
                        value={formData.firstName}
                        onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                        className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                        Last Name
                      </label>
                      <input
                        type="text"
                        name="lastName"
                        required
                        value={formData.lastName}
                        onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                        className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">Email</label>
                    <input
                      type="email"
                      name="email"
                      required
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                    />
                  </div>

                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">Phone</label>
                    <input
                      type="tel"
                      name="phone"
                      value={formData.phone}
                      onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                    />
                  </div>

                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">Message</label>
                    <textarea
                      name="message"
                      required
                      rows={4}
                      value={formData.message}
                      onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors resize-none"
                    />
                  </div>

                  {formStatus === "error" && (
                    <div className="flex items-center gap-2 text-destructive text-xs bg-destructive/10 rounded-lg px-3 py-2">
                      <AlertTriangle className="w-4 h-4" />
                      Failed to send to Formspree. Please try again.
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={formStatus === "loading"}
                    className="btn-cyan w-full flex items-center justify-center gap-2"
                  >
                    {formStatus === "loading" ? (
                      <span className="animate-pulse">Sending...</span>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        Submit
                      </>
                    )}
                  </button>
                </form>
              )}
            </div>
          </motion.div>

          {/* Right Panel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <AnimatePresence mode="wait">
              {!chatOpen ? (
                <motion.div
                  key="info"
                  exit={{ opacity: 0, x: 20 }}
                  className="glass-card rounded-2xl p-8 h-full flex flex-col justify-between"
                >
                  <div>
                    <h3 className="font-display font-bold text-foreground mb-6">Contact Info</h3>
                    <div className="space-y-5">
                      <div className="flex items-start gap-3">
                        <Mail className="w-5 h-5 text-primary mt-0.5" />
                        <div>
                          <span className="font-mono-tactical text-[10px] text-muted-foreground block">Email</span>
                          <span className="text-sm text-foreground">71382505004.caleb@sritcbe.ac.in</span>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Phone className="w-5 h-5 text-primary mt-0.5" />
                        <div>
                          <span className="font-mono-tactical text-[10px] text-muted-foreground block">Phone</span>
                          <span className="text-sm text-foreground">+91 1234567890</span>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <MapPin className="w-5 h-5 text-primary mt-0.5" />
                        <div>
                          <span className="font-mono-tactical text-[10px] text-muted-foreground block">Office</span>
                          <span className="text-sm text-foreground">To be updated yet</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => setChatOpen(true)}
                    className="mt-8 w-full glass-card rounded-xl px-4 py-3 flex items-center justify-center gap-2 text-sm font-display font-semibold text-primary hover:bg-primary/5 transition-colors border border-primary/20"
                  >
                    <MessageCircle className="w-4 h-4" />
                    Chat with us
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="chat"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="glass-card rounded-2xl overflow-hidden flex flex-col h-[520px]"
                >
                  <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-background/50">
                    <div className="flex items-center gap-2">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-success" />
                      </span>
                      <span className="font-display font-semibold text-foreground text-sm">Surya Sys Realtime</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={exportChat}
                        className="text-muted-foreground hover:text-primary transition-colors"
                        title="Export Chat"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                      <button
                        onClick={clearChat}
                        className="text-muted-foreground hover:text-destructive transition-colors"
                        title="Clear History"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      <button onClick={() => setChatOpen(false)} className="text-muted-foreground hover:text-foreground">
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                    {chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex ${msg.type === "sent" ? "justify-end" : msg.type === "system" ? "justify-center" : "justify-start"}`}
                      >
                        {msg.type === "system" ? (
                          <div className="text-[10px] text-muted-foreground/50 font-mono-tactical py-1 uppercase tracking-tighter">
                            {msg.text}
                          </div>
                        ) : (
                          <div
                            className={`max-w-[85%] rounded-xl px-4 py-2.5 ${msg.type === "sent"
                              ? "bg-primary/20 border border-primary/30"
                              : "bg-muted/50 border border-border"
                              }`}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-mono-tactical text-[9px] text-muted-foreground">
                                {msg.sender}
                              </span>
                              <span className="text-[9px] text-muted-foreground/40">{msg.time}</span>
                            </div>
                            <p className="text-sm text-foreground break-words">{msg.text}</p>
                          </div>
                        )}
                      </div>
                    ))}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Chat input */}
                  <div className="px-4 py-3 border-t border-border bg-background/50">
                    {isAuthenticated ? (
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && sendChat()}
                          placeholder="Type a message..."
                          className="flex-1 bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 transition-colors"
                        />
                        <button onClick={sendChat} className="btn-cyan !px-3 !py-2">
                          <Send className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => showAuthModal("Please login to send messages in live chat.")}
                        className="w-full flex items-center justify-center gap-2 bg-muted/30 border border-border rounded-lg px-3 py-2.5 text-muted-foreground/50 hover:border-primary/20 transition-colors cursor-pointer"
                      >
                        <Lock className="w-4 h-4" />
                        <span className="text-xs">Login to use Support Chat</span>
                      </button>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>


    </div>
  );
};

export default Contact;
