import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, X, Send, Sparkles, Bot, User, Loader2, ChevronDown } from "lucide-react";


type Message = {
    id: string;
    role: "bot" | "user";
    content: string;
};

const INITIAL_MESSAGES: Message[] = [
    {
        id: "init-1",
        role: "bot",
        content: "Hi there! I am your AI Solar Assistant. How can I help you today? \n\nYou can ask me about:\n- How our satellite analysis works\n- Subsidy details (PM Surya Ghar Yojana)\n- Return on investment & cost savings",
    },
];

const SAMPLE_PROMPTS = [
    "How accurate is the satellite imagery?",
    "What is the PM Surya Ghar subsidy?",
    "How do I start an analysis?",
    "Is the data shared with third parties?",
    "What roof types are compatible with solar?",
    "How long does solar panel installation take?",
    "What is net metering and how does it work?",
    "Do solar panels work on cloudy days?",
    "How much roof space do I need for 3kW?",
    "What are the maintenance costs for solar?",
    "Can solar panels withstand heavy hail?",
    "What happens during a grid power outage?",
    "How long do solar panels last?",
    "What is the ROI for a residential solar setup?",
    "Are there any state-level subsidies available?",
    "How do I clean my solar panels?",
    "Do solar panels damage the roof?",
    "Can I go completely off-grid?",
    "What is the average efficiency of residential panels?",
    "How do I apply for the PM Surya Ghar scheme?",
    "Which inverter type is best for my house?",
    "Do I need a battery backup system?",
    "Will solar panels increase my property value?",
    "How are solar panels recycled at end-of-life?"
];

// Lightweight markdown-lite formatter to bypass ESM crashes from heavy libraries
const renderFormattedText = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={i} className="text-secondary opacity-90">{part.slice(2, -2)}</strong>;
        }
        return <span key={i}>{part}</span>;
    });
};

// Helper to simulate a highly intelligent AI by using robust pattern matching
const generateAIResponse = async (userMsg: string): Promise<string> => {
    const lowerMsg = userMsg.toLowerCase();

    // Simulate network delay for realism
    await new Promise(r => setTimeout(r, 1000 + Math.random() * 1000));

    if (lowerMsg.includes("subsidy") || lowerMsg.includes("surya ghar") || lowerMsg.includes("gov") || lowerMsg.includes("government")) {
        return "**PM Surya Ghar Muft Bijli Yojana** is a central government scheme providing up to ₹78,000 in subsidies for rooftop solar installations.\n\n- **1-2 kW:** ₹30,000 per kW\n- **3 kW:** ₹78,000 total\n- **>3 kW:** Capped at ₹78,000\n\nYou can use our **Subsidy Calculator** page to estimate your exact benefits based on your state and energy needs!";
    }
    if (lowerMsg.includes("satellite") || lowerMsg.includes("accurate") || lowerMsg.includes("accuracy") || lowerMsg.includes("imagery")) {
        return "Our system uses high-resolution multispectral orbital data. We process this data through a custom convolutional neural network (CNN) that identifies rooftop structures and calculates the usable area for solar panels.\n\nWhile highly precise (usually within 5-10% of physical surveys), shading from trees or very sudden structural changes might require slight manual adjustments during final installation validation.";
    }
    if (lowerMsg.includes("cost") || lowerMsg.includes("price") || lowerMsg.includes("investment") || lowerMsg.includes("roi")) {
        return "The cost of a solar installation typically ranges from ₹50,000 to ₹60,000 per kW before subsidies. \n\nAfter applying the PM Surya Ghar subsidy, the effective cost drops significantly. Most residential users see a **Return on Investment (ROI) within 3 to 5 years**, after which electricity generation is practically free for the 25-year lifespan of the panels.";
    }
    if (lowerMsg.includes("start") || lowerMsg.includes("how to") || lowerMsg.includes("analyze") || lowerMsg.includes("analysis")) {
        return "It's super easy to get started!\n\n1. Go to the **Solar Analysis** page.\n2. Enter your address or drop a pin on the map.\n3. Click **Scan Area**.\n\nOur AI will instantly fetch the latest imagery, detect your roof area, and generate a customized power generation report.";
    }
    if (lowerMsg.includes("save") || lowerMsg.includes("history") || lowerMsg.includes("account")) {
        return "Yes! If you are logged in, all your Solar SCans and Subsidy Calculations are automatically saved to your **History**. You can access them anytime across any device where you log into Surya Sys.";
    }
    if (lowerMsg.includes("hello") || lowerMsg.includes("hi") || lowerMsg.includes("hey")) {
        return "Hello! I'm here to answer any questions you have about going solar, our detection technology, or government subsidies. What's on your mind?";
    }
    if (lowerMsg.includes("thanks") || lowerMsg.includes("thank you")) {
        return "You're very welcome! Let me know if you need anything else.";
    }
    if (lowerMsg.includes("contact") || lowerMsg.includes("support")) {
        return "If you need human assistance, you can visit our **Contact** page to submit a form, or use the live Ably chat support portal there to talk directly to one of our solar technicians.";
    }

    // Fallback intelligent response
    return "That's an interesting question. Based on our solar database, residential solar is highly dependent on your specific roof metrics and local irradiance.\n\nI recommend running a free scan on our **Solar Analysis** page to get personalized data for your exact location. Can I clarify anything about the timeline, costs, or technology for you?";
};

export default function AIChatbot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
    const [inputValue, setInputValue] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatWindowRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const [randomPrompts, setRandomPrompts] = useState<string[]>([]);

    const shufflePrompts = () => {
        const shuffled = [...SAMPLE_PROMPTS].sort(() => 0.5 - Math.random());
        setRandomPrompts(shuffled.slice(0, 2));
    };

    useEffect(() => {
        // Initial shuffle
        shufflePrompts();

        // Only set up interval if we haven't sent a message yet
        if (messages.length === 1) {
            const interval = setInterval(shufflePrompts, 10000);
            return () => clearInterval(interval);
        }
    }, [messages.length]);

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen, isTyping]);

    const handleSend = async (text: string = inputValue) => {
        if (!text.trim()) return;

        const userMsg: Message = { id: Date.now().toString(), role: "user", content: text };
        setMessages((prev) => [...prev, userMsg]);
        setInputValue("");
        setIsTyping(true);

        const reply = await generateAIResponse(text);

        setIsTyping(false);
        setMessages((prev) => [...prev, { id: (Date.now() + 1).toString(), role: "bot", content: reply }]);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            handleSend();
        }
    };

    // Close on click outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            // Check if chat is open, we have a ref, and the click was outside the ref
            if (isOpen && chatWindowRef.current && !chatWindowRef.current.contains(event.target as Node)) {

                // Don't close if they clicked the toggle button (prevent double toggle)
                const isToggleButton = (event.target as Element).closest('button[data-chat-toggle="true"]');
                if (!isToggleButton) {
                    setIsOpen(false);
                }
            }
        };

        if (isOpen) {
            document.addEventListener("mousedown", handleClickOutside);
        }

        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [isOpen]);

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        ref={chatWindowRef}
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="fixed bottom-24 right-4 md:right-8 w-[350px] max-w-[calc(100vw-32px)] h-[500px] max-h-[calc(100vh-120px)] bg-background/90 backdrop-blur-xl border border-primary/20 rounded-2xl shadow-2xl flex flex-col overflow-hidden z-[100]"
                    >
                        {/* Header */}
                        <div className="bg-primary/10 border-b border-primary/20 p-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="bg-primary/20 p-2 rounded-lg">
                                    <Sparkles className="w-5 h-5 text-primary" />
                                </div>
                                <div>
                                    <h3 className="font-display font-bold text-foreground text-sm">Surya AI Assistant</h3>
                                    <p className="text-[10px] text-primary">Powered by Smart Logic</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="text-muted-foreground hover:text-foreground transition-colors p-1"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth custom-scrollbar pr-2">
                            {messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                                >
                                    <div className={`flex gap-3 max-w-[85%] ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                                        <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted border border-primary/20"}`}>
                                            {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-primary" />}
                                        </div>
                                        <div className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${msg.role === "user"
                                            ? "bg-primary text-primary-foreground rounded-tr-sm"
                                            : "glass-card text-foreground rounded-tl-sm prose prose-invert prose-sm"
                                            }`}>
                                            {msg.role === "user" ? (
                                                msg.content
                                            ) : (
                                                <div className="whitespace-pre-wrap markdown-override text-[13px] leading-6 break-words">
                                                    {renderFormattedText(msg.content)}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {isTyping && (
                                <div className="flex justify-start">
                                    <div className="flex gap-3 max-w-[85%]">
                                        <div className="shrink-0 w-8 h-8 rounded-full bg-muted border border-primary/20 flex items-center justify-center">
                                            <Bot className="w-4 h-4 text-primary" />
                                        </div>
                                        <div className="px-4 py-2.5 rounded-2xl glass-card rounded-tl-sm flex items-center gap-2">
                                            <Loader2 className="w-4 h-4 text-primary animate-spin" />
                                            <span className="text-xs text-muted-foreground">AI is thinking...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Prompts (only show 2 randomly, stacked vertically, hiding after 1st message) */}
                        {messages.length === 1 && !isTyping && (
                            <div className="px-4 pb-2">
                                <div className="flex flex-col gap-2">
                                    {randomPrompts.map((prompt: string, i: number) => (
                                        <button
                                            key={i}
                                            onClick={() => handleSend(prompt)}
                                            className="text-left bg-muted/50 hover:bg-primary/20 border border-primary/10 hover:border-primary/50 text-xs text-foreground px-4 py-2 rounded-xl transition-colors w-full"
                                        >
                                            {prompt}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Input */}
                        <div className="p-3 border-t border-primary/20 bg-background">
                            <div className="relative flex items-center">
                                <input
                                    type="text"
                                    value={inputValue}
                                    onChange={(e) => setInputValue(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Ask anything about solar..."
                                    className="w-full bg-muted/50 border border-primary/20 focus:border-primary rounded-full pl-4 pr-12 py-3 text-sm outline-none transition-colors"
                                    disabled={isTyping}
                                />
                                <button
                                    onClick={() => handleSend()}
                                    disabled={!inputValue.trim() || isTyping}
                                    className="absolute right-2 w-8 h-8 flex items-center justify-center bg-primary text-primary-foreground rounded-full hover:bg-primary/90 disabled:opacity-50 disabled:hover:bg-primary transition-colors"
                                >
                                    <Send className="w-4 h-4 ml-0.5" />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <motion.button
                data-chat-toggle="true"
                onClick={() => setIsOpen(!isOpen)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="fixed bottom-6 right-6 md:right-8 w-14 h-14 bg-primary text-primary-foreground rounded-full flex items-center justify-center shadow-[0_0_20px_hsla(168,100%,48%,0.4)] hover:shadow-[0_0_30px_hsla(168,100%,48%,0.6)] transition-shadow z-[100]"
            >
                <AnimatePresence mode="wait">
                    {isOpen ? (
                        <motion.div
                            key="close"
                            initial={{ rotate: -90, opacity: 0 }}
                            animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: 90, opacity: 0 }}
                            transition={{ duration: 0.15 }}
                        >
                            <ChevronDown className="w-7 h-7" />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="open"
                            initial={{ rotate: 90, opacity: 0 }}
                            animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: -90, opacity: 0 }}
                            transition={{ duration: 0.15 }}
                        >
                            <MessageSquare className="w-6 h-6" />
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.button>
        </>
    );
}
