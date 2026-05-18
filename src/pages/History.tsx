import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, MapPin, Zap, BarChart3, Eye, Trash2, Loader2, Calculator, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";


const History = () => {
  const { user } = useAuth();
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [activeCategory, setActiveCategory] = useState<"analysis" | "subsidy">("analysis");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const isAdmin = user?.email === "2204caleb2007@gmail.com";

  useEffect(() => {
    if (user) loadHistory();
  }, [user]);

  const loadHistory = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.getMyAnalyses(user?.id || "", isAdmin);
      setHistoryItems(data);
    } catch (err: any) {
      setError(err.message || "Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteAnalysis(id, user?.id || "", isAdmin);
      setHistoryItems((prev) => prev.filter((a) => a.id !== id));
    } catch (err: any) {
      setError(err.message || "Failed to delete from history");
    }
  };

  const currentItems = historyItems.filter(item => item.record_type === activeCategory);

  return (
    <div className="min-h-screen py-16">
      <div className="container max-w-6xl mx-auto px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-3">
            <Clock className="inline w-8 h-8 text-primary mr-3 -mt-1" />
            Analysis History
          </h1>
          <p className="text-muted-foreground flex items-center gap-2">
            Logged in as <span className="text-primary font-mono text-sm">{user?.name || user?.email}</span>
            {isAdmin && <span className="glass-card px-2 py-0.5 rounded text-[10px] text-amber-500 border-amber-500/30 flex items-center gap-1 uppercase tracking-wider"><ShieldCheck className="w-3 h-3" /> Admin View Active</span>}
          </p>
        </motion.div>

        <div className="flex gap-4 border-b border-border/50 mb-8 pb-3 overflow-x-auto custom-scrollbar">
          <button
            onClick={() => setActiveCategory("analysis")}
            className={`font-display text-sm font-semibold tracking-wide transition-colors whitespace-nowrap px-4 py-2 rounded-lg ${activeCategory === "analysis" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
          >
            Solar Analysis Reports
          </button>
          <button
            onClick={() => setActiveCategory("subsidy")}
            className={`font-display text-sm font-semibold tracking-wide transition-colors whitespace-nowrap px-4 py-2 rounded-lg ${activeCategory === "subsidy" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
          >
            Subsidy Reports
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
          </div>
        ) : error ? (
          <div className="glass-card rounded-2xl p-8 text-center">
            <p className="text-destructive text-sm mb-4">{error}</p>
            <button onClick={loadHistory} className="btn-cyan !px-6 !py-2 text-sm">
              Retry
            </button>
          </div>
        ) : currentItems.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl p-12 text-center mt-12"
          >
            <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-4">
              {activeCategory === "analysis" ? (
                <BarChart3 className="w-7 h-7 text-muted-foreground" />
              ) : (
                <Calculator className="w-7 h-7 text-muted-foreground" />
              )}
            </div>
            <h3 className="font-display font-bold text-foreground mb-2">No {activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)} Reports Yet</h3>
            <p className="text-sm text-muted-foreground mb-6">
              You haven't generated any {activeCategory} reports yet.
            </p>
            <Link to={activeCategory === "analysis" ? "/analysis" : "/calculator"} className="btn-cyan inline-flex items-center gap-2">
              {activeCategory === "analysis" ? "Run New Scan" : "Calculate Subsidy"}
            </Link>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence mode="popLayout">
              {currentItems.map((item, i) => (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: i * 0.05 }}
                  className="glass-card flex flex-col rounded-2xl p-6 group hover:border-primary/30 transition-colors"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <span className="font-mono text-xs text-primary">{item.sample_id || item.id}</span>
                      <p className="text-[10px] text-muted-foreground font-mono-tactical mt-1">
                        {new Date(item.created_at).toLocaleDateString()} · {new Date(item.created_at).toLocaleTimeString()}
                      </p>
                    </div>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="text-muted-foreground/40 hover:text-destructive transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {isAdmin && (
                    <div className="mb-4 text-xs font-mono text-muted-foreground bg-muted/20 px-3 py-1.5 rounded-lg border border-border/50 truncate">
                      User: {item.user_id}
                    </div>
                  )}

                  {activeCategory === "analysis" && (
                    <>
                      {item.latitude && item.longitude && (
                        <div className="flex items-center gap-2 mb-4 text-xs text-muted-foreground font-mono">
                          <MapPin className="w-3.5 h-3.5 text-primary" />
                          <span>{Number(item.latitude).toFixed(4)}°N</span>
                          <span className="text-muted-foreground/30">|</span>
                          <span>{Number(item.longitude).toFixed(4)}°E</span>
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-3 mb-5 mt-auto">
                        <div className="bg-muted/30 rounded-lg p-3">
                          <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-1">Panels</span>
                          <span className="font-display font-bold text-foreground">{item.panelCount || item.panel_count || '0'}</span>
                        </div>
                        <div className="bg-muted/30 rounded-lg p-3">
                          <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-1">Capacity</span>
                          <div className="flex items-baseline gap-1">
                            <span className="font-display font-bold text-secondary">{item.powerEstimateKw || item.estimated_capacity || '0'}</span>
                            <span className="text-[10px] text-muted-foreground">kW</span>
                          </div>
                        </div>
                      </div>

                      <Link
                        to={`/analysis?id=${item.id}`}
                        className="mt-auto w-full glass-card rounded-lg px-4 py-2 text-sm font-display font-medium text-primary flex items-center justify-center gap-2 hover:bg-primary/5 transition-colors border border-primary/20"
                      >
                        <Eye className="w-4 h-4" />
                        View Scan
                      </Link>
                    </>
                  )}

                  {activeCategory === "subsidy" && (
                    <>
                      <div className="flex items-center justify-between mb-4 border-b border-border/50 pb-3">
                        <span className="text-sm font-mono-tactical text-muted-foreground uppercase">{item.state}</span>
                        <span className="font-bold text-foreground font-display">{item.capacity} kW</span>
                      </div>

                      <div className="mb-5 mt-auto text-center py-4 bg-muted/20 rounded-xl border border-border/50">
                        <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-2 uppercase tracking-wider">Estimated Subsidy</span>
                        <div className="font-display font-bold text-xl text-secondary">
                          ₹{item.subsidy_min?.toLocaleString()} - ₹{item.subsidy_max?.toLocaleString()}
                        </div>
                      </div>
                    </>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>


    </div>
  );
};

export default History;
