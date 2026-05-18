import { motion, AnimatePresence } from "framer-motion";
import { X, AlertCircle } from "lucide-react";
import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";

const GuestBanner = () => {
  const [visible, setVisible] = useState(false);
  const { isAuthenticated, showAuthModal } = useAuth();
  const location = useLocation();

  useEffect(() => {
    // Show banner on route change if not authenticated and not on index page
    if (!isAuthenticated && location.pathname !== "/") {
      setVisible(true);
      // Auto-hide after 5 seconds
      const timer = setTimeout(() => {
        setVisible(false);
      }, 5000);
      return () => clearTimeout(timer);
    } else {
      setVisible(false);
    }
  }, [location.pathname, isAuthenticated]);

  return (
    <AnimatePresence>
      {visible && !isAuthenticated && location.pathname !== "/" && (
        <motion.div
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
          className="fixed top-20 right-4 md:right-6 z-[100] max-w-sm w-full shadow-2xl"
        >
          <div className="glass-card-amber rounded-xl overflow-hidden pointer-events-auto border border-amber/30">
            <div className="p-4 relative">
              <button
                onClick={() => setVisible(false)}
                className="absolute top-2 right-2 text-muted-foreground hover:text-foreground transition-colors p-1"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>

              <div className="flex items-start gap-3">
                <div className="mt-0.5 relative flex items-center justify-center">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75" />
                  <AlertCircle className="w-5 h-5 text-secondary relative z-10" />
                </div>
                <div className="flex-1 pr-6">
                  <h4 className="font-mono-tactical text-xs font-semibold text-secondary mb-1">
                    Guest Mode
                  </h4>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Results will not be saved — some features restricted
                  </p>

                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default GuestBanner;
