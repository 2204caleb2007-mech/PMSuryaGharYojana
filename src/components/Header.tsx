import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Satellite, Settings, LogIn, LogOut, Menu, X, Clock, Bell } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const Header = () => {
  const location = useLocation();
  const { user, isAuthenticated, logout, showAuthModal } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => {
    return localStorage.getItem("surya_notifications") === "true";
  });

  const navLinks = [
    { label: "Home", path: "/" },
    { label: "Solar Analysis", path: "/analysis" },
    { label: "Subsidy Calculator", path: "/calculator" },
    { label: "How It Works", path: "/process" },
    { label: "Contact", path: "/contact" },
  ];

  // Persist notification preference
  const toggleNotifications = () => {
    const next = !notificationsEnabled;
    setNotificationsEnabled(next);
    localStorage.setItem("surya_notifications", String(next));
    if (next) {
      // Placeholder SMS API call - replace with real provider (e.g. Twilio, MSG91)
      console.log("[SMS API placeholder] Notifications enabled for user:", user?.email);
      // Example: POST /api/sms/subscribe { phone: user.phone, email: user.email }
    }
  };

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close settings on route change
  useEffect(() => {
    setSettingsOpen(false);
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <header
      className={`sticky top-0 z-50 w-full transition-all duration-300 ${scrolled ? "bg-background/70 backdrop-blur-2xl" : "bg-background/50 backdrop-blur-lg"
        }`}
    >
      <div className="container mx-auto px-4 md:px-6 relative h-16 flex items-center">
        {/* Brand - absolutely pinned left so it never affects center nav */}
        <div className="absolute left-4 md:left-6">
          <Link
            to="/"
            className="flex items-center gap-2 group"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <Satellite className="w-6 h-6 text-primary transition-transform group-hover:rotate-12" />
            <span className="font-display font-bold text-lg text-foreground">
              Surya <span className="neon-text">Sys.</span>
            </span>
          </Link>
        </div>

        {/* Desktop nav - TRUE center, unaffected by brand or right-side content */}
        <nav className="hidden md:flex items-center gap-1 mx-auto">
          {navLinks.map((link) => {
            const active = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className={`relative px-3 py-2 text-sm font-medium transition-colors ${active ? "text-primary" : "text-muted-foreground hover:text-foreground"
                  }`}
              >
                {link.label}
                {active && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute bottom-0 left-3 right-3 h-0.5 bg-primary rounded-full"
                    style={{ boxShadow: "0 0 8px hsl(168, 100%, 48%, 0.5)" }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Right side - absolutely pinned right so it NEVER shifts the center nav */}
        <div className="absolute right-4 md:right-6 flex items-center gap-2">
          {isAuthenticated ? (
            <>
              {/* Username chip - no green dot */}
              <div className="hidden md:flex items-center px-3 py-1.5 glass-card rounded-full">
                <span className="font-mono-tactical text-[10px] text-foreground">
                  {user?.name || user?.email?.split("@")[0]}
                </span>
              </div>

              {/* Settings dropdown */}
              <div className="relative hidden md:block">
                <button
                  onClick={() => setSettingsOpen(!settingsOpen)}
                  className="text-muted-foreground hover:text-foreground transition-colors p-1"
                >
                  <Settings className="w-5 h-5" />
                </button>

                <AnimatePresence>
                  {settingsOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 5, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 5, scale: 0.95 }}
                      className="absolute right-0 top-10 w-64 glass-card rounded-xl neon-border overflow-hidden"
                    >
                      <div className="p-2">
                        <Link
                          to="/history"
                          className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-muted/50 transition-colors"
                        >
                          <Clock className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm text-foreground">Analysis History</span>
                        </Link>

                        {/* Notifications toggle */}
                        <button
                          onClick={toggleNotifications}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-muted/50 transition-colors text-left"
                        >
                          <Bell className={`w-4 h-4 ${notificationsEnabled ? "text-primary" : "text-muted-foreground"}`} />
                          <div className="flex-1">
                            <span className="text-sm text-foreground block">SMS Notifications</span>
                            <span className="text-[10px] text-muted-foreground">{notificationsEnabled ? "Enabled" : "Disabled"}</span>
                          </div>
                          <div className={`w-8 h-4 rounded-full transition-colors flex items-center px-0.5 ${notificationsEnabled ? "bg-primary" : "bg-muted"
                            }`}>
                            <div className={`w-3 h-3 rounded-full bg-white transition-transform ${notificationsEnabled ? "translate-x-4" : "translate-x-0"
                              }`} />
                          </div>
                        </button>

                        <button
                          onClick={async () => {
                            await logout();
                            setSettingsOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-destructive/10 transition-colors text-left"
                        >
                          <LogOut className="w-4 h-4 text-destructive" />
                          <span className="text-sm text-destructive">Logout</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </>
          ) : (
            <button
              onClick={() => showAuthModal()}
              className="btn-cyan text-sm !px-4 !py-2 hidden md:flex items-center gap-2"
            >
              <LogIn className="w-4 h-4" />
              Login
            </button>
          )}

          {/* Mobile toggle */}
          <button
            className="md:hidden text-muted-foreground hover:text-foreground"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile nav */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden overflow-hidden glass-card border-t border-border"
          >
            <nav className="flex flex-col p-4 gap-1">
              {navLinks.map((link) => {
                const active = location.pathname === link.path;
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    onClick={() => setMobileOpen(false)}
                    className={`px-3 py-2 rounded-md text-sm font-medium ${active
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                      }`}
                  >
                    {link.label}
                  </Link>
                );
              })}
              {isAuthenticated ? (
                <button
                  onClick={async () => {
                    await logout();
                    setMobileOpen(false);
                  }}
                  className="px-3 py-2 rounded-md text-sm font-medium text-destructive hover:bg-destructive/10 text-left mt-2"
                >
                  <LogOut className="w-4 h-4 inline mr-2" />
                  Logout
                </button>
              ) : (
                <button
                  onClick={() => {
                    showAuthModal();
                    setMobileOpen(false);
                  }}
                  className="btn-cyan text-sm !px-4 !py-2 mt-2 flex items-center justify-center gap-2"
                >
                  <LogIn className="w-4 h-4" />
                  Login
                </button>
              )}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

export default Header;
