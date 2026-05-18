import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Lock, AlertTriangle, Eye, EyeOff, Loader2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

// Google Identity Services + Maps globals are declared in src/vite-env.d.ts

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string;

const AuthModal = () => {
  const { authModalOpen, hideAuthModal, authModalReason, login, loginWithGoogle, register } = useAuth();
  const [tab, setTab] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);

  /* ── Google callback — receives credential from GSI popup ──────────────── */
  const handleGoogleCredential = useCallback(
    async (response: { credential: string }) => {
      setGoogleLoading(true);
      setError("");
      try {
        await loginWithGoogle(response.credential);
        toast.success("Signed in with Google successfully!");
      } catch (err: any) {
        const msg = err?.message || "Google sign-in failed. Please try again.";
        setError(msg);
        toast.error(msg);
      } finally {
        setGoogleLoading(false);
      }
    },
    [loginWithGoogle]
  );

  /* ── Initialise GSI every time the modal opens ─────────────────────────── */
  useEffect(() => {
    const initGSI = () => {
      if (!authModalOpen || !GOOGLE_CLIENT_ID || !window.google?.accounts?.id) return;

      try {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: handleGoogleCredential,
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        // Render the official Google button into the container
        const buttonDiv = document.getElementById("google-button-container");
        if (buttonDiv) {
          window.google.accounts.id.renderButton(buttonDiv, {
            theme: "outline",
            size: "large",
            width: buttonDiv.offsetWidth || 350,
            text: "continue_with",
            shape: "rectangular",
            logo_alignment: "left",
          });
        }
      } catch (err) {
        console.error("GSI Init Error:", err);
      }
    };

    // Small delay to ensure the DOM element is rendered and the script is ready
    const timer = setTimeout(initGSI, 150);
    return () => clearTimeout(timer);
  }, [authModalOpen, handleGoogleCredential]);

  const resetForm = () => {
    setName(""); setEmail(""); setPassword(""); setConfirmPassword("");
    setError(""); setShowPassword(false);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      toast.success("Welcome back!");
      resetForm();
    } catch (err: any) {
      setError(err.message || "Login failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) { setError("Passwords do not match."); return; }
    if (password.length < 8)          { setError("Password must be at least 8 characters."); return; }
    setError("");
    setLoading(true);
    try {
      await register(name, email, password);
      toast.success("Account created successfully!");
      resetForm();
    } catch (err: any) {
      setError(err.message || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const switchTab = (t: "login" | "register") => { setTab(t); setError(""); };

  return (
    <AnimatePresence>
      {authModalOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-background/80 backdrop-blur-xl" onClick={hideAuthModal} />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", damping: 25 }}
            className="relative w-full max-w-md glass-card rounded-2xl neon-border overflow-hidden"
          >
            {/* Close */}
            <button
              onClick={hideAuthModal}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors z-10"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Header */}
            <div className="px-8 pt-8 pb-4 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center mx-auto mb-4">
                <Lock className="w-6 h-6 text-primary" />
              </div>
              <h2 className="font-display text-xl font-bold text-foreground">
                {tab === "login" ? "Welcome Back" : "Create Account"}
              </h2>
              {authModalReason && (
                <div className="mt-3 flex items-center gap-2 justify-center text-xs text-secondary bg-secondary/10 rounded-lg px-3 py-2">
                  <Lock className="w-3.5 h-3.5 shrink-0" />
                  {authModalReason}
                </div>
              )}
            </div>

            {/* ── Google Sign-In button ──────────────────────────────────────── */}
            <div className="px-8 pb-2">
              <div className="relative group w-full">
                {/* 1. Custom Visual Button (Cyan) */}
                <button
                  type="button"
                  className="btn-cyan w-full flex items-center justify-center gap-3 relative pointer-events-none"
                >
                  {/* White background circle behind the G logo to make it pop against cyan */}
                  <div className="bg-white rounded-full p-[2px]">
                    <svg width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                      <path fill="none" d="M0 0h48v48H0z"/>
                    </svg>
                  </div>
                  Continue with Google
                </button>

                {/* 2. Official Google Button (Invisible overlay to catch clicks) */}
                <div className="absolute inset-0 z-10 w-full h-full opacity-[0.01] overflow-hidden flex justify-center items-center cursor-pointer">
                  <div id="google-button-container" className="w-[400px] flex justify-center transform scale-y-150" />
                </div>
              </div>

              {googleLoading && (
                <div className="mt-2 flex items-center justify-center gap-2 text-[10px] text-primary animate-pulse font-mono">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Verifying Identity...
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="flex items-center gap-3 px-8 py-2">
              <div className="flex-1 h-px bg-border" />
              <span className="text-[10px] font-mono-tactical text-muted-foreground">OR</span>
              <div className="flex-1 h-px bg-border" />
            </div>

            {/* Tabs */}
            <div className="flex mx-8 rounded-lg overflow-hidden border border-border">
              <button
                onClick={() => switchTab("login")}
                className={`flex-1 py-2 text-sm font-display font-semibold transition-colors ${tab === "login"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
                  }`}
              >
                Login
              </button>
              <button
                onClick={() => switchTab("register")}
                className={`flex-1 py-2 text-sm font-display font-semibold transition-colors ${tab === "register"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
                  }`}
              >
                Register
              </button>
            </div>

            {/* Form */}
            <div className="p-8 pt-6">
              {error && (
                <div className="flex items-center gap-2 text-destructive text-xs bg-destructive/10 rounded-lg px-3 py-2 mb-4">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  {error}
                </div>
              )}

              {tab === "login" ? (
                <form onSubmit={handleLogin} className="space-y-4">
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Email
                    </label>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                      placeholder="eg: operator@gmail.com"
                    />
                  </div>
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? "text" : "password"}
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="btn-cyan w-full flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {loading ? (
                      <><Loader2 className="w-4 h-4 animate-spin" /> Authenticating…</>
                    ) : "Login"}
                  </button>
                </form>
              ) : (
                <form onSubmit={handleRegister} className="space-y-4">
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Full Name
                    </label>
                    <input
                      type="text"
                      required
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Email
                    </label>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                      placeholder="eg: operator@gmail.com"
                    />
                  </div>
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? "text" : "password"}
                        required
                        minLength={8}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div>
                    <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                      Confirm Password
                    </label>
                    <input
                      type="password"
                      required
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="btn-cyan w-full flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {loading ? (
                      <><Loader2 className="w-4 h-4 animate-spin" /> Creating account…</>
                    ) : "Register"}
                  </button>
                </form>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AuthModal;
