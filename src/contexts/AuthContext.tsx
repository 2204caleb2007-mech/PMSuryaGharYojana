import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { api, type ApiUser } from "@/lib/api";

interface AuthContextType {
  user: ApiUser | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  loginWithGoogle: (credential: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  showAuthModal: (reason?: string) => void;
  hideAuthModal: () => void;
  authModalOpen: boolean;
  authModalReason: string;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<ApiUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalReason, setAuthModalReason] = useState("");

  // Restore session from stored token on mount
  useEffect(() => {
    const token = api.getToken();
    if (token) {
      api.me()
        .then(setUser)
        .catch(() => {
          api.clearToken();
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  // ── Local email/password login ───────────────────────────────────────────
  const login = useCallback(async (email: string, password: string) => {
    const result = await api.login({ email, password });
    api.setToken(result.access_token);
    setUser(result.user);
    setAuthModalOpen(false);
  }, []);

  // ── Google One-Tap / popup login ─────────────────────────────────────────
  const loginWithGoogle = useCallback(async (credential: string) => {
    const result = await api.googleAuth(credential);
    api.setToken(result.access_token);
    setUser(result.user);
    setAuthModalOpen(false);
  }, []);

  // ── Register ─────────────────────────────────────────────────────────────
  const register = useCallback(async (name: string, email: string, password: string) => {
    const result = await api.register({ name, email, password });
    api.setToken(result.access_token);
    setUser(result.user);
    setAuthModalOpen(false);
  }, []);

  // ── Logout ───────────────────────────────────────────────────────────────
  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // Ignore logout errors — client-side cleanup is enough
    }
    api.clearToken();
    setUser(null);
  }, []);

  // ── Auth modal helpers ───────────────────────────────────────────────────
  const showAuthModal = useCallback((reason = "") => {
    setAuthModalReason(reason);
    setAuthModalOpen(true);
  }, []);

  const hideAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthModalReason("");
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated: !!user,
        login,
        loginWithGoogle,
        register,
        logout,
        showAuthModal,
        hideAuthModal,
        authModalOpen,
        authModalReason,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
