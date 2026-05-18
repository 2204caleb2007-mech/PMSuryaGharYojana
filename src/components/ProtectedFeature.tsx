import { Lock } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import type { ReactNode } from "react";

interface ProtectedFeatureProps {
  children: ReactNode;
  reason?: string;
  /** Render a lock overlay instead of hiding content */
  mode?: "overlay" | "replace" | "disable";
  /** Custom fallback for replace mode */
  fallback?: ReactNode;
}

/**
 * Wraps content that requires authentication.
 * - overlay: Shows content with a blurred lock overlay on top
 * - replace: Replaces content with a lock message
 * - disable: Renders children but disables interaction with pointer-events-none
 */
const ProtectedFeature = ({
  children,
  reason = "Login required to access this feature",
  mode = "replace",
  fallback,
}: ProtectedFeatureProps) => {
  const { isAuthenticated, showAuthModal } = useAuth();

  if (isAuthenticated) return <>{children}</>;

  if (mode === "overlay") {
    return (
      <div className="relative">
        <div className="blur-sm pointer-events-none select-none opacity-60">
          {children}
        </div>
        <button
          onClick={() => showAuthModal(reason)}
          className="absolute inset-0 flex items-center justify-center bg-background/40 backdrop-blur-sm rounded-lg cursor-pointer z-10"
        >
          <div className="glass-card rounded-xl px-4 py-3 flex items-center gap-2 neon-border">
            <Lock className="w-4 h-4 text-primary" />
            <span className="font-mono-tactical text-[10px] text-primary">Login Required</span>
          </div>
        </button>
      </div>
    );
  }

  if (mode === "disable") {
    return (
      <div
        className="pointer-events-none opacity-50 select-none cursor-not-allowed"
        onClick={() => showAuthModal(reason)}
      >
        {children}
      </div>
    );
  }

  // replace mode
  if (fallback) return <>{fallback}</>;

  return (
    <button
      onClick={() => showAuthModal(reason)}
      className="glass-card rounded-lg px-3 py-3 flex items-center justify-center gap-2 w-full text-muted-foreground/50 hover:text-muted-foreground hover:border-primary/20 transition-colors cursor-pointer"
    >
      <Lock className="w-4 h-4" />
      <span className="text-xs">{reason}</span>
    </button>
  );
};

export default ProtectedFeature;
