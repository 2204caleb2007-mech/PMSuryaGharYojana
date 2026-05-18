import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { AuthProvider } from "./contexts/AuthContext";
import Header from "./components/Header";
import Footer from "./components/Footer";
import GuestBanner from "./components/GuestBanner";
import AuthModal from "./components/AuthModal";
import ProtectedRoute from "./components/ProtectedRoute";
import Index from "./pages/Index";
import Analysis from "./pages/Analysis";
import Calculator from "./pages/Calculator";
import Process from "./pages/Process";
import Contact from "./pages/Contact";
import History from "./pages/History";
import AIChatbot from "./components/AIChatbot";

// Scroll to top whenever the route changes
const ScrollToTop = () => {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname]);
  return null;
};

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <AuthModal />
        <BrowserRouter>
          <ScrollToTop />
          <div className="flex flex-col min-h-screen">
            <Header />
            <GuestBanner />
            <main className="flex-1">
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/analysis" element={<Analysis />} />
                <Route path="/calculator" element={<Calculator />} />
                <Route path="/process" element={<Process />} />
                <Route path="/contact" element={<Contact />} />
                <Route
                  path="/history"
                  element={
                    <ProtectedRoute>
                      <History />
                    </ProtectedRoute>
                  }
                />

              </Routes>
            </main>
            <Footer />
            <AIChatbot />
          </div>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
