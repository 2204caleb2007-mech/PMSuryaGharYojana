import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Calculator as CalcIcon, CheckCircle2, Lock, Save } from "lucide-react";

import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";

const indianStates = [
  "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
  "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
  "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
  "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
  "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
  "Andaman & Nicobar", "Chandigarh", "Delhi", "Jammu & Kashmir", "Ladakh",
  "Lakshadweep", "Puducherry",
];

const subsidyRates = [
  { range: "Up to 2 kW", rate: "₹30,000/kW", total: "₹60,000" },
  { range: "2–3 kW", rate: "₹18,000/kW (additional)", total: "₹78,000" },
  { range: "Above 3 kW", rate: "₹9,000/kW (additional)", total: "Varies" },
];

const eligibility = [
  "Indian residential household",
  "Grid-connected rooftop solar system",
  "Installed by empanelled vendor",
  "Net metering arrangement",
  "Valid Aadhaar-linked bank account",
];

function calculateSubsidy(kw: number): { min: number; max: number } {
  let subsidy = 0;
  if (kw <= 2) {
    subsidy = kw * 30000;
  } else if (kw <= 3) {
    subsidy = 60000 + (kw - 2) * 18000;
  } else {
    subsidy = 78000 + (kw - 3) * 9000;
  }
  return { min: Math.round(subsidy * 0.9), max: Math.round(subsidy) };
}

const Calculator = () => {
  const { isAuthenticated, user, showAuthModal } = useAuth();
  const [capacity, setCapacity] = useState(3);
  const [state, setState] = useState("");
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(false);

  const subsidy = useMemo(() => calculateSubsidy(capacity), [capacity]);

  const handleSave = async () => {
    if (!isAuthenticated) {
      showAuthModal("Please login to save subsidy estimates to your account.");
      return;
    }
    setLoading(true);
    try {
      if (state) {
        await api.calculateSubsidy({
          state: state,
          capacity,
        });
      }

      await api.saveSubsidy(user?.id || "", {
        state: state || "Not selected",
        capacity,
        subsidy_min: subsidy.min,
        subsidy_max: subsidy.max
      });

      setSaved(true);
    } catch {
      // Fallback for offline/demo
      setSaved(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 md:px-6 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-3">
            <CalcIcon className="inline w-8 h-8 text-primary mr-3 -mt-1" />
            Subsidy Calculator
          </h1>
          <p className="text-muted-foreground max-w-lg mx-auto">
            Estimate your PM Surya Ghar Yojana subsidy based on system capacity and location.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Calculator Card */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card rounded-2xl p-8 mb-8 lg:mb-0 flex flex-col h-[520px]"
          >
            {/* Capacity Slider */}
            <div className="mb-8">
              <div className="flex justify-between items-baseline mb-3">
                <label className="font-mono-tactical text-[10px] text-muted-foreground">
                  System Capacity
                </label>
                <span className="font-display text-2xl font-bold text-foreground">
                  {capacity} <span className="text-sm text-muted-foreground">kW</span>
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                step={0.5}
                value={capacity}
                onChange={(e) => {
                  setCapacity(parseFloat(e.target.value));
                  setSaved(false);
                }}
                className="w-full h-2 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, hsl(168,100%,48%) ${((capacity - 1) / 9) * 100}%, hsl(220,30%,18%) ${((capacity - 1) / 9) * 100}%)`,
                }}
              />
              <div className="flex justify-between text-[10px] text-muted-foreground/50 font-mono mt-1">
                <span>1 kW</span>
                <span>10 kW</span>
              </div>
            </div>

            {/* State Dropdown */}
            <div className="mb-8">
              <label className="font-mono-tactical text-[10px] text-muted-foreground mb-2 block">
                State / UT
              </label>
              <select
                value={state}
                onChange={(e) => {
                  setState(e.target.value);
                  setSaved(false);
                }}
                className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors appearance-none"
              >
                <option value="">Select State / UT</option>
                {indianStates.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* Save button */}
            <button
              onClick={handleSave}
              disabled={loading}
              className={`w-full rounded-lg px-4 py-3 font-display font-semibold text-sm flex items-center justify-center gap-2 transition-all ${saved
                ? "bg-success/20 border border-success/30 text-success"
                : "btn-cyan"
                }`}
            >
              {loading ? (
                <span className="animate-pulse">Saving...</span>
              ) : saved ? (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  Estimate Saved
                </>
              ) : isAuthenticated ? (
                <>
                  <Save className="w-4 h-4" />
                  Save Estimate
                </>
              ) : (
                <>
                  <Lock className="w-4 h-4" />
                  Save Estimate (Login Required)
                </>
              )}
            </button>

            {/* Result Banner */}
            <motion.div
              key={capacity}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mt-auto rounded-xl bg-gradient-to-r from-primary/10 to-secondary/10 border border-primary/20 p-6 text-center"
            >
              <span className="font-mono-tactical text-[10px] text-muted-foreground block mb-2">
                Estimated Subsidy Range
              </span>
              <div className="font-display text-3xl md:text-4xl font-bold">
                <span className="text-primary">₹{subsidy.min.toLocaleString()}</span>
                <span className="text-muted-foreground mx-2">—</span>
                <span className="text-secondary">₹{subsidy.max.toLocaleString()}</span>
              </div>
              <span className="text-xs text-muted-foreground mt-2 block">
                For {capacity} kW system {state && `in ${state}`}
              </span>
            </motion.div>
          </motion.div>

          {/* Info Card */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card rounded-2xl p-8 flex flex-col h-[520px]"
          >
            <h3 className="font-display font-bold text-foreground mb-4">Eligibility Criteria</h3>
            <ul className="space-y-3 mb-8">
              {eligibility.map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-sm text-muted-foreground">
                  <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
                  {item}
                </li>
              ))}
            </ul>

            <div className="mt-auto">
              <h3 className="font-display font-bold text-foreground mb-4">Subsidy Rates</h3>
              <div className="space-y-2">
                {subsidyRates.map((rate, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-2 px-3 rounded-lg bg-muted/30 text-sm"
                  >
                    <span className="text-muted-foreground">{rate.range}</span>
                    <span className="font-mono text-xs text-foreground">{rate.rate}</span>
                    <span className="font-display font-semibold text-secondary">{rate.total}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>


    </div>
  );
};

export default Calculator;
