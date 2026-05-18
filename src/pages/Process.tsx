import { motion } from "framer-motion";
import {
  Satellite, MapPin, Cpu, Zap, BarChart3, FileCheck,
  ArrowDown
} from "lucide-react";
import { Link } from "react-router-dom";
import TiltedCard from "@/components/TiltedCard";
import SolarpanelImg from "./img/Solarpanel.png";
import EcofriendlyImg from "./img/Ecofriendly.png";

const steps = [
  {
    icon: MapPin,
    title: "Target Acquisition",
    description: "Input GPS coordinates or search for a location on the interactive satellite map. The system locks onto the target area for analysis.",
  },
  {
    icon: Satellite,
    title: "Satellite Imagery Fetch",
    description: "High-resolution multispectral satellite imagery is retrieved from our orbital data sources, processed and tiled for analysis.",
    // Tilted card floats to the LEFT of this step
    card: {
      side: "left" as const,
      imageSrc: SolarpanelImg,
      altText: "Satellite in orbit capturing Earth imagery",
      captionText: "Orbital Data Capture",
    },
  },
  {
    icon: Cpu,
    title: "Neural Inference",
    description: "Our trained convolutional neural network processes the imagery tiles, identifying solar panel signatures through spectral and geometric analysis.",
  },
  {
    icon: Zap,
    title: "Power Estimation",
    description: "Detected panels are measured and cross-referenced with local irradiance data, orientation factors, and seasonal patterns to estimate generation capacity.",
  },
  {
    icon: BarChart3,
    title: "Subsidy Mapping",
    description: "The system automatically maps detected capacity against PM Surya Ghar Yojana subsidy rates, calculating eligibility and estimated benefits.",
    // Tilted card floats to the RIGHT of this step
    card: {
      side: "right" as const,
      imageSrc: EcofriendlyImg,
      altText: "Solar panels generating electricity",
      captionText: "Solar Power Output",
    },
  },
  {
    icon: FileCheck,
    title: "Report Generation",
    description: "A comprehensive scan report is compiled with detection metrics, confidence scores, power estimates, and exportable JSON data logs.",
  },
];

const Process = () => {
  return (
    <div className="min-h-screen py-16">
      {/* max-w-3xl matches original — overflow-visible lets cards bleed out to sides */}
      <div className="container max-w-3xl mx-auto px-4 md:px-6 overflow-visible">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-3">
            How It Works
          </h1>
          <p className="text-muted-foreground max-w-lg mx-auto">
            From satellite acquisition to subsidy calculation — our end-to-end solar analysis pipeline.
          </p>
        </motion.div>

        {/* Timeline */}
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-8 top-4 bottom-20 w-px bg-primary/50 md:block hidden" />

          <div className="space-y-6">
            {steps.map((step, i) => (
              <div key={i}>
                {/* Wrapper is relative so we can absolute-position the tilted card */}
                <div className="relative">
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.45, delay: i * 0.08 }}
                    className="relative flex gap-6"
                  >
                    {/* Number circle */}
                    <div className="hidden md:flex shrink-0 w-16 h-16 items-center justify-center relative z-10">
                      <div
                        className="w-10 h-10 rounded-full bg-background border border-primary/40 flex items-center justify-center"
                        style={{ boxShadow: "0 0 12px hsl(168,100%,48%,0.2)" }}
                      >
                        <span className="font-display font-bold text-primary text-sm">{i + 1}</span>
                      </div>
                    </div>

                    {/* Card — full original width */}
                    <div className="flex-1 glass-card rounded-2xl p-6 relative overflow-hidden group hover:border-primary/30 transition-colors">
                      {/* Left accent bar */}
                      <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-primary to-primary/20 rounded-l-2xl" />
                      <div className="pl-4">
                        <div className="flex items-center gap-3 mb-3">
                          <step.icon className="w-5 h-5 text-primary" />
                          <h3 className="font-display font-bold text-foreground">{step.title}</h3>
                          <span className="md:hidden font-mono-tactical text-[10px] text-primary/50">0{i + 1}</span>
                        </div>
                        <p className="text-sm text-muted-foreground leading-relaxed">{step.description}</p>
                      </div>
                    </div>
                  </motion.div>

                  {/* Tilted card — absolutely floated outside the container, hidden on small screens */}
                  {step.card && (
                    <motion.div
                      initial={{
                        opacity: 0,
                        x: step.card.side === "left" ? -40 : 40,
                        rotate: step.card.side === "left" ? -6 : 6,
                      }}
                      animate={{
                        opacity: 1,
                        x: 0,
                        rotate: step.card.side === "left" ? -2 : 2,
                      }}
                      transition={{ duration: 0.6, delay: i * 0.08 + 0.2 }}
                      className={`absolute top-1/2 -translate-y-1/2 hidden xl:block ${step.card.side === "left"
                        ? "right-[calc(100%+48px)]"
                        : "left-[calc(100%+48px)]"
                        }`}
                    >
                      <TiltedCard
                        imageSrc={step.card.imageSrc}
                        altText={step.card.altText}
                        captionText={step.card.captionText}
                        containerHeight="280px"
                        containerWidth="260px"
                        imageHeight="280px"
                        imageWidth="260px"
                        rotateAmplitude={12}
                        scaleOnHover={1.06}
                        showMobileWarning={false}
                        showTooltip={false}
                      />
                    </motion.div>
                  )}
                </div>

                {/* Arrow connector */}
                {i < steps.length - 1 && (
                  <div className="flex justify-center md:ml-8 py-1">
                    <ArrowDown className="w-4 h-4 text-primary/30" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-center mt-16"
        >
          <Link to="/analysis" className="btn-cyan inline-flex items-center gap-2 text-lg !px-10 !py-4">
            <Cpu className="w-5 h-5" />
            Start Analysis Now
          </Link>
        </motion.div>
      </div>


    </div>
  );
};

export default Process;
