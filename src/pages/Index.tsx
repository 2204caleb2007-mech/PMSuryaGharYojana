import { motion } from "framer-motion";
import CountUp from "react-countup";
import { Rocket, Zap, BarChart3, Shield, ArrowRight, ArrowDown, Cpu, Calculator, Network, Home, Building, DollarSign, PiggyBank, Leaf, ChevronsRight, ChevronsLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import LightPillar from "@/components/LightPillar";
import teamSimple from "./img/teamsimple.jpg";

const stats = [
  { label: "Panels Detected", value: 2847500, suffix: "+", icon: BarChart3 },
  { label: "States Covered", value: 28, suffix: "", icon: Shield },
  { label: "Accuracy Rate", value: 97.3, suffix: "%", decimals: 1, icon: Zap },
  { label: "Scans Today", value: 1482, suffix: "", icon: Cpu },
];

const initiatives = [
  {
    icon: Home,
    title: "For Residential Users",
    color: "amber"
  },
  {
    icon: Building,
    title: "For Commercial Users",
    color: "amber"
  },
  {
    icon: Zap,
    title: "Free Electricity",
  },
  {
    icon: DollarSign,
    title: "Income Opportunity",
  },
  {
    icon: PiggyBank,
    title: "Substantial Subsidies",
  },
  {
    icon: Leaf,
    title: "Green Energy",
  }
];

const HERO_HEADLINES = [
  "Democratizing solar intelligence through satellite AI. Empowering millions of households to navigate the PM Surya Ghar Yojana with surgical precision.",
  "Converting raw satellite data into actionable energy blueprints. Providing real-time neural analysis to accelerate India's transition to a decentralized solar grid.",
  "Mapping the future of Indian energy with multispectral precision. Our AI-driven engine identifies untapped rooftop potential to maximize your Solar Energy.",
  "Advanced geospatial analytics for deploying cutting-edge inference models to simplify subsidy eligibility and system architecture for every citizen.",
  "Orchestrating a nationwide solar transformation through scalable AI. We bridge the gap between orbital data and local generation."
];

const HeroSection = () => {
  const [ctaText, setCtaText] = useState("View Features");
  const [headlineIndex, setHeadlineIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setHeadlineIndex((prev) => (prev + 1) % HERO_HEADLINES.length);
    }, 20000);
    return () => clearInterval(timer);
  }, []);



  const handleCtaClick = () => {
    const featuresSection = document.getElementById("features");
    if (featuresSection) {
      featuresSection.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Light Pillar background effect */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <LightPillar
          topColor="#ffb703"
          bottomColor="#00f5d4"
          intensity={1}
          rotationSpeed={0.5}
          interactive={false}
          glowAmount={0.005}
          pillarWidth={3}
          pillarHeight={0.3}
          noiseIntensity={0.5}
          pillarRotation={90}
        />
      </div>
      {/* Smooth fade transition to next section */}
      <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-background to-transparent pointer-events-none z-0" />

      <div className="container mx-auto px-4 md:px-6 relative z-10 text-center py-20">
        {/* Status chip */}


        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="font-anton tracking-wide text-5xl md:text-7xl lg:text-8xl font-normal mb-6 -mt-16 whitespace-nowrap"
        >
          <span className="text-secondary">
            POWERING{" "}
          </span>
          <span className="text-foreground">
            INDIA'S SOLAR{" "}
          </span>
          <span className="text-primary">
            REVOLUTION
          </span>
        </motion.h1>

        {/* Subtitle - chevrons stay persistent, only text animates */}
        <div className="flex items-center justify-center gap-4 mt-6 px-4 min-h-[120px]">
          {/* Amber left chevron with nudge animation */}
          <motion.div
            animate={{ x: [0, -6, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          >
            <ChevronsRight className="w-12 h-12 md:w-20 md:h-20 text-secondary shrink-0" />
          </motion.div>

          {/* Only the text animates on headline change */}
          <motion.p
            key={headlineIndex}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.8 }}
            className="text-white/90 italic font-bold text-lg md:text-xl max-w-3xl leading-relaxed"
          >
            {HERO_HEADLINES[headlineIndex]}
          </motion.p>

          {/* Cyan right chevron with nudge animation - stays outside key block */}
          <motion.div
            animate={{ x: [0, 6, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          >
            <ChevronsLeft className="w-12 h-12 md:w-20 md:h-20 text-primary shrink-0" />
          </motion.div>
        </div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-58"
        >
          <button
            onClick={handleCtaClick}
            className="bg-white text-black hover:bg-white/90 shadow-[0_0_30px_rgba(255,255,255,0.2)] hover:shadow-[0_0_40px_rgba(255,255,255,0.4)] transition-all font-display font-bold rounded-lg text-lg px-10 py-4 relative group overflow-hidden"
          >
            <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 translate-y-full group-hover:translate-y-0 transition-all duration-300 pointer-events-none" />
            <ArrowDown className="inline w-5 h-5 mr-2 -mt-0.5" />
            {ctaText}
          </button>
        </motion.div>


      </div>
    </section>
  );
};

const Index = () => {
  return (
    <div className="w-full overflow-x-hidden">
      <HeroSection />

      {/* Stats Section */}
      <section className="w-full relative z-10 -mt-10 pb-12 pt-16">
        <div className="w-full px-4 md:px-8 lg:px-16 mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-8 md:gap-12 lg:gap-16"
          >
            {stats.map((stat, i) => (
              <div key={i} className="text-center flex flex-col items-center">
                <stat.icon className="w-8 h-8 text-primary mb-4" />
                <div className="font-display text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mb-2">
                  <CountUp
                    end={stat.value}
                    duration={2.5}
                    separator=","
                    decimals={stat.decimals || 0}
                    suffix={stat.suffix}
                    enableScrollSpy
                    scrollSpyOnce
                  />
                </div>
                <div className="font-mono-tactical text-xs md:text-sm text-muted-foreground tracking-widest uppercase">
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Quick Actions / Features */}
      <section className="py-24 relative" id="features">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-16">
            <h2 className="font-display text-4xl md:text-5xl font-bold text-foreground">
              Features
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Subsidy Estimator */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="glass-card rounded-2xl p-8 relative overflow-hidden group h-full flex flex-col hover:border-secondary/40 transition-all border border-border/50"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-secondary/5 to-transparent" />
              <div className="relative z-10 flex flex-col justify-between h-full flex-grow">
                <div>
                  <Calculator className="w-8 h-8 text-secondary mb-4" />
                  <h3 className="font-display text-2xl font-bold text-foreground mb-3">
                    Subsidy Estimator
                  </h3>
                  <p className="text-muted-foreground mb-8 text-sm leading-relaxed">
                    Calculate your PM Surya Ghar subsidy eligibility. Our platform analyzes your setup to map it to the correct central financing and state-level benefits automatically.
                  </p>
                </div>
                <div>
                  <Link to="/calculator" className="inline-flex items-center gap-2 px-8 py-3 rounded-lg font-display font-semibold transition-all duration-300 bg-secondary text-secondary-foreground shadow-[0_0_20px_rgba(255,183,3,0.3)] hover:shadow-[0_0_30px_rgba(255,183,3,0.5)] hover:-translate-y-0.5">
                    Launch Estimator <ArrowRight className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            </motion.div>

            {/* AI Box */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="glass-card rounded-2xl p-8 relative overflow-hidden group h-full flex flex-col"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent" />
              <div className="relative z-10 flex flex-col justify-between h-full flex-grow">
                <div>
                  <Cpu className="w-8 h-8 text-primary mb-4" />
                  <h3 className="font-display text-2xl font-bold text-foreground mb-3">
                    AI Solar{" "}
                    <span className="neon-text">Analysis Engine</span>
                  </h3>
                  <p className="text-muted-foreground mb-8 text-sm leading-relaxed">
                    Upload coordinates or click the satellite map to initiate a neural
                    scan. Our AI processes multispectral imagery to detect and analyze
                    solar installations in real-time.
                  </p>
                </div>
                <div>
                  <Link to="/analysis" className="btn-cyan inline-flex items-center gap-2">
                    Launch Analysis <ArrowRight className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            </motion.div>

            {/* System Architecture */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="glass-card rounded-2xl p-8 relative overflow-hidden group h-full flex flex-col hover:border-primary/40 transition-all border border-border/50"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent flex" />
              <div className="relative z-10 flex flex-col justify-between h-full flex-grow">
                <div>
                  <Network className="w-8 h-8 text-primary mb-4" />
                  <h3 className="font-display text-2xl font-bold text-foreground mb-3">
                    System Architecture
                  </h3>
                  <p className="text-muted-foreground mb-8 text-sm leading-relaxed">
                    Explore how our analysis pipeline uniquely connects satellite imagery, neural networks in real-time, and robust backend infrastructure to process data and yield accurate scans.
                  </p>
                </div>
                <div>
                  <Link to="/process" className="btn-cyan inline-flex items-center gap-2">
                    View Architecture <ArrowRight className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Static Features List */}
          <div className="mt-16 w-full">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-y-12 gap-x-4 px-2">
              {initiatives.map((item, i) => (
                <div key={i} className="flex flex-col items-center justify-start text-center px-2">
                  <item.icon className={`w-10 h-10 mb-4 drop-shadow-lg ${item.color === 'amber' ? 'text-secondary' : 'text-primary'}`} />
                  <h4 className="font-display text-base md:text-lg font-bold text-foreground mb-2 tracking-wide leading-tight">
                    {item.title}
                  </h4>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Initiative Section */}
      <section className="pb-10 pt-12 relative">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-8">
            <h2 className="font-display text-4xl md:text-5xl font-bold text-foreground">
              About us
            </h2>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-6">
                Our Vision:{" "}
                <span className="amber-glow">Solar Future</span>
              </h2>
              <p className="text-muted-foreground leading-relaxed mb-6">
                Surya Systems leverages cutting-edge satellite imagery and neural
                network inference to provide comprehensive solar analytics.
                Supporting the PM Surya Ghar Yojana initiative with AI-driven
                insights for rooftop solar adoption across India.
              </p>
              <p className="text-muted-foreground leading-relaxed mb-6">
                Made by Team Simple, this platform processes real-time satellite data to detect existing
                installations, estimate generation potential, and navigate subsidy
                eligibility. Our mission is to catalyze a nation-wide transition to sustainable, self-reliant power generation
                through the power of accessible AI.
              </p>
              <div className="mt-8">
                <a
                  href="/AI-Powered-Solar-Analysis.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-cyan inline-flex items-center gap-2"
                >
                  Read more <ArrowRight className="w-4 h-4" />
                </a>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative h-full flex flex-col items-center justify-center pt-10"
            >
              <div className="relative z-10 w-full aspect-video rounded-2xl overflow-hidden glass-card border-primary/20 p-2 shadow-2xl">
                <img
                  src={teamSimple}
                  alt="Team Surya Systems"
                  className="w-full h-full object-cover rounded-xl"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/40 to-transparent pointer-events-none" />
              </div>
              {/* Decorative element */}
              <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-primary/10 blur-3xl rounded-full" />
              <div className="absolute -top-6 -left-6 w-32 h-32 bg-secondary/10 blur-3xl rounded-full" />
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
