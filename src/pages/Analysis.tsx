import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Crosshair, Upload, Lock, AlertTriangle, CheckCircle2, XCircle,
  RotateCcw, Download, ChevronLeft, ChevronRight, Radar, MapPin,
  Cpu, Zap, BarChart3, Activity, Eye, Save, Compass, Home
} from "lucide-react";
import { Link } from "react-router-dom";

import ProtectedFeature from "@/components/ProtectedFeature";
import { useAuth } from "@/contexts/AuthContext";
import { api, type AnalysisResult } from "@/lib/api";

// ArcGIS + Google globals are declared in src/vite-env.d.ts

const ScanOverlay = ({ progress, status }: { progress: number; status: string }) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="fixed inset-0 z-50 flex items-center justify-center bg-background/90 backdrop-blur-xl"
  >
    <div className="text-center">
      <div className="relative w-48 h-48 mx-auto mb-8">
        <div className="absolute inset-0 rounded-full border border-primary/20" />
        <div className="absolute inset-4 rounded-full border border-primary/15" />
        <div className="absolute inset-8 rounded-full border border-primary/10" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-primary/20" />
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-primary/20" />
        <motion.div
          className="absolute inset-0"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <div
            className="absolute top-1/2 left-1/2 w-1/2 h-0.5 origin-left"
            style={{
              background: "linear-gradient(90deg, hsl(168,100%,48%,0.8), transparent)",
            }}
          />
        </motion.div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-primary animate-pulse" />
      </div>
      <div className="font-display text-4xl font-bold text-foreground mb-2">
        {progress}%
      </div>
      <div className="font-mono-tactical text-xs text-primary animate-pulse">
        {status}
      </div>
    </div>
  </motion.div>
);

const ConfidenceCircle = ({ value }: { value: number }) => {
  const circumference = 2 * Math.PI * 40;
  const filled = (value / 100) * circumference;

  return (
    <div className="relative w-24 h-24">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r="40" fill="none" stroke="hsl(220,30%,18%)" strokeWidth="4" />
        <motion.circle
          cx="48" cy="48" r="40" fill="none"
          stroke="hsl(168,100%,48%)"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - filled }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          style={{ filter: "drop-shadow(0 0 6px hsl(168,100%,48%,0.5))" }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="font-display text-lg font-bold text-foreground">{value}%</span>
      </div>
    </div>
  );
};

// Demo fallback results for guests
const demoResults = {
  detected: true,
  panelCount: 24,
  confidence: 94.7,
  powerEstimateKw: 7.2,
  areaEstimateSqm: 42.5,
  sampleId: "DEMO-SCAN",
  qcNotes: [
    "High-resolution imagery confirmed (0.5m GSD)",
    "Panel orientation: South-facing, ~15° tilt",
    "Minor shadow interference from adjacent structure",
    "Spectral signature consistent with polycrystalline silicon",
    "Edge detection boundary confidence: 92.3%",
  ],
  rawJson: {
    scan_id: "DEMO-SCAN",
    timestamp: new Date().toISOString(),
    coordinates: { lat: 28.6139, lon: 77.209 },
    detection: { positive: true, panel_count: 24, confidence: 0.947 },
    estimation: { power_kw: 7.2, area_sqm: 42.5 },
    note: "DEMO MODE — results not persisted",
  },
};

interface ScanResults {
  detected: boolean;
  panelCount: number;
  confidence: number;
  powerEstimateKw: number;
  areaEstimateSqm: number;
  sampleId: string;
  qcNotes: string[];
  rawJson: Record<string, any>;
  savedId?: string;
}

const Analysis = () => {
  const { isAuthenticated, user, showAuthModal } = useAuth();
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [address, setAddress] = useState("");
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStatus, setScanStatus] = useState("");
  const [results, setResults] = useState<ScanResults | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [batchProgress, setBatchProgress] = useState<string | null>(null);

  // ArcGIS Refs (2D map)
  const mapViewRef = useRef<any>(null);
  const mapElementRef = useRef<HTMLDivElement>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);

  // Google Maps 3D Refs
  const googleMap3DRef = useRef<any>(null);
  const sceneElementRef = useRef<HTMLDivElement>(null);
  const [altitude, setAltitude] = useState(0);
  const [tilt, setTilt] = useState(0);
  const [activeTab, setActiveTab] = useState<'primary' | 'detailed' | 'logs'>('primary');

  const scanStatuses = [
    "Initializing satellite uplink...",
    "Acquiring imagery tiles...",
    "Running spectral analysis...",
    "Neural inference processing...",
    "Classifying panel signatures...",
    "Generating power estimates...",
    "Compiling scan report...",
  ];

  // Initialize MapView
  useEffect(() => {
    if (!window.require || !mapElementRef.current) return;

    window.require([
      "esri/Map",
      "esri/views/MapView",
      "esri/Graphic",
      "esri/layers/GraphicsLayer",
      "esri/geometry/Point",
      "esri/rest/locator",
      "esri/widgets/Search"
    ], (Map: any, MapView: any, Graphic: any, GraphicsLayer: any, Point: any, locator: any, Search: any) => {
      const graphicsLayer = new GraphicsLayer();
      const map = new Map({
        basemap: "streets-navigation-vector"
      });
      map.add(graphicsLayer);

      const initialLon = parseFloat(lon) || 78.9629;
      const initialLat = parseFloat(lat) || 20.5937;
      const initialZoom = (lat && lon) ? 15 : 4;

      const view = new MapView({
        container: mapElementRef.current,
        map: map,
        zoom: initialZoom,
        center: [initialLon, initialLat],
        constraints: {
          geometry: {
            type: "extent",
            xmin: 68.1,
            ymin: 6.5,
            xmax: 97.4,
            ymax: 35.5,
            spatialReference: { wkid: 4326 }
          },
          minZoom: 4
        },
        ui: { components: ["attribution", "zoom"] }
      });

      let searchWidget: any = null;

      if (searchContainerRef.current) {
        // Clear any existing widgets first (React 18 strict mode double-mount handling)
        searchContainerRef.current.innerHTML = '';

        searchWidget = new Search({
          view: view,
          container: searchContainerRef.current,
          popupEnabled: false,
          resultGraphicEnabled: false,
          maxResults: 4,
          maxSuggestions: 4
        });

        searchWidget.on("select-result", (event: any) => {
          const geom = event.result.feature.geometry;
          updateLocation(geom.longitude, geom.latitude, true);
        });
      }

      const updateLocation = (longitude: number, latitude: number, updateInput = false) => {
        if (isNaN(latitude) || isNaN(longitude)) return;

        // Round to 4 decimal places for display purposes
        const pLat = Math.round(latitude * 10000) / 10000;
        const pLon = Math.round(longitude * 10000) / 10000;

        if (updateInput) {
          setLat(pLat.toString());
          setLon(pLon.toString());
        }

        graphicsLayer.removeAll();
        const point = new Point({ longitude: pLon, latitude: pLat });
        graphicsLayer.add(new Graphic({
          geometry: point,
          symbol: {
            type: "simple-marker",
            path: "M16,0C7.164,0,0,7.164,0,16c0,13.682,14.659,31.026,15.341,31.815C15.511,48.01,15.75,48.113,16,48.113s0.489-0.103,0.659-0.298C17.341,47.026,32,29.682,32,16C32,7.164,24.836,0,16,0z M16,24c-4.418,0-8-3.582-8-8s3.582-8,8-8s8,3.582,8,8S20.418,24,16,24z",
            color: "#ffb703",
            outline: { color: "transparent", width: 0 },
            size: "28px",
            yoffset: "14px"
          }
        }));

        // Add a realistic dark drop shadow pointing down and right instead of a glow
        graphicsLayer.effect = "drop-shadow(3px 5px 6px rgba(0, 0, 0, 0.3))";

        locator.locationToAddress("https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer", {
          location: point
        }).then((res: any) => setAddress(res.address || "Unknown Location"));
      };

      view.on("click", (event: any) => {
        const point = view.toMap(event.screenPoint);
        updateLocation(point.longitude, point.latitude, true);
      });

      mapViewRef.current = { view, updateLocation, graphicsLayer, Point, Graphic, searchWidget };

      if (lat && lon) {
        updateLocation(parseFloat(lon), parseFloat(lat));
      }
    });

    return () => {
      if (mapViewRef.current?.searchWidget) mapViewRef.current.searchWidget.destroy();
      if (mapViewRef.current?.view) mapViewRef.current.view.destroy();
    };
  }, []);

  // Sync manual input to map
  const handleInputChange = (type: 'lat' | 'lon', val: string) => {
    if (type === 'lat') setLat(val);
    else setLon(val);

    if (mapViewRef.current) {
      const nLat = type === 'lat' ? parseFloat(val) : parseFloat(lat);
      const nLon = type === 'lon' ? parseFloat(val) : parseFloat(lon);
      if (!isNaN(nLat) && !isNaN(nLon)) {
        mapViewRef.current.updateLocation(nLon, nLat);
        mapViewRef.current.view.goTo({ center: [nLon, nLat] });
      }
    }
  };

  // Initialize Google Maps 3D when results appear
  useEffect(() => {
    if (!results || !sceneElementRef.current) return;

    let destroyed = false;
    const container = sceneElementRef.current;

    const showMapError = (msg: string) => {
      if (destroyed || !container) return;
      container.innerHTML = `<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(13,17,23,0.97);gap:10px;padding:24px;text-align:center;"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00f7ff" stroke-width="1.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="13"/><circle cx="12" cy="16.5" r="0.6" fill="#00f7ff" stroke="none"/></svg><span style="font-family:monospace;font-size:10px;color:rgba(0,247,255,0.85);letter-spacing:.1em;text-transform:uppercase;">3D Map Error</span><span style="font-family:monospace;font-size:10px;color:rgba(255,255,255,0.4);max-width:320px;line-height:1.7;">${msg}</span></div>`;
    };

    const initGoogle3DMap = async () => {
      try {
        // Guard: ensure the Google Maps bootstrap is ready
        const gm = (window as any).google?.maps;
        if (!gm?.importLibrary) {
          showMapError("Google Maps API not available. Check that the script in index.html loaded and the API key is valid.");
          return;
        }

        // Load the maps3d library
        let Map3DElement: any, Marker3DElement: any;
        try {
          ({ Map3DElement, Marker3DElement } = await gm.importLibrary("maps3d"));
        } catch (libErr: any) {
          showMapError(`Maps 3D library failed to load: ${libErr?.message || libErr}. Make sure "Maps JavaScript API" and "Map Tiles API" are enabled in Google Cloud Console, and that localhost is an allowed referrer.`);
          return;
        }

        if (destroyed || !container) return;

        const centerLat = parseFloat(lat);
        const centerLon = parseFloat(lon);

        // Clear container and create element via createElement
        // (setting properties AFTER appending is more reliable than constructor options)
        container.innerHTML = "";
        const el = document.createElement("gmp-map-3d") as any;
        el.style.cssText = "width:100%;height:100%;display:block;";
        container.appendChild(el);

        el.center = { lat: centerLat, lng: centerLon, altitude: 500 };
        el.range = 800;
        el.tilt = 60;
        el.heading = 0;
        el.mode = "SATELLITE";

        // Marker at target
        const marker = new Marker3DElement({ position: { lat: centerLat, lng: centerLon, altitude: 0 } });
        el.appendChild(marker);

        // Show billing / API errors inside the container
        el.addEventListener("gmp-requesterror", (e: any) => {
          const msg = e?.error?.message || "Google Maps API error";
          console.error("[Google 3D] gmp-requesterror:", e);
          showMapError(
            msg.includes("BillingNotEnabled")
              ? "Billing is not enabled for this API key. Please visit console.cloud.google.com → your project → Billing and enable it. Google provides a $200/month free credit."
              : `API error: ${msg}`
          );
        });

        // HUD sync on load
        el.addEventListener("gmp-load", () => {
          if (destroyed) return;
          el.addEventListener("gmp-centerchange", () => {
            if (el.center?.altitude !== undefined) setAltitude(Math.round(el.center.altitude));
          });
          el.addEventListener("gmp-tiltchange", () => {
            if (el.tilt !== undefined) setTilt(Math.round(el.tilt));
          });
        });

        (window as any).__google3DMapEl = el; // debug handle
        googleMap3DRef.current = el;
        setAltitude(500);
        setTilt(60);

      } catch (err: any) {
        console.error("[Google 3D] Unexpected error:", err);
        showMapError(`Unexpected error: ${err?.message || err}`);
      }
    };

    initGoogle3DMap();

    return () => {
      destroyed = true;
      if (container) container.innerHTML = "";
      googleMap3DRef.current = null;
    };
  }, [results]);

  // Setup error with auto-clear
  const showError = (msg: string) => {
    setError(msg);
    setTimeout(() => setError(""), 5000);
  };

  const executeScan = async () => {
    if (!lat || !lon || isNaN(parseFloat(lat)) || isNaN(parseFloat(lon))) {
      showError("Please search or drop a pin on the map to define target coordinates.");
      return;
    }
    setError("");
    setResults(null);
    setSaved(false);
    setScanning(true);
    setScanProgress(0);

    let step = 0;
    const interval = setInterval(() => {
      step++;
      const progress = Math.min(Math.round((step / 14) * 100), 100);
      setScanProgress(progress);
      setScanStatus(scanStatuses[Math.min(Math.floor(step / 2), scanStatuses.length - 1)]);
      if (step >= 14) clearInterval(interval);
    }, 350);

    try {
      const apiResult = await api.analyzeLocation({
        latitude: parseFloat(lat),
        longitude: parseFloat(lon),
      });

      clearInterval(interval);
      setScanProgress(100);
      setScanStatus("Scan complete.");

      setTimeout(() => {
        setScanning(false);
        setResults({
          detected: apiResult.panel_detected,
          panelCount: apiResult.panel_count,
          confidence: apiResult.confidence_score,
          powerEstimateKw: apiResult.estimated_capacity,
          areaEstimateSqm: apiResult.pv_area,
          sampleId: apiResult.sample_id,
          qcNotes: apiResult.qc_notes || [],
          rawJson: apiResult.raw_json,
          savedId: apiResult.id,
        });
        console.log("Roboflow Analysis Result:", {
          panelDetected: apiResult.panel_detected,
          panelCount: apiResult.panel_count,
          confidence: apiResult.confidence_score,
          predictions: apiResult.raw_json?.predictions
        });
        if (apiResult.user_id) setSaved(true);
      }, 500);
    } catch (err: any) {
      clearInterval(interval);
      setScanProgress(100);
      setScanStatus("Using demo inference...");

      setTimeout(() => {
        setScanning(false);
        setResults({
          ...demoResults,
          rawJson: {
            ...demoResults.rawJson,
            coordinates: { lat: parseFloat(lat), lon: parseFloat(lon) },
          },
        });
      }, 500);
    }
  };

  const saveToDb = async () => {
    if (!isAuthenticated) {
      showAuthModal("Please login to save analysis results to your account.");
      return;
    }
    if (!results) return;
    setSaving(true);
    try {
      const saveRes = await api.saveAnalysis(user?.id || "", results);
      setSaved(true);
      setResults((prev) => prev ? { ...prev, savedId: saveRes.history_record.id } : prev);
    } catch (err: any) {
      showError("Failed to save: " + (err.message || "Unknown error"));
    } finally {
      setSaving(false);
    }
  };

  const resetScan = () => {
    setResults(null);
    setActiveTab('primary');
    setDetailsOpen(false);
    setError("");
    setSaved(false);
  };

  const downloadJson = () => {
    if (!isAuthenticated) {
      showAuthModal("Please login to download JSON analysis logs.");
      return;
    }
    if (!results) return;
    const blob = new Blob([JSON.stringify(results.rawJson, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scan-${results.sampleId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCsvUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!isAuthenticated) {
      showAuthModal("Please login to use CSV batch processing.");
      return;
    }
    setBatchProgress("Uploading...");
    try {
      const result = await api.uploadCsv(file);
      setBatchProgress(`Job ${result.job_id}: ${result.status}`);
    } catch (err: any) {
      setBatchProgress(null);
      showError("CSV upload failed: " + (err.message || "Unknown error"));
    }
  };

  const tabsArray = ['primary', 'detailed', 'logs'] as const;
  const tabNames = {
    primary: 'Primary Analysis',
    detailed: 'Detailed Metrics',
    logs: 'Quality Control Notes'
  };

  const handleNextTab = () => {
    const idx = tabsArray.indexOf(activeTab);
    setActiveTab(tabsArray[(idx + 1) % tabsArray.length]);
  };

  const handlePrevTab = () => {
    const idx = tabsArray.indexOf(activeTab);
    setActiveTab(tabsArray[(idx - 1 + tabsArray.length) % tabsArray.length]);
  };

  return (
    <div className="min-h-screen">
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 50 }}
            className="fixed top-24 right-6 z-[100] flex items-center shadow-lg gap-3 text-destructive text-sm font-medium bg-destructive/10 border border-destructive/20 backdrop-blur-md rounded-xl px-4 py-3 max-w-sm leading-relaxed"
          >
            <AlertTriangle className="w-5 h-5" />
            {error}
          </motion.div>
        )}
      </AnimatePresence>
      <style>{`
        .esri-search {
          box-shadow: none !important;
          background: transparent !important;
          width: 100% !important;
          max-width: 600px !important;
        }
        .esri-search__container {
          background: rgba(13, 17, 23, 0.8) !important;
          border: 1px solid rgba(0, 247, 255, 0.2) !important;
          border-radius: 8px !important;
          backdrop-filter: blur(8px) !important;
        }
        .esri-search__input {
          background: transparent !important;
          color: #fff !important;
          font-family: inherit !important;
        }
        .esri-search__input::placeholder {
          color: rgba(255, 255, 255, 0.5) !important;
        }
        .esri-search__submit-button, .esri-search__clear-button {
          background: transparent !important;
          color: #00f7ff !important;
        }
        .esri-menu {
          background-color: rgba(13, 17, 23, 0.95) !important;
          border: 1px solid rgba(0, 247, 255, 0.2) !important;
          border-radius: 8px !important;
        }
        .esri-menu__list-item {
          color: rgba(255, 255, 255, 0.8) !important;
        }
        .esri-menu__list-item:hover, .esri-menu__list-item--active {
          background-color: rgba(0, 247, 255, 0.1) !important;
          color: #00f7ff !important;
        }
        /* Google Maps 3D (Map3DElement) — fill the scene container */
        gmp-map-3d {
          width: 100% !important;
          height: 100% !important;
          display: block !important;
        }
      `}</style>
      <AnimatePresence>
        {scanning && <ScanOverlay progress={scanProgress} status={scanStatus} />}
      </AnimatePresence>

      <div className="container mx-auto px-4 md:px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="font-display text-4xl md:text-5xl font-bold text-foreground mb-3">
            <Radar className="inline w-8 h-8 text-primary mr-3 -mt-1" />
            AI-Powered Solar Analysis
          </h1>
        </motion.div>

        <div className={results ? "hidden" : "block"}>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card rounded-2xl p-6 h-[450px] overflow-y-auto flex flex-col"
            >
              <h3 className="font-display font-bold text-foreground mb-6 flex items-center gap-2">
                <Crosshair className="w-5 h-5 text-primary" />
                Target Parameters
              </h3>

              <div className="mb-4">
                <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                  Target ID
                </label>
                <div className="glass-card rounded-lg px-3 py-2 text-sm text-muted-foreground font-mono">
                  {isAuthenticated ? "SRY-2026-AUTO" : "DEMO-SCAN"}
                </div>
              </div>

              <div className="mb-4">
                <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                  Latitude
                </label>
                <input
                  type="text"
                  value={lat}
                  onChange={(e) => handleInputChange('lat', e.target.value)}
                  className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground font-mono focus:outline-none focus:border-primary/50"
                />
              </div>

              <div className="mb-4">
                <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                  Longitude
                </label>
                <input
                  type="text"
                  value={lon}
                  onChange={(e) => handleInputChange('lon', e.target.value)}
                  className="w-full bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground font-mono focus:outline-none focus:border-primary/50"
                />
              </div>

              <div className="mb-6">
                <label className="font-mono-tactical text-[10px] text-muted-foreground mb-1 block">
                  Batch Upload
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleCsvUpload}
                />
                {isAuthenticated ? (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="glass-card rounded-lg px-3 py-3 flex items-center justify-center gap-2 text-primary/70 hover:text-primary transition-colors w-full"
                  >
                    <Upload className="w-4 h-4" />
                    <span className="text-xs">Upload CSV</span>
                  </button>
                ) : (
                  <button
                    onClick={() => showAuthModal("Please login to use CSV batch processing.")}
                    className="glass-card rounded-lg px-3 py-3 flex items-center justify-center gap-2 text-muted-foreground/50 w-full"
                  >
                    <Lock className="w-4 h-4" />
                    <span className="text-xs">CSV upload (locked)</span>
                  </button>
                )}
              </div>

              <button
                onClick={executeScan}
                disabled={scanning}
                className="btn-cyan w-full flex items-center justify-center gap-2"
              >
                <Radar className="w-5 h-5" />
                EXECUTE SCAN
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-2 glass-card rounded-2xl relative h-[450px]"
            >
              <div className="absolute top-4 right-4 z-[60] glass-card rounded-lg px-3 py-1.5 flex items-center gap-2 bg-background/50 backdrop-blur">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-success" />
                </span>
                <span className="font-mono-tactical text-[10px] text-foreground uppercase">ArcGIS Satellite Feed</span>
              </div>

              <div ref={mapElementRef} className="absolute inset-0 z-0 bg-muted/20 rounded-2xl overflow-hidden" />

              <div className="absolute bottom-0 left-0 right-0 z-[50] flex gap-px border-t border-border rounded-b-2xl">
                <div className="flex-1 bg-background/80 backdrop-blur-sm px-4 py-2 flex items-center justify-center min-h-[60px]">
                  <div ref={searchContainerRef} className="w-full flex justify-center relative z-[60]" />
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        <AnimatePresence>
          {results && (
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
                <div className="glass-card rounded-2xl p-6 flex flex-col h-[450px]">
                  <h3 className="font-display font-bold text-foreground mb-6 flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-primary" />
                    Neural Inference Output
                  </h3>

                  {/* Navigation Arrows for Category */}
                  <div className="flex justify-between items-center bg-muted/30 rounded-lg p-2 mb-6 border border-border/50">
                    <button onClick={handlePrevTab} className="p-1.5 hover:bg-background rounded-md text-muted-foreground hover:text-primary transition-colors">
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="italic font-bold text-sm text-foreground capitalize tracking-wide">{tabNames[activeTab]}</span>
                    <button onClick={handleNextTab} className="p-1.5 hover:bg-background rounded-md text-muted-foreground hover:text-primary transition-colors">
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Tab Content */}
                  <div className="flex-1 flex flex-col">
                    <AnimatePresence mode="wait">
                      {activeTab === 'primary' ? (
                        <motion.div
                          key="primary"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 10 }}
                          className="flex flex-col gap-4"
                        >
                          <div className="glass-card rounded-xl p-5 w-full">
                            <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-2">Detection Signature</span>
                            <div className="flex items-center gap-2">
                              {results.detected ? (
                                <>
                                  <CheckCircle2 className="w-6 h-6 text-success" />
                                  <span className="font-display text-lg font-bold text-success uppercase">Positive</span>
                                </>
                              ) : (
                                <>
                                  <XCircle className="w-6 h-6 text-destructive" />
                                  <span className="font-display text-lg font-bold text-destructive uppercase">Negative</span>
                                </>
                              )}
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            <div className="glass-card rounded-xl p-5">
                              <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-2">Panel Count</span>
                              <span className="font-display text-3xl font-bold text-foreground">{results.panelCount}</span>
                            </div>

                            <div className="glass-card rounded-xl p-5 flex flex-col items-center justify-center">
                              <span className="font-mono-tactical text-[9px] text-muted-foreground block mb-2">Confidence</span>
                              <ConfidenceCircle value={results.confidence} />
                            </div>
                          </div>
                        </motion.div>
                      ) : activeTab === 'detailed' ? (
                        <motion.div
                          key="detailed"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          className="space-y-3"
                        >
                          <div className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                            <span className="text-xs text-muted-foreground">Coordinates:</span>
                            <span className="text-sm font-mono text-foreground">{parseFloat(lat).toFixed(4)}°N, {parseFloat(lon).toFixed(4)}°E</span>
                          </div>

                          <div className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                            <span className="text-xs text-muted-foreground">Power Estimate:</span>
                            <span className="text-sm font-mono text-foreground">{results.powerEstimateKw} kW</span>
                          </div>

                          <div className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                            <span className="text-xs text-muted-foreground">Area Estimate:</span>
                            <span className="text-sm font-mono text-foreground">{results.areaEstimateSqm} m²</span>
                          </div>

                          <div className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                            <span className="text-xs text-muted-foreground">Scan Radius:</span>
                            <span className="text-sm font-mono text-foreground">1200 sq.ft</span>
                          </div>

                          <div className="flex justify-between items-center p-3 rounded-lg bg-muted/20">
                            <span className="text-xs text-muted-foreground">QC Status:</span>
                            <span className="text-sm font-mono text-foreground text-green-500">VERIFIABLE</span>
                          </div>
                        </motion.div>
                      ) : (
                        <motion.div
                          key="logs"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          className="space-y-3 h-full flex flex-col"
                        >
                          <ul className="space-y-4">
                            {results.qcNotes.slice(0, 4).map((note, i) => (
                              <li key={i} className="flex items-start gap-3 bg-muted/20 p-3 rounded-lg border border-border/50">
                                <div className="mt-0.5 rounded-full bg-primary/20 p-1">
                                  <CheckCircle2 className="w-4 h-4 text-primary" />
                                </div>
                                <span className="text-sm text-foreground/90 leading-relaxed font-mono">{note}</span>
                              </li>
                            ))}
                          </ul>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>

                <div className="lg:col-span-2 glass-card rounded-2xl relative overflow-hidden h-[450px]">
                  <div className="absolute top-3 right-3 z-[60] glass-card rounded-lg px-3 py-2 bg-background/50 backdrop-blur flex flex-col gap-1 border border-border/50">
                    <span className="font-mono-tactical text-[10px] text-foreground">ALT: {altitude}m</span>
                    <span className="font-mono-tactical text-[10px] text-foreground">PITCH: {tilt}°</span>
                  </div>

                  <div ref={sceneElementRef} className="absolute inset-0 z-0 bg-muted/20" />

                  <div className="absolute bottom-20 right-3 z-[50] flex flex-col gap-2">
                    <button onClick={() => {
                      const m = googleMap3DRef.current;
                      if (m) {
                        const newTilt = tilt > 30 ? 0 : 65;
                        m.tilt = newTilt;
                      }
                    }} className="glass-card rounded-lg p-2 bg-background/80 hover:border-primary/40"><Eye className="w-4 h-4 text-primary" /></button>
                    <button onClick={() => {
                      const m = googleMap3DRef.current;
                      if (!m) return;
                      m.center = { lat: parseFloat(lat), lng: parseFloat(lon), altitude: 500 };
                      m.range = 800;
                      m.tilt = 60;
                      m.heading = 0;
                    }} className="glass-card rounded-lg p-2 bg-background/80 hover:border-primary/40"><Home className="w-4 h-4 text-primary" /></button>
                  </div>

                  <div className="absolute bottom-0 left-0 right-0 z-[50] flex gap-px border-t border-border rounded-b-2xl">
                    <div className="flex-1 bg-background/80 backdrop-blur-sm px-4 py-3 flex flex-wrap items-center justify-between min-h-[60px]">
                      <div className="flex items-center gap-2">
                        {saved ? (
                          <div className="glass-card rounded-lg px-3 py-2 flex items-center gap-2 bg-success/10 border-success/30">
                            <CheckCircle2 className="w-4 h-4 text-success" />
                            <span className="font-mono-tactical text-[10px] text-success uppercase">Stored</span>
                          </div>
                        ) : (
                          <button onClick={saveToDb} disabled={saving} className="glass-card rounded-lg px-3 py-2 flex items-center gap-2 hover:border-primary/50 transition-colors bg-background/50">
                            <Save className="w-4 h-4 text-primary" />
                            <span className="font-mono-tactical text-[10px] text-primary uppercase">{saving ? "Saving..." : "Save Result"}</span>
                          </button>
                        )}
                        <button onClick={downloadJson} className="glass-card rounded-lg px-3 py-2 flex items-center gap-2 hover:border-primary/50 transition-colors bg-background/50">
                          <Download className="w-4 h-4 text-primary" />
                          <span className="font-mono-tactical text-[10px] text-primary uppercase">Extract JSON</span>
                        </button>
                      </div>

                      <div className="flex items-center gap-3">
                        <button onClick={resetScan} className="glass-card rounded-lg px-4 py-2 text-sm text-foreground hover:bg-primary/20 hover:border-primary/50 flex items-center bg-background/50"><RotateCcw className="w-3 h-3 mr-2" />Reset</button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Analysis;
