"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function PrivacyMoat() {
  const [isLocal, setIsLocal] = useState(true);

  const localBadges = [
    { label: "Data Residency", value: "Offline / Air-gapped" },
    { label: "Processing Latency", value: "< 12ms (RTX Series)" },
    { label: "Uplink Requirement", value: "0 Mbps (Fully Disconnected)" },
    { label: "Telemetry", value: "Disabled by default" },
    { label: "Model Architecture", value: "Quantized Edge Vis-Transformer" }
  ];

  const cloudBadges = [
    { label: "Data Residency", value: "AWS us-east-1 (Encrypted)" },
    { label: "Processing Latency", value: "142ms + Network Jitter" },
    { label: "Uplink Requirement", value: "15 Mbps minimum (Video stream)" },
    { label: "Telemetry", value: "Opt-in Analytics" },
    { label: "Model Architecture", value: "Massive Cloud API Model" }
  ];

  const activeBadges = isLocal ? localBadges : cloudBadges;

  return (
    <section className="py-32 bg-pitch text-chalk relative overflow-hidden">
      
      {/* Background Grid */}
      <div className="absolute inset-0 z-0 bg-grid opacity-30 pointer-events-none" />

      <div className="relative z-10 max-w-5xl mx-auto px-6">
        
        {/* Toggle Header */}
        <div className="flex flex-col items-center text-center mb-16">
          <div className="flex bg-pitch border border-slate-dark p-1 rounded-full shadow-lg mb-8">
            <button
              onClick={() => setIsLocal(true)}
              className={`px-8 py-3 rounded-full font-mono text-sm tracking-wider transition-all ${
                isLocal 
                  ? "bg-neon/20 text-neon shadow-[0_0_15px_rgba(0,230,118,0.3)]" 
                  : "text-chalk/50 hover:text-chalk"
              }`}
            >
              LOCALLY HOSTED GPU
            </button>
            <button
              onClick={() => setIsLocal(false)}
              className={`px-8 py-3 rounded-full font-mono text-sm tracking-wider transition-all ${
                !isLocal 
                  ? "bg-[#0ea5e9]/20 text-[#0ea5e9] shadow-[0_0_15px_rgba(14,165,233,0.3)]" 
                  : "text-chalk/50 hover:text-chalk"
              }`}
            >
              CLOUD LLM CLUSTER
            </button>
          </div>

          <AnimatePresence mode="wait">
             <motion.div
               key={isLocal ? "local" : "cloud"}
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               exit={{ opacity: 0, y: -10 }}
               transition={{ duration: 0.2 }}
             >
                <h3 className="text-4xl md:text-5xl font-heading font-bold uppercase mb-4">
                  {isLocal ? "Total Data Sovereignty." : "Infinite Scalability (Opt-in)."}
                </h3>
                <p className="text-chalk/70 font-sans max-w-2xl mx-auto text-lg font-light leading-relaxed">
                  {isLocal 
                    ? "Your tactical IP never leaves your machine. The Gaffer's Guide engine runs a quantized visual transformer directly on your local silicon, completely offline."
                    : "For lower-end machines, offload the processing to our encrypted, ephemeral cloud cluster. Data is processed in-memory and instantly destroyed post-match."
                  }
                </p>
             </motion.div>
          </AnimatePresence>
        </div>

        {/* Badges Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <AnimatePresence mode="wait">
            {activeBadges.map((badge, idx) => (
              <motion.div
                key={`${isLocal}-${idx}`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2, delay: idx * 0.05 }}
                className={`p-4 border bg-pitch rounded flex flex-col items-center text-center justify-center ${
                  isLocal ? "border-neon/30 shadow-[0_0_10px_rgba(0,230,118,0.05)]" : "border-[#0ea5e9]/30 shadow-[0_0_10px_rgba(14,165,233,0.05)]"
                }`}
              >
                <div className="font-mono text-xs text-chalk/50 uppercase mb-2">{badge.label}</div>
                <div className={`font-mono text-sm font-bold ${isLocal ? "text-neon" : "text-[#0ea5e9]"}`}>
                  {badge.value}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

      </div>
    </section>
  );
}
