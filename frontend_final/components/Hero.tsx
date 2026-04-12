"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue, animate } from "framer-motion";
import { TextReveal } from "@/components/TextReveal";

export function Hero() {
  const [pipelineStep, setPipelineStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPipelineStep((prev) => (prev + 1) % 3);
    }, 5000); // 5 seconds per step
    return () => clearInterval(interval);
  }, [pipelineStep]);

  return (
    <section id="engine" className="relative min-h-screen pt-32 pb-16 px-6 overflow-hidden flex flex-col items-center justify-center">
      {/* Content */}
      <div className="relative z-10 w-full max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-16">
        
        {/* Text Side */}
        <div className="flex-1 text-center lg:text-left">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 border border-slate-light bg-surface rounded text-neon font-mono text-xs uppercase tracking-widest"
          >
            <span className="w-2 h-2 bg-neon animate-pulse shadow-[0_0_8px_#00e676]" />
            <span>AI Coaching Engine — v1.0</span>
          </motion.div>
          
          <div className="relative inline-block mb-8">
            <h1 className="text-5xl md:text-7xl lg:text-[6rem] font-sans font-bold tracking-tight leading-[1.1] uppercase flex flex-col items-center lg:items-start text-chalk">
              <TextReveal text="Your Tactics" />
              <TextReveal 
                text="Decoded" 
                delay={0.65} 
                className="text-neon"
              />
            </h1>
          </div>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="text-lg md:text-xl text-chalk/80 max-w-xl mx-auto lg:mx-0 font-sans font-normal leading-relaxed"
          >
            Ingest raw match footage. Extract coordinate data. Identify strategic vulnerabilities instantly. Processed securely offline.
          </motion.p>
        </div>

        {/* Dynamic Pipeline Visual Side */}
        <div className="flex-1 w-full max-w-2xl">
          <div className="relative aspect-video rounded border border-slate-light bg-surface shadow-[0_0_40px_rgba(0,230,118,0.05)] overflow-hidden flex flex-col">
             
             {/* Header with Step Indicators */}
             <div className="flex bg-slate-dark h-10 border-b border-slate-light divide-x divide-slate-light">
                {["1. Detect", "2. Analyze", "3. Instruct"].map((label, idx) => (
                  <button 
                    key={idx} 
                    onClick={() => setPipelineStep(idx)}
                    className={`flex-1 flex items-center justify-center font-mono text-xs transition-colors hover:bg-surface/50 cursor-pointer ${pipelineStep === idx ? 'text-neon bg-surface' : 'text-chalk/40 bg-transparent'}`}
                  >
                    {label}
                  </button>
                ))}
             </div>

             {/* Dynamic SVG Canvas */}
             <div className="flex-1 relative bg-pitch">
               <svg viewBox="0 0 800 450" className="w-full h-full opacity-90" xmlns="http://www.w3.org/2000/svg">
                  {/* Pitch Background Elements */}
                  <rect x="20" y="20" width="760" height="410" fill="none" stroke="#2a4030" strokeWidth="2" />
                  <line x1="400" y1="20" x2="400" y2="430" stroke="#2a4030" strokeWidth="2" />
                  <circle cx="400" cy="225" r="70" fill="none" stroke="#2a4030" strokeWidth="2" />
                  
                  {/* Common players on the field */}
                  {[
                    { cx: 300, cy: 120, team: "opp" }, { cx: 280, cy: 220, team: "opp" }, { cx: 320, cy: 350, team: "opp" },
                    { cx: 150, cy: 100, team: "us" },  { cx: 130, cy: 220, team: "us" },  { cx: 160, cy: 340, team: "us" }
                  ].map((p, i) => (
                    <circle key={`p-${i}`} cx={p.cx} cy={p.cy} r="6" fill={p.team === "us" ? "#00e676" : "#e8f0e9"} />
                  ))}

                  <AnimatePresence mode="wait">
                    {/* STEP 1: DETECT (Draw bounding boxes and scan line) */}
                    {pipelineStep === 0 && (
                      <motion.g key="step-1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <rect x="280" y="100" width="40" height="40" fill="none" stroke="#00e676" strokeWidth="2" />
                        <text x="285" y="95" fill="#00e676" fontSize="12" fontFamily="monospace">ID:09</text>
                        
                        <rect x="260" y="200" width="40" height="40" fill="none" stroke="#00e676" strokeWidth="2" />
                        <text x="265" y="195" fill="#00e676" fontSize="12" fontFamily="monospace">ID:10</text>

                        <rect x="300" y="330" width="40" height="40" fill="none" stroke="#00e676" strokeWidth="2" />
                        <text x="305" y="325" fill="#00e676" fontSize="12" fontFamily="monospace">ID:11</text>
                        
                        {/* Scanning Laser */}
                        <motion.line x1="0" y1="0" x2="0" y2="450" stroke="#00e676" strokeWidth="2" opacity="0.5"
                          animate={{ x1: [0, 800, 0], x2: [0, 800, 0] }} transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        />
                      </motion.g>
                    )}

                    {/* STEP 2: ANALYZE (Network Lines and Highlight Vulnerability) */}
                    {pipelineStep === 1 && (
                      <motion.g key="step-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        {/* Network Lines */}
                        <line x1="150" y1="100" x2="130" y2="220" stroke="#00e676" strokeWidth="2" strokeDasharray="5 5" opacity="0.4" />
                        <line x1="130" y1="220" x2="160" y2="340" stroke="#00e676" strokeWidth="2" strokeDasharray="5 5" opacity="0.4" />
                        
                        {/* Huge Gap Highlight */}
                        <motion.rect x="350" y="80" width="200" height="250" fill="none" stroke="#ffb300" strokeWidth="3" strokeDasharray="10 10" 
                          animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity }}
                        />
                        <rect x="350" y="80" width="200" height="250" fill="#ffb300" opacity="0.1" />
                        <text x="360" y="100" fill="#ffb300" fontSize="14" fontFamily="monospace" fontWeight="bold">! MASSIVE SPACE DETECTED</text>
                        
                        {/* Vectors showing movement of opponent into space */}
                        <motion.path d="M 300 120 Q 380 90 450 150" fill="none" stroke="#e8f0e9" strokeWidth="2" strokeDasharray="4 4" markerEnd="url(#arrow)" 
                          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 2, repeat: Infinity }}
                        />
                      </motion.g>
                    )}

                    {/* STEP 3: INSTRUCT (Dialogue / Instructions) */}
                    {pipelineStep === 2 && (
                      <motion.g key="step-3" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        {/* Coach Query Bubble */}
                        <rect x="400" y="60" width="350" height="80" fill="#111a12" stroke="#2a4030" strokeWidth="2" rx="8" />
                        <text x="420" y="90" fill="#e8f0e9" fontSize="14" fontFamily="sans-serif">
                           <tspan fontWeight="bold" fill="#ffb300">Manager:</tspan> "They are bypassing our midfield line. 
                        </text>
                        <text x="420" y="115" fill="#e8f0e9" fontSize="14" fontFamily="sans-serif">
                           How do we plug the gap?"
                        </text>

                        {/* Engine Response Bubble */}
                        <motion.rect x="400" y="170" width="350" height="120" fill="#0a0f0a" stroke="#00e676" strokeWidth="2" rx="8" 
                          initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.5 }}
                        />
                        <motion.text x="420" y="200" fill="#00e676" fontSize="14" fontFamily="monospace"
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
                        >
                           &gt; ENGINE RESPONSE
                        </motion.text>
                        <motion.text x="420" y="230" fill="#e8f0e9" fontSize="14" fontFamily="sans-serif"
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}
                        >
                           Drop your #6 deeper into a defensive pivot.
                        </motion.text>
                        <motion.text x="420" y="260" fill="#e8f0e9" fontSize="14" fontFamily="sans-serif"
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.5 }}
                        >
                           Force their #10 out wide. (+18% Win Prob)
                        </motion.text>
                        
                        {/* Suggested Movement Vector */}
                        <motion.path d="M 160 340 L 250 250" fill="none" stroke="#00e676" strokeWidth="4" 
                          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 3 }}
                        />
                      </motion.g>
                    )}

                  </AnimatePresence>
               </svg>
             </div>

          </div>
        </div>

      </div>
    </section>
  );
}
