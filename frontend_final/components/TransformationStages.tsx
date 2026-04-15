"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const stages = [
  {
    id: "ingest",
    label: "Ingest",
    title: "Raw Footage Pipeline",
    description: "Drop your standard match broadcast or tactical camera feed. The engine immediately begins frame-by-frame processing without sending data to the cloud.",
    visual: () => (
      <svg viewBox="0 0 800 600" className="w-full h-full opacity-80">
        {/* Video Player Mockup */}
        <rect x="50" y="50" width="700" height="400" fill="#111a12" stroke="#2a4030" strokeWidth="2" rx="8" />
        <rect x="50" y="450" width="700" height="40" fill="#0a0f0a" stroke="#2a4030" strokeWidth="2" />
        <circle cx="90" cy="470" r="10" fill="#00e676" />
        <rect x="120" y="468" width="600" height="4" fill="#2a4030" rx="2" />
        <motion.rect x="120" y="468" width="150" height="4" fill="#00e676" rx="2" 
          animate={{ width: [100, 400] }} transition={{ duration: 4, repeat: Infinity }}
        />
        <text x="350" y="250" fill="#1a2420" fontSize="48" fontFamily="monospace">NO SIGNAL</text>
        <motion.g animate={{ opacity: [1, 0, 1] }} transition={{ duration: 0.1, repeat: Infinity, repeatDelay: 1 }}>
          <text x="350" y="250" fill="#fff" fontSize="48" fontFamily="monospace" opacity="0.1">PROCESSING</text>
        </motion.g>
      </svg>
    )
  },
  {
    id: "detect",
    label: "Detect",
    title: "Computer Vision Tracking",
    description: "Every player, the ball, and the referee are identified. Abstract YOLO bounding boxes snap coordinates to moving targets in real-time.",
    visual: () => (
      <svg viewBox="0 0 800 600" className="w-full h-full opacity-90">
        {/* Bounding Boxes Mockup */}
        <rect x="50" y="50" width="700" height="500" fill="none" stroke="#2a4030" strokeWidth="2" rx="8" />
        {/* Simulated targets */}
        {[
          {x: 200, y: 150}, {x: 350, y: 300}, {x: 500, y: 200}, {x: 600, y: 400}
        ].map((pos, i) => (
          <motion.g key={i} animate={{ x: [0, Math.random()*40-20], y: [0, Math.random()*40-20] }} transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}>
             <rect x={pos.x} y={pos.y} width="50" height="100" fill="none" stroke="#00e676" strokeWidth="2" />
             <text x={pos.x} y={pos.y - 5} fill="#00e676" fontSize="12" fontFamily="monospace">ID:{140+i} 0.9{i}</text>
             {/* Center dot */}
             <circle cx={pos.x + 25} cy={pos.y + 50} r="3" fill="#00e676" />
          </motion.g>
        ))}
      </svg>
    )
  },
  {
    id: "map",
    label: "Map",
    title: "2D Coordinate Projection",
    description: "The chaos of the broadcast angle is flattened into a perfect mathematical 2D pitch map, filtering out perspective distortion.",
    visual: () => (
      <svg viewBox="0 0 800 600" className="w-full h-full opacity-90">
        {/* 2D Map */}
        <rect x="100" y="50" width="600" height="500" fill="none" stroke="#2a4030" strokeWidth="2" />
        <line x1="400" y1="50" x2="400" y2="550" stroke="#2a4030" strokeWidth="2" />
        <circle cx="400" cy="300" r="80" fill="none" stroke="#2a4030" strokeWidth="2" />
        {/* Network connections */}
        <motion.path 
          d="M 200 150 L 350 300 L 300 450 L 200 150" 
          fill="none" stroke="#00e676" strokeWidth="2" strokeDasharray="5 5" opacity="0.5"
          animate={{ strokeDashoffset: [0, 50] }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
        {/* Team A */}
        <circle cx="200" cy="150" r="8" fill="#00e676" />
        <circle cx="350" cy="300" r="8" fill="#00e676" />
        <circle cx="300" cy="450" r="8" fill="#00e676" />
        {/* Team B */}
        <circle cx="600" cy="200" r="8" fill="#e8f0e9" />
        <circle cx="500" cy="350" r="8" fill="#e8f0e9" />
      </svg>
    )
  },
  {
    id: "diagnose",
    label: "Diagnose",
    title: "Algorithmic Vulnerability Scan",
    description: "The engine runs deterministic tactical frameworks (Positional Play, High Block, Low Block) to highlight exploitable areas on the pitch.",
    visual: () => (
      <svg viewBox="0 0 800 600" className="w-full h-full opacity-90">
         <rect x="100" y="50" width="600" height="500" fill="none" stroke="#2a4030" strokeWidth="2" />
         {/* Danger Zone */}
         <motion.rect 
            x="450" y="200" width="200" height="200" 
            fill="none" stroke="#ffb300" strokeWidth="3" strokeDasharray="10 10"
            animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.5, repeat: Infinity }}
         />
         <rect x="450" y="200" width="200" height="20s0" fill="#ffb300" opacity="0.1" />
         <text x="460" y="220" fill="#ffb300" fontSize="14" fontFamily="monospace">! EXPOSED FLANK</text>
         {/* Vectors */}
         <motion.path 
            d="M 200 300 Q 400 100 600 250" 
            fill="none" stroke="#00e676" strokeWidth="3" markerEnd="url(#arrow)"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 2, repeat: Infinity }}
         />
      </svg>
    )
  },
  {
    id: "instruct",
    label: "Instruct",
    title: "Touchline Briefing",
    description: "Instead of raw data, the system outputs immediate, actionable human-language instructions you can shout from the technical area.",
    visual: () => (
      <svg viewBox="0 0 800 600" className="w-full h-full opacity-90">
        <rect x="150" y="100" width="500" height="400" fill="#111a12" stroke="#2a4030" strokeWidth="2" rx="12" />
        <rect x="150" y="100" width="500" height="40" fill="#1a2420" rx="12" style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }} />
        <text x="170" y="125" fill="#e8f0e9" fontSize="14" fontFamily="monospace">COACHING_BRIEF.txt</text>
        
        <text x="180" y="180" fill="#ffb300" fontSize="18" fontFamily="monospace">ACTION: TUCK IN THE WINGER</text>
        <motion.text x="180" y="210" fill="#e8f0e9" fontSize="14" fontFamily="monospace"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        >
          &gt; The opponent left back is overlapping freely.
        </motion.text>
        <motion.text x="180" y="240" fill="#e8f0e9" fontSize="14" fontFamily="monospace"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
        >
          &gt; Instruct #7 to invert deeply during defensive phases.
        </motion.text>
        <motion.text x="180" y="270" fill="#00e676" fontSize="14" fontFamily="monospace"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}
        >
          SUCCESS PROBABILITY: 89%
        </motion.text>
      </svg>
    )
  }
];

export function TransformationStages() {
  const [activeStage, setActiveStage] = useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setActiveStage((prev) => (prev + 1) % stages.length);
    }, 6000); // 6 seconds per stage
    return () => clearInterval(interval);
  }, [activeStage]); // Dependency resets the timer if user clicks a manual tab

  return (
    <section className="relative py-32 bg-pitch border-y border-slate-dark text-chalk">
      <div className="max-w-7xl mx-auto px-6">
        
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-heading font-bold uppercase tracking-wide mb-4">
            The Transformation Layer
          </h2>
          <p className="text-chalk/60 font-mono text-sm max-w-xl mx-auto">
            From raw pixels to actionable touchline insights in &lt; 200ms.
          </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-12 lg:gap-24 items-center">
          
          {/* Left: SVG Visual */}
          <div className="flex-1 w-full aspect-square md:aspect-video lg:aspect-square bg-surface border border-slate-dark rounded-xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden p-6 relative">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeStage}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
                transition={{ duration: 0.3 }}
                className="absolute inset-0"
              >
                {stages[activeStage].visual()}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Right: Interactive Tabs & Content */}
          <div className="flex-1 w-full max-w-xl">
            <div className="flex flex-wrap items-center gap-2 mb-10 pb-4">
              {stages.map((stage, idx) => (
                <button
                  key={stage.id}
                  onClick={() => setActiveStage(idx)}
                  className={`px-4 py-2 rounded font-mono text-sm transition-all whitespace-nowrap ${
                    activeStage === idx 
                      ? "bg-neon/10 text-neon border border-neon shadow-[0_0_15px_rgba(0,230,118,0.2)]" 
                      : "bg-surface border border-slate-dark text-chalk/50 hover:text-chalk"
                  }`}
                >
                  0{idx + 1}. {stage.label}
                </button>
              ))}
            </div>

            <div className="min-h-[200px]">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeStage}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <h3 className="text-3xl font-heading font-bold uppercase tracking-wide mb-4 text-neon">
                    {stages[activeStage].title}
                  </h3>
                  <p className="text-lg text-chalk/80 leading-relaxed font-sans font-light">
                    {stages[activeStage].description}
                  </p>
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
}
