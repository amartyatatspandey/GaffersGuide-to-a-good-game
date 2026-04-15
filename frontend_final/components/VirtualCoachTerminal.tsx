"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const scenarios = [
  {
    id: "high_line",
    label: "High Line Trap",
    text: "> ANALYZING DEFENSIVE SHAPE...\n> DETECTED: Opponent forward (#9) is lingering offside by 2.3m.\n> INSTRUCTION: Hold the line. Do not drop. The trap is successfully set. Prepare to counter if possession is won.\n> WIN PROBABILITY DELTA: +12%",
    renderPitch: () => (
      <svg viewBox="0 0 500 400" className="w-full h-full">
        {/* Field */}
        <rect x="10" y="10" width="480" height="380" fill="none" stroke="#2a4030" strokeWidth="2" />
        <line x1="250" y1="10" x2="250" y2="390" stroke="#2a4030" strokeWidth="2" />
        {/* Defensive Line (Our Team - Neon) */}
        <line x1="150" y1="50" x2="150" y2="350" stroke="#00e676" strokeWidth="2" strokeDasharray="4 4" />
        {[80, 160, 240, 320].map((y, i) => (
          <circle key={i} cx="150" cy={y} r="6" fill="#00e676" />
        ))}
        {/* Offside Attacker (Opponent - Chalk) */}
        <motion.circle cx="120" cy="200" r="6" fill="#e8f0e9"
          animate={{ cx: [130, 120, 130] }} transition={{ duration: 2, repeat: Infinity }}
        />
        <text x="110" y="190" fill="#ffb300" fontSize="12" fontFamily="monospace">OFFSIDE</text>
        <motion.rect x="115" y="195" width="10" height="10" fill="none" stroke="#ffb300" strokeWidth="1"
          animate={{ scale: [1, 2], opacity: [1, 0] }} transition={{ duration: 1, repeat: Infinity }}
        />
      </svg>
    )
  },
  {
    id: "false_9",
    label: "False 9 Vacancy",
    text: "> ANALYZING OPPOSITION ATTACK...\n> DETECTED: Opponent #10 has dropped into Midfield Zone 8.\n> INSTRUCTION: Center Backs (#4, #5) must HOLD position. Midfielder (#6) track the runner. Do not break the defensive shape.\n> STRUCTURAL INTEGRITY: 94%",
    renderPitch: () => (
      <svg viewBox="0 0 500 400" className="w-full h-full">
        <rect x="10" y="10" width="480" height="380" fill="none" stroke="#2a4030" strokeWidth="2" />
        <line x1="250" y1="10" x2="250" y2="390" stroke="#2a4030" strokeWidth="2" />
        
        <circle cx="100" cy="180" r="6" fill="#00e676" />
        <text x="85" y="175" fill="#00e676" fontSize="10" fontFamily="monospace">CB#4</text>
        <circle cx="100" cy="220" r="6" fill="#00e676" />
        <text x="85" y="235" fill="#00e676" fontSize="10" fontFamily="monospace">CB#5</text>

        {/* Dropping False 9 */}
        <motion.circle cx="200" cy="200" r="6" fill="#e8f0e9"
          animate={{ cx: [150, 250] }} transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
        />
        <motion.path d="M 150 200 L 250 200" fill="none" stroke="#e8f0e9" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
        
        {/* Tracking Midfielder */}
        <motion.circle cx="220" cy="180" r="6" fill="#00e676"
          animate={{ cx: [170, 270] }} transition={{ duration: 2, repeat: Infinity, repeatType: "reverse", delay: 0.2 }}
        />
        <text x="220" y="170" fill="#00e676" fontSize="10" fontFamily="monospace">CDM#6</text>
      </svg>
    )
  }
];

export function VirtualCoachTerminal() {
  const [activeScenario, setActiveScenario] = useState(0);
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isLocalMode, setIsLocalMode] = useState(true);

  // Typewriter effect logic
  useEffect(() => {
    setIsTyping(true);
    setDisplayedText("");
    
    const fullText = scenarios[activeScenario].text;
    let i = 0;
    
    const interval = setInterval(() => {
      setDisplayedText(fullText.substring(0, i));
      i++;
      if (i > fullText.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, 20); // typing speed

    return () => clearInterval(interval);
  }, [activeScenario]);

  return (
    <section id="terminal" className="py-32 bg-pitch border-b border-slate-dark">
      <div className="max-w-7xl mx-auto px-6">
        
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
          <div>
            <h2 className="text-4xl md:text-5xl font-heading font-black uppercase tracking-widest text-chalk mb-2">
              Virtual Coach Terminal
            </h2>
            <p className="font-mono text-chalk/50 text-sm">
              [SYSTEM.READY] Running tactical evaluation models in real-time.
            </p>
          </div>
          
          {/* Toggle Switch */}
          <div className="flex bg-surface border border-slate-dark p-1 rounded-md max-w-xs self-start md:self-auto shadow-sm">
            <button
              onClick={() => setIsLocalMode(true)}
              className={`flex-1 px-4 py-2 font-mono text-xs uppercase tracking-wider transition-colors ${
                isLocalMode ? 'bg-neon/10 text-neon' : 'text-chalk/40 hover:text-chalk/70'
              }`}
            >
              Local GPU
            </button>
            <button
              onClick={() => setIsLocalMode(false)}
              className={`flex-1 px-4 py-2 font-mono text-xs uppercase tracking-wider transition-colors ${
                !isLocalMode ? 'bg-[#0ea5e9]/10 text-[#0ea5e9]' : 'text-chalk/40 hover:text-chalk/70'
              }`}
            >
              Cloud LLM
            </button>
          </div>
        </div>

        {/* Two Column Terminal */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-[500px]">
          
          {/* Left: Pitch View */}
          <div className="bg-surface border border-slate-dark rounded shadow-lg overflow-hidden flex flex-col">
            <div className="bg-slate-dark px-4 py-2 flex items-center justify-between border-b border-slate-light">
              <span className="font-mono text-xs text-chalk/70 uppercase">Live_Pitch_View.svg</span>
              <div className="flex gap-2">
                <span className="w-3 h-3 rounded-full bg-slate-light" />
                <span className="w-3 h-3 rounded-full bg-slate-light" />
                <span className="w-3 h-3 rounded-full bg-slate-light" />
              </div>
            </div>
            <div className="relative flex-1 bg-[#111a12] p-4 flex items-center justify-center">
               <AnimatePresence mode="popLayout">
                 <motion.div
                   key={activeScenario}
                   initial={{ opacity: 0, scale: 0.95 }}
                   animate={{ opacity: 1, scale: 1 }}
                   exit={{ opacity: 0, scale: 0.95 }}
                   transition={{ duration: 0.3 }}
                   className="w-full h-full"
                 >
                   {scenarios[activeScenario].renderPitch()}
                 </motion.div>
               </AnimatePresence>
            </div>
          </div>

          {/* Right: Coach Engine Output */}
          <div className="bg-[#0a0f0a] border border-slate-dark rounded shadow-lg overflow-hidden flex flex-col relative before:absolute before:inset-0 before:bg-[linear-gradient(rgba(26,36,32,0.5)_1px,transparent_1px),linear-gradient(90deg,rgba(26,36,32,0.5)_1px,transparent_1px)] before:bg-[size:20px_20px] before:opacity-20 before:pointer-events-none">
            
            <div className="bg-surface px-4 py-2 flex items-center justify-between border-b border-slate-dark z-10">
              <span className="font-mono text-xs text-neon uppercase flex items-center gap-2">
                <span className="w-2 h-2 bg-neon rounded-full animate-pulse" />
                Coach_Engine_Output
              </span>
            </div>

            {/* Scenario Selector */}
            <div className="p-4 border-b border-slate-dark flex gap-2 overflow-x-auto z-10">
              {scenarios.map((scenario, idx) => (
                <button
                  key={scenario.id}
                  onClick={() => setActiveScenario(idx)}
                  className={`px-3 py-1.5 font-mono text-xs uppercase tracking-wider whitespace-nowrap border transition-colors ${
                    activeScenario === idx 
                      ? "bg-neon/10 border-neon text-neon" 
                      : "bg-surface border-slate-dark text-chalk/50 hover:text-chalk"
                  }`}
                >
                  {scenario.label}
                </button>
              ))}
            </div>

            {/* Terminal Output */}
            <div className="flex-1 p-6 font-mono text-sm leading-relaxed text-chalk/90 z-10 relative">
               <div className="whitespace-pre-wrap break-words">
                 {displayedText}
                 <span className={`inline-block w-2.5 h-4 ml-1 bg-neon ${isTyping ? '' : 'animate-pulse'}`} />
               </div>
               
               <div className="absolute bottom-6 left-6 font-mono text-xs text-slate-light">
                 [MODE: {isLocalMode ? 'OFFLINE_NATIVE' : 'AWS_CLOUD_NODE'}] | LATENCY: {isLocalMode ? '8ms' : '142ms'}
               </div>
            </div>

          </div>

        </div>

      </div>
    </section>
  );
}
