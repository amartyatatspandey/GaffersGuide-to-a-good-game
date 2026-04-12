"use client";

import React, { useRef } from "react";
import { useScroll, useTransform, motion } from "framer-motion";
import { TextReveal } from "@/components/TextReveal";

function BackgroundPitch() {
  // Static Base Formations
  const initialTeamA = [
    {id: 'A1', bx: 60, by: 250}, 
    {id: 'A2', bx: 150, by: 100}, {id: 'A3', bx: 130, by: 200}, {id: 'A4', bx: 130, by: 300}, {id: 'A5', bx: 150, by: 400},
    {id: 'A6', bx: 250, by: 150}, {id: 'A7', bx: 220, by: 250}, {id: 'A8', bx: 250, by: 350},
    {id: 'A9', bx: 350, by: 100}, {id: 'A10', bx: 330, by: 250}, {id: 'A11', bx: 350, by: 400} 
  ].map(p => ({ ...p, x: p.bx, y: p.by }));
  
  const initialTeamB = [
    {id: 'B1', bx: 740, by: 250}, 
    {id: 'B2', bx: 650, by: 100}, {id: 'B3', bx: 670, by: 200}, {id: 'B4', bx: 670, by: 300}, {id: 'B5', bx: 650, by: 400}, 
    {id: 'B6', bx: 550, by: 100}, {id: 'B7', bx: 570, by: 200}, {id: 'B8', bx: 570, by: 300}, {id: 'B9', bx: 550, by: 400}, 
    {id: 'B10', bx: 450, by: 200}, {id: 'B11', bx: 450, by: 300} 
  ].map(p => ({ ...p, x: p.bx, y: p.by }));

  const [nodesA, setNodesA] = React.useState(initialTeamA);
  const [nodesB, setNodesB] = React.useState(initialTeamB);

  // Animate the structure dynamically over time (Simulated Tactical Movement)
  React.useEffect(() => {
    const interval = setInterval(() => {
      const applyJitter = (node: any) => ({
        ...node,
        // Players constantly adjust within a 25px radius of their base structure
        x: node.bx + (Math.random() * 50 - 25),
        y: node.by + (Math.random() * 50 - 25)
      });
      setNodesA(prev => prev.map(applyJitter));
      setNodesB(prev => prev.map(applyJitter));
    }, 3000); // Shift every 3 seconds

    return () => clearInterval(interval);
  }, []);

  const ballPathX = [130, 220, 250, 350, 450, 570, 670, 550, 350, 250, 130];
  const ballPathY = [300, 250, 150, 100, 200, 200, 300, 400, 400, 350, 300];

  const CONNECTION_DISTANCE = 180; // slightly higher to keep lines connected during jitter
  const getConnections = (team: any[], prefix: string) => {
    const lines = [];
    for (let i = 0; i < team.length; i++) {
      for (let j = i + 1; j < team.length; j++) {
        const dx = team[i].x - team[j].x;
        const dy = team[i].y - team[j].y;
        if (Math.sqrt(dx * dx + dy * dy) < CONNECTION_DISTANCE) {
          lines.push({ id: `${prefix}-${team[i].id}-${team[j].id}`, x1: team[i].x, y1: team[i].y, x2: team[j].x, y2: team[j].y });
        }
      }
    }
    return lines;
  };

  const linesA = getConnections(nodesA, 'A');
  const linesB = getConnections(nodesB, 'B');

  return (
    <div className="absolute inset-0 z-0 flex items-center justify-center opacity-[0.50] pointer-events-none mix-blend-screen overflow-hidden">
      <svg viewBox="-10 -10 820 520" className="w-[95vw] max-w-6xl h-auto" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" 
           style={{ transform: "perspective(1000px) rotateX(15deg) scale(1.05)" }}>
        
        <g stroke="#00e676" fill="none" strokeWidth="2" opacity="0.4">
          <rect x="20" y="20" width="760" height="460" />
          <line x1="400" y1="20" x2="400" y2="480" />
          <circle cx="400" cy="250" r="70" />
          <rect x="20" y="100" width="130" height="300" />
          <rect x="650" y="100" width="130" height="300" />
          <rect x="20" y="180" width="45" height="140" />
          <rect x="735" y="180" width="45" height="140" />
          <circle cx="400" cy="250" r="3" fill="#00e676" />
        </g>

        {/* Neural Maps (Animated dynamically via SVG transitions!) */}
        <g stroke="#e8f0e9" strokeWidth="1" strokeDasharray="4 4" opacity="0.4">
          {linesA.map((line) => (
            <motion.line key={line.id} animate={{ x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2 }} transition={{ duration: 3, ease: 'easeInOut' }} />
          ))}
        </g>
        <g stroke="#00e676" strokeWidth="1" strokeDasharray="4 4" opacity="0.4">
          {linesB.map((line) => (
            <motion.line key={line.id} animate={{ x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2 }} transition={{ duration: 3, ease: 'easeInOut' }} />
          ))}
        </g>

        {/* Moving Players */}
        {nodesA.map((p) => (
          <motion.circle key={p.id} r="5" fill="#e8f0e9" opacity="0.9" animate={{ cx: p.x, cy: p.y }} transition={{ duration: 3, ease: 'easeInOut' }} />
        ))}
        {nodesB.map((p) => (
          <motion.circle key={p.id} r="5" fill="#00e676" opacity="0.9" animate={{ cx: p.x, cy: p.y }} transition={{ duration: 3, ease: 'easeInOut' }} />
        ))}

        {/* Simulated Ball Sequence */}
        <motion.circle 
          r="4" fill="#ffb300"
          animate={{ cx: ballPathX, cy: ballPathY }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          style={{ filter: "drop-shadow(0px 0px 5px rgba(255,179,0,0.9))" }}
        />
      </svg>
    </div>
  );
}

function HeroScrollytelling() {
  const containerRef = useRef<HTMLElement>(null);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });

  const scale = useTransform(scrollYProgress, [0, 1], [1, 1.1]);

  return (
    <section ref={containerRef} className="relative min-h-[75vh] md:min-h-[85vh] bg-pitch flex items-center justify-center overflow-hidden py-12">
      
      {/* Super subtle background animation */}
      <BackgroundPitch />

      {/* Layer 2 (Foreground Text) */}
      <motion.div 
        style={{ scale }}
        className="relative z-10 flex flex-col items-center justify-center px-4 w-full h-full"
      >
        <div className="text-chalk font-sans font-bold text-[13vw] md:text-[12vw] leading-[1.1] tracking-tight uppercase whitespace-nowrap">
          <TextReveal text="GAFFER'S GUIDE" />
        </div>
      </motion.div>

    </section>
  );
}

function AboutUs() {
  return (
    <section className="relative bg-pitch z-20 pt-48 pb-12 px-6">
      <div className="max-w-7xl mx-auto flex flex-col items-center text-center">
        
        {/* Top: Header */}
        <motion.h2 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-4xl md:text-6xl lg:text-[5rem] font-sans font-black tracking-tight leading-[1.1] text-chalk mb-8"
        >
          Math meets the <span className="text-neon">Beautiful Game</span>
        </motion.h2>

        {/* Middle: Mission Paragraphs */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ delay: 0.2 }}
          className="text-lg md:text-xl text-chalk/70 font-light max-w-4xl mb-4 space-y-6 leading-relaxed font-sans"
        >
          <p>
            Bridging computer vision tracking with universal tactical philosophies. 
            We eliminate the noise by tracking high-fidelity player coordinate data in 
            real-time, mapping complex spatial relationships into actionable tactical maneuvers. 
            No guesswork, just pure geometric dominance.
          </p>
          <p>
            Built by engineers from elite quantitative trading firms alongside seasoned tactical analysts, our engine processes raw match footage securely on your hardware. It identifies structural vulnerabilities and exposes defensive gaps before the opposition even surfaces them, seamlessly aligning computational power with world-class coaching.
          </p>
        </motion.div>

      </div>
    </section>
  );
}

export function OpeningSequence() {
  return (
    <div className="w-full relative bg-pitch">
      <HeroScrollytelling />
      <AboutUs />
    </div>
  );
}
