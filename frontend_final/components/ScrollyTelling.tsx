"use client";

import React, { useRef } from "react";
import { useScroll, useTransform, motion } from "framer-motion";

export function ScrollyTelling() {
  const containerRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  // Typography Opacity Maps
  // 0.0 - 0.2: "YOUR TACTICS. DECODED."
  const opText0 = useTransform(scrollYProgress, [0, 0.15, 0.2], [1, 1, 0]);
  
  // 0.2 - 0.4: "RAW FOOTAGE PIPELINE."
  const opText1 = useTransform(scrollYProgress, [0.15, 0.2, 0.35, 0.4], [0, 1, 1, 0]);
  
  // 0.4 - 0.6: "COMPUTER VISION TRACKING."
  const opText2 = useTransform(scrollYProgress, [0.35, 0.4, 0.55, 0.6], [0, 1, 1, 0]);
  
  // 0.6 - 0.8: "2D COORDINATE PROJECTION."
  const opText3 = useTransform(scrollYProgress, [0.55, 0.6, 0.75, 0.8], [0, 1, 1, 0]);
  
  // 0.8 - 1.0: "VULNERABILITY SCAN."
  const opText4 = useTransform(scrollYProgress, [0.75, 0.8, 1], [0, 1, 1]);


  // Visual Stage Opacity Maps
  // Stage 1 (Raw)
  const opVis1 = useTransform(scrollYProgress, [0, 0.4, 0.45], [1, 1, 0]);
  
  // Stage 2 (Tracking - Neon Boxes) starts at 0.35, ends around 0.6
  const opVis2 = useTransform(scrollYProgress, [0.35, 0.4, 0.6, 0.65], [0, 1, 1, 0]);
  
  // Stage 3 (Map - Top-down 2D clean SVG) starts at 0.55, stays active for Stage 4
  const opVis3 = useTransform(scrollYProgress, [0.55, 0.6, 1], [0, 1, 1]);
  
  // Stage 4 (Scan - Zone 14 glows in amber) starts at 0.75
  const opVis4 = useTransform(scrollYProgress, [0.75, 0.8, 1], [0, 1, 1]);

  return (
    <section ref={containerRef} className="relative h-[500vh] bg-black">
      
      {/* BackgroundVisuals (z-index: 0) */}
      <div className="sticky top-0 h-screen w-full bg-black z-0 overflow-hidden flex items-center justify-center">
         
         {/* Visual Stage 1 (Raw) */}
         <motion.div style={{ opacity: opVis1 }} className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-full h-full bg-grid opacity-30 filter blur-sm grayscale" />
         </motion.div>

         {/* Visual Stage 2 (Tracking) */}
         <motion.div style={{ opacity: opVis2 }} className="absolute inset-0 flex items-center justify-center pointer-events-none p-8">
            <svg viewBox="0 0 800 500" className="w-[80vw] max-w-4xl h-auto opacity-80" xmlns="http://www.w3.org/2000/svg">
              {/* Animated Bounding Boxes simulating YOLO */}
              {[
                { x: 300, y: 150 }, { x: 450, y: 300 }, { x: 200, y: 250 }, { x: 550, y: 100 }
              ].map((pos, i) => (
                <motion.g key={i}>
                  <motion.rect 
                     width="40" height="80" 
                     fill="none" stroke="#00e676" strokeWidth="3"
                     animate={{ 
                       x: [pos.x, pos.x + 30, pos.x], 
                       y: [pos.y, pos.y - 15, pos.y],
                       opacity: [0.3, 1, 0.3]
                     }}
                     transition={{ duration: 2 + i * 0.5, repeat: Infinity, ease: "linear" }}
                  />
                  <motion.text 
                     fill="#00e676" fontSize="14" fontFamily="monospace" fontWeight="bold"
                     animate={{ x: [pos.x, pos.x + 30, pos.x], y: [pos.y - 10, pos.y - 25, pos.y - 10] }}
                     transition={{ duration: 2 + i * 0.5, repeat: Infinity, ease: "linear" }}
                  >
                    ID:{10 + i}
                  </motion.text>
                </motion.g>
              ))}
            </svg>
         </motion.div>

         {/* Visual Stage 3 (Map) -> Stays pinned for Stage 4 */}
         <motion.div style={{ opacity: opVis3 }} className="absolute inset-0 flex items-center justify-center p-8 pointer-events-none">
            <svg viewBox="0 0 800 500" className="w-full max-w-5xl h-auto opacity-90" xmlns="http://www.w3.org/2000/svg">
               {/* 2D Pitch Diagram */}
               <rect x="20" y="20" width="760" height="460" fill="none" stroke="#e8f0e9" strokeWidth="2" opacity="0.4" />
               <line x1="400" y1="20" x2="400" y2="480" stroke="#e8f0e9" strokeWidth="2" opacity="0.4" />
               <circle cx="400" cy="250" r="70" fill="none" stroke="#e8f0e9" strokeWidth="2" opacity="0.4" />
               <rect x="20" y="100" width="130" height="300" fill="none" stroke="#e8f0e9" strokeWidth="2" opacity="0.4" />
               <rect x="650" y="100" width="130" height="300" fill="none" stroke="#e8f0e9" strokeWidth="2" opacity="0.4" />

               {/* Stage 4 (Scan) Overlay inside Map Stage SVG to align perfectly */}
               <motion.rect 
                 style={{ opacity: opVis4 }}
                 x="220" y="150" width="160" height="200" 
                 fill="#ffb300" fillOpacity="0.2" stroke="#ffb300" strokeWidth="3" strokeDasharray="10 10"
               />
               
               {/* Pulse Animation for Zone 14 */}
               <motion.rect 
                 style={{ opacity: opVis4, transformOrigin: "300px 250px" }}
                 x="220" y="150" width="160" height="200" 
                 fill="none" stroke="#ffb300" strokeWidth="1"
                 animate={{ scale: [1, 1.1, 1], opacity: [1, 0, 1] }}
                 transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
               />

               <motion.text 
                 style={{ opacity: opVis4 }}
                 x="230" y="140" fill="#ffb300" fontSize="16" fontFamily="monospace" fontWeight="bold"
               >
                 ! ZONE 14 EXPLOITED
               </motion.text>

               {/* GNN Graph Network Lines */}
               {[
                 { x1: 150, y1: 150, x2: 280, y2: 250 },
                 { x1: 150, y1: 360, x2: 280, y2: 250 },
                 { x1: 280, y1: 250, x2: 480, y2: 150 },
                 { x1: 280, y1: 250, x2: 350, y2: 120 }
               ].map((line, i) => (
                 <line key={`line-${i}`} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke="#e8f0e9" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />
               ))}

               {/* GNN Nodes */}
               {[
                  { cx: 150, cy: 150 }, { cx: 140, cy: 260 }, { cx: 150, cy: 360 }, { cx: 280, cy: 250 },
                  { cx: 350, cy: 120 }, { cx: 340, cy: 250 }, { cx: 480, cy: 150 }, { cx: 470, cy: 260 }
                ].map((pos, i) => (
                  <circle key={`node-${i}`} cx={pos.cx} cy={pos.cy} r="8" fill="#e8f0e9" />
                ))}
            </svg>
         </motion.div>
      </div>

      {/* ForegroundTextMask (z-index: 10) */}
      {/* 
        Must use `mix-blend-difference` alongside `text-white` over the dark background.
        The underlying SVG lines are white (#e8f0e9) and green (#00e676), so where the text 
        overlaps them, it will invert, slicing through dynamically.
      */}
      <div className="sticky top-0 h-screen w-full flex items-center justify-center pointer-events-none mix-blend-difference z-10 px-4">
        
        <motion.h1 style={{ opacity: opText0 }} className="absolute text-white font-black text-5xl md:text-[6vw] leading-none text-center tracking-tighter uppercase whitespace-pre-wrap">
          YOUR TACTICS. DECODED.
        </motion.h1>

        <motion.h1 style={{ opacity: opText1 }} className="absolute text-white font-black text-5xl md:text-[6vw] leading-none text-center tracking-tighter uppercase whitespace-pre-wrap">
          RAW FOOTAGE PIPELINE.
        </motion.h1>

        <motion.h1 style={{ opacity: opText2 }} className="absolute text-white font-black text-5xl md:text-[6vw] leading-none text-center tracking-tighter uppercase whitespace-pre-wrap">
          COMPUTER VISION TRACKING.
        </motion.h1>

        <motion.h1 style={{ opacity: opText3 }} className="absolute text-white font-black text-5xl md:text-[6vw] leading-none text-center tracking-tighter uppercase whitespace-pre-wrap">
          2D COORDINATE PROJECTION.
        </motion.h1>

        <motion.h1 style={{ opacity: opText4 }} className="absolute text-white font-black text-5xl md:text-[6vw] leading-none text-center tracking-tighter uppercase whitespace-pre-wrap">
          VULNERABILITY SCAN.
        </motion.h1>

      </div>

    </section>
  );
}
