"use client";

import React, { useRef, useState } from "react";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import Image from "next/image";

const stagesData = [
  {
    step: "Step 1",
    title: "Ingest the raw reality of the pitch.",
    img: "/images/step1.png"
  },
  {
    step: "Step 2",
    title: "100% offline computer vision maps the chaos into coordinates.",
    img: "/images/step2.png"
  },
  {
    step: "Step 3",
    title: "Our deterministic math engine calculates the structural integrity of the match.",
    img: "/images/step3.png"
  },
  {
    step: "Step 4",
    title: "AI identifies the exact tactical vulnerability in real-time.",
    alert: "! Suicidal High Line",
    img: "/images/step4.png"
  },
  {
    step: "Step 5",
    title: "Gaffer's Guide gives you the exact touchline instruction to fix it.",
    img: "/images/step4.png" // Reusing step4 image for the instruction
  }
];

export function ScrollAnimation() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeStage, setActiveStage] = useState(0);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    // 5 stages uniformly distributed over the scroll height
    let stage = Math.floor(latest * 5);
    if (stage >= 5) stage = 4;
    setActiveStage(stage);
  });

  return (
    <section ref={containerRef} className="relative bg-gray-950 border-t border-white/5" style={{ height: "300vh" }}>
      <div className="sticky top-0 h-screen w-full overflow-hidden flex items-center">
        
        <div className="max-w-7xl mx-auto px-6 w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-24 items-center">
          
          {/* Left Side: Images (Sticky inside the sticky container) */}
          <div className="relative w-full aspect-square md:aspect-video lg:aspect-square rounded-2xl overflow-hidden shadow-2xl border border-white/10 bg-gray-900 group">
             {stagesData.map((data, idx) => (
                <motion.div
                  key={idx}
                  className="absolute inset-0 w-full h-full"
                  initial={{ opacity: 0, scale: 1.05 }}
                  animate={{ 
                    opacity: activeStage === idx ? 1 : 0,
                    scale: activeStage === idx ? 1 : 1.05
                  }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                >
                  <Image 
                    src={data.img}
                    alt={data.title}
                    fill
                    className="object-cover"
                    priority
                  />
                  {/* Subtle vignette overlay */}
                  <div className="absolute inset-0 bg-radial-[at_50%_50%] from-transparent to-gray-900/60 pointer-events-none" />
                </motion.div>
             ))}

             {/* Insight Card overlay for Step 5 */}
             <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: activeStage === 4 ? 1 : 0, y: activeStage === 4 ? 0 : 50 }}
                transition={{ duration: 0.5, delay: activeStage === 4 ? 0.3 : 0 }}
                className="absolute bottom-6 mx-6 right-0 left-0 lg:left-auto lg:right-6 lg:w-80 glass p-5 rounded-xl border-emerald-500/40 shadow-[0_10px_40px_-10px_rgba(16,185,129,0.3)] z-20"
             >
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-emerald-400 text-xs font-mono uppercase tracking-wider">Tactical Action</span>
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Drop the Defensive Line</h3>
                <p className="text-xs text-gray-300 mb-3 leading-relaxed">
                  Your high block is disjointed. The pivot (#4) is pressing aggressively, leaving Zone 14 exposed. Drop the line by 15 meters to compact the shape.
                </p>
             </motion.div>
          </div>

          {/* Right Side: Text Scroller */}
          <div className="relative flex flex-col justify-center">
            {/* The text changes based on the activeStage */}
            <div className="relative min-h-[300px] flex items-center">
               {stagesData.map((data, idx) => (
                 <motion.div
                   key={idx}
                   initial={{ opacity: 0, y: 30 }}
                   animate={{ 
                    opacity: activeStage === idx ? 1 : 0,
                    y: activeStage === idx ? 0 : (activeStage < idx ? 30 : -30),
                    pointerEvents: activeStage === idx ? "auto" : "none"
                   }}
                   transition={{ duration: 0.4 }}
                   className="absolute left-0 right-0"
                 >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-emerald-400 text-sm font-medium mb-4 backdrop-blur-md">
                      {data.step}
                    </div>
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight text-white mb-6 leading-tight">
                      {data.title}
                    </h2>
                    {data.alert && (
                      <div className="inline-flex items-center gap-2 px-4 py-2 mt-4 rounded bg-amber-500/10 border border-amber-500/50 text-amber-400 font-mono text-sm animate-pulse">
                        {data.alert}
                      </div>
                    )}
                 </motion.div>
               ))}
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}

