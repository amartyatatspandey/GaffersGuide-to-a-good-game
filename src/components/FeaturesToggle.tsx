"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, Cloud, Server, ShieldCheck } from "lucide-react";

export function FeaturesToggle() {
  const [isLocal, setIsLocal] = useState(true);

  return (
    <section className="relative py-32 bg-gray-950 overflow-hidden border-t py-20 border-white/5">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          
          {/* Left Column Text */}
          <div className="relative z-10">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Elite analytics shouldn't be a <span className="text-gradient">luxury.</span>
            </h2>
            <p className="text-xl text-gray-400 mb-8 font-light leading-relaxed">
              We built Gaffer's Guide to democratize the data. Top-flight clubs spend millions on server banks. We wrote a more efficient math layer so you can run the exact same models <span className="font-semibold text-emerald-400">directly on your laptop.</span>
            </p>

            <div className="mb-10 p-6 glass-card border border-emerald-500/20 bg-emerald-500/5">
              <h3 className="text-2xl font-semibold mb-4 text-white">The Hybrid Moat</h3>
              <p className="text-gray-300 mb-6">
                Your Tactics. Your Hardware. 100% Private. Switch between cloud processing for speed or strict local mode for absolute tactical secrecy.
              </p>
              
              {/* Interactive Toggle Switch */}
              <div className="flex items-center gap-4 bg-gray-900/80 p-2 rounded-2xl w-max border border-white/10 relative">
                <button
                  onClick={() => setIsLocal(false)}
                  className={`relative z-10 flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-colors ${!isLocal ? 'text-cyan-300' : 'text-gray-500'}`}
                >
                  <Cloud className="w-5 h-5" />
                  Cloud AI
                </button>
                <button
                  onClick={() => setIsLocal(true)}
                  className={`relative z-10 flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-colors ${isLocal ? 'text-emerald-400' : 'text-gray-500'}`}
                >
                  <Server className="w-5 h-5" />
                  Local Private AI
                </button>
                
                {/* Highlight Pill */}
                <motion.div
                  layout
                  className={`absolute top-2 bottom-2 w-[calc(50%-8px)] rounded-xl border ${isLocal ? 'bg-emerald-500/10 border-emerald-500/30 right-2' : 'bg-cyan-500/10 border-cyan-500/30 left-2'}`}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                />
              </div>
            </div>
          </div>

          {/* Right Column Abstract Visual */}
          <div className="relative h-[500px] flex items-center justify-center">
            {/* Visual Container */}
            <div className="absolute inset-0 max-w-md mx-auto aspect-square rounded-full border border-white/10 flex items-center justify-center shadow-[inset_0_0_100px_rgba(0,0,0,0.8)]">
              
              {/* Abstract Data Flow Animation */}
              <AnimatePresence mode="popLayout">
                {!isLocal ? (
                  // Cloud Flow
                  <motion.div
                    key="cloud"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.5 }}
                    className="absolute inset-0 flex items-center justify-center flex-col"
                  >
                    <div className="relative w-40 h-40 flex items-center justify-center">
                      <Cloud className="w-20 h-20 text-cyan-400 opacity-80" />
                      <div className="absolute inset-0 border-2 border-cyan-400 rounded-full animate-ping opacity-20" />
                      {/* Flowing dots going UP */}
                      <div className="absolute top-full flex flex-col items-center gap-2 pt-8">
                        {[...Array(4)].map((_, i) => (
                          <motion.div
                            key={i}
                            animate={{ y: [-40, 0], opacity: [0, 1, 0] }}
                            transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                            className="w-1.5 h-6 bg-gradient-to-t from-transparent to-cyan-400 rounded-full"
                          />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  // Local Flow
                  <motion.div
                    key="local"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.5 }}
                    className="absolute inset-0 flex items-center justify-center"
                  >
                    <div className="relative w-64 h-64 flex items-center justify-center">
                      <motion.div 
                        animate={{ rotate: 360 }} 
                        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-0 rounded-full border-t border-b border-emerald-500/50"
                      />
                      <motion.div 
                        animate={{ rotate: -360 }} 
                        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-4 rounded-full border-r border-l border-emerald-400/40"
                      />
                      
                      <div className="relative z-10 flex flex-col items-center">
                        <Lock className="w-16 h-16 text-emerald-400 mb-2" />
                        <ShieldCheck className="w-6 h-6 text-emerald-300 absolute -bottom-2 -right-2" />
                      </div>

                      {/* Data flowing INwards to lock */}
                      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
                        {[0, 60, 120, 180, 240, 300].map((deg, i) => (
                          <motion.circle
                            key={i}
                            cx="50" cy="15" r="1.5"
                            fill="#34d399"
                            animate={{ cy: [15, 35], opacity: [0, 1, 0] }}
                            transition={{ duration: 2, repeat: Infinity, delay: i * 0.4 }}
                            style={{ transformOrigin: '50px 50px', rotate: `${deg}deg` }}
                          />
                        ))}
                      </svg>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
