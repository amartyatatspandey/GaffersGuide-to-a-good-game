"use client";

import React from "react";
import { motion } from "framer-motion";

export function DownloadCTA() {
  return (
    <footer id="download-footer" className="relative py-32 bg-pitch border-t border-slate-dark flex flex-col items-center justify-center text-center overflow-hidden">
      
      {/* Background ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neon/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative z-10 px-6 w-full max-w-4xl mx-auto">
        
        <h2 className="text-5xl md:text-7xl font-sans font-bold tracking-tight uppercase mb-6">
          Install <span className="text-neon">Drop</span> Analyze
        </h2>
        
        <p className="text-xl text-chalk/80 font-normal mb-12 max-w-2xl mx-auto leading-relaxed">
          The ultimate unfair advantage is ready to boot. Download the engine, import your match footage, and let the algorithm reveal the truth.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-3 px-8 py-4 bg-neon text-pitch font-mono font-bold uppercase tracking-widest rounded shadow-[0_0_30px_rgba(0,230,118,0.4)] hover:shadow-[0_0_50px_rgba(0,230,118,0.6)] transition-shadow"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
               <path d="M12 2C17.52 2 22 6.48 22 12C22 17.52 17.52 22 12 22C6.48 22 2 17.52 2 12C2 6.48 6.48 2 12 2ZM11 19.93C7.05 19.43 4 16.05 4 12C4 7.95 7.05 4.57 11 4.07V19.93ZM13 4.07C16.95 4.57 20 7.95 20 12C20 16.05 16.95 19.43 13 19.93V4.07Z"/>
            </svg>
            Download for macOS
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-3 px-8 py-4 bg-transparent border-2 border-neon text-neon font-mono font-bold uppercase tracking-widest rounded hover:bg-neon/10 transition-colors"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
               <path d="M0 3.449L9.75 2.1L9.75 11.5L0 11.5L0 3.449ZM24 0L24 11.5L10.5 11.5L10.5 1.992L24 0ZM0 12.5L9.75 12.5L9.75 21.9L0 20.551L0 12.5ZM10.5 12.5L24 12.5L24 24L10.5 22.008L10.5 12.5Z"/>
            </svg>
            Windows Binary (.exe)
          </motion.button>
        </div>

        <div className="mt-16 pt-8 border-t border-slate-dark text-sm font-mono text-slate-light flex flex-col md:flex-row justify-between items-center gap-4">
          <p>© 2026 GAF_FER_SYS. All Rights Reserved.</p>
          <div className="flex gap-6">
             <a href="#" className="hover:text-chalk transition-colors">Documentation</a>
             <a href="#" className="hover:text-chalk transition-colors">EULA</a>
             <a href="#" className="hover:text-chalk transition-colors">Telemetry Opt-Out</a>
          </div>
        </div>

      </div>
    </footer>
  );
}
