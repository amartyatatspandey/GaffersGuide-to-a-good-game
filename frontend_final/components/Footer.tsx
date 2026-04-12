"use client";

import React from "react";
import { Download, Monitor } from "lucide-react";

export function Footer() {
  return (
    <footer id="download-footer" className="relative py-32 bg-gray-950 overflow-hidden">
      {/* Background Abstract Glows */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-50" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-xs h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent blur-sm" />
      
      <div className="absolute bottom-0 left-0 right-0 h-64 bg-emerald-950/20 blur-[100px] pointer-events-none" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
        <h2 className="text-4xl md:text-6xl font-bold tracking-tight text-white mb-8">
          Ready to change the way you see the game?
        </h2>
        
        <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto font-light">
          Join the early access cohort of analysts and coaches building the future of tactical football.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
          <a
            href="#"
            className="group relative inline-flex items-center justify-center gap-3 px-8 py-4 rounded-2xl bg-white text-gray-950 font-bold overflow-hidden transition-all hover:scale-105 active:scale-95"
          >
            <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-gray-200 to-white opacity-0 group-hover:opacity-100 transition-opacity" />
            <svg className="w-5 h-5 relative z-10" viewBox="0 0 24 24" fill="currentColor">
              <path d="M14.71 14.59c.09-3.72 3.03-5.59 3.16-5.67-1.74-2.58-4.44-2.93-5.4-3-2.31-.22-4.52 1.35-5.71 1.35-1.19 0-3.03-1.33-4.95-1.3-2.5.03-4.8 1.48-6.11 3.75-2.65 4.62-.68 11.45 1.9 15.21 1.25 1.83 2.73 3.88 4.7 3.8 1.88-.08 2.61-1.22 4.88-1.22 2.26 0 2.94 1.22 4.9 1.18 2.02-.04 3.28-1.85 4.52-3.69 1.42-2.12 2-4.17 2.03-4.28-.05-.02-3.95-1.54-4.02-6.13zm-3.03-9.58c1.03-1.26 1.73-3.03 1.54-4.81-1.5.06-3.34 1.01-4.41 2.28-.95 1.14-1.79 2.92-1.57 4.66 1.67.13 3.39-.85 4.44-2.13z"/>
            </svg>
            <span className="relative z-10">Download for macOS (.dmg)</span>
          </a>

          <a
            href="#"
            className="group relative inline-flex items-center justify-center gap-3 px-8 py-4 rounded-2xl bg-gray-900 border border-white/10 text-white font-bold overflow-hidden transition-all hover:scale-105 active:scale-95 hover:border-emerald-500/50 hover:shadow-[0_0_30px_rgba(16,185,129,0.2)]"
          >
            <Monitor className="w-5 h-5 relative z-10 text-cyan-400 group-hover:text-emerald-400 transition-colors" />
            <span className="relative z-10">Download for Windows (.exe)</span>
          </a>
        </div>
        
        <div className="mt-20 pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between text-sm text-gray-500">
          <div className="flex items-center gap-2 mb-4 md:mb-0">
            {/* Small subtle logo */}
            <div className="w-5 h-5 rounded-sm bg-gradient-to-tr from-emerald-500 to-emerald-400 rotate-12 flex items-center justify-center opacity-50">
              <div className="w-2.5 h-2.5 bg-gray-950 rounded-sm -rotate-12" />
            </div>
            <span>© 2026 Gaffer's Guide. All rights reserved.</span>
          </div>
          <div className="flex gap-6">
            <a href="#" className="hover:text-white transition-colors">Privacy</a>
            <a href="#" className="hover:text-white transition-colors">Terms</a>
            <a href="#" className="hover:text-white transition-colors">Twitter (X)</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
