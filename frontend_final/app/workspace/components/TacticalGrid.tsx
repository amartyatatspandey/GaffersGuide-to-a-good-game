"use client";
import React from 'react';

export function TacticalGrid({ activeRole }: { activeRole: string | null }) {
  // A CSS Grid representing the pitch
  const isFalse9 = activeRole === 'False 9';
  const isZone14 = activeRole === 'Zone 14';
  const isInvertedFB = activeRole === 'Inverted Fullback';

  return (
    <div className="w-full h-full p-4 flex flex-col items-center justify-center bg-[#111a12] border border-[#1a2420] rounded-xl relative overflow-hidden">
      <h3 className="absolute top-4 left-4 text-xs font-bold text-gray-500 uppercase tracking-widest font-mono">Spatial Mapping</h3>
      
      <div className="w-full max-w-xs aspect-[3/4] grid grid-rows-6 grid-cols-4 gap-1 p-2 border border-gray-800 bg-pitch rounded-md relative shadow-2xl">
        {/* Draw a grid of cells representing positional zones */}
        {Array.from({ length: 24 }).map((_, i) => {
          const r = Math.floor(i / 4);
          const c = i % 4;
          
          let highlightClass = "border border-gray-800/20 bg-[#0a0f0a] transition-all duration-500";
          
          // Zone 14 is conceptually the center-attacking block (Row 2, Col 1&2 for 4 columns)
          if (r === 2 && (c === 1 || c === 2) && (isZone14 || isFalse9)) {
            highlightClass = "border border-emerald-500 shadow-[inset_0_0_20px_rgba(16,185,129,0.4)] bg-emerald-900/30 animate-pulse";
          }
          
          // Inverted FB highlight (Row 4, Col 2 or 1 ... cutting inside into midfield)
          if (r === 3 && (c === 1 || c === 2) && isInvertedFB) {
            highlightClass = "border border-emerald-500 shadow-[inset_0_0_20px_rgba(16,185,129,0.4)] bg-emerald-900/30 animate-pulse";
          }
          
          return <div key={i} className={highlightClass} />;
        })}
        
        {/* Midline representation */}
        <div className="absolute top-1/2 left-0 w-full h-[1px] bg-gray-700 pointer-events-none"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 rounded-full border border-gray-700 pointer-events-none"></div>
      </div>
      
      {activeRole && (
        <div className="absolute bottom-4 right-4 bg-emerald-900/50 text-emerald-400 border border-emerald-500 px-3 py-1 font-mono text-xs font-bold shadow-lg uppercase bounce-in">
          TRACKING: {activeRole}
        </div>
      )}
    </div>
  );
}
