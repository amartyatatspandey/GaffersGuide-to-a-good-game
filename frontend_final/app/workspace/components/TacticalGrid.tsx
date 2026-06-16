"use client";
import React, { useState } from 'react';
import { Target, Shield, Zap, Activity, TrendingUp, Info } from 'lucide-react';

export function TacticalGrid({ zonalData }: { zonalData?: any[] }) {
  const [hoveredZone, setHoveredZone] = useState<any>(null);

  if (!zonalData || zonalData.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-[#0a0f0a] border border-gray-900 rounded-xl relative overflow-hidden">
          <Activity className="text-gray-800 animate-pulse mb-2" size={32} />
          <p className="text-[10px] font-mono text-gray-600 uppercase tracking-widest font-bold">Awaiting spatial telemetry...</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-[#050805] border border-gray-900 rounded-xl relative overflow-hidden group">
      
      <div className="flex-1 w-full grid grid-cols-4 grid-rows-4 gap-1 p-1 bg-black/40 border border-gray-800/50 rounded-lg relative overflow-hidden">
        {/* Pitch markings */}
        <div className="absolute top-1/2 left-0 w-full h-[1px] bg-white/5 pointer-events-none" />
        <div className="absolute top-0 left-1/2 w-[1px] h-full bg-white/5 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 rounded-full border border-white/5 pointer-events-none" />

        {zonalData.slice(0, 16).map((zone, i) => {
          const control = zone.avg_control_pct;
          const threat = zone.threat_level;
          
          let colorClass = "bg-gray-900/10";
          let borderClass = "border-gray-800/10";
          
          if (control > 65) {
            colorClass = "bg-emerald-500/10 hover:bg-emerald-500/25";
            borderClass = "border-emerald-500/30";
          } else if (control < 35) {
            colorClass = "bg-red-500/10 hover:bg-red-500/25";
            borderClass = "border-red-500/30";
          } else if (control < 50) {
             colorClass = "bg-amber-500/10 hover:bg-amber-500/25";
             borderClass = "border-amber-500/30";
          }

          return (
            <div 
              key={i} 
              className={`relative transition-all duration-300 border ${borderClass} ${colorClass} cursor-pointer overflow-hidden flex items-center justify-center`}
              onMouseEnter={() => setHoveredZone(zone)}
              onMouseLeave={() => setHoveredZone(null)}
            >
               {/* Intensity indicator */}
               <div 
                 className="absolute bottom-0 left-0 h-[1.5px] bg-white/20 transition-all duration-500"
                 style={{ width: `${zone.avg_pressure_index || 0}%` }}
               />

               <span className="text-[9px] font-mono text-gray-600 font-bold opacity-30">
                  {zone.zone_id}
               </span>

               {/* Hover Stats */}
               {hoveredZone === zone && (
                 <div className="absolute inset-0 bg-black/90 backdrop-blur-sm flex flex-col items-center justify-center p-2 z-20">
                    <div className="text-[9px] font-mono text-gray-400 uppercase mb-1">Control</div>
                    <div className="text-xs font-bold text-emerald-500">{Math.round(control)}%</div>
                 </div>
               )}
            </div>
          );
        })}
      </div>

      <div className="p-2 border-t border-gray-900 bg-black/20 flex justify-between items-center">
         <div className="flex gap-4">
            <div className="flex items-center gap-1"><div className="w-1.5 h-1.5 bg-emerald-500/30 rounded-sm" /><span className="text-[8px] text-gray-600 uppercase font-mono">Dominance</span></div>
            <div className="flex items-center gap-1"><div className="w-1.5 h-1.5 bg-amber-500/30 rounded-sm" /><span className="text-[8px] text-gray-600 uppercase font-mono">Contested</span></div>
            <div className="flex items-center gap-1"><div className="w-1.5 h-1.5 bg-red-500/30 rounded-sm" /><span className="text-[8px] text-gray-600 uppercase font-mono">Vulnerable</span></div>
         </div>
         <span className="text-[8px] font-mono text-gray-700 uppercase italic">Interactive Zonal Matrix</span>
      </div>
    </div>
  );
}
