import React, { useEffect, useState } from 'react';
import { Play, Pause, FastForward, SkipBack, Radio, Video, Expand } from 'lucide-react';

export function VideoHUD() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [progress, setProgress] = useState(0);

  // Mock timeline and radar movement
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setProgress(prev => (prev >= 100 ? 0 : prev + 0.1));
    }, 50);
    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className="w-full max-w-6xl h-[280px] flex gap-4 items-center justify-center font-mono animate-fade-in z-20">
      
      {/* 1. Left Side: Raw Uploaded Match Video */}
      <div className="flex-1 h-full relative bg-[#0a0f0a] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)] flex flex-col group">
        
        {/* Placeholder Pitch Background representing video fill */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#111a12] to-[#0a1a10] opacity-80">
           {/* Abstract grass stripes */}
           <div className="w-full h-full flex flex-col justify-around opacity-10">
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
           </div>
        </div>

        {/* Top Badge */}
        <div className="absolute top-3 left-3 bg-black/80 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-gray-800">
           <Video size={12} className="text-gray-400" />
           <span className="text-[10px] text-gray-300 font-bold uppercase tracking-widest">Raw Source Feed</span>
        </div>

        {/* Center Mock Subjects (Silhouettes) */}
        <div className="absolute inset-0 flex items-center justify-center opacity-60">
             <div className="w-8 h-12 bg-gray-400/20 blur-sm rounded translate-x-12 -translate-y-4"></div>
             <div className="w-8 h-10 bg-gray-400/20 blur-sm rounded -translate-x-16 translate-y-8"></div>
        </div>

        {/* Video Controls (Bottom) */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-4 transition-all opacity-100 sm:opacity-0 sm:group-hover:opacity-100 z-10">
           <div className="w-full h-1bg-gray-800 rounded-full mb-3 cursor-pointer group/bar relative">
              <div className="h-1 bg-gray-600 rounded-full absolute w-full"></div>
              <div className="h-1 bg-emerald-500 rounded-full absolute" style={{ width: `${progress}%` }}></div>
           </div>
           
           <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-gray-300">
                 <button onClick={() => setIsPlaying(!isPlaying)} className="hover:text-white">
                   {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                 </button>
                 <SkipBack size={16} className="cursor-pointer hover:text-white" />
                 <FastForward size={16} className="cursor-pointer hover:text-white" />
                 <span className="text-xs font-mono ml-2">12:45 / 94:12</span>
              </div>
              <Expand size={14} className="text-gray-400 cursor-pointer hover:text-white" />
           </div>
        </div>
      </div>

      {/* 2. Right Side: Telemetry 2D Radar */}
      <div className="flex-1 h-full relative bg-[#050805] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
        
        {/* Top Badge */}
        <div className="absolute top-3 left-3 bg-emerald-900/40 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-emerald-800/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
           <Radio size={12} className="text-emerald-400 animate-pulse" />
           <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-widest">Telemetry Engine</span>
        </div>
        
        {/* Right Stats */}
        <div className="absolute bottom-3 right-3 text-[9px] text-gray-500 flex gap-3 z-10 uppercase tracking-widest font-bold bg-black/60 px-2 py-1 rounded">
           <span>Model: V4_Geo</span>
           <span className="text-emerald-500">Sync: OK</span>
        </div>

        {/* Radar Pitch Board */}
        <div className="absolute inset-4 border border-emerald-900/30 rounded bg-[#030603] perspective-1000">
             {/* Center Lines */}
             <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-emerald-900/30 -translate-x-1/2"></div>
             <div className="absolute left-1/2 top-1/2 w-20 h-20 border border-emerald-900/30 rounded-full -translate-x-1/2 -translate-y-1/2"></div>
             
             {/* Dynamic Vectors / Passing Lanes */}
             {isPlaying && (
               <svg className="absolute inset-0 w-full h-full opacity-40 z-0">
                 <line x1="30%" y1="60%" x2="45%" y2="40%" stroke="#10b981" strokeWidth="1" strokeDasharray="4" className="animate-pulse" />
                 <line x1="45%" y1="40%" x2="65%" y2="55%" stroke="#10b981" strokeWidth="1" strokeDasharray="4" className="animate-pulse" style={{ animationDelay: '0.2s' }} />
               </svg>
             )}

             {/* Team A (Blue) Dots - Using math to jitter their positions slightly */}
             <div className="absolute w-3 h-3 bg-blue-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.8)] z-10 transition-all duration-300" style={{ left: `calc(30% + ${Math.sin(progress) * 5}px)`, top: `calc(60% + ${Math.cos(progress) * 3}px)` }}></div>
             <div className="absolute w-3 h-3 bg-blue-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.8)] z-10 transition-all duration-300" style={{ left: `calc(20% + ${Math.cos(progress * 1.5) * 4}px)`, top: `calc(30% + ${Math.sin(progress) * 6}px)` }}></div>
             {/* Active Highlighted Player */}
             <div className="absolute w-4 h-4 bg-emerald-400 rounded-full ring-2 ring-white shadow-[0_0_15px_rgba(16,185,129,1)] z-20 transition-all duration-500" style={{ left: `calc(45% + ${Math.sin(progress * 0.8) * 8}px)`, top: `calc(40% + ${Math.cos(progress * 0.8) * 8}px)` }}>
               <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] text-emerald-400 font-bold tracking-widest">FALSE_9</div>
             </div>

             {/* Team B (Red) Dots */}
             <div className="absolute w-3 h-3 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)] z-10 transition-all duration-300" style={{ left: `calc(70% + ${Math.cos(progress) * 5}px)`, top: `calc(50% + ${Math.sin(progress) * 3}px)` }}></div>
             <div className="absolute w-3 h-3 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)] z-10 transition-all duration-300" style={{ left: `calc(65% + ${Math.sin(progress * 1.2) * 4}px)`, top: `calc(75% + ${Math.cos(progress) * 6}px)` }}></div>
             <div className="absolute w-3 h-3 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.8)] z-10 transition-all duration-300" style={{ left: `calc(80% + ${Math.sin(progress * 0.5) * 7}px)`, top: `calc(25% + ${Math.cos(progress * 0.5) * 7}px)` }}></div>
        </div>

      </div>

    </div>
  );
}
