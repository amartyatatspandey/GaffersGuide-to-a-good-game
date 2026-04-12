import React from 'react';
import { Minus, Square, X } from 'lucide-react';

export function Titlebar() {
  return (
    <div className="h-8 bg-black flex items-center justify-between border-b border-[#1a2420] select-none shrink-0" style={{ WebkitAppRegion: 'drag' } as any}>
      <div className="pl-4 flex items-center h-full">
        <span className="text-xs font-bold font-mono tracking-tight text-gray-300">
          Gaffers<span className="text-neon">.</span>Guide
        </span>
      </div>
      <div className="flex-1 flex justify-center items-center h-full">
        <span className="text-xs text-gray-500 font-mono tracking-wider">Match_Analysis_Final.mp4</span>
      </div>
      <div className="flex h-full" style={{ WebkitAppRegion: 'no-drag' } as any}>
        <button className="h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center">
          <Minus size={14} />
        </button>
        <button className="h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center">
          <Square size={12} />
        </button>
        <button className="h-full px-4 hover:bg-red-600 hover:text-white text-gray-400 transition-colors flex items-center justify-center">
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
