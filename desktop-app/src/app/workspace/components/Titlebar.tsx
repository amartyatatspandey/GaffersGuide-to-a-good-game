import React, { type CSSProperties } from "react";
import { Minus, Square, X } from "lucide-react";

interface ElectronDragStyle extends CSSProperties {
  WebkitAppRegion?: "drag" | "no-drag";
}

export function Titlebar(): React.JSX.Element {
  const dragRegionStyle: ElectronDragStyle = { WebkitAppRegion: "drag" };
  const noDragRegionStyle: ElectronDragStyle = { WebkitAppRegion: "no-drag" };

  return (
    <div className="h-8 bg-black flex items-center justify-between border-b border-[#1a2420] select-none shrink-0" style={dragRegionStyle}>
      <div className="pl-4 flex items-center h-full">
        <span className="text-xs font-bold font-mono tracking-tight text-gray-300">
          Gaffers<span className="text-neon">.</span>Guide
        </span>
      </div>
      <div className="flex-1 flex justify-center items-center h-full">
        <span className="text-xs text-gray-500 font-mono tracking-wider">Match_Analysis_Final.mp4</span>
      </div>
      <div className="flex h-full" style={noDragRegionStyle}>
        <button
          className="h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center"
          onClick={() => window.desktopWindow?.minimize()}
        >
          <Minus size={14} />
        </button>
        <button
          className="h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center"
          onClick={() => window.desktopWindow?.maximize()}
        >
          <Square size={12} />
        </button>
        <button
          className="h-full px-4 hover:bg-red-600 hover:text-white text-gray-400 transition-colors flex items-center justify-center"
          onClick={() => window.desktopWindow?.close()}
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
