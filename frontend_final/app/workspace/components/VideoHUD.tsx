import React, { useEffect, useState, useRef } from 'react';
import { Play, Pause, Radio, Video } from 'lucide-react';
import RadarWidget from './radar/RadarWidget';

export function VideoHUD({ file, trackingData }: { file?: File | null, trackingData?: any }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoSrc, setVideoSrc] = useState<string>('');

  useEffect(() => {
    if (file) {
      const src = URL.createObjectURL(file);
      setVideoSrc(src);
      return () => {
        URL.revokeObjectURL(src);
      };
    }
  }, [file]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  return (
    <div className="w-full max-w-6xl h-[280px] flex gap-4 items-center justify-center font-mono animate-fade-in z-20">
      
      {/* 1. Left Side: Raw Uploaded Match Video */}
      <div className="flex-1 h-full relative bg-[#0a0f0a] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)] flex flex-col group">
        <div className="absolute top-3 left-3 bg-black/80 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-gray-800">
           <Video size={12} className="text-gray-400" />
           <span className="text-[10px] text-gray-300 font-bold uppercase tracking-widest">Raw Source Feed</span>
        </div>

        {videoSrc ? (
          <video 
            ref={videoRef}
            src={videoSrc}
            className="w-full h-full object-cover"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
            playsInline
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600 bg-[#111a12]">No Source Video</div>
        )}

        {/* Video Controls Overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-4 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 z-10 transition-all">
           <div className="flex items-center gap-4 text-gray-300">
              <button onClick={togglePlay} className="hover:text-white transition-colors">
                {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
              </button>
           </div>
        </div>
      </div>

      {/* 2. Right Side: Telemetry 2D Radar */}
      <div className="flex-1 h-full relative bg-[#050805] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
        <div className="absolute top-3 left-3 bg-emerald-900/40 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-emerald-800/50">
           <Radio size={12} className="text-emerald-400 animate-pulse" />
           <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-widest">Telemetry Engine</span>
        </div>
        
        {trackingData ? (
          <div className="w-full h-full pt-10 px-4 pb-4">
            <RadarWidget trackingData={trackingData} videoRef={videoRef} />
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-emerald-900/50 font-mono text-sm tracking-widest">
             AWAITING TELEMETRY SYNC
          </div>
        )}
      </div>
    </div>
  );
}
