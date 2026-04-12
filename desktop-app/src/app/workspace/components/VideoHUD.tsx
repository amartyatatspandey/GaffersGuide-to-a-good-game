import React, { useEffect, useMemo, useRef, useState } from "react";
import { FastForward, Pause, Play, Radio, SkipBack, Video } from "lucide-react";

import {
  type JobArtifactsResponse,
  type TrackingDataResponse,
  getBetaSourceVideoUrl,
  getBetaTracking,
} from "@/lib/api";

interface VideoHUDProps {
  jobId: string | null;
  artifacts: JobArtifactsResponse | null;
}

function overlayStatusLabel(
  state: JobArtifactsResponse["overlay_state"] | undefined,
  reason: string | null | undefined,
): string {
  if (state === "ready") {
    return "Annotated overlay available";
  }
  if (state === "unavailable") {
    return reason ?? "Annotated overlay not generated";
  }
  if (state === "not_ready") {
    return "Annotated overlay not ready";
  }
  return "Annotated overlay —";
}

export function VideoHUD({ jobId, artifacts }: VideoHUDProps): React.JSX.Element {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [progressPct, setProgressPct] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [trackingData, setTrackingData] = useState<TrackingDataResponse | null>(null);
  const [trackingError, setTrackingError] = useState<string | null>(null);
  const [trackingFrameIndex, setTrackingFrameIndex] = useState(0);

  const sourceVideoUrl = useMemo(() => {
    if (!jobId) {
      return null;
    }
    return getBetaSourceVideoUrl(jobId);
  }, [jobId]);

  const activeFrame = useMemo(() => {
    if (!trackingData?.frames?.length) {
      return null;
    }
    const idx = Math.max(0, Math.min(trackingFrameIndex, trackingData.frames.length - 1));
    return trackingData.frames[idx];
  }, [trackingData, trackingFrameIndex]);

  useEffect(() => {
    if (!jobId) {
      setTrackingData(null);
      setTrackingError(null);
      return;
    }
    if (artifacts?.tracking_state !== "ready") {
      setTrackingData(null);
      setTrackingError(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const payload = await getBetaTracking(jobId);
        if (!cancelled) {
          setTrackingData(payload);
          setTrackingError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setTrackingError(error instanceof Error ? error.message : "Failed to load tracking data");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [artifacts?.tracking_state, jobId]);

  useEffect(() => {
    if (!trackingData?.frames?.length || duration <= 0) {
      return;
    }
    const ratio = Math.max(0, Math.min(1, currentTime / duration));
    const idx = Math.floor((trackingData.frames.length - 1) * ratio);
    setTrackingFrameIndex(idx);
  }, [currentTime, duration, trackingData]);

  useEffect(() => {
    if (!jobId) {
      setSourceError(null);
      setDuration(0);
      setCurrentTime(0);
      setProgressPct(0);
    }
  }, [jobId]);

  const radarEmptyMessage = useMemo(() => {
    if (trackingError) {
      return "Tracking load failed.";
    }
    if (artifacts?.tracking_state && artifacts.tracking_state !== "ready") {
      return "Tracking pending…";
    }
    if (artifacts?.tracking_state === "ready" && !activeFrame?.players?.length) {
      return "No player points in this frame.";
    }
    return "Tracking radar unavailable.";
  }, [activeFrame?.players?.length, artifacts?.tracking_state, trackingError]);

  const handlePlayPause = (): void => {
    const node = videoRef.current;
    if (!node) {
      return;
    }
    if (node.paused) {
      void node.play();
      setIsPlaying(true);
    } else {
      node.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (deltaSeconds: number): void => {
    const node = videoRef.current;
    if (!node) {
      return;
    }
    node.currentTime = Math.max(0, Math.min((node.duration || 0) - 0.1, node.currentTime + deltaSeconds));
  };

  const handleScrub = (event: React.MouseEvent<HTMLDivElement>): void => {
    const node = videoRef.current;
    if (!node || !duration) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    node.currentTime = ratio * duration;
  };

  return (
    <div className="w-full max-w-6xl h-[280px] flex gap-4 items-center justify-center font-mono animate-fade-in z-20">

      <div className="flex-1 h-full relative bg-[#0a0f0a] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)] flex flex-col group">

        <div className="absolute inset-0 bg-gradient-to-b from-[#111a12] to-[#0a1a10] opacity-40 z-[2] pointer-events-none">
           <div className="w-full h-full flex flex-col justify-around opacity-10">
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
              <div className="w-full h-10 bg-emerald-500"></div>
           </div>
        </div>

        <div className="absolute top-3 left-3 bg-black/80 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-gray-800">
           <Video size={12} className="text-gray-400" />
           <span className="text-[10px] text-gray-300 font-bold uppercase tracking-widest">Match feed</span>
        </div>

        <div className="absolute top-3 right-3 z-10 max-w-[55%] text-right">
          <div className="rounded border border-gray-700 bg-black/70 px-2 py-1 text-[9px] text-gray-400 leading-snug">
            {overlayStatusLabel(artifacts?.overlay_state, artifacts?.overlay_reason)}
          </div>
        </div>

        {sourceVideoUrl && (
          <video
            key={sourceVideoUrl}
            ref={videoRef}
            src={sourceVideoUrl}
            className="absolute inset-0 h-full w-full object-cover z-[1]"
            controls={false}
            autoPlay={false}
            muted
            onLoadedMetadata={(event) => {
              setDuration(event.currentTarget.duration || 0);
              setSourceError(null);
            }}
            onTimeUpdate={(event) => {
              const t = event.currentTarget.currentTime || 0;
              const d = event.currentTarget.duration || 0;
              setCurrentTime(t);
              setProgressPct(d > 0 ? (t / d) * 100 : 0);
            }}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onError={() => {
              setSourceError("Source video failed to load.");
              setIsPlaying(false);
            }}
          />
        )}
        {!jobId && (
          <div className="absolute inset-0 z-[3] flex items-center justify-center bg-black/40">
            <div className="rounded border border-gray-700 bg-black/50 px-3 py-2 text-xs text-gray-400">
              Upload a match clip to preview video.
            </div>
          </div>
        )}
        {jobId && sourceError && (
          <div className="absolute inset-0 z-[3] flex items-center justify-center bg-black/50">
            <div className="rounded border border-red-800 bg-red-950/70 px-3 py-2 text-xs text-red-200">
              {sourceError}
            </div>
          </div>
        )}

        <div className="absolute inset-0 flex items-center justify-center opacity-20 z-[2] pointer-events-none">
             <div className="w-8 h-12 bg-gray-400/20 blur-sm rounded translate-x-12 -translate-y-4"></div>
             <div className="w-8 h-10 bg-gray-400/20 blur-sm rounded -translate-x-16 translate-y-8"></div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-4 transition-all opacity-100 sm:opacity-0 sm:group-hover:opacity-100 z-10">
           <div
             className="w-full h-2 bg-gray-800 rounded-full mb-3 cursor-pointer relative"
             onClick={handleScrub}
           >
              <div className="h-1 bg-gray-600 rounded-full absolute w-full"></div>
              <div className="h-1 bg-emerald-500 rounded-full absolute" style={{ width: `${progressPct}%` }}></div>
           </div>

           <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-gray-300">
                 <button onClick={handlePlayPause} className="hover:text-white">
                   {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                 </button>
                 <button onClick={() => handleSeek(-5)} className="hover:text-white">
                   <SkipBack size={16} />
                 </button>
                 <button onClick={() => handleSeek(5)} className="hover:text-white">
                   <FastForward size={16} />
                 </button>
                 <span className="text-xs font-mono ml-2">
                   {sourceError ? "Source unavailable" : jobId ? "Source ready" : "No clip"}
                 </span>
                 <span className="text-[10px] text-gray-500">
                   {Math.floor(currentTime)}s / {Math.floor(duration)}s
                 </span>
              </div>
           </div>
        </div>
      </div>

      <div className="flex-1 h-full relative bg-[#050805] rounded-xl overflow-hidden border border-gray-800 shadow-[0_0_30px_rgba(0,0,0,0.5)]">

        <div className="absolute top-3 left-3 bg-emerald-900/40 px-2 py-1.5 rounded flex items-center gap-2 z-10 border border-emerald-800/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
           <Radio size={12} className="text-emerald-400 animate-pulse" />
           <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-widest">Telemetry Engine</span>
        </div>

        <div className="absolute top-3 right-3 z-10 rounded border border-emerald-900/40 bg-black/60 px-2 py-1 text-[9px] text-emerald-600/90 uppercase tracking-widest font-bold">
          {artifacts?.tracking_state === "ready"
            ? "Tracking ready"
            : artifacts?.tracking_state
              ? "Tracking pending"
              : "Tracking —"}
        </div>

        <div className="absolute bottom-3 right-3 text-[9px] text-gray-500 flex gap-3 z-10 uppercase tracking-widest font-bold bg-black/60 px-2 py-1 rounded">
           <span>{jobId ? `Job: ${jobId.slice(0, 8)}` : "Job: None"}</span>
           <span className="text-emerald-500">{artifacts?.status ?? "pending"}</span>
        </div>

        <div className="absolute inset-4 border border-emerald-900/30 rounded bg-[#030603] perspective-1000 overflow-hidden">
             <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-emerald-900/30 -translate-x-1/2"></div>
             <div className="absolute left-1/2 top-1/2 w-20 h-20 border border-emerald-900/30 rounded-full -translate-x-1/2 -translate-y-1/2"></div>
             {!activeFrame?.players?.length && (
               <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-500 px-4 text-center">
                 {radarEmptyMessage}
               </div>
             )}
             {activeFrame?.players?.map((player, idx) => {
               const x = Math.max(0, Math.min(1, Number(player.x_pitch ?? 0.5)));
               const y = Math.max(0, Math.min(1, Number(player.y_pitch ?? 0.5)));
               const teamId = Number(player.team_id ?? 0);
               const teamClass = teamId === 1 ? "bg-red-500" : "bg-blue-500";
               return (
                 <div
                   key={`${activeFrame.frame_idx}-${idx}`}
                   className={`absolute w-2.5 h-2.5 rounded-full z-10 ${teamClass}`}
                   style={{ left: `${x * 100}%`, top: `${y * 100}%`, transform: "translate(-50%, -50%)" }}
                 />
               );
             })}
        </div>

      </div>

    </div>
  );
}
