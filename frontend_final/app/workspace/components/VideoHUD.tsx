import type { ChangeEvent } from "react";
import { RefObject, useCallback, useEffect, useState } from "react";

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${m}:${String(sec).padStart(2, "0")}`;
}

interface VideoHUDProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  videoSrc: string | null;
  jobId: string | null;
  status: string;
  onDownload: () => void;
}

export default function VideoHUD({
  videoRef,
  videoSrc,
  jobId,
  status,
  onDownload,
}: VideoHUDProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const syncFromVideo = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const d = v.duration;
    setDuration(Number.isFinite(d) ? d : 0);
    setCurrentTime(Number.isFinite(v.currentTime) ? v.currentTime : 0);
    setIsPlaying(!v.paused);
  }, [videoRef]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoSrc) {
      setCurrentTime(0);
      setDuration(0);
      setIsPlaying(false);
      return;
    }

    const onLoadedMetadata = (): void => syncFromVideo();
    const onTimeUpdate = (): void => syncFromVideo();
    const onSeeked = (): void => syncFromVideo();
    const onPlay = (): void => setIsPlaying(true);
    const onPause = (): void => setIsPlaying(false);
    const onEnded = (): void => {
      setIsPlaying(false);
      syncFromVideo();
    };

    v.addEventListener("loadedmetadata", onLoadedMetadata);
    v.addEventListener("durationchange", onLoadedMetadata);
    v.addEventListener("timeupdate", onTimeUpdate);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onEnded);

    if (v.readyState >= HTMLMediaElement.HAVE_METADATA) {
      syncFromVideo();
    }

    return () => {
      v.removeEventListener("loadedmetadata", onLoadedMetadata);
      v.removeEventListener("durationchange", onLoadedMetadata);
      v.removeEventListener("timeupdate", onTimeUpdate);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onEnded);
    };
  }, [videoRef, videoSrc, syncFromVideo]);

  const togglePlay = (): void => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
    } else {
      v.pause();
    }
  };

  const onSeekInput = (e: ChangeEvent<HTMLInputElement>): void => {
    const v = videoRef.current;
    if (!v) return;
    const t = parseFloat(e.target.value);
    if (!Number.isFinite(t)) return;
    v.currentTime = t;
    setCurrentTime(t);
  };

  const onVideoAreaClick = (): void => {
    togglePlay();
  };

  const maxT = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const rangeValue = Number.isFinite(currentTime) ? Math.min(Math.max(currentTime, 0), maxT || 0) : 0;

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col gap-2">
      <div className="flex flex-1 min-h-0 flex-col overflow-hidden rounded-xl border border-gray-800 bg-black shadow-2xl">
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div className="absolute left-3 top-3 z-10 flex gap-2">
            <span className="rounded bg-black/60 px-2 py-1 text-xs font-bold text-white backdrop-blur-md">
              LIVE FEED
            </span>
            {jobId && (
              <span className="rounded bg-blue-900/80 px-2 py-1 text-xs font-mono text-blue-100 backdrop-blur-md">
                {jobId.slice(0, 8)}…
              </span>
            )}
          </div>

          {videoSrc ? (
            <>
              <div
                className="relative min-h-0 flex-1 cursor-pointer bg-black"
                onClick={onVideoAreaClick}
                onKeyDown={(e) => {
                  if (e.key === " " || e.key === "Enter") {
                    e.preventDefault();
                    togglePlay();
                  }
                }}
                role="button"
                tabIndex={0}
                aria-label="Play or pause video"
              >
                <video
                  ref={videoRef}
                  src={videoSrc}
                  className="pointer-events-none h-full w-full object-cover"
                  playsInline
                  preload="metadata"
                />
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                  {!isPlaying && (
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm transition-transform hover:scale-110">
                      <svg className="ml-1 h-8 w-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  )}
                </div>
              </div>

              <div
                className="flex shrink-0 items-center gap-2 border-t border-gray-800 bg-black/90 px-2 py-2"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              >
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePlay();
                  }}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/10 text-white transition hover:bg-white/20"
                  aria-label={isPlaying ? "Pause" : "Play"}
                >
                  {isPlaying ? (
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                    </svg>
                  ) : (
                    <svg className="ml-0.5 h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                </button>
                <input
                  type="range"
                  min={0}
                  max={maxT > 0 ? maxT : 0}
                  step="any"
                  value={maxT > 0 ? rangeValue : 0}
                  onChange={onSeekInput}
                  className="h-2 min-w-0 flex-1 cursor-pointer accent-emerald-500"
                  aria-label="Seek video"
                  disabled={maxT <= 0}
                />
                <span className="shrink-0 font-mono text-xs tabular-nums text-gray-300">
                  {formatClock(currentTime)} / {formatClock(duration)}
                </span>
              </div>
            </>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center text-gray-500">
              <div className="rounded-full bg-gray-900 p-6">
                <svg className="h-12 w-12 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-400">No video loaded</p>
                <p className="mt-1 text-xs">Upload a clip to begin analysis</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between px-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-gray-500">Source</span>
        <div className="flex items-center gap-2">
          <span
            className={`text-[10px] font-bold uppercase ${
              status === "completed" ? "text-emerald-500" : "text-gray-600"
            }`}
          >
            {status}
          </span>
          <button
            type="button"
            onClick={onDownload}
            disabled={status !== "completed"}
            className="rounded border border-gray-800 bg-gray-900 px-2 py-1 text-[10px] font-bold uppercase text-gray-300 transition hover:bg-gray-800 disabled:opacity-50"
          >
            Download JSON
          </button>
        </div>
      </div>
    </div>
  );
}
