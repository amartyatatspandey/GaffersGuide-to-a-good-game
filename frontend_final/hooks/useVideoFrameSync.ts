import { RefObject, useEffect, useRef } from "react";

interface UseVideoFrameSyncOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  fps: number;
  maxFrameIndex: number;
  onFrameChange: (frameIndex: number) => void;
}

export function useVideoFrameSync({
  videoRef,
  fps,
  maxFrameIndex,
  onFrameChange,
}: UseVideoFrameSyncOptions): void {
  const rafIdRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(-1);
  const fpsRef = useRef<number>(fps);
  const maxFrameRef = useRef<number>(maxFrameIndex);
  const onFrameRef = useRef<(frameIndex: number) => void>(onFrameChange);

  fpsRef.current = fps;
  maxFrameRef.current = maxFrameIndex;
  onFrameRef.current = onFrameChange;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const stop = (): void => {
      if (rafIdRef.current != null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };

    const emitCurrent = (): void => {
      const curFps = Math.max(1, fpsRef.current);
      const idx = Math.max(
        0,
        Math.min(maxFrameRef.current, Math.round(video.currentTime * curFps)),
      );
      if (idx !== lastFrameRef.current) {
        lastFrameRef.current = idx;
        onFrameRef.current(idx);
      }
    };

    const tick = (): void => {
      emitCurrent();
      if (!video.paused && !video.ended) {
        rafIdRef.current = requestAnimationFrame(tick);
      } else {
        rafIdRef.current = null;
      }
    };

    const onPlay = (): void => {
      stop();
      rafIdRef.current = requestAnimationFrame(tick);
    };
    const onPauseOrEnd = (): void => stop();
    const onSeeked = (): void => emitCurrent();

    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPauseOrEnd);
    video.addEventListener("ended", onPauseOrEnd);
    video.addEventListener("seeked", onSeeked);

    // Render one frame immediately when hook attaches.
    emitCurrent();
    if (!video.paused && !video.ended) {
      rafIdRef.current = requestAnimationFrame(tick);
    }

    return () => {
      stop();
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPauseOrEnd);
      video.removeEventListener("ended", onPauseOrEnd);
      video.removeEventListener("seeked", onSeeked);
    };
  }, [videoRef]);
}

