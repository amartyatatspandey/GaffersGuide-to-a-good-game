import { RefObject, useEffect, useRef } from "react";

interface UseVideoFrameSyncOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  fps: number;
  maxFrameIndex: number;
  onFrameChange: (frameIndex: number) => void;
}

const DEFAULT_FPS = 30;
const MAX_REF_RETRY_FRAMES = 360;

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
    let cancelled = false;
    let retryRafId: number | null = null;
    let retryCount = 0;
    let detachVideo: (() => void) | null = null;

    const stop = (): void => {
      if (rafIdRef.current != null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };

    const safeFps = (): number => {
      const f = fpsRef.current;
      if (Number.isFinite(f) && f > 0) return f;
      return DEFAULT_FPS;
    };

    const emitCurrent = (video: HTMLVideoElement): void => {
      const curFps = safeFps();
      const t = video.currentTime;
      const timeOk = Number.isFinite(t) && t >= 0 ? t : 0;
      const rawIdx = Math.round(timeOk * curFps);
      const idx = Math.max(
        0,
        Math.min(maxFrameRef.current, Number.isFinite(rawIdx) ? rawIdx : 0),
      );
      if (idx !== lastFrameRef.current) {
        lastFrameRef.current = idx;
        onFrameRef.current(idx);
      }
    };

    const tick = (video: HTMLVideoElement): void => {
      emitCurrent(video);
      if (!cancelled && !video.paused && !video.ended) {
        rafIdRef.current = requestAnimationFrame(() => tick(video));
      } else {
        rafIdRef.current = null;
      }
    };

    const attach = (video: HTMLVideoElement): (() => void) => {
      const onPlay = (): void => {
        stop();
        rafIdRef.current = requestAnimationFrame(() => tick(video));
      };
      const onPauseOrEnd = (): void => stop();
      const onSeeked = (): void => emitCurrent(video);
      const onSeeking = (): void => emitCurrent(video);
      const onTimeUpdate = (): void => emitCurrent(video);

      video.addEventListener("play", onPlay);
      video.addEventListener("pause", onPauseOrEnd);
      video.addEventListener("ended", onPauseOrEnd);
      video.addEventListener("seeked", onSeeked);
      video.addEventListener("seeking", onSeeking);
      video.addEventListener("timeupdate", onTimeUpdate);

      emitCurrent(video);
      if (!video.paused && !video.ended) {
        rafIdRef.current = requestAnimationFrame(() => tick(video));
      }

      return () => {
        stop();
        video.removeEventListener("play", onPlay);
        video.removeEventListener("pause", onPauseOrEnd);
        video.removeEventListener("ended", onPauseOrEnd);
        video.removeEventListener("seeked", onSeeked);
        video.removeEventListener("seeking", onSeeking);
        video.removeEventListener("timeupdate", onTimeUpdate);
      };
    };

    const tryAttach = (): void => {
      if (cancelled) return;
      const video = videoRef.current;
      if (!video) {
        if (retryCount < MAX_REF_RETRY_FRAMES) {
          retryCount += 1;
          retryRafId = requestAnimationFrame(tryAttach);
        }
        return;
      }
      retryRafId = null;
      detachVideo = attach(video);
    };

    tryAttach();

    return () => {
      cancelled = true;
      if (retryRafId != null) {
        cancelAnimationFrame(retryRafId);
        retryRafId = null;
      }
      stop();
      detachVideo?.();
      detachVideo = null;
    };
  }, [videoRef, fps, maxFrameIndex]);
}
