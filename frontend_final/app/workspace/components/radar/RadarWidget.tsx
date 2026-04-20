import { RefObject, useEffect, useMemo, useState } from "react";
import { useVideoFrameSync } from "@/hooks/useVideoFrameSync";
import { adaptTrackingPayload, buildFrameLookup } from "@/lib/trackingAdapter";
import type { TrackingFrame, TrackingPayload } from "@/lib/types/trackingTypes";
import { debugSessionLog } from "@/lib/debugSessionLog";
import RadarCanvas from "./RadarCanvas";

interface RadarWidgetProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  trackingData: TrackingPayload | null;
  onFrameChange?: (frameIndex: number, frame: TrackingFrame | undefined) => void;
}

export default function RadarWidget({
  videoRef,
  trackingData,
  onFrameChange,
}: RadarWidgetProps) {
  const adaptedTracking = useMemo(
    () => (trackingData ? adaptTrackingPayload(trackingData) : null),
    [trackingData],
  );
  const lookup = useMemo(
    () => (adaptedTracking ? buildFrameLookup(adaptedTracking) : null),
    [adaptedTracking],
  );
  const maxFrameIndex = useMemo(() => {
    if (!lookup) return 0;
    const n = lookup.totalFrames;
    const tel = adaptedTracking?.telemetry?.total_frames_processed;
    if (typeof tel === "number" && Number.isFinite(tel) && tel > 0) {
      return Math.max(0, Math.min(Math.floor(tel), n) - 1);
    }
    return Math.max(0, n - 1);
  }, [adaptedTracking, lookup]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [syncFps, setSyncFps] = useState(30);

  useEffect(() => {
    const raf = window.requestAnimationFrame(() => {
      setCurrentFrame(0);
    });
    return () => window.cancelAnimationFrame(raf);
  }, [adaptedTracking]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !adaptedTracking) return;
    const totalFrames =
      adaptedTracking.telemetry?.total_frames_processed ??
      adaptedTracking.frames.length;
    const onLoadedMetadata = (): void => {
      if (video.duration > 0 && totalFrames > 0) {
        const derived = totalFrames / video.duration;
        setSyncFps(Number.isFinite(derived) && derived > 0 ? derived : 30);
      } else {
        setSyncFps(30);
      }
    };
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    // Metadata may already be available (blob URL warmed before Radar mounts).
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      onLoadedMetadata();
    }
    return () => video.removeEventListener("loadedmetadata", onLoadedMetadata);
  }, [videoRef, adaptedTracking]);

  useVideoFrameSync({
    videoRef,
    fps: syncFps,
    maxFrameIndex,
    onFrameChange: setCurrentFrame,
  });

  const frame = lookup?.getFrameByIndex(currentFrame);

  useEffect(() => {
    onFrameChange?.(currentFrame, frame);
  }, [currentFrame, frame, onFrameChange]);

  useEffect(() => {
    // #region agent log
    debugSessionLog({
      sessionId: "bb63ae",
      hypothesisId: "H1",
      location: "RadarWidget.tsx:mount",
      message: "RadarWidget mounted",
      data: {
        hasTrackingPayload: Boolean(trackingData),
        adaptedOk: Boolean(adaptedTracking),
        totalFrames: lookup?.totalFrames ?? 0,
      },
    });
    // #endregion
  }, [trackingData, adaptedTracking, lookup?.totalFrames]);

  return (
    <div data-testid="tactical-radar" className="rounded-lg border border-gray-800 bg-[#0a0f0a] p-3 shadow-sm">
      <h2 className="mb-2 text-sm font-semibold text-gray-200">Tactical Radar</h2>
      <RadarCanvas frame={frame} />
      <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
        <span>Frame {currentFrame}</span>
        <span>Sync FPS {syncFps.toFixed(2)}</span>
      </div>
    </div>
  );
}

